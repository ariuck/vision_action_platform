"""API·권한·Celery 작업 테스트 (FastAPI와 Redis 없이 돌도록 가짜 응답으로 대체)."""
import shutil
import tempfile
from unittest import mock

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from rest_framework.test import APIClient

from .models import AnalysisJob, PersonPrediction
from .services import AIServiceError
from .tasks import run_analysis

User = get_user_model()
TMP_MEDIA = tempfile.mkdtemp(prefix="vap_test_media_")

FAKE_RESULT = {
    "status": "ok",
    "message": "2명 예측",
    "processing_sec": 1.5,
    "video": {"fps": 30.0, "frames": 90, "duration_sec": 3.0},
    "predictions": [
        {"track_id": 1, "action_id": "A007", "action_label": "sit_floor", "action_name": "바닥에 앉기",
         "confidence": 0.72, "uncertain": False, "frames": 90, "valid_frames": 88, "first_sec": 0.0,
         "last_sec": 3.0, "top3": [{"action_label": "sit_floor", "action_name": "바닥에 앉기", "confidence": 0.72}]},
        {"track_id": 2, "action_id": "A016", "action_label": "fall_backward", "action_name": "뒤로 넘어지기",
         "confidence": 0.15, "uncertain": True, "frames": 40, "valid_frames": 35, "first_sec": 1.0,
         "last_sec": 2.3, "top3": []},
    ],
}


def video(name="clip.mp4", size=1024):
    return SimpleUploadedFile(name, b"\x00" * size, content_type="video/mp4")


@override_settings(MEDIA_ROOT=TMP_MEDIA, ALLOWED_CAMERAS={"phone": "http://cam.test/video"})
class AnalysisApiTests(TestCase):
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TMP_MEDIA, ignore_errors=True)

    def setUp(self):
        self.alice = User.objects.create_user("alice", password="pw-alice-123")
        self.bob = User.objects.create_user("bob", password="pw-bob-123")
        self.admin = User.objects.create_user("admin1", password="pw-admin-123", is_staff=True)
        self.client = APIClient()
        self.delay = mock.patch("analyses.views.run_analysis.delay",
                                return_value=mock.Mock(id="task-123")).start()
        self.addCleanup(mock.patch.stopall)

    def make_job(self, user, **kw):
        return AnalysisJob.objects.create(user=user, input_type="UPLOAD", video_file=video(), **kw)

    # 로그인하지 않으면 API를 쓸 수 없다
    def test_anonymous_is_rejected(self):
        res = self.client.get("/api/analyses/")
        self.assertIn(res.status_code, (401, 403))

    # 업로드하면 202와 작업 ID를 바로 받고, 커밋 뒤 Celery 작업이 등록된다
    def test_create_upload_returns_202_and_enqueues(self):
        self.client.force_authenticate(self.alice)
        with self.captureOnCommitCallbacks(execute=True):
            res = self.client.post("/api/analyses/", {"input_type": "UPLOAD", "title": "테스트",
                                                      "video_file": video()}, format="multipart")
        self.assertEqual(res.status_code, 202)
        job = AnalysisJob.objects.get(pk=res.data["id"])
        self.assertEqual(job.status, "PENDING")
        self.assertEqual(job.celery_task_id, "task-123")
        self.delay.assert_called_once_with(job.pk)
        self.assertEqual(res.data["status_url"], f"/api/analyses/{job.pk}/status/")

    def test_upload_rejects_wrong_extension(self):
        self.client.force_authenticate(self.alice)
        res = self.client.post("/api/analyses/", {"input_type": "UPLOAD", "video_file": video("a.txt")},
                               format="multipart")
        self.assertEqual(res.status_code, 400)
        self.assertIn("video_file", res.data)

    def test_upload_requires_file(self):
        self.client.force_authenticate(self.alice)
        res = self.client.post("/api/analyses/", {"input_type": "UPLOAD"}, format="multipart")
        self.assertEqual(res.status_code, 400)

    def test_camera_must_be_whitelisted(self):
        self.client.force_authenticate(self.alice)
        bad = self.client.post("/api/analyses/", {"input_type": "CAMERA", "camera_name": "http://1.2.3.4"})
        self.assertEqual(bad.status_code, 400)
        with self.captureOnCommitCallbacks(execute=True):
            ok = self.client.post("/api/analyses/", {"input_type": "CAMERA", "camera_name": "phone"})
        self.assertEqual(ok.status_code, 202)

    # Redis가 꺼져 있으면 작업을 FAILURE로 남긴다
    def test_queue_down_marks_failure(self):
        self.delay.side_effect = ConnectionError("redis down")
        self.client.force_authenticate(self.alice)
        with self.captureOnCommitCallbacks(execute=True):
            res = self.client.post("/api/analyses/", {"input_type": "UPLOAD", "video_file": video()},
                                   format="multipart")
        job = AnalysisJob.objects.get(pk=res.data["id"])
        self.assertEqual(job.status, "FAILURE")
        self.assertEqual(job.error_code, "QUEUE_UNAVAILABLE")

    # 다른 회원의 작업은 보이지 않고 수정·삭제도 안 된다
    def test_user_sees_only_own_jobs(self):
        mine = self.make_job(self.alice)
        other = self.make_job(self.bob)
        self.client.force_authenticate(self.alice)
        ids = [j["id"] for j in self.client.get("/api/analyses/").data["results"]]
        self.assertEqual(ids, [mine.pk])
        self.assertEqual(self.client.get(f"/api/analyses/{other.pk}/").status_code, 404)
        self.assertEqual(self.client.patch(f"/api/analyses/{other.pk}/", {"title": "x"}).status_code, 404)
        self.assertEqual(self.client.delete(f"/api/analyses/{other.pk}/").status_code, 404)

    def test_staff_sees_all_jobs(self):
        self.make_job(self.alice)
        self.make_job(self.bob)
        self.client.force_authenticate(self.admin)
        self.assertEqual(self.client.get("/api/analyses/").data["count"], 2)

    # PATCH는 제목·메모만 바뀌고 상태는 바뀌지 않는다
    def test_patch_changes_only_title_and_memo(self):
        job = self.make_job(self.alice)
        self.client.force_authenticate(self.alice)
        res = self.client.patch(f"/api/analyses/{job.pk}/", {"title": "새 제목", "memo": "메모",
                                                             "status": "SUCCESS"}, format="json")
        self.assertEqual(res.status_code, 200)
        job.refresh_from_db()
        self.assertEqual((job.title, job.memo, job.status), ("새 제목", "메모", "PENDING"))

    def test_status_endpoint(self):
        job = self.make_job(self.alice)
        self.client.force_authenticate(self.alice)
        res = self.client.get(f"/api/analyses/{job.pk}/status/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.data["status"], "PENDING")
        self.assertFalse(res.data["is_finished"])

    def test_delete_removes_predictions(self):
        job = self.make_job(self.alice)
        PersonPrediction.objects.create(job=job, track_id=1, action_id="A007", action_label="sit_floor",
                                        action_name="바닥에 앉기", confidence=0.7)
        self.client.force_authenticate(self.alice)
        self.assertEqual(self.client.delete(f"/api/analyses/{job.pk}/").status_code, 204)
        self.assertFalse(PersonPrediction.objects.exists())


@override_settings(MEDIA_ROOT=TMP_MEDIA)
class RunAnalysisTaskTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("carol", password="pw-carol-123")
        self.job = AnalysisJob.objects.create(user=self.user, input_type="UPLOAD", video_file=video())

    @mock.patch("analyses.tasks.request_prediction", return_value=FAKE_RESULT)
    def test_success_saves_predictions(self, _):
        self.assertEqual(run_analysis.apply(args=[self.job.pk]).get(), "SUCCESS")
        self.job.refresh_from_db()
        self.assertEqual((self.job.status, self.job.stage), ("SUCCESS", "DONE"))
        self.assertEqual(self.job.predictions.count(), 2)
        self.assertTrue(self.job.predictions.get(track_id=2).uncertain)
        self.assertEqual(self.job.video_frames, 90)

    @mock.patch("analyses.tasks.request_prediction", side_effect=AIServiceError("AI_UNREACHABLE", "연결 실패"))
    def test_ai_down_marks_failure(self, _):
        self.assertEqual(run_analysis.apply(args=[self.job.pk]).get(), "FAILURE")
        self.job.refresh_from_db()
        self.assertEqual((self.job.status, self.job.error_code), ("FAILURE", "AI_UNREACHABLE"))
        self.assertEqual(self.job.stage, "FAILED")
        self.assertIsNotNone(self.job.finished_at)


@override_settings(ALLOWED_CAMERAS={"phone": "http://cam.test/video"})
class CameraViewTests(TestCase):
    def setUp(self):
        User.objects.create_user("erin", password="pw-erin-123")

    def test_camera_pages_need_login(self):
        res = self.client.get(reverse("analyses:camera_status", args=["phone"]))
        self.assertEqual(res.status_code, 302)

    def test_unknown_camera_is_404(self):
        # 등록 안 된 이름으로 다른 주소에 접속시키지 못하게 막는다
        self.client.login(username="erin", password="pw-erin-123")
        for view in ("camera_stream", "camera_status", "camera_pose"):
            self.assertEqual(self.client.get(reverse(f"analyses:{view}", args=["other"])).status_code, 404)

    @mock.patch("analyses.camera_views.camera_hub")
    def test_status_reports_live(self, hub):
        hub.latest.return_value = (3, b"jpg")
        hub.last_error.return_value = None
        self.client.login(username="erin", password="pw-erin-123")
        res = self.client.get(reverse("analyses:camera_status", args=["phone"]))
        self.assertEqual(res.json(), {"live": True, "error": None})
        hub.keep_alive.assert_called_once_with("phone")


class PageTests(TestCase):
    def test_index_requires_login(self):
        res = self.client.get(reverse("analyses:index"))
        self.assertEqual(res.status_code, 302)

    def test_dashboard_is_staff_only(self):
        User.objects.create_user("dave", password="pw-dave-123")
        self.client.login(username="dave", password="pw-dave-123")
        self.assertEqual(self.client.get(reverse("analyses:index")).status_code, 200)
        self.assertEqual(self.client.get(reverse("analyses:dashboard")).status_code, 302)
        User.objects.create_user("boss", password="pw-boss-123", is_staff=True)
        self.client.login(username="boss", password="pw-boss-123")
        self.assertEqual(self.client.get(reverse("analyses:dashboard")).status_code, 200)
