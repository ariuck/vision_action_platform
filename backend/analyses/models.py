"""분석 작업(AnalysisJob)과 사람별 예측(PersonPrediction).

- 회원(auth_user) 1명 → 분석 작업 여러 개
- 분석 작업 1개 → 사람(Track ID)별 예측 여러 개
- 영상 파일은 Media 폴더에 두고 DB에는 경로만 저장한다.
- track_id는 영상 한 개 안에서만 의미 있는 추적 번호다 (실제 사람 ID 아님).
- expected_action은 사용자가 '하려던 동작'으로 고른 값일 뿐, 검증된 정답이 아니다.
"""
from django.conf import settings
from django.db import models

from .actions import ACTION_CHOICES


class AnalysisJob(models.Model):
    class InputType(models.TextChoices):
        UPLOAD = "UPLOAD", "영상 업로드"
        CAMERA = "CAMERA", "IP Webcam"

    class Status(models.TextChoices):
        PENDING = "PENDING", "대기"
        STARTED = "STARTED", "처리 중"
        SUCCESS = "SUCCESS", "완료"
        FAILURE = "FAILURE", "실패"

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="analysis_jobs")
    title = models.CharField("제목", max_length=100, blank=True)
    memo = models.TextField("메모", blank=True)
    input_type = models.CharField("입력 방식", max_length=10, choices=InputType.choices)
    video_file = models.FileField("영상 파일", upload_to="videos/%Y/%m/", blank=True)
    camera_name = models.CharField("카메라", max_length=50, blank=True)
    capture_seconds = models.FloatField("수집 시간(초)", null=True, blank=True)
    expected_action = models.CharField("수행한 동작(선택)", max_length=40, choices=ACTION_CHOICES, blank=True)

    class Stage(models.TextChoices):
        # 화면 안내용 세부 단계 (status는 Celery 작업 상태 그대로 둔다)
        QUEUED = "QUEUED", "대기"
        CAPTURING = "CAPTURING", "영상 수집"
        ANALYZING = "ANALYZING", "AI 분석"
        DONE = "DONE", "완료"
        FAILED = "FAILED", "실패"

    status = models.CharField("상태", max_length=10, choices=Status.choices, default=Status.PENDING, db_index=True)
    stage = models.CharField("진행 단계", max_length=10, choices=Stage.choices, default=Stage.QUEUED)
    capture_started_at = models.DateTimeField("녹화 시작", null=True, blank=True)
    celery_task_id = models.CharField(max_length=64, blank=True)
    result_code = models.CharField("결과 구분", max_length=30, blank=True)  # ok / no_person / no_prediction
    message = models.CharField("결과 메시지", max_length=255, blank=True)
    error_code = models.CharField("오류 코드", max_length=40, blank=True)
    error_message = models.TextField("오류 내용", blank=True)

    video_fps = models.FloatField(null=True, blank=True)
    video_frames = models.IntegerField(null=True, blank=True)
    video_duration_sec = models.FloatField(null=True, blank=True)
    ai_processing_sec = models.FloatField("AI 추론 시간(초)", null=True, blank=True)
    total_processing_sec = models.FloatField("전체 처리 시간(초)", null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"#{self.pk} {self.get_input_type_display()} {self.get_status_display()}"

    @property
    def is_finished(self) -> bool:
        return self.status in (self.Status.SUCCESS, self.Status.FAILURE)


class PersonPrediction(models.Model):
    job = models.ForeignKey(AnalysisJob, on_delete=models.CASCADE, related_name="predictions")
    track_id = models.IntegerField("Track ID")
    action_id = models.CharField(max_length=10)
    action_label = models.CharField(max_length=40, db_index=True)
    action_name = models.CharField(max_length=40)
    confidence = models.FloatField()
    uncertain = models.BooleanField("판단 불확실", default=False)
    top3 = models.JSONField(default=list, blank=True)
    frames = models.IntegerField("보인 Frame 수", default=0)
    valid_frames = models.IntegerField("예측에 쓴 Frame 수", default=0)
    first_sec = models.FloatField(null=True, blank=True)
    last_sec = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["job", "track_id"]
        constraints = [models.UniqueConstraint(fields=["job", "track_id"], name="unique_track_per_job")]

    def __str__(self):
        return f"job#{self.job_id} track {self.track_id} {self.action_label} {self.confidence:.2f}"
