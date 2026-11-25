"""Celery 작업: 영상 확보 → FastAPI 추론 요청 → Django ORM으로 결과 저장.

상태 변화: PENDING(등록) → STARTED(처리 중) → SUCCESS / FAILURE
실패해도 원인(error_code, error_message)을 DB에 남겨 화면에서 보여 준다.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from celery import shared_task
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from .camera_hub import CameraError, capture_clip
from .models import AnalysisJob, PersonPrediction
from .services import AIServiceError, request_prediction


def _set_stage(job: AnalysisJob, stage: str, **extra) -> None:
    job.stage = stage
    for k, v in extra.items():
        setattr(job, k, v)
    job.save(update_fields=["stage", *extra])


def _fail(job: AnalysisJob, code: str, message: str, started: float) -> None:
    job.status = AnalysisJob.Status.FAILURE
    job.stage = AnalysisJob.Stage.FAILED
    job.error_code = code
    job.error_message = message
    job.finished_at = timezone.now()
    job.total_processing_sec = round(time.perf_counter() - started, 3)
    job.save()


@shared_task(bind=True)
def run_analysis(self, job_id: int) -> str:
    started = time.perf_counter()
    job = AnalysisJob.objects.get(pk=job_id)
    job.status = AnalysisJob.Status.STARTED
    job.started_at = timezone.now()
    job.celery_task_id = job.celery_task_id or (self.request.id or "")
    job.save(update_fields=["status", "started_at", "celery_task_id"])

    # 1) 분석할 영상 확보
    try:
        if job.input_type == AnalysisJob.InputType.CAMERA:
            seconds = job.capture_seconds or settings.CAMERA_CAPTURE_SECONDS
            rel = Path("videos") / "camera" / f"job{job.pk}_{uuid.uuid4().hex[:8]}.mp4"
            # 카메라 연결이 확인된 순간부터 '영상 수집' 단계로 표시 (화면의 남은 시간 계산 기준)
            info = capture_clip(job.camera_name, seconds, Path(settings.MEDIA_ROOT) / rel,
                                on_start=lambda: _set_stage(job, AnalysisJob.Stage.CAPTURING,
                                                            capture_started_at=timezone.now()))
            job.video_file.name = rel.as_posix()
            job.message = f"카메라에서 {info['seconds']}초 동안 {info['frames']}개 Frame을 받았습니다."
            job.save(update_fields=["video_file", "message"])
        if not job.video_file:
            _fail(job, "NO_VIDEO", "분석할 영상이 없습니다.", started)
            return "FAILURE"
        video_path = Path(job.video_file.path)
        if not video_path.exists():
            _fail(job, "VIDEO_MISSING", "저장된 영상 파일을 찾을 수 없습니다.", started)
            return "FAILURE"
    except CameraError as exc:
        _fail(job, exc.code, str(exc), started)
        return "FAILURE"

    # 2) FastAPI 추론 요청
    _set_stage(job, AnalysisJob.Stage.ANALYZING)
    try:
        result = request_prediction(video_path)
    except AIServiceError as exc:
        _fail(job, exc.code, str(exc), started)
        return "FAILURE"

    # 3) Django ORM으로 저장 (작업 상태 + 사람별 예측을 한 번에)
    video = result.get("video", {})
    with transaction.atomic():
        job.predictions.all().delete()
        PersonPrediction.objects.bulk_create([
            PersonPrediction(
                job=job, track_id=p["track_id"], action_id=p["action_id"], action_label=p["action_label"],
                action_name=p["action_name"], confidence=p["confidence"], uncertain=p["uncertain"],
                no_action=p.get("no_action", False), is_main=p.get("is_main", True),
                motion_level=p.get("motion_level"), height_ratio=p.get("height_ratio"),
                action_group=p.get("action_group", ""), group_scores=p.get("group_scores", {}),
                action_start_sec=p.get("action_start_sec"), action_end_sec=p.get("action_end_sec"),
                top3=p.get("top3", []), frames=p.get("frames", 0), valid_frames=p.get("valid_frames", 0),
                first_sec=p.get("first_sec"), last_sec=p.get("last_sec"))
            for p in result.get("predictions", [])
        ])
        job.status = AnalysisJob.Status.SUCCESS
        job.stage = AnalysisJob.Stage.DONE
        job.result_code = result.get("status", "")
        job.message = (job.message + " " if job.message else "") + result.get("message", "")
        job.video_fps = video.get("fps")
        job.video_frames = video.get("frames")
        job.video_duration_sec = video.get("duration_sec")
        job.ai_processing_sec = result.get("processing_sec")
        job.finished_at = timezone.now()
        job.total_processing_sec = round(time.perf_counter() - started, 3)
        job.save()
    return "SUCCESS"
