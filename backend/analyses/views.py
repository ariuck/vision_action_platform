"""DRF API: 분석 요청 CRUD와 상태 조회.

- 일반 회원은 자기 작업만 보고 수정·삭제할 수 있다 (다른 사람 것은 404).
- 관리자(is_staff)는 전체 작업을 조회할 수 있다.
- 생성 시 DB 저장이 확정된 뒤(on_commit) Celery 작업을 등록하고 작업 ID를 바로 돌려준다.
"""
from django.db import transaction
from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import AnalysisJob
from .serializers import AnalysisJobSerializer, AnalysisJobUpdateSerializer, AnalysisStatusSerializer
from .tasks import run_analysis


class AnalysisJobViewSet(mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin,
                         mixins.UpdateModelMixin, mixins.DestroyModelMixin, viewsets.GenericViewSet):
    serializer_class = AnalysisJobSerializer

    def get_queryset(self):
        qs = AnalysisJob.objects.select_related("user").prefetch_related("predictions")
        if self.request.user.is_staff:
            return qs
        return qs.filter(user=self.request.user)

    def get_serializer_class(self):
        if self.action in ("update", "partial_update"):
            return AnalysisJobUpdateSerializer
        if self.action == "status":
            return AnalysisStatusSerializer
        return AnalysisJobSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        with transaction.atomic():
            job = serializer.save(user=request.user)
            # 커밋 전에 Worker가 먼저 실행되면 작업을 못 찾으므로 커밋 뒤에 등록한다.
            transaction.on_commit(lambda: self._enqueue(job.pk))
        job.refresh_from_db()
        return Response({"id": job.pk, "status": job.status, "celery_task_id": job.celery_task_id,
                         "status_url": f"/api/v1/analyses/{job.pk}/status/"}, status=status.HTTP_202_ACCEPTED)

    @staticmethod
    def _enqueue(job_id: int) -> None:
        try:
            result = run_analysis.delay(job_id)
        except Exception as exc:  # Redis가 꺼져 있으면 작업을 실패로 남긴다
            AnalysisJob.objects.filter(pk=job_id).update(
                status=AnalysisJob.Status.FAILURE, stage=AnalysisJob.Stage.FAILED, error_code="QUEUE_UNAVAILABLE",
                error_message=f"작업 대기열(Redis)에 등록하지 못했습니다: {exc}")
            return
        AnalysisJob.objects.filter(pk=job_id, celery_task_id="").update(celery_task_id=result.id or "")

    def perform_destroy(self, instance):
        # 사람별 예측은 CASCADE로 함께 지워지고, 영상 파일도 저장소에서 지운다.
        if instance.video_file:
            instance.video_file.delete(save=False)
        instance.delete()

    @action(detail=True, methods=["get"])
    def status(self, request, pk=None):
        return Response(AnalysisStatusSerializer(self.get_object()).data)
