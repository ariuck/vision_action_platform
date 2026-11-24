"""HTML 화면: 분석 화면(로그인 회원)과 관리자 대시보드(is_staff만)."""
from django.conf import settings
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.decorators import login_required
from django.db.models import Avg, Count
from django.shortcuts import render

from .actions import ACTION_CHOICES, ACTIONS
from .models import AnalysisJob, PersonPrediction


@login_required
def index(request):
    jobs = AnalysisJob.objects.filter(user=request.user).prefetch_related("predictions")[:20]
    return render(request, "analyses/index.html", {
        "jobs": jobs,
        "cameras": list(settings.ALLOWED_CAMERAS),
        "capture_seconds": settings.CAMERA_CAPTURE_SECONDS,
        "action_choices": ACTION_CHOICES,
    })


@staff_member_required
def dashboard(request):
    # 차트 숫자는 Django ORM 집계로 만든다 (FastAPI는 DB에 접근하지 않음).
    by_action = dict(PersonPrediction.objects.values_list("action_label").annotate(n=Count("id")))
    by_status = dict(AnalysisJob.objects.values_list("status").annotate(n=Count("id")))
    by_input = dict(AnalysisJob.objects.values_list("input_type").annotate(n=Count("id")))
    avg_sec = dict(AnalysisJob.objects.filter(status=AnalysisJob.Status.SUCCESS)
                   .values_list("input_type").annotate(s=Avg("total_processing_sec")))
    chart = {
        "actions": {"labels": [f"{aid} {name}" for aid, _, name, _ in ACTIONS],
                    "values": [by_action.get(label, 0) for _, label, _, _ in ACTIONS]},
        "status": {"labels": [s.label for s in AnalysisJob.Status],
                   "values": [by_status.get(s.value, 0) for s in AnalysisJob.Status]},
        "input": {"labels": [t.label for t in AnalysisJob.InputType],
                  "values": [by_input.get(t.value, 0) for t in AnalysisJob.InputType],
                  "avg_sec": [round(avg_sec.get(t.value) or 0, 1) for t in AnalysisJob.InputType]},
    }
    recent = AnalysisJob.objects.select_related("user").prefetch_related("predictions")[:15]
    return render(request, "analyses/dashboard.html", {
        "chart": chart, "recent": recent,
        "total_jobs": AnalysisJob.objects.count(),
        "total_predictions": PersonPrediction.objects.count(),
        "uncertain": PersonPrediction.objects.filter(uncertain=True).count(),
    })
