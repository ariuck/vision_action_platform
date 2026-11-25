"""HTML 화면: 분석 화면(로그인 회원)과 관리자 대시보드(is_staff만)."""
from django.conf import settings
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from .actions import ACTION_CHOICES, ACTIONS
from .camera_registry import all_cameras
from .models import AnalysisJob
from .stats import dashboard_stats, user_summary


@login_required
def index(request):
    jobs = AnalysisJob.objects.filter(user=request.user).prefetch_related("predictions")[:20]
    return render(request, "analyses/index.html", {
        "jobs": jobs,
        "cameras": list(all_cameras()),
        "capture_seconds": settings.CAMERA_CAPTURE_SECONDS,
        "action_choices": ACTION_CHOICES,
        "summary": user_summary(request.user),
        "action_groups": {label: group for _, label, _, group in ACTIONS},
    })


@staff_member_required
def dashboard(request):
    # 차트 숫자는 Django ORM 집계로 만든다 (FastAPI는 DB에 접근하지 않음).
    stats = dashboard_stats()
    recent = AnalysisJob.objects.select_related("user").prefetch_related("predictions")[:15]
    return render(request, "analyses/dashboard.html", {
        "kpi": stats["kpi"], "labeled": stats["labeled"], "chart": stats["chart"], "recent": recent,
    })
