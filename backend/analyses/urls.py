"""주소는 교재 개발 가이드의 권장 규칙에 맞춘다.

화면:  /analyses/ (사용자 분석), /dashboard/ (관리자)
API:   /api/v1/analyses/ (작업 생성·목록·상세·수정·삭제), /api/v1/analyses/<id>/status/ (상태)
       /api/v1/cameras/ (카메라 등록·목록)
카메라 미리보기: /cameras/<이름>/stream|status|pose/ (브라우저 화면용)
"""
from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import camera_views, page_views
from .views import AnalysisJobViewSet

app_name = "analyses"

router = DefaultRouter()
router.register("analyses", AnalysisJobViewSet, basename="analysis")

urlpatterns = [
    path("analyses/", page_views.index, name="index"),
    path("dashboard/", page_views.dashboard, name="dashboard"),
    path("api/v1/cameras/", camera_views.camera_list, name="camera_list"),
    path("api/v1/cameras/<str:name>/", camera_views.camera_delete, name="camera_delete"),
    path("api/v1/", include(router.urls)),
    path("cameras/<str:name>/stream/", camera_views.camera_stream, name="camera_stream"),
    path("cameras/<str:name>/status/", camera_views.camera_status, name="camera_status"),
    path("cameras/<str:name>/pose/", camera_views.camera_pose, name="camera_pose"),
]
