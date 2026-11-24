from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import camera_views, page_views
from .views import AnalysisJobViewSet

app_name = "analyses"

router = DefaultRouter()
router.register("analyses", AnalysisJobViewSet, basename="analysis")

urlpatterns = [
    path("", page_views.index, name="index"),
    path("dashboard/", page_views.dashboard, name="dashboard"),
    path("api/", include(router.urls)),
    path("api/cameras/", camera_views.camera_list, name="camera_list"),
    path("api/cameras/<str:name>/", camera_views.camera_delete, name="camera_delete"),
    path("cameras/<str:name>/stream/", camera_views.camera_stream, name="camera_stream"),
    path("cameras/<str:name>/status/", camera_views.camera_status, name="camera_status"),
    path("cameras/<str:name>/pose/", camera_views.camera_pose, name="camera_pose"),
]
