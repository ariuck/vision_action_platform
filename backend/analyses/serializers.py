"""입력 검증과 응답 JSON 모양.

- 생성: 영상 업로드는 video_file 필수, IP Webcam은 허용된 camera_name 필수
- 수정: 제목·메모만 바꿀 수 있다 (상태·예측 결과는 읽기 전용)
"""
from pathlib import Path

from django.conf import settings
from rest_framework import serializers

from .models import AnalysisJob, PersonPrediction

ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class PersonPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = PersonPrediction
        fields = ["track_id", "action_id", "action_label", "action_name", "confidence", "uncertain", "no_action",
                  "is_main", "motion_level", "height_ratio", "top3", "frames", "valid_frames", "first_sec", "last_sec"]


class AnalysisJobSerializer(serializers.ModelSerializer):
    predictions = PersonPredictionSerializer(many=True, read_only=True)
    input_type_display = serializers.CharField(source="get_input_type_display", read_only=True)
    status_display = serializers.CharField(source="get_status_display", read_only=True)
    expected_action_display = serializers.CharField(source="get_expected_action_display", read_only=True)
    username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = AnalysisJob
        fields = ["id", "username", "title", "memo", "input_type", "input_type_display", "video_file",
                  "camera_name", "capture_seconds", "expected_action", "expected_action_display", "status",
                  "status_display", "stage", "celery_task_id", "result_code", "message", "error_code", "error_message",
                  "video_fps", "video_frames", "video_duration_sec", "ai_processing_sec", "total_processing_sec",
                  "created_at", "started_at", "finished_at", "predictions"]
        read_only_fields = ["status", "stage", "celery_task_id", "result_code", "message", "error_code", "error_message",
                            "video_fps", "video_frames", "video_duration_sec", "ai_processing_sec",
                            "total_processing_sec", "created_at", "started_at", "finished_at", "capture_seconds"]
        extra_kwargs = {"video_file": {"required": False}}

    def validate(self, attrs):
        input_type = attrs.get("input_type")
        if input_type == AnalysisJob.InputType.UPLOAD:
            video = attrs.get("video_file")
            if not video:
                raise serializers.ValidationError({"video_file": "분석할 영상 파일을 올려 주세요."})
            if Path(video.name).suffix.lower() not in ALLOWED_EXT:
                raise serializers.ValidationError({"video_file": "MP4·MOV·AVI·MKV·WEBM 영상만 올릴 수 있습니다."})
            if video.size > settings.MAX_UPLOAD_MB * 1024 * 1024:
                raise serializers.ValidationError({"video_file": f"영상은 {settings.MAX_UPLOAD_MB}MB 이하만 됩니다."})
            attrs["camera_name"] = ""
        elif input_type == AnalysisJob.InputType.CAMERA:
            from .camera_registry import get_url

            name = attrs.get("camera_name", "")
            if not name or not get_url(name):
                raise serializers.ValidationError({"camera_name": "등록된 카메라 중에서 골라 주세요."})
            attrs["video_file"] = None
            attrs["capture_seconds"] = settings.CAMERA_CAPTURE_SECONDS
        return attrs


class AnalysisJobUpdateSerializer(serializers.ModelSerializer):
    """PATCH/PUT: 제목·메모만 수정"""

    class Meta:
        model = AnalysisJob
        fields = ["title", "memo"]


class AnalysisStatusSerializer(serializers.ModelSerializer):
    status_display = serializers.CharField(source="get_status_display", read_only=True)
    stage_display = serializers.CharField(source="get_stage_display", read_only=True)
    is_finished = serializers.BooleanField(read_only=True)
    server_time = serializers.SerializerMethodField()  # 녹화 남은 시간을 서버 시계 기준으로 계산하기 위해

    class Meta:
        model = AnalysisJob
        fields = ["id", "input_type", "status", "status_display", "stage", "stage_display", "is_finished",
                  "capture_seconds", "capture_started_at", "server_time", "result_code", "message",
                  "error_code", "error_message", "started_at", "finished_at"]

    def get_server_time(self, obj):
        from django.utils import timezone
        return timezone.now().isoformat()
