from django.contrib import admin

from .models import AnalysisJob, Camera, PersonPrediction


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ["name", "url", "created_by", "created_at"]
    search_fields = ["name", "url"]


class PersonPredictionInline(admin.TabularInline):
    model = PersonPrediction
    extra = 0
    fields = ["track_id", "action_id", "action_name", "confidence", "uncertain", "valid_frames", "first_sec", "last_sec"]
    readonly_fields = fields
    can_delete = False


@admin.register(AnalysisJob)
class AnalysisJobAdmin(admin.ModelAdmin):
    list_display = ["id", "user", "input_type", "status", "result_code", "expected_action", "people",
                    "total_processing_sec", "created_at"]
    list_filter = ["status", "input_type", "result_code", "created_at"]
    search_fields = ["user__username", "title", "memo", "camera_name", "error_code"]
    readonly_fields = ["celery_task_id", "started_at", "finished_at", "created_at", "video_fps", "video_frames",
                       "video_duration_sec", "ai_processing_sec", "total_processing_sec"]
    inlines = [PersonPredictionInline]

    @admin.display(description="사람 수")
    def people(self, obj):
        return obj.predictions.count()


@admin.register(PersonPrediction)
class PersonPredictionAdmin(admin.ModelAdmin):
    list_display = ["id", "job", "track_id", "action_id", "action_name", "confidence", "uncertain", "created_at"]
    list_filter = ["action_label", "uncertain", "job__input_type"]
    search_fields = ["job__user__username", "action_label", "action_name"]
