"""IP Webcam 미리보기용 화면 API (로그인 회원만).

- stream : 카메라 영상을 MJPEG로 브라우저에 보낸다. 카메라 주소는 서버만 알고 화면에는 드러나지 않는다.
- status : 카메라 연결 여부 (미리보기가 안 나올 때 이유 표시용)
- pose   : 지금 Frame의 사람 관절 위치 → 화면에서 뼈대를 그린다 (DB에는 저장하지 않음)
"""
import time

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import Http404, JsonResponse, StreamingHttpResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from . import camera_hub
from .camera_registry import build_url, describe, get_url, validate_camera
from .models import Camera
from .services import AIServiceError, request_frame_pose

STREAM_FPS = 15
STREAM_MAX_SEC = 15 * 60


def _check(name: str) -> None:
    if not get_url(name):
        raise Http404("등록되지 않은 카메라")


@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def camera_list(request):
    """GET: 고를 수 있는 카메라 목록 / POST: IP·포트·경로로 새 카메라 등록"""
    if request.method == "POST":
        name = str(request.data.get("name", "")).strip()
        ip = str(request.data.get("ip", "")).strip()
        path = str(request.data.get("path", "/video")).strip()
        try:
            port = int(request.data.get("port", 4747))
        except (TypeError, ValueError):
            port = 0
        errors = validate_camera(name, ip, port, path)
        if not errors and Camera.objects.filter(name=name).exists():
            errors.append("같은 이름의 카메라가 이미 있습니다.")
        if errors:
            return Response({"errors": errors}, status=400)
        Camera.objects.create(name=name, url=build_url(ip, port, path), created_by=request.user)
        return Response({"name": name}, status=201)

    items = [{"name": n, "address": "(.env 등록)", "deletable": False} for n in settings.ALLOWED_CAMERAS]
    for cam in Camera.objects.select_related("created_by"):
        if cam.name in settings.ALLOWED_CAMERAS:
            continue
        mine = request.user.is_staff or cam.created_by_id == request.user.id
        items.append({"name": cam.name, "address": describe(cam.url) if mine else "-", "deletable": mine})
    return Response({"cameras": items})


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def camera_delete(request, name):
    cam = Camera.objects.filter(name=name).first()
    if cam is None or not (request.user.is_staff or cam.created_by_id == request.user.id):
        raise Http404
    cam.delete()
    return Response(status=204)


@login_required
def camera_stream(request, name):
    _check(name)

    def frames():
        last, end, next_alive = -1, time.monotonic() + STREAM_MAX_SEC, 0.0
        while (now := time.monotonic()) < end:
            if now >= next_alive:
                camera_hub.keep_alive(name)
                next_alive = now + 1.0
            seq, jpg = camera_hub.latest(name)
            if jpg is not None and seq != last:
                last = seq
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            time.sleep(1 / STREAM_FPS)

    response = StreamingHttpResponse(frames(), content_type="multipart/x-mixed-replace; boundary=frame")
    response["Cache-Control"] = "no-store"
    return response


@login_required
def camera_status(request, name):
    _check(name)
    camera_hub.keep_alive(name)
    seq, jpg = camera_hub.latest(name)
    return JsonResponse({"live": jpg is not None, "error": camera_hub.last_error(name)})


@login_required
def camera_pose(request, name):
    _check(name)
    camera_hub.keep_alive(name)
    _, jpg = camera_hub.latest(name)
    if jpg is None:
        return JsonResponse({"people": [], "live": False})
    try:
        data = request_frame_pose(jpg)
    except AIServiceError as exc:
        return JsonResponse({"people": [], "live": True, "error": exc.code}, status=502)
    data["live"] = True
    return JsonResponse(data)
