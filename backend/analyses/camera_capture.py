"""허용된 IP Webcam에서 정해진 시간만 Frame을 읽어 MP4로 저장한다 (Celery Worker가 실행).

- 카메라 주소는 settings.ALLOWED_CAMERAS에 등록된 것만 쓴다. 사용자가 임의의 URL을 넣을 수 없다.
- 연결이 안 되거나 Frame이 하나도 안 들어오면 CameraError로 실패 사유를 남긴다.
- 저장 FPS는 실제로 받은 Frame 수 ÷ 수집 시간으로 정한다 (네트워크 때문에 원래 FPS보다 낮을 수 있음).
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
from django.conf import settings


class CameraError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def camera_url(name: str) -> str:
    try:
        return settings.ALLOWED_CAMERAS[name]
    except KeyError as exc:
        raise CameraError("CAMERA_NOT_ALLOWED", f"허용되지 않은 카메라입니다: {name}") from exc


def capture_clip(name: str, seconds: float, out_path: Path) -> dict:
    url = camera_url(name)
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, settings.CAMERA_CONNECT_TIMEOUT * 1000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, settings.CAMERA_CONNECT_TIMEOUT * 1000)
    if not cap.isOpened():
        raise CameraError("CAMERA_UNREACHABLE", "카메라에 연결할 수 없습니다. 폰 카메라 앱이 켜져 있는지 확인하세요.")

    frames = []
    started = time.perf_counter()
    while time.perf_counter() - started < seconds:
        ok, frame = cap.read()
        if not ok:
            if not frames:
                break
            continue
        frames.append(frame)
    elapsed = time.perf_counter() - started
    cap.release()
    if not frames:
        raise CameraError("CAMERA_NO_FRAME", "카메라에서 영상이 들어오지 않았습니다. 연결 상태를 확인하세요.")
    if elapsed < seconds * 0.5:
        raise CameraError("CAMERA_DISCONNECTED", f"수집 중 연결이 끊겼습니다 ({elapsed:.1f}초 수집).")

    fps = len(frames) / elapsed
    height, width = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return {"frames": len(frames), "fps": round(fps, 2), "seconds": round(elapsed, 2), "width": width, "height": height}
