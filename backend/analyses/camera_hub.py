"""IP Webcam 연결을 하나만 열고, 최신 Frame을 Redis에 올려 여러 곳이 같이 쓰게 한다.

왜 필요한가:
  DroidCam은 동시에 한 곳만 접속할 수 있다 (두 번째 연결은 실패하는 것을 직접 확인함).
  그런데 웹 미리보기(Django)와 녹화(Celery Worker)는 서로 다른 컨테이너에서 같은 카메라가 필요하다.

방식:
  - 카메라를 실제로 읽는 '리더'는 전체에서 하나만 돈다 (Redis 잠금 cam:<이름>:owner).
  - 리더는 읽은 Frame을 JPEG로 바꿔 cam:<이름>:frame 에 계속 덮어쓴다 (앞 8바이트 = Frame 번호).
  - 미리보기·녹화 쪽은 cam:<이름>:want 를 몇 초마다 갱신해 '아직 보고 있다'고 알린다.
    아무도 갱신하지 않으면 리더가 스스로 연결을 닫는다.
"""
from __future__ import annotations

import os
import socket
import struct
import threading
import time
import uuid

import cv2
import redis
from django.conf import settings

OWNER_TTL = 5      # 리더가 죽으면 이 시간 뒤 다른 프로세스가 이어받을 수 있음
WANT_TTL = 6       # 보는 쪽이 사라지고 이 시간이 지나면 연결 종료
FRAME_TTL = 3      # 이 시간 넘게 새 Frame이 없으면 '끊김'으로 본다

_client = None


class CameraError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _redis() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis.from_url(settings.CAMERA_REDIS_URL)
    return _client


def _key(name: str, part: str) -> str:
    return f"cam:{name}:{part}"


def camera_url(name: str) -> str:
    from .camera_registry import get_url

    url = get_url(name)
    if not url:
        raise CameraError("CAMERA_NOT_ALLOWED", f"등록되지 않은 카메라입니다: {name}")
    return url


def keep_alive(name: str) -> None:
    """미리보기·녹화 중이라고 알리고, 리더가 없으면 이 프로세스에서 하나 띄운다."""
    url = camera_url(name)
    r = _redis()
    r.set(_key(name, "want"), 1, ex=WANT_TTL)
    token = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"
    if r.set(_key(name, "owner"), token, nx=True, ex=OWNER_TTL):
        threading.Thread(target=_reader_loop, args=(name, url, token), daemon=True).start()


def latest(name: str) -> tuple[int, bytes | None]:
    data = _redis().get(_key(name, "frame"))
    if not data:
        return -1, None
    return struct.unpack(">Q", data[:8])[0], data[8:]


def last_error(name: str) -> str | None:
    err = _redis().get(_key(name, "error"))
    return err.decode() if err else None


def _reader_loop(name: str, url: str, token: str) -> None:
    r = _redis()
    owner, want, frame_key, err_key = (_key(name, p) for p in ("owner", "want", "frame", "error"))
    timeout_ms = int(settings.CAMERA_CONNECT_TIMEOUT * 1000)  # OpenCV params는 정수만 받는다
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
                                                  cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms])
    try:
        if not cap.isOpened():
            r.set(err_key, "CAMERA_UNREACHABLE", ex=10)
            return
        r.delete(err_key)
        seq, fails = 0, 0
        while r.exists(want):
            if r.get(owner) != token.encode():
                break
            r.expire(owner, OWNER_TTL)
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails > 20:
                    r.set(err_key, "CAMERA_DISCONNECTED", ex=10)
                    break
                time.sleep(0.05)
                continue
            fails = 0
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                seq += 1
                r.set(frame_key, struct.pack(">Q", seq) + jpg.tobytes(), ex=FRAME_TTL)
    finally:
        cap.release()
        if r.get(owner) == token.encode():
            r.delete(owner)


def wait_first_frame(name: str, timeout: float) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        keep_alive(name)
        if latest(name)[1] is not None:
            return
        if last_error(name) == "CAMERA_UNREACHABLE":
            break
        time.sleep(0.2)
    raise CameraError("CAMERA_UNREACHABLE", "카메라에 연결할 수 없습니다. 폰 카메라 앱과 Wi-Fi 연결을 확인하세요.")


def capture_clip(name: str, seconds: float, out_path, on_start=None) -> dict:
    """미리보기와 같은 연결에서 seconds초 동안 Frame을 모아 MP4로 저장한다."""
    wait_first_frame(name, settings.CAMERA_CONNECT_TIMEOUT)
    if on_start:
        on_start()
    jpgs: list[bytes] = []
    last_seq, last_new = -1, time.monotonic()
    started = time.monotonic()
    next_alive = 0.0
    while (now := time.monotonic()) - started < seconds:
        if now >= next_alive:
            keep_alive(name)
            next_alive = now + 1.0
        seq, jpg = latest(name)
        if jpg is not None and seq != last_seq:
            last_seq, last_new = seq, now
            jpgs.append(jpg)
        elif now - last_new > FRAME_TTL:
            raise CameraError("CAMERA_DISCONNECTED", f"수집 중 카메라 연결이 끊겼습니다 ({now - started:.1f}초 수집).")
        time.sleep(0.004)
    elapsed = time.monotonic() - started
    if not jpgs:
        raise CameraError("CAMERA_NO_FRAME", "카메라에서 영상이 들어오지 않았습니다.")

    import numpy as np
    first = cv2.imdecode(np.frombuffer(jpgs[0], np.uint8), cv2.IMREAD_COLOR)
    height, width = first.shape[:2]
    fps = len(jpgs) / elapsed
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for jpg in jpgs:
        writer.write(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR))
    writer.release()
    return {"frames": len(jpgs), "fps": round(fps, 2), "seconds": round(elapsed, 2), "width": width, "height": height}
