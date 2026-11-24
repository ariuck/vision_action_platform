"""FastAPI 추론 서버에 영상을 HTTP로 보내고 응답을 받는다.

서로 다른 컨테이너라 파일 경로를 공유한다고 가정하지 않고, 영상 파일 자체를 multipart로 보낸다.
"""
from __future__ import annotations

from pathlib import Path

import requests
from django.conf import settings


class AIServiceError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def request_prediction(video_path: Path) -> dict:
    url = f"{settings.AI_SERVICE_URL.rstrip('/')}/predict/video"
    try:
        with open(video_path, "rb") as f:
            response = requests.post(url, files={"file": (video_path.name, f, "video/mp4")},
                                     timeout=settings.AI_REQUEST_TIMEOUT)
    except requests.Timeout as exc:
        raise AIServiceError("AI_TIMEOUT", f"AI 서버 응답이 {settings.AI_REQUEST_TIMEOUT}초 안에 오지 않았습니다.") from exc
    except requests.ConnectionError as exc:
        raise AIServiceError("AI_UNREACHABLE", "AI 서버에 연결할 수 없습니다.") from exc

    if response.status_code == 200:
        return response.json()
    try:
        detail = response.json().get("detail", {})
    except ValueError:
        detail = {}
    if isinstance(detail, dict):
        raise AIServiceError(detail.get("code", f"AI_HTTP_{response.status_code}"),
                             detail.get("message", response.text[:300]))
    raise AIServiceError(f"AI_HTTP_{response.status_code}", str(detail)[:300])
