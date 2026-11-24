"""사용할 수 있는 카메라 목록 = .env의 ALLOWED_CAMERAS + 화면에서 등록한 Camera.

주소를 아무거나 받으면 서버가 내부 서비스(DB·Redis 등)나 외부 사이트로 접속하는 통로가 될 수 있어서
등록할 때 아래 조건을 검사한다.
  - http 주소만
  - 호스트는 IP 숫자로, CAMERA_ALLOWED_NETWORKS(기본 192.168.0.0/16) 안에 있어야 함
    → 같은 Wi-Fi의 휴대폰은 되고, 인터넷 주소·컨테이너 내부망은 막힌다
"""
from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

from django.conf import settings

NAME_RE = re.compile(r"^[0-9A-Za-z가-힣_-]{1,30}$")


def all_cameras() -> dict[str, str]:
    from .models import Camera

    cameras = dict(settings.ALLOWED_CAMERAS)
    for cam in Camera.objects.all():
        cameras.setdefault(cam.name, cam.url)
    return cameras


def get_url(name: str) -> str | None:
    if name in settings.ALLOWED_CAMERAS:
        return settings.ALLOWED_CAMERAS[name]
    from .models import Camera

    cam = Camera.objects.filter(name=name).only("url").first()
    return cam.url if cam else None


def build_url(ip: str, port: int, path: str) -> str:
    path = "/" + (path or "video").lstrip("/")
    return f"http://{ip}:{port}{path}"


def validate_camera(name: str, ip: str, port: int, path: str) -> list[str]:
    errors = []
    if not NAME_RE.match(name or ""):
        errors.append("이름은 한글·영문·숫자·_- 로 30자 이내여야 합니다.")
    elif name in settings.ALLOWED_CAMERAS:
        errors.append("이미 .env에 등록된 이름입니다.")
    try:
        addr = ipaddress.ip_address(ip)
        networks = [ipaddress.ip_network(n) for n in settings.CAMERA_ALLOWED_NETWORKS]
        if not any(addr in n for n in networks):
            errors.append(f"허용된 내부망({', '.join(settings.CAMERA_ALLOWED_NETWORKS)}) 주소만 등록할 수 있습니다.")
    except ValueError:
        errors.append("IP는 192.168.0.10 같은 숫자 주소로 입력해 주세요.")
    if not (1 <= int(port or 0) <= 65535):
        errors.append("포트 번호가 올바르지 않습니다.")
    if path and not re.match(r"^/?[0-9A-Za-z_./-]{0,60}$", path):
        errors.append("경로에는 영문·숫자·/ . _ - 만 쓸 수 있습니다.")
    return errors


def describe(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.hostname}:{parts.port}{parts.path}"
