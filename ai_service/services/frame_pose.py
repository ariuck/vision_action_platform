"""카메라 미리보기용: 사진 1장에서 사람별 관절 17개 위치만 구한다 (동작 예측은 하지 않음).

영상 분석에 쓰는 YOLO 객체는 Tracker 상태를 들고 있어서, 미리보기용으로 따로 하나 더 불러 섞이지 않게 한다.
"""
from __future__ import annotations

import time
from threading import Lock

import cv2
import numpy as np

from .model_loader import MODEL_DIR, ModelBundle

_live_model = None
_live_lock = Lock()


def _model(bundle: ModelBundle):
    global _live_model
    if _live_model is None:
        from ultralytics import YOLO
        _live_model = YOLO(str(MODEL_DIR / bundle.project_config["pose_model"]))
    return _live_model


def pose_from_jpeg(data: bytes, bundle: ModelBundle) -> dict:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("이미지를 읽을 수 없습니다.")
    h, w = image.shape[:2]
    cfg = bundle.project_config
    started = time.perf_counter()
    with _live_lock:
        result = _model(bundle).predict(image, imgsz=480, conf=float(cfg["person_confidence"]),
                                        device=bundle.device, verbose=False)[0]
    people = []
    if result.keypoints is not None and result.boxes is not None:
        kxy = result.keypoints.xy.cpu().numpy()
        kconf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else np.ones(kxy.shape[:2])
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        for i in range(len(boxes)):
            people.append({
                "box": [round(float(v) / s, 4) for v, s in zip(boxes[i], (w, h, w, h))],
                "score": round(float(scores[i]), 3),
                # 화면 크기와 상관없이 그릴 수 있게 0~1 비율로 보낸다
                "keypoints": [[round(float(x) / w, 4), round(float(y) / h, 4), round(float(c), 3)]
                              for (x, y), c in zip(kxy[i], kconf[i])],
            })
    return {"width": w, "height": h, "people": people,
            "processing_ms": round((time.perf_counter() - started) * 1000, 1)}
