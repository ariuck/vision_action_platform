"""AI 추론 서버. 영상을 받아 사람별 동작 예측 JSON만 돌려준다 (DB에는 접근하지 않음)."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from routers.predictions import router as prediction_router
from services.action_inference import UNCERTAIN_BELOW
from services.model_loader import load_bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 모델을 한 번만 불러와 모든 요청에서 재사용한다.
    app.state.bundle = None
    app.state.load_error = None
    try:
        app.state.bundle = load_bundle()
    except Exception as exc:  # 모델이 없어도 서버는 떠서 /health로 원인을 알려 준다
        app.state.load_error = str(exc)
    yield


app = FastAPI(title="Vision Action AI Service", version="1.0.0", lifespan=lifespan)
app.include_router(prediction_router)


@app.get("/health")
def health():
    bundle = app.state.bundle
    if bundle is None:
        return {"status": "error", "model_ready": False, "error": app.state.load_error}
    return {
        "status": "ok",
        "model_ready": True,
        "classes": len(bundle.classes),
        "feature_count": len(bundle.feature_columns),
        "frame_feature_count": bundle.frame_feature_count,
        "sequence_steps": bundle.sequence_steps,
        "train_clips": bundle.training_info.get("train_clips"),
        "pose_model": bundle.project_config["pose_model"],
        "device": bundle.device,
        "uncertain_below": UNCERTAIN_BELOW,
    }
