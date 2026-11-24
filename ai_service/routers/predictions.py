"""POST /predict/video — Celery Worker가 보낸 영상 1건을 받아 사람별 예측 JSON을 돌려준다."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from services.action_inference import predict_video
from services.pose_pipeline import VideoReadError

router = APIRouter(tags=["predictions"])

ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "300")) * 1024 * 1024


@router.post("/predict/video")
async def predict_video_endpoint(request: Request, file: UploadFile = File(...)):
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail={"code": "MODEL_NOT_READY", "message": "모델이 아직 준비되지 않았습니다."})

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail={
            "code": "UNSUPPORTED_FORMAT", "message": f"지원하지 않는 형식입니다 ({ext or '확장자 없음'}). MP4로 올려 주세요."})

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        size = 0
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_BYTES:
                tmp.close()
                os.unlink(tmp.name)
                raise HTTPException(status_code=413, detail={
                    "code": "FILE_TOO_LARGE", "message": f"영상이 {MAX_BYTES // 1024 // 1024}MB보다 큽니다."})
            tmp.write(chunk)
        path = tmp.name
    try:
        if size == 0:
            raise HTTPException(status_code=400, detail={"code": "EMPTY_FILE", "message": "빈 파일입니다."})
        # YOLO·Random Forest 계산은 오래 걸리므로 이벤트 루프를 막지 않게 별도 스레드에서 실행
        from starlette.concurrency import run_in_threadpool
        result = await run_in_threadpool(predict_video, path, bundle)
    except VideoReadError as exc:
        raise HTTPException(status_code=422, detail={"code": "VIDEO_UNREADABLE", "message": str(exc)}) from exc
    finally:
        if os.path.exists(path):
            os.unlink(path)
    result["filename"] = file.filename
    result["file_bytes"] = size
    return result


@router.post("/pose/frame")
async def pose_frame_endpoint(request: Request, file: UploadFile = File(...)):
    """IP Webcam 미리보기 위에 뼈대를 그리기 위한 사진 1장 관절 추출 (DB 저장 없음)."""
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail={"code": "MODEL_NOT_READY", "message": "모델이 아직 준비되지 않았습니다."})
    data = await file.read(5 * 1024 * 1024 + 1)
    if not data or len(data) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail={"code": "BAD_IMAGE", "message": "5MB 이하 JPEG 한 장을 보내 주세요."})
    from starlette.concurrency import run_in_threadpool
    from services.frame_pose import pose_from_jpeg
    try:
        return await run_in_threadpool(pose_from_jpeg, data, bundle)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"code": "BAD_IMAGE", "message": str(exc)}) from exc
