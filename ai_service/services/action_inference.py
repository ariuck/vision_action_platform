"""사람별 1,350 Feature → Random Forest 예측 → Track ID별 결과 정리."""
from __future__ import annotations

import os
import time

import numpy as np

from services.pose_pipeline import track_features, track_to_frames, track_video

# 이 값보다 Confidence가 낮으면 화면에 '판단 불확실'로 표시한다.
# 17개 동작이라 표가 나뉘어 Confidence가 원래 낮게 나온다 (5명 교차 평가 평균 약 0.33).
UNCERTAIN_BELOW = float(os.getenv("UNCERTAIN_BELOW", "0.20"))
# 이보다 짧게 보인 Track은 잠깐 생긴 중복 박스로 보고 예측하지 않는다.
MIN_TRACK_SEC = float(os.getenv("MIN_TRACK_SEC", "0.5"))


def predict_video(path: str, bundle) -> dict:
    started = time.perf_counter()
    info, tracks = track_video(path, bundle)
    steps = bundle.sequence_steps
    min_valid = int(bundle.training_info.get("min_valid_frames", 10))
    classes = np.asarray(bundle.classes)

    predictions, skipped = [], []
    for tid, track in sorted(tracks.items()):
        seen_sec = (track.frame_ids[-1] - track.frame_ids[0] + 1) / info.fps
        if seen_sec < MIN_TRACK_SEC:
            skipped.append({"track_id": tid, "reason": f"{seen_sec:.2f}초만 보여 예측하지 않았습니다."})
            continue
        vector, used = track_features(track_to_frames(track), info.fps, steps, min_valid)
        if vector is None:
            skipped.append({"track_id": tid, "reason": used})
            continue
        proba = bundle.model.predict_proba(vector.reshape(1, -1))[0]
        order = np.argsort(proba)[::-1]
        top = classes[order[0]]
        action = bundle.actions[top]
        conf = float(proba[order[0]])
        predictions.append({
            "track_id": tid,
            "action_id": action["action_id"],
            "action_label": top,
            "action_name": action["action_name"],
            "action_group": action.get("group", ""),
            "confidence": round(conf, 4),
            "uncertain": conf < UNCERTAIN_BELOW,
            "top3": [{"action_label": str(classes[j]), "action_name": bundle.actions[classes[j]]["action_name"],
                      "confidence": round(float(proba[j]), 4)} for j in order[:3]],
            "frames": len(track.frame_ids),
            "valid_frames": int(used),
            "first_sec": round(track.frame_ids[0] / info.fps, 3),
            "last_sec": round(track.frame_ids[-1] / info.fps, 3),
        })

    if predictions:
        status, message = "ok", f"{len(predictions)}명의 동작을 예측했습니다."
    elif tracks:
        status, message = "no_prediction", "사람은 찾았지만 예측에 필요한 만큼 관절이 잡히지 않았습니다."
    else:
        status, message = "no_person", "영상에서 사람을 찾지 못했습니다."
    return {
        "status": status,
        "message": message,
        "video": {"fps": round(info.fps, 3), "frames": info.frames, "width": info.width,
                  "height": info.height, "duration_sec": info.duration_sec},
        "predictions": predictions,
        "skipped_tracks": skipped,
        "uncertain_below": UNCERTAIN_BELOW,
        "processing_sec": round(time.perf_counter() - started, 3),
    }
