"""사람별 1,350 Feature → Random Forest 예측 → Track ID별 결과 정리."""
from __future__ import annotations

import os
import time

import numpy as np

from services.pose_pipeline import track_features, track_to_frames, track_video
from services.track_quality import measure

NO_ACTION_LABEL = "no_action"

# 이 값보다 Confidence가 낮으면 화면에 '판단 불확실'로 표시한다.
# 17개 동작이라 표가 나뉘어 Confidence가 원래 낮게 나온다 (5명 교차 평가 평균 약 0.33).
UNCERTAIN_BELOW = float(os.getenv("UNCERTAIN_BELOW", "0.20"))
# 이보다 짧게 보인 Track은 잠깐 생긴 중복 박스로 보고 예측하지 않는다.
MIN_TRACK_SEC = float(os.getenv("MIN_TRACK_SEC", "0.5"))
# 사람 인식 신뢰도(BBox confidence) 중앙값이 이보다 낮은 Track은 사람이 아닌 것으로 보고 뺀다.
# 학습 영상의 실제 사람은 모두 0.85 이상, 카메라에서 의자에 걸린 외투 같은 가짜 사람은 0.31~0.73이었다.
MIN_PERSON_SCORE = float(os.getenv("MIN_PERSON_SCORE", "0.75"))


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
        if track.median_score < MIN_PERSON_SCORE:
            # Frame마다 자르지 않고 사람(Track) 단위로 거른다 → 진짜 사람의 Frame은 학습 때와 똑같이 남는다
            skipped.append({"track_id": tid, "reason": f"사람일 가능성이 낮아 제외했습니다 (인식 신뢰도 {track.median_score:.2f})."})
            continue
        frames = track_to_frames(track)
        # 모델이 동작 구간만으로 학습됐으면 예측할 때도 같은 규칙으로 잘라서 넣는다
        vector, used, segment = track_features(frames, info.fps, steps, min_valid,
                                               segment_only=bool(bundle.training_info.get("segment_only")),
                                               segment_floor=float(bundle.training_info.get("segment_floor", 0.0)),
                                               summary_names=bundle.frame_feature_names if bundle.summary_count else None)
        if vector is None:
            skipped.append({"track_id": tid, "reason": used})
            continue
        quality = measure(frames, info.fps, info.width, info.height)
        proba = bundle.model.predict_proba(vector.reshape(1, -1))[0]
        order = np.argsort(proba)[::-1]
        top3 = [{"action_label": str(classes[j]), "action_name": bundle.actions[classes[j]]["action_name"],
                 "confidence": round(float(proba[j]), 4)} for j in order[:3]]
        if quality["no_action"]:
            # 거의 안 움직였으면 17개 중 억지로 고르지 않는다 (모델 후보는 참고용으로만 남김)
            result = {"action_id": "-", "action_label": NO_ACTION_LABEL, "action_name": "동작 없음(정상)",
                      "action_group": "", "confidence": 0.0, "uncertain": False}
        else:
            top = classes[order[0]]
            action = bundle.actions[top]
            conf = float(proba[order[0]])
            result = {"action_id": action["action_id"], "action_label": top, "action_name": action["action_name"],
                      "action_group": action.get("group", ""), "confidence": round(conf, 4),
                      "uncertain": conf < UNCERTAIN_BELOW}
        predictions.append({
            "track_id": tid, **result, "top3": top3,
            "no_action": quality["no_action"],
            "is_main": quality["is_main"],
            "motion_level": quality["motion_level"],
            "height_ratio": quality["height_ratio"],
            "frames": len(track.frame_ids),
            "valid_frames": int(used),
            "first_sec": round(track.frame_ids[0] / info.fps, 3),
            "last_sec": round(track.frame_ids[-1] / info.fps, 3),
            # 실제로 모델에 넣은 동작 구간 (구간 자르기를 안 쓰면 None)
            "action_start_sec": round(segment[0] / info.fps, 3) if segment else None,
            "action_end_sec": round(segment[1] / info.fps, 3) if segment else None,
        })

    main = [p for p in predictions if p["is_main"]]
    if predictions:
        status = "ok"
        acted = sum(not p["no_action"] for p in main)
        still = len(main) - acted
        parts = ([f"{acted}명의 동작을 예측했습니다."] if acted else []) + \
                ([f"{still}명은 움직임이 거의 없어 '동작 없음'으로 판단했습니다."] if still else [])
        message = " ".join(parts)
        background = len(predictions) - len(main)
        not_person = sum("사람일 가능성이 낮아" in s["reason"] for s in skipped)
        excluded = ([f"배경 인물 {background}명"] if background else []) + \
                   ([f"사람이 아닌 대상 {not_person}개"] if not_person else [])
        if excluded:
            message += f" ({' · '.join(excluded)} 제외)"
        if not main:
            message = "화면 앞쪽에서 몸 전체가 보이는 사람이 없습니다. 카메라 위치를 확인하세요."
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
