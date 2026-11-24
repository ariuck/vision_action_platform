"""외부 테스트 영상(학습에 안 쓴 새 영상) → 서비스와 같은 방식으로 주인공을 고르고 Feature 계산.

영상은 얼굴이 나와서 Git에 올리지 않는다. 경로와 정답은 아래 목록으로만 관리한다.
    ai_service/.venv/bin/python training/build_external.py
출력: training/cache/external_floor0.4.pkl
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402

from dataset import AI  # noqa: E402
from services.action_inference import MIN_PERSON_SCORE, MIN_TRACK_SEC  # noqa: E402
from services.model_loader import load_bundle  # noqa: E402
from services.pose_pipeline import track_features, track_to_frames, track_video  # noqa: E402
from services.track_quality import measure  # noqa: E402

# (파일, 정답, 출처) — 새 사람이 가로 1920×1080 30fps로 찍은 5개 + 웹캠(DroidCam) 앞차기 5개
EXTERNAL = [
    ("/tmp/vap_ext/ext_push.mp4", "push", "new_video"),
    ("/tmp/vap_ext/ext_grab_shake.mp4", "grab_shake", "new_video"),
    ("/tmp/vap_ext/ext_kiosk_tap.mp4", "kiosk_tap", "new_video"),
    ("/tmp/vap_ext/ext_crouch_take_item.mp4", "crouch_take_item", "new_video"),
    ("/tmp/vap_ext/ext_lie_forward.mp4", "lie_forward", "new_video"),
    ("/tmp/vap_cam/job15_dc9bace0.mp4", "front_kick", "webcam"),
    ("/tmp/vap_cam/job16_c1ffdbb5.mp4", "front_kick", "webcam"),
    ("/tmp/vap_cam/job17_b271e0b5.mp4", "front_kick", "webcam"),
    ("/tmp/vap_cam/job18_faaa48ea.mp4", "front_kick", "webcam"),
    ("/tmp/vap_cam/job24_d4d6c284.mp4", "front_kick", "webcam"),
]


def main():
    bundle = load_bundle()
    cols = json.loads((AI / "models" / "rf_feature_columns.json").read_text())
    frame_names = [c.split("_", 1)[1] for c in cols[:45]]
    rows = []
    for path, label, source in EXTERNAL:
        if not Path(path).exists():
            print("없음:", path)
            continue
        info, tracks = track_video(path, bundle)
        cands = []
        for tid, tr in tracks.items():
            frames = track_to_frames(tr)
            if len(frames) / info.fps < MIN_TRACK_SEC or tr.median_score < MIN_PERSON_SCORE:
                continue
            q = measure(frames, info.fps, info.width, info.height)
            if q["is_main"]:
                cands.append((len(tr.frame_ids), frames))
        if not cands:
            print("주인공 없음:", path)
            continue
        frames = max(cands, key=lambda x: x[0])[1]
        vec, used, seg = track_features(frames, info.fps, 30, 10, segment_only=True, segment_floor=0.4,
                                        summary_names=frame_names)
        rows.append({"clip_id": Path(path).stem, "action_label": label, "source": source,
                     **{f"f{i}": float(v) for i, v in enumerate(vec)}})
        print(f"{Path(path).name}: {info.width}x{info.height} {info.fps:.0f}fps 주인공 {len(cands)}명, 구간 {seg}")
    pd.DataFrame(rows).to_pickle(Path(__file__).resolve().parent / "cache" / "external_floor0.4.pkl")


if __name__ == "__main__":
    main()
