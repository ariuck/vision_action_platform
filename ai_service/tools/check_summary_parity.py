"""결정적인 순간 특징을 쓰는 모델: 서비스 1,575개 계산이 학습 코드와 같은지 확인.

    python tools/check_summary_parity.py --train-repo ~/ai_vision/subject10_pose_action --clips 30
(ai_service 폴더에서, 학습 레포 가상환경으로 실행)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "tools"))

from check_feature_parity import track_from_keypoint_csv  # noqa: E402
from services.pose_pipeline import track_features, track_to_frames  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-repo", required=True)
    p.add_argument("--clips", type=int, default=30)
    args = p.parse_args()
    repo = Path(args.train_repo).expanduser()
    sys.path.append(str(repo / "scripts" / "team03"))
    import team_day06_baseline as tb  # noqa: E402
    from team_loso_segment_v2 import load_clips  # noqa: E402
    from team_loso_summary import build  # noqa: E402

    info = json.loads((BASE / "models" / "rf_training_info.json").read_text())
    cols = json.loads((BASE / "models" / "rf_feature_columns.json").read_text())
    floor = float(info.get("segment_floor", 0.0))
    frame_names = [c.split("_", 1)[1] for c in cols[:45]]
    config = tb.load_config()
    split = pd.read_csv(repo / "data/team_projects/team03/splits/test.csv")
    sample = split.sample(n=min(args.clips, len(split)), random_state=1)
    ref = build(load_clips(sample, tb.SEQ_DIR), 30, int(config["min_valid_frames"]), floor)
    worst = 0.0
    for _, row in ref.iterrows():
        item = sample[sample.clip_id == row.clip_id].iloc[0]
        track, fps = track_from_keypoint_csv(repo / item["keypoint_csv"])
        vec, used, seg = track_features(track_to_frames(track), fps, 30, int(config["min_valid_frames"]),
                                        segment_only=True, segment_floor=floor, summary_names=frame_names)
        diff = float(np.max(np.abs(vec - row[cols].to_numpy(float))))
        worst = max(worst, diff)
        print(f"{row.clip_id}  Feature {len(vec)}  max|diff|={diff:.6f}")
    print(f"\n최대 차이: {worst:.6f}  →  {'같음 (반올림 수준)' if worst < 1e-3 else '다름 — 계산 확인 필요'}")


if __name__ == "__main__":
    main()
