"""기준선 최소값(segment_floor)을 쓴 모델: 서비스 Feature 계산이 학습 코드와 같은지 확인.

학습 레포의 rows_for_floor(학습 코드)와 서비스 track_features(segment_floor=...)를 같은 Keypoint CSV로 비교한다.
    python tools/check_floor_parity.py --train-repo ~/ai_vision/subject10_pose_action --floor 0.4 --clips 30
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
    p.add_argument("--floor", type=float, required=True)
    p.add_argument("--clips", type=int, default=30)
    args = p.parse_args()
    repo = Path(args.train_repo).expanduser()
    sys.path.append(str(repo / "scripts" / "team03"))
    import team_day06_baseline as tb  # noqa: E402
    from team_loso_segment_floor import rows_for_floor  # noqa: E402
    from team_loso_segment_v2 import load_clips  # noqa: E402

    config = tb.load_config()
    split = pd.read_csv(repo / "data/team_projects/team03/splits/test.csv")
    sample = split.sample(n=min(args.clips, len(split)), random_state=0)
    ref, _ = rows_for_floor(load_clips(sample, tb.SEQ_DIR), args.floor, 30, int(config["min_valid_frames"]))
    cols = json.loads((BASE / "models" / "rf_feature_columns.json").read_text())
    worst = 0.0
    for _, row in ref.iterrows():
        item = sample[sample.clip_id == row.clip_id].iloc[0]
        track, fps = track_from_keypoint_csv(repo / item["keypoint_csv"])
        vec, used, seg = track_features(track_to_frames(track), fps, 30, int(config["min_valid_frames"]),
                                        segment_only=True, segment_floor=args.floor)
        diff = float(np.max(np.abs(vec - row[cols].to_numpy(float))))
        worst = max(worst, diff)
        print(f"{row.clip_id}  frames={used:4d} (학습 {int(row.valid_frames)})  segment={seg}  max|diff|={diff:.6f}")
    print(f"\n최대 차이: {worst:.6f}  →  {'같음 (반올림 수준)' if worst < 1e-3 else '다름 — 계산 확인 필요'}")


if __name__ == "__main__":
    main()
