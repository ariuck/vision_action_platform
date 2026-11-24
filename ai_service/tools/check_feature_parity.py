"""서비스의 45-D 계산이 학습 때 계산과 같은지 확인한다 (T01 모델 불러오기 · 입력 규격 확인용).

학습 레포(subject10_pose_action)의 Keypoint CSV를 서비스 코드(pose_pipeline.track_features)에 넣어 만든 1,350개 숫자와,
학습 코드(team_day06_baseline.build_split)가 만든 숫자를 Clip마다 비교한다.

사용 (ai_service 폴더에서, 학습 레포의 가상환경으로):
    python tools/check_feature_parity.py --train-repo ~/ai_vision/subject10_pose_action --clips 12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from services.pose_pipeline import Track, track_features, track_to_frames  # noqa: E402
from src.pose_utils import COCO_KEYPOINT_NAMES  # noqa: E402


def track_from_keypoint_csv(path: Path) -> tuple[Track, float]:
    df = pd.read_csv(path)
    t = df["time_sec"].to_numpy(float)
    fps = 1.0 / float(np.median(np.diff(t)))
    track = Track(1)
    for row in df.itertuples():
        if int(row.pose_detected) != 1:
            continue
        kp = np.array([[getattr(row, f"{n}_x_px"), getattr(row, f"{n}_y_px")] for n in COCO_KEYPOINT_NAMES])
        conf = np.array([getattr(row, f"{n}_conf") for n in COCO_KEYPOINT_NAMES])
        box = np.array([row.box_x1, row.box_y1, row.box_x2, row.box_y2])
        track.add(row.frame_id, kp, conf, box)
    return track, fps


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-repo", required=True)
    p.add_argument("--clips", type=int, default=12)
    args = p.parse_args()
    repo = Path(args.train_repo).expanduser()
    sys.path.append(str(repo / "scripts" / "team03"))
    import team_day06_baseline as tb  # noqa: E402  (학습 코드)

    config = tb.load_config()
    cond = tb.CONDITIONS["team_45d"]
    split = pd.read_csv(repo / "data/team_projects/team03/splits/test.csv")
    sample = split.groupby("take").head(max(1, args.clips // 3)).head(args.clips)
    cols = json.loads((BASE / "models" / "rf_feature_columns.json").read_text())

    worst = 0.0
    for item in sample.itertuples():
        old = tb.SPLIT_DIR
        one = pd.DataFrame([item._asdict()]).drop(columns=["Index"])
        tmp = repo / "data/team_projects/team03/splits/_parity.csv"
        one.to_csv(tmp, index=False)
        try:
            ref = tb.build_split("_parity", cond, config)
        finally:
            tmp.unlink()
            tb.SPLIT_DIR = old
        ref_vec = ref[cols].to_numpy(float)[0]
        track, fps = track_from_keypoint_csv(repo / item.keypoint_csv)
        vec, used = track_features(track_to_frames(track), fps, 30, int(config["min_valid_frames"]))
        diff = float(np.max(np.abs(vec - ref_vec)))
        worst = max(worst, diff)
        print(f"{item.clip_id}  frames={used:4d}  max|diff|={diff:.6f}")
    print(f"\n최대 차이: {worst:.6f}  →  {'같음 (반올림 수준)' if worst < 1e-3 else '다름 — 계산 확인 필요'}")


if __name__ == "__main__":
    main()
