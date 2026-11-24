"""'결정적인 순간' 특징 (학습 레포 scripts/team03/team_summary_features.py와 같은 코드).

30 Step 시간표는 모든 칸을 똑같이 봐서, 서 있는 시간이 길면 '대부분 서 있는 동작'(예: 던지기)처럼 보였다.
던지기는 던지는 순간, 발차기는 차는 순간이 중요하므로 아래 값을 시간표 뒤에 더한다.

  - 45개 Frame 특징마다 최대·최소·평균·표준편차 (180개)
  - 가장 크게 움직인 순간의 45개 값 (45개)
    가장 크게 움직인 순간 = 손목·발목 속도 4개의 합이 가장 큰 Frame
"""
from __future__ import annotations

import numpy as np

STATS = ("max", "min", "mean", "std")
SPEED_NAMES = ("left_ankle_speed", "right_ankle_speed", "left_wrist_speed", "right_wrist_speed")


def summary_feature_names(frame_names: list[str]) -> list[str]:
    return [f"sum_{s}_{n}" for s in STATS for n in frame_names] + [f"peak_{n}" for n in frame_names]


def summary_features(matrix: np.ndarray, frame_names: list[str]) -> np.ndarray:
    """matrix: (Frame 수, 45) 동작 구간 안의 Frame 특징 (시간 순서) → (225,)"""
    m = np.asarray(matrix, dtype=np.float64)
    stats = [m.max(0), m.min(0), m.mean(0), m.std(0)]
    speed_idx = [frame_names.index(n) for n in SPEED_NAMES]
    peak = m[int(np.argmax(m[:, speed_idx].sum(1)))]
    return np.concatenate(stats + [peak]).astype(np.float32)
