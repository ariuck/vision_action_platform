"""예측 전에 사람(Track)마다 '얼마나 움직였는지'와 '얼마나 크게·오래 보였는지'를 잰다.

움직임 크기는 학습 데이터의 동작 구간을 정할 때 쓴 방식(subject10 team_action_segments.py)과 같다.
  1. 몸 관절 12개(어깨·팔꿈치·손목·골반·무릎·발목)가 0.1초 동안 움직인 평균 거리 (Pixel)
  2. 그 사람의 상체 길이 중앙값으로 나눔 → 멀리 있든 가까이 있든 같은 기준
  3. 0.5초 이동 평균으로 부드럽게
  4. 상위 95% 값(motion_p95)을 그 사람의 '가장 크게 움직인 정도'로 사용

사용처
- 동작 없음: motion_p95가 NO_ACTION_MOTION보다 작으면 예측 대신 '동작 없음(정상)'
- 주인공/배경: 아래를 모두 만족하면 주인공, 아니면 배경 인물 (몇 명이든 주인공이 될 수 있음)
  - 키(BBox 높이 상위 90%) ÷ 화면 높이 ≥ MAIN_MIN_HEIGHT
    (누우면 BBox 높이가 줄어서 평균 대신 가장 크게 보였을 때에 가까운 상위 90%를 쓴다)
  - MAIN_MIN_SEC 이상 보임
  - 화면 왼쪽·오른쪽 끝에 걸쳐 잘린 Frame이 절반 미만
  - 두 발목이 보인 Frame이 절반 이상 (책상 뒤에 앉아 다리가 가려진 사람 제외)
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from src.team_features import torso_length

BODY_JOINTS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
LAG_SEC = 0.1
SMOOTH_SEC = 0.5

NO_ACTION_MOTION = float(os.getenv("NO_ACTION_MOTION", "0.30"))
MAIN_MIN_HEIGHT = float(os.getenv("MAIN_MIN_HEIGHT", "0.25"))
# 기준값은 학습 영상 255개(실제 동작한 사람)와 카메라 영상 4개로 정했다 (docs/dev_log.md 14번)
#  - 학습 영상 최소: 움직임 0.428, 키 비율 0.291, 보인 시간 1.62초, 발목 보인 비율 0.532
MAIN_MIN_SEC = float(os.getenv("MAIN_MIN_SEC", "1.0"))
MAIN_MAX_EDGE = float(os.getenv("MAIN_MAX_EDGE", "0.5"))   # 화면 끝에 걸친 Frame 비율이 이보다 작아야 주인공
MAIN_MIN_LEGS = float(os.getenv("MAIN_MIN_LEGS", "0.3"))   # 두 발목이 보인 Frame 비율이 이 이상이어야 주인공


def motion_curve(frames: pd.DataFrame, fps: float) -> np.ndarray:
    pose = frames["pose_ok"].to_numpy(bool)
    n = len(frames)
    xy = np.full((n, len(BODY_JOINTS), 2), np.nan)
    torso = np.full(n, np.nan)
    for i, (ok, k) in enumerate(zip(pose, frames["kp"])):
        if ok:
            xy[i] = k[BODY_JOINTS]
            torso[i] = torso_length(k)
    if not np.isfinite(torso).any():
        return np.array([])
    torso_med = float(np.nanmedian(torso))
    lag = max(1, round(LAG_SEC * fps))
    dist = np.full(n, np.nan)
    if n > lag and torso_med > 1e-8:
        step = np.linalg.norm(xy[lag:] - xy[:-lag], axis=2)
        with warnings.catch_warnings():  # 한쪽이 안 보인 Frame은 NaN → 경고만 숨김
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_step = np.nanmean(step, axis=1)
        dist[lag:] = mean_step / torso_med / (lag / fps)
    s = pd.Series(dist).interpolate(limit_direction="both")
    win = max(1, round(SMOOTH_SEC * fps))
    return s.rolling(win, center=True, min_periods=1).mean().to_numpy()


THRESH_FRAC = 0.2      # 하위 5% ~ 상위 95% 사이 20% 높이를 기준선으로
MIN_ACTIVE_SEC = 0.15  # 기준선을 이 시간 이상 넘어야 움직임으로 인정
PAD_SEC = 0.2          # 시작·끝 앞뒤 여유


def find_segment(curve: np.ndarray, fps: float) -> tuple[int, int]:
    """학습 때 동작 구간(team_action_segments.py find_segment)과 같은 규칙. Frame 위치(index) 시작·끝."""
    n = len(curve)
    if n == 0 or not np.isfinite(curve).any():
        return 0, max(0, n - 1)
    p5, p95 = np.nanpercentile(curve, [5, 95])
    thr = p5 + THRESH_FRAC * (p95 - p5)
    above = np.nan_to_num(curve, nan=0.0) > thr
    min_len = max(1, round(MIN_ACTIVE_SEC * fps))
    runs, start = [], None
    for i, a in enumerate(np.append(above, False)):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if i - start >= min_len:
                runs.append((start, i - 1))
            start = None
    s_idx, e_idx = (runs[0][0], runs[-1][1]) if runs else (0, n - 1)
    pad = round(PAD_SEC * fps)
    return max(0, s_idx - pad), min(n - 1, e_idx + pad)


def measure(frames: pd.DataFrame, fps: float, video_width: int, video_height: int) -> dict:
    curve = motion_curve(frames, fps)
    finite = curve[np.isfinite(curve)] if curve.size else curve
    motion_p95 = float(np.percentile(finite, 95)) if finite.size else 0.0
    seen = frames[frames["pose_ok"]]
    boxes = np.stack(seen["box"].to_list()) if len(seen) else np.zeros((0, 4))
    confs = np.stack(seen["conf"].to_list()) if len(seen) else np.zeros((0, 17))
    height_ratio = float(np.percentile(boxes[:, 3] - boxes[:, 1], 90)) / video_height if len(boxes) else 0.0
    # 화면 왼쪽·오른쪽 끝에 걸쳐 잘린 사람 (책상에 앉은 뒷사람이 주로 이렇게 보였다)
    margin = 0.01 * video_width
    edge_ratio = float(np.mean((boxes[:, 0] <= margin) | (boxes[:, 2] >= video_width - margin))) if len(boxes) else 1.0
    # 두 발목이 보인 Frame 비율 (학습 영상은 몸 전체가 보이게 찍었다)
    legs_ratio = float(np.mean((confs[:, 15] >= 0.5) & (confs[:, 16] >= 0.5))) if len(confs) else 0.0
    seen_sec = len(frames) / fps
    return {
        "motion_level": round(motion_p95, 3),
        "height_ratio": round(height_ratio, 3),
        "edge_ratio": round(edge_ratio, 3),
        "legs_ratio": round(legs_ratio, 3),
        "seen_sec": round(seen_sec, 2),
        "no_action": motion_p95 < NO_ACTION_MOTION,
        "is_main": (height_ratio >= MAIN_MIN_HEIGHT and seen_sec >= MAIN_MIN_SEC
                    and edge_ratio < MAIN_MAX_EDGE and legs_ratio >= MAIN_MIN_LEGS),
    }
