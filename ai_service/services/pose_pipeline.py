"""영상 → YOLO Pose + ByteTrack → 사람(Track ID)별 45-D × 30 Step 입력.

학습 때(subject10_pose_action의 team_build_feature_sequences.py · team_day06_baseline.py)와
같은 순서로 계산한다. 다르게 계산하면 모델이 학습 때 본 적 없는 숫자를 받게 된다.

Frame 한 장의 Feature
- 38-D : 골반 중심 + 어깨 폭 정규화 좌표 34 + 팔꿈치·무릎 각도 4
- +7   : 상체 기울기, BBox 가로/세로 비율, 골반-바닥 높이, 왼·오른발목 속도, 왼·오른손목 속도
         (속도는 0.1초 간격, 길이는 그 사람의 상체 길이 중앙값으로 나눔)
Track 하나의 입력
- 38-D가 계산된 Frame만 사용, 팀 Feature 빈 값은 앞뒤로 보간
- Track 전체를 30 Step으로 보간 → 1,350개 숫자
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd

from src.feature_utils import normalize_by_shoulder_width, selected_joint_angles
from src.sequence_utils import flatten_sequence, resample_sequence
from src.team_features import torso_length, torso_tilt_deg

from .track_quality import find_segment, motion_curve

LAG_SEC = 0.1
TEAM_FEATURES = ["torso_tilt_deg", "bbox_aspect_ratio", "hip_floor_height",
                 "left_ankle_speed", "right_ankle_speed", "left_wrist_speed", "right_wrist_speed"]
SPEED_JOINTS = {"left_ankle_speed": 15, "right_ankle_speed": 16, "left_wrist_speed": 9, "right_wrist_speed": 10}


class VideoReadError(Exception):
    """영상 파일을 열 수 없거나 Frame을 읽을 수 없을 때"""


@dataclass
class VideoInfo:
    fps: float
    frames: int
    width: int
    height: int

    @property
    def duration_sec(self) -> float:
        return round(self.frames / self.fps, 3) if self.fps else 0.0


@dataclass
class Track:
    track_id: int
    frame_ids: list[int] = field(default_factory=list)
    keypoints: list[np.ndarray] = field(default_factory=list)   # (17, 2) Pixel 좌표
    confidences: list[np.ndarray] = field(default_factory=list)  # (17,)
    boxes: list[np.ndarray] = field(default_factory=list)       # (4,) x1, y1, x2, y2

    def add(self, frame_id, kp, conf, box):
        self.frame_ids.append(int(frame_id))
        self.keypoints.append(np.asarray(kp, dtype=np.float64))
        self.confidences.append(np.asarray(conf, dtype=np.float64))
        self.boxes.append(np.asarray(box, dtype=np.float64))


def track_video(path: str, bundle) -> tuple[VideoInfo, dict[int, Track]]:
    """영상 전체를 YOLO Pose + ByteTrack으로 추적해 Track ID별 관절 기록을 모은다."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoReadError("영상 파일을 열 수 없습니다. 지원하는 형식(MP4 등)인지 확인하세요.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cfg = bundle.project_config
    tracks: dict[int, Track] = {}
    frame_id = 0
    width = height = 0
    with bundle.lock:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            # 첫 Frame은 persist=False로 Tracker를 새로 만들어 이전 요청의 ID가 이어지지 않게 한다.
            result = bundle.pose_model.track(
                source=frame, persist=frame_id > 0, tracker="bytetrack.yaml",
                imgsz=int(cfg["image_size"]), conf=float(cfg["person_confidence"]),
                device=bundle.device, verbose=False)[0]
            boxes, kps = result.boxes, result.keypoints
            if boxes is not None and boxes.id is not None and kps is not None:
                ids = boxes.id.int().cpu().numpy()
                xyxy = boxes.xyxy.cpu().numpy()
                kp_xy = kps.xy.cpu().numpy()
                kp_conf = kps.conf.cpu().numpy() if kps.conf is not None else np.ones(kp_xy.shape[:2])
                for i, tid in enumerate(ids):
                    tracks.setdefault(int(tid), Track(int(tid))).add(frame_id, kp_xy[i], kp_conf[i], xyxy[i])
            frame_id += 1
    cap.release()
    if frame_id == 0:
        raise VideoReadError("영상에서 Frame을 읽지 못했습니다. 파일이 손상됐을 수 있습니다.")
    if fps <= 0:
        fps = 30.0
    return VideoInfo(fps=fps, frames=frame_id, width=width, height=height), tracks


def track_to_frames(track: Track) -> pd.DataFrame:
    """Track의 첫 Frame부터 마지막 Frame까지 빠짐없이 한 줄씩 만든다 (안 보인 Frame은 pose_ok=0).

    학습 때 Keypoint CSV가 영상의 모든 Frame을 한 줄씩 가졌던 것과 같은 모양이다.
    """
    first, last = track.frame_ids[0], track.frame_ids[-1]
    index = {f: i for i, f in enumerate(track.frame_ids)}
    rows = []
    for f in range(first, last + 1):
        i = index.get(f)
        rows.append({"frame_id": f, "pose_ok": i is not None, "kp": track.keypoints[i] if i is not None else None,
                     "conf": track.confidences[i] if i is not None else None,
                     "box": track.boxes[i] if i is not None else None})
    return pd.DataFrame(rows)


def track_features(frames: pd.DataFrame, fps: float, steps: int, min_valid_frames: int,
                   segment_only: bool = False):
    """한 사람의 Frame들 → (1,350 Feature, 사용한 Frame 수, 사용한 구간). 입력을 만들 수 없으면 (None, 사유, None).

    segment_only=True면 학습 때처럼 움직임이 큰 동작 구간만 남긴다.
    (속도 Feature는 구간을 자르기 전에 전체 Frame으로 계산한 뒤 고른다 — 학습 코드와 같은 순서)
    """
    lag = max(1, round(LAG_SEC * fps))
    pose = frames["pose_ok"].to_numpy(bool)
    kps = list(frames["kp"])
    torso = np.array([torso_length(k) if p else np.nan for k, p in zip(kps, pose)])
    torso_clip = float(np.nanmedian(torso)) if np.isfinite(torso).any() else math.nan

    base_rows, base_ids = [], []
    team = {name: np.full(len(frames), np.nan) for name in TEAM_FEATURES}
    for i, row in enumerate(frames.itertuples()):
        if not pose[i]:
            continue
        k, box = row.kp, row.box
        try:
            normalized, _, _ = normalize_by_shoulder_width(k)
            angles = selected_joint_angles(k)
            base = [float(v) for p in normalized for v in p] + [
                angles["left_elbow_angle"], angles["right_elbow_angle"],
                angles["left_knee_angle"], angles["right_knee_angle"]]
            if len(base) == 38 and np.all(np.isfinite(base)):
                base_rows.append(base)
                base_ids.append(int(row.frame_id))
        except ValueError:
            pass
        team["torso_tilt_deg"][i] = torso_tilt_deg(k)
        bw, bh = box[2] - box[0], box[3] - box[1]
        team["bbox_aspect_ratio"][i] = bw / bh if bh > 1e-8 else math.nan
        hip_y = (k[11][1] + k[12][1]) / 2
        team["hip_floor_height"][i] = (box[3] - hip_y) / torso_clip if torso_clip > 1e-8 else math.nan
        for name, j in SPEED_JOINTS.items():
            if i >= lag and pose[i - lag] and torso_clip > 1e-8:
                team[name][i] = float(np.linalg.norm(k[j] - kps[i - lag][j])) / torso_clip / (lag / fps)

    team_df = pd.DataFrame(team, index=frames["frame_id"].to_numpy())
    team_df = team_df.where(np.isfinite(team_df)).interpolate(limit_direction="both")

    segment = None
    if segment_only:
        s_idx, e_idx = find_segment(motion_curve(frames, fps), fps)
        f0, f1 = int(frames["frame_id"].iloc[s_idx]), int(frames["frame_id"].iloc[e_idx])
        keep = [i for i, f in enumerate(base_ids) if f0 <= f <= f1]
        base_rows = [base_rows[i] for i in keep]
        base_ids = [base_ids[i] for i in keep]
        segment = (f0, f1)

    if len(base_rows) < min_valid_frames:
        return None, f"관절이 제대로 잡힌 Frame이 {len(base_rows)}개뿐입니다 (최소 {min_valid_frames}개).", None

    extra = team_df.loc[base_ids].to_numpy(np.float32)
    matrix = np.hstack([np.asarray(base_rows, dtype=np.float32), extra])
    if not np.all(np.isfinite(matrix)):
        return None, "팀 Feature를 계산할 수 없는 Frame이 있습니다 (사람이 너무 짧게 보였을 수 있음).", None
    vector = flatten_sequence(resample_sequence(matrix, np.asarray(base_ids), target_steps=steps))
    return vector, len(base_rows), segment
