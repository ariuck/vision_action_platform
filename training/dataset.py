"""학습 데이터 만들기 + 증강 (AI 서버와 똑같은 Feature 계산 코드 사용).

왜 여기서 다시 만드나
- 지금까지는 학습 레포(subject10_pose_action) 코드로 Feature를 만들고, 서비스가 같은지 따로 확인했다.
- 증강을 여러 가지 시험하려면 Keypoint를 바꾼 뒤 Feature를 다시 계산해야 하는데,
  서비스 코드(ai_service/services/pose_pipeline.py)를 그대로 쓰면 학습·서비스가 어긋날 일이 없다.
- 원본 데이터(Keypoint CSV)는 학습 레포에 있고, Git에는 올리지 않는다.

증강 (Keypoint 단계, 학습에만 사용)
- hflip   : 좌우 반전 (모든 방향)
- fast/slow: 동작 속도 1.25배 / 0.8배 (같은 Frame을 다른 fps로 계산 → 속도 Feature가 바뀜)
- jitter  : 관절 좌표에 작은 잡음 (상체 길이의 2%) — 화질이 낮은 카메라 흉내
- rot±    : 화면을 ±7도 기울임 — 카메라가 살짝 기울어진 경우
- pad     : 앞뒤에 1.5~3초 서 있는 Frame 추가 (잡음 포함) — 카메라 녹화처럼 서 있는 시간이 긴 경우
- fps30   : 한 Frame 건너 하나씩 → 30fps 영상 흉내 (학습 영상은 전부 60fps)
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AI = Path(__file__).resolve().parents[1] / "ai_service"
sys.path.insert(0, str(AI))

from services.pose_pipeline import Track, track_features, track_to_frames  # noqa: E402
from src.pose_utils import COCO_KEYPOINT_NAMES  # noqa: E402

TRAIN_REPO = Path.home() / "ai_vision/subject10_pose_action"
MANIFEST = TRAIN_REPO / "data/team_projects/team03/manifests/keypoint_dataset_manifest.csv"
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
AUGS = ["orig", "hflip", "fast", "slow", "jitter", "rot+", "rot-", "pad", "fps30"]


def track_from_csv(path: Path) -> tuple[Track, float]:
    df = pd.read_csv(path)
    fps = 1.0 / float(np.median(np.diff(df["time_sec"].to_numpy(float))))
    df = df[df["pose_detected"] == 1]
    kx = df[[f"{n}_x_px" for n in COCO_KEYPOINT_NAMES]].to_numpy(float)
    ky = df[[f"{n}_y_px" for n in COCO_KEYPOINT_NAMES]].to_numpy(float)
    kc = df[[f"{n}_conf" for n in COCO_KEYPOINT_NAMES]].to_numpy(float)
    boxes = df[["box_x1", "box_y1", "box_x2", "box_y2"]].to_numpy(float)
    tr = Track(1)
    for i, fid in enumerate(df["frame_id"].to_numpy()):
        tr.add(int(fid), np.stack([kx[i], ky[i]], 1), kc[i], boxes[i])
    return tr, fps


def _torso(tr: Track) -> float:
    kp = np.stack(tr.keypoints)
    return float(np.nanmedian(np.linalg.norm(kp[:, [5, 6]].mean(1) - kp[:, [11, 12]].mean(1), axis=1)))


def augment(tr: Track, fps: float, kind: str, width: float, seed: int) -> tuple[Track, float]:
    """Track 복사본에 증강 적용 → (새 Track, 계산에 쓸 fps)"""
    rng = np.random.default_rng(seed)
    t = copy.deepcopy(tr)
    if kind == "orig":
        return t, fps
    if kind == "fast":   # 같은 동작을 1.25배 빠르게 한 것처럼
        return t, fps * 1.25
    if kind == "slow":
        return t, fps * 0.8
    if kind == "hflip":
        for i in range(len(t.keypoints)):
            k, c = t.keypoints[i].copy(), t.confidences[i].copy()
            k[:, 0] = width - k[:, 0]
            for a, b in FLIP_PAIRS:
                k[[a, b]] = k[[b, a]]
                c[[a, b]] = c[[b, a]]
            x1, y1, x2, y2 = t.boxes[i]
            t.keypoints[i], t.confidences[i] = k, c
            t.boxes[i] = np.array([width - x2, y1, width - x1, y2])
        return t, fps
    if kind == "jitter":
        s = 0.02 * _torso(tr)
        t.keypoints = [k + rng.normal(0, s, k.shape) for k in t.keypoints]
        return t, fps
    if kind in ("rot+", "rot-"):
        ang = np.radians(7 if kind == "rot+" else -7)
        r = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        center = np.nanmedian(np.stack([k[[11, 12]].mean(0) for k in t.keypoints]), axis=0)
        for i in range(len(t.keypoints)):
            t.keypoints[i] = (t.keypoints[i] - center) @ r.T + center
            x1, y1, x2, y2 = t.boxes[i]
            corners = (np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]]) - center) @ r.T + center
            t.boxes[i] = np.array([*corners.min(0), *corners.max(0)])
        return t, fps
    if kind == "fps30":   # 60fps 학습 영상을 한 Frame 건너 하나씩 → 30fps 카메라·휴대폰 영상 흉내
        new = Track(1)
        for fid, k, c, b in zip(tr.frame_ids, tr.keypoints, tr.confidences, tr.boxes):
            if fid % 2 == 0:
                new.add(fid // 2, k, c, b)
        return new, fps / 2
    if kind == "pad":   # 앞뒤로 서 있는 시간 추가 (첫·마지막 Frame + 작은 잡음)
        s = 0.01 * _torso(tr)
        n_before = int(rng.uniform(1.5, 3.0) * fps)
        n_after = int(rng.uniform(1.5, 3.0) * fps)
        f0, f1 = tr.frame_ids[0], tr.frame_ids[-1]
        new = Track(1)
        for j in range(n_before):
            new.add(f0 - n_before + j + 100000, tr.keypoints[0] + rng.normal(0, s, (17, 2)), tr.confidences[0], tr.boxes[0])
        for fid, k, c, b in zip(tr.frame_ids, tr.keypoints, tr.confidences, tr.boxes):
            new.add(fid + 100000, k, c, b)
        for j in range(n_after):
            new.add(f1 + 1 + j + 100000, tr.keypoints[-1] + rng.normal(0, s, (17, 2)), tr.confidences[-1], tr.boxes[-1])
        return new, fps
    raise ValueError(kind)


def build_cache(out: Path, floor: float, frame_names: list[str]) -> pd.DataFrame:
    """원본 255개 × 증강 8가지 → Feature 1,575개 (서비스 계산과 같음). 결과를 pickle로 저장."""
    man = pd.read_csv(MANIFEST)
    rows = []
    for n, (_, r) in enumerate(man.iterrows()):
        tr, fps = track_from_csv(TRAIN_REPO / r["keypoint_csv"])
        for a_i, kind in enumerate(AUGS):
            t2, fps2 = augment(tr, fps, kind, float(r["width"]), seed=n * 100 + a_i)
            vec, used, seg = track_features(track_to_frames(t2), fps2, 30, 10, segment_only=True,
                                            segment_floor=floor, summary_names=frame_names)
            if vec is None:
                continue
            rows.append({"clip_id": r["clip_id"], "subject": r["subject"], "take": r["take"],
                         "action_label": r["action_label"], "aug": kind, **{f"f{i}": float(v) for i, v in enumerate(vec)}})
        if n % 50 == 0:
            print(f"  {n}/{len(man)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_pickle(out)
    return df
