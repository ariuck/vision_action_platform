"""추론에 필요한 모델과 학습 정보를 서버 시작 시 한 번 불러온다.

- rf_action_baseline.joblib : 3팀 17개 동작 Random Forest (45-D × 30 Step = 1,350 Feature)
- rf_feature_columns.json   : 학습 때 Feature 이름과 순서
- rf_training_info.json     : Class 목록, sequence_steps, min_valid_frames 등
- yolo11n-pose.pt           : 관절 17개 추출용 YOLO Pose 가중치
- config/action_classes.csv : action_label → 번호·한국어 이름

joblib.load()는 직접 학습한 모델 파일에만 사용한다.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))
CONFIG_DIR = BASE_DIR / "config"


@dataclass
class ModelBundle:
    model: object
    feature_columns: list[str]
    training_info: dict
    project_config: dict
    actions: dict[str, dict]
    pose_model: object
    device: str
    lock: Lock = field(default_factory=Lock)  # YOLO Tracker 상태를 요청끼리 섞지 않기 위해

    @property
    def sequence_steps(self) -> int:
        return int(self.training_info["sequence_steps"])

    @property
    def summary_count(self) -> int:
        """결정적인 순간 특징 수 (없는 예전 모델은 0)"""
        return int(self.training_info.get("summary_feature_count", 0)) if self.training_info.get("summary_features") else 0

    @property
    def frame_feature_count(self) -> int:
        return (len(self.feature_columns) - self.summary_count) // self.sequence_steps

    @property
    def frame_feature_names(self) -> list[str]:
        """Frame 특징 45개 이름 (학습 때 순서). 결정적인 순간 특징 계산에 쓴다."""
        return [c.split("_", 1)[1] for c in self.feature_columns[: self.frame_feature_count]]

    @property
    def classes(self) -> list[str]:
        return [str(c) for c in self.model.classes_]


def _resolve_device(requested: str) -> str:
    import torch

    if requested in ("auto", "", None):
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return requested


def load_bundle() -> ModelBundle:
    from ultralytics import YOLO

    model = joblib.load(MODEL_DIR / "rf_action_baseline.joblib")
    feature_columns = json.loads((MODEL_DIR / "rf_feature_columns.json").read_text(encoding="utf-8"))
    training_info = json.loads((MODEL_DIR / "rf_training_info.json").read_text(encoding="utf-8"))
    project_config = json.loads((CONFIG_DIR / "project.json").read_text(encoding="utf-8"))
    with (CONFIG_DIR / "action_classes.csv").open(encoding="utf-8-sig") as f:
        actions = {row["action_label"]: row for row in csv.DictReader(f)}

    # 학습 때와 입력 규격이 같은지 시작할 때 확인한다.
    steps = int(training_info["sequence_steps"])
    if len(feature_columns) != int(training_info["feature_count"]):
        raise RuntimeError("Feature 순서 파일과 학습 정보의 Feature 수가 다릅니다.")
    if getattr(model, "n_features_in_", len(feature_columns)) != len(feature_columns):
        raise RuntimeError("모델 입력 Feature 수가 Feature 순서 파일과 다릅니다.")
    summary = int(training_info.get("summary_feature_count", 0)) if training_info.get("summary_features") else 0
    if (len(feature_columns) - summary) % steps:
        raise RuntimeError("시간표 Feature 수가 sequence_steps로 나누어떨어지지 않습니다.")
    if summary and not all(c.startswith(("sum_", "peak_")) for c in feature_columns[-summary:]):
        raise RuntimeError("결정적인 순간 특징 이름이 Feature 순서 파일 끝에 없습니다.")
    missing = [c for c in model.classes_ if c not in actions]
    if missing:
        raise RuntimeError(f"동작표에 없는 Class: {missing}")

    device = _resolve_device(os.getenv("POSE_DEVICE", project_config.get("device", "auto")))
    pose_model = YOLO(str(MODEL_DIR / project_config["pose_model"]))
    return ModelBundle(model, feature_columns, training_info, project_config, actions, pose_model, device)
