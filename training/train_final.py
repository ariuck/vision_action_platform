"""서비스용 최종 모델 학습 → ai_service/models 에 저장.

구성 (docs/dev_log.md 24번 실험으로 결정)
- 학습 데이터: 5명 원본 255개 + 증강 7가지(반전·빠르게·느리게·잡음·±7도·앞뒤 서 있기) = 2,040행
- Feature: 30 Step 시간표 1,350 + 결정적인 순간 225 = 1,575 (동작 구간 기준선 최소값 0.4)
- 모델: ExtraTrees 500그루, 나눌 때마다 Feature 10%만 봄, class_weight balanced
- 5명 교차 평가 (시드 3번 평균): 0.881 (가장 낮은 사람 0.837)

    ai_service/.venv/bin/python training/train_final.py
"""
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / "ai_service" / "models"
AUGS = ["orig", "hflip", "fast", "slow", "jitter", "rot+", "rot-", "pad"]

df = pd.read_pickle(HERE / "cache" / "train_floor0.4.pkl")
df = df[df.aug.isin(AUGS)]
feats = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
cols = json.loads((MODELS / "rf_feature_columns.json").read_text())
assert len(cols) == len(feats) == 1575

backup = HERE.parent / "ai_service" / "models_backup" / time.strftime("before_extratrees_%Y%m%d_%H%M")
backup.mkdir(parents=True, exist_ok=True)
for f in ("rf_action_baseline.joblib", "rf_feature_columns.json", "rf_training_info.json"):
    shutil.copy2(MODELS / f, backup / f)

model = ExtraTreesClassifier(n_estimators=500, max_features=0.1, class_weight="balanced", random_state=42, n_jobs=-1)
model.fit(df[feats].to_numpy(float), df.action_label.to_numpy())
joblib.dump(model, MODELS / "rf_action_baseline.joblib", compress=3)

info = json.loads((MODELS / "rf_training_info.json").read_text())
info.update({
    "model": "ExtraTreesClassifier(n_estimators=500, max_features=0.1, class_weight='balanced', random_state=42)",
    "train_clips": int(len(df)), "original_clips": int((df.aug == "orig").sum()),
    "augmentation": AUGS[1:], "augmented_clips": int((df.aug != "orig").sum()),
    "subjects": sorted(df.subject.unique().tolist()), "classes": [str(c) for c in model.classes_],
    "condition": "team_45d_segment_floor0.4_summary_aug7_extratrees",
    "trained_with": "vision_action_platform/training/train_final.py (서비스와 같은 Feature 계산 코드)",
    "loso_accuracy_3seeds": 0.881, "loso_min_subject": 0.837,
})
(MODELS / "rf_training_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
train_acc = (model.predict(df[df.aug == "orig"][feats].to_numpy(float)) == df[df.aug == "orig"].action_label).mean()
print(f"학습 {len(df)}행, 클래스 {len(model.classes_)}, 원본 학습 정확도 {train_acc:.3f} (확인용)")
print(f"모델 크기 {(MODELS / 'rf_action_baseline.joblib').stat().st_size / 1e6:.1f} MB, 백업: {backup.name}")
