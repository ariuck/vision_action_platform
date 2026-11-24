"""증강·모델 비교 실험.

평가 두 가지
- LOSO: 5명 중 1명씩 빼고 학습 → 빠진 사람의 원본 51개로 평가 (증강본은 평가에 안 씀). 5번 평균.
- 외부: 학습에 한 번도 안 쓴 새 영상 10개 (새 사람 가로 30fps 5개 + 웹캠 앞차기 5개).
  5명 전체로 학습한 모델로 평가. 개수가 적어 참고용, 주 기준은 LOSO.

    ai_service/.venv/bin/python training/experiment.py aug
    ai_service/.venv/bin/python training/experiment.py model --augs orig,hflip,...
출력: training/results/*.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
SUBJECTS = ["S01", "S02", "S03", "S04", "S05"]

train_df = pd.read_pickle(CACHE / "train_floor0.4.pkl")
ext_df = pd.read_pickle(CACHE / "external_floor0.4.pkl")
FEATS = [c for c in train_df.columns if c.startswith("f") and c[1:].isdigit()]


def rf(**kw):
    return RandomForestClassifier(n_estimators=kw.pop("n", 300), class_weight="balanced", random_state=42,
                                  n_jobs=-1, **kw)


MODELS = {
    "RF (지금)": lambda: rf(),
    "RF 500·특징10%": lambda: rf(n=500, max_features=0.1),
    "RF 500·잎2": lambda: rf(n=500, min_samples_leaf=2),
    "ExtraTrees 500": lambda: ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1),
    "GradientBoosting": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, class_weight="balanced",
                                                               random_state=42),
    "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced")),
    "SVM": lambda: make_pipeline(StandardScaler(), SVC(C=5, gamma="scale", probability=True, class_weight="balanced",
                                                      random_state=42)),
    "앙상블 RF+ET": lambda: VotingClassifier([("rf", rf(n=500)), ("et", ExtraTreesClassifier(
        n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1))], voting="soft"),
    "앙상블 RF+ET+LR": lambda: VotingClassifier([("rf", rf(n=500)), ("et", ExtraTreesClassifier(
        n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)), ("lr", make_pipeline(
            StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced")))], voting="soft"),
    "앙상블 RF+ET+SVM": lambda: VotingClassifier([("rf", rf(n=500)), ("et", ExtraTreesClassifier(
        n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)), ("svm", make_pipeline(
            StandardScaler(), SVC(C=5, probability=True, class_weight="balanced", random_state=42)))], voting="soft"),
}


def select(augs: list[str]) -> pd.DataFrame:
    """증강 목록에 맞는 학습 행. 'hflipT02'는 측면만 반전 (지금 서비스 구성)."""
    parts = []
    for a in augs:
        if a == "hflipT02":
            parts.append(train_df[(train_df.aug == "hflip") & (train_df.take == "T02")])
        else:
            parts.append(train_df[train_df.aug == a])
    return pd.concat(parts, ignore_index=True)


def evaluate(make_model, augs: list[str]) -> dict:
    data = select(augs)
    orig = train_df[train_df.aug == "orig"]
    accs, f1s, confs, per_action = [], [], [], []
    for s in SUBJECTS:
        tr = data[data.subject != s]
        te = orig[orig.subject == s]
        m = make_model()
        m.fit(tr[FEATS].to_numpy(float), tr.action_label.to_numpy())
        p = m.predict_proba(te[FEATS].to_numpy(float))
        pred = np.asarray(m.classes_)[p.argmax(1)]
        y = te.action_label.to_numpy()
        accs.append(accuracy_score(y, pred))
        f1s.append(f1_score(y, pred, average="macro", zero_division=0))
        confs.append(p.max(1).mean())
        per_action += list(zip(y, pred == y))
    m = make_model()
    m.fit(data[FEATS].to_numpy(float), data.action_label.to_numpy())
    p = m.predict_proba(ext_df[FEATS].to_numpy(float))
    cls = np.asarray(m.classes_)
    ext_pred = cls[p.argmax(1)]
    ext_ok = ext_pred == ext_df.action_label.to_numpy()
    true_p = [p[i, list(cls).index(lbl)] for i, lbl in enumerate(ext_df.action_label)]
    pa = pd.DataFrame(per_action, columns=["action", "ok"]).groupby("action").ok.mean()
    return {"loso_acc": np.mean(accs), "loso_min": np.min(accs), "loso_f1": np.mean(f1s), "conf": np.mean(confs),
            "ext_acc": ext_ok.mean(), "ext_new5": ext_ok[ext_df.source.to_numpy() == "new_video"].mean(),
            "ext_webcam5": ext_ok[ext_df.source.to_numpy() == "webcam"].mean(), "ext_true_p": np.mean(true_p),
            "rows": len(data), "per_action": pa.to_dict(),
            "ext_detail": dict(zip(ext_df.clip_id, ext_pred))}


def show(results: dict, name: str):
    df = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk not in ("per_action", "ext_detail")}
                       for k, v in results.items()}).T
    print(df.round(3).to_string())
    df.to_csv(RESULTS / f"{name}.csv", encoding="utf-8-sig")
    pa = pd.DataFrame({k: v["per_action"] for k, v in results.items()}).round(3)
    pa.to_csv(RESULTS / f"{name}_per_action.csv", encoding="utf-8-sig")
    det = pd.DataFrame({k: v["ext_detail"] for k, v in results.items()})
    det.insert(0, "정답", ext_df.set_index("clip_id").action_label)
    det.to_csv(RESULTS / f"{name}_external.csv", encoding="utf-8-sig")
    return df, pa, det


def main():
    stage = sys.argv[1]
    t0 = time.time()
    if stage == "aug":
        configs = {
            "A0 지금 (원본+측면반전)": ["orig", "hflipT02"],
            "A1 반전 전부": ["orig", "hflip"],
            "A2 A1+속도": ["orig", "hflip", "fast", "slow"],
            "A3 A1+잡음": ["orig", "hflip", "jitter"],
            "A4 A1+기울임": ["orig", "hflip", "rot+", "rot-"],
            "A5 A1+서있기": ["orig", "hflip", "pad"],
            "A6 전부": ["orig", "hflip", "fast", "slow", "jitter", "rot+", "rot-", "pad"],
            "A7 A0+서있기": ["orig", "hflipT02", "pad"],
        }
        res = {}
        for k, augs in configs.items():
            res[k] = evaluate(MODELS["RF (지금)"], augs)
            print(f"{k}: LOSO {res[k]['loso_acc']:.3f} 외부 {res[k]['ext_acc']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
        df, pa, det = show(res, "aug_compare")
        print("\n동작별 LOSO:\n" + pa.to_string())
        print("\n외부 영상 예측:\n" + det.to_string())
    elif stage == "model":
        augs = sys.argv[sys.argv.index("--augs") + 1].split(",")
        names = sys.argv[sys.argv.index("--models") + 1].split("|") if "--models" in sys.argv else list(MODELS)
        res = {}
        for k in names:
            res[k] = evaluate(MODELS[k], augs)
            print(f"{k}: LOSO {res[k]['loso_acc']:.3f} 외부 {res[k]['ext_acc']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
        df, pa, det = show(res, "model_compare")
        print("\n동작별 LOSO:\n" + pa.to_string())
        print("\n외부 영상 예측:\n" + det.to_string())


if __name__ == "__main__":
    main()
