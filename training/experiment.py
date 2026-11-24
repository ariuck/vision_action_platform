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
ext_all = pd.read_pickle(CACHE / "external_floor0.4.pkl")
if "aug" not in ext_all.columns:
    ext_all["aug"] = "orig"
ext_df = ext_all[ext_all.aug == "orig"].reset_index(drop=True)
FEATS = [c for c in train_df.columns if c.startswith("f") and c[1:].isdigit()]


def rf(**kw):
    return RandomForestClassifier(n_estimators=kw.pop("n", 300), class_weight="balanced", random_state=42,
                                  n_jobs=-1, **kw)


MODELS = {
    "RF (지금)": lambda: rf(),
    "RF 500·특징10%": lambda: rf(n=500, max_features=0.1),
    "RF 500·잎2": lambda: rf(n=500, min_samples_leaf=2),
    "ExtraTrees 500": lambda: ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1),
    "ExtraTrees 1000": lambda: ExtraTreesClassifier(n_estimators=1000, class_weight="balanced", random_state=42, n_jobs=-1),
    "ExtraTrees 500·특징10%": lambda: ExtraTreesClassifier(n_estimators=500, max_features=0.1, class_weight="balanced",
                                                         random_state=42, n_jobs=-1),
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


SEEDS = [42, 7, 123]


def evaluate_seeds(make_model, augs: list[str]) -> dict:
    """Random Forest는 무작위로 나무를 만들어서, 데이터 순서만 바뀌어도 1~2%p 달라진다. 3번 돌려 평균."""
    runs = []
    for sd in SEEDS:
        def mk(sd=sd):
            m = make_model()
            for est in ([m] + [e for _, e in getattr(m, "estimators", [])] + list(getattr(m, "named_steps", {}).values())):
                if hasattr(est, "random_state"):
                    est.set_params(random_state=sd)
            return m
        runs.append(evaluate(mk, augs))
    out = {k: float(np.mean([r[k] for r in runs])) for k in runs[0] if k not in ("per_action", "ext_detail")}
    out["loso_std_seed"] = float(np.std([r["loso_acc"] for r in runs]))
    out["per_action"] = pd.DataFrame([r["per_action"] for r in runs]).mean().to_dict()
    out["ext_detail"] = runs[0]["ext_detail"]
    return out


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


def evaluate_tta(make_model, augs: list[str], tta: list[str]) -> dict:
    """예측할 때 증강(TTA): 같은 영상의 원본·반전·기울임 버전 확률을 평균 내 답을 정한다."""
    data = select(augs)
    accs = []
    for s in SUBJECTS:
        tr = data[data.subject != s]
        m = make_model()
        m.fit(tr[FEATS].to_numpy(float), tr.action_label.to_numpy())
        te = train_df[(train_df.subject == s) & (train_df.aug.isin(tta))]
        p = pd.DataFrame(m.predict_proba(te[FEATS].to_numpy(float)), columns=m.classes_)
        p["clip_id"] = te.clip_id.to_numpy()
        avg = p.groupby("clip_id").mean()
        truth = te.drop_duplicates("clip_id").set_index("clip_id").action_label
        accs.append(float((avg.idxmax(axis=1) == truth.loc[avg.index]).mean()))
    m = make_model()
    m.fit(data[FEATS].to_numpy(float), data.action_label.to_numpy())
    te = ext_all[ext_all.aug.isin(tta)]
    p = pd.DataFrame(m.predict_proba(te[FEATS].to_numpy(float)), columns=m.classes_)
    p["clip_id"] = te.clip_id.to_numpy()
    avg = p.groupby("clip_id").mean()
    truth = te.drop_duplicates("clip_id").set_index("clip_id").loc[avg.index]
    ok = avg.idxmax(axis=1) == truth.action_label
    return {"loso_acc": float(np.mean(accs)), "loso_min": float(np.min(accs)), "ext_acc": float(ok.mean()),
            "ext_new5": float(ok[truth.source == "new_video"].mean()), "ext_detail": avg.idxmax(axis=1).to_dict()}


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
        ev = evaluate_seeds if "--seeds" in sys.argv else evaluate
        for k in names:
            res[k] = ev(MODELS[k], augs)
            print(f"{k}: LOSO {res[k]['loso_acc']:.3f} 외부 {res[k]['ext_acc']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
        df, pa, det = show(res, "model_compare_3seeds" if "--seeds" in sys.argv else "model_compare")
        print("\n동작별 LOSO:\n" + pa.to_string())
        print("\n외부 영상 예측:\n" + det.to_string())
    elif stage == "aug3":   # 상위 후보를 무작위 시드 3개로 다시 비교
        configs = {
            "A0 지금": ["orig", "hflipT02"],
            "A1 반전 전부": ["orig", "hflip"],
            "A2 A1+속도": ["orig", "hflip", "fast", "slow"],
            "A3 A1+잡음": ["orig", "hflip", "jitter"],
            "A4 A1+기울임": ["orig", "hflip", "rot+", "rot-"],
            "A8 A4+잡음": ["orig", "hflip", "rot+", "rot-", "jitter"],
            "A9 A4+속도": ["orig", "hflip", "rot+", "rot-", "fast", "slow"],
            "A6 전부": ["orig", "hflip", "fast", "slow", "jitter", "rot+", "rot-", "pad"],
        }
        res = {}
        for k, augs in configs.items():
            res[k] = evaluate_seeds(MODELS["RF (지금)"], augs)
            print(f"{k}: LOSO {res[k]['loso_acc']:.3f} (±{res[k]['loso_std_seed']:.3f}) 외부 {res[k]['ext_acc']:.2f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
        df, pa, det = show(res, "aug_compare_3seeds")
        print("\n동작별 LOSO:\n" + pa.to_string())
    elif stage == "fps30":   # 30fps 흉내 증강 효과 + 30fps로 바꾼 학습 영상으로도 평가
        model = MODELS["ExtraTrees 500·특징10%"]
        base = ["orig", "hflip", "fast", "slow", "jitter", "rot+", "rot-", "pad"]
        res = {"A6 (30fps 없음)": evaluate_seeds(model, base), "A6 + fps30": evaluate_seeds(model, base + ["fps30"])}
        for k, r in res.items():
            print(f"{k}: LOSO {r['loso_acc']:.3f} 최저 {r['loso_min']:.3f} 외부 {r['ext_acc']:.2f} (새 영상 {r['ext_new5']:.2f})")
        # 빠진 사람의 영상을 30fps로 바꿔 평가: 30fps 입력에서 얼마나 떨어지는지
        for k, augs in (("A6 (30fps 없음)", base), ("A6 + fps30", base + ["fps30"])):
            accs = []
            for sd in SEEDS:
                for s in SUBJECTS:
                    tr = select(augs)
                    tr = tr[tr.subject != s]
                    te = train_df[(train_df.subject == s) & (train_df.aug == "fps30")]
                    m = model()
                    m.set_params(random_state=sd)
                    m.fit(tr[FEATS].to_numpy(float), tr.action_label.to_numpy())
                    accs.append(accuracy_score(te.action_label, m.predict(te[FEATS].to_numpy(float))))
            print(f"  {k} → 30fps로 바꾼 영상 LOSO {np.mean(accs):.3f}")
        show(res, "fps30_compare")
    elif stage == "tta":    # 예측할 때 증강 비교 (모델·학습 증강은 고정)
        augs = sys.argv[sys.argv.index("--augs") + 1].split(",")
        model = sys.argv[sys.argv.index("--model") + 1]
        sets = {"원본만": ["orig"], "원본+반전": ["orig", "hflip"], "원본+반전+기울임": ["orig", "hflip", "rot+", "rot-"]}
        rows = {}
        for k, tta in sets.items():
            runs = []
            for sd in SEEDS:
                def mk(sd=sd):
                    m = MODELS[model]()
                    for est in [m] + [e for _, e in getattr(m, "estimators", [])]:
                        if hasattr(est, "random_state"):
                            est.set_params(random_state=sd)
                    return m
                runs.append(evaluate_tta(mk, augs, tta))
            rows[k] = {x: float(np.mean([r[x] for r in runs])) for x in ("loso_acc", "loso_min", "ext_acc", "ext_new5")}
            rows[k]["ext_detail"] = runs[0]["ext_detail"]
            print(f"TTA {k}: LOSO {rows[k]['loso_acc']:.3f} 외부 {rows[k]['ext_acc']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
        df = pd.DataFrame({k: {x: v for x, v in r.items() if x != "ext_detail"} for k, r in rows.items()}).T
        print(df.round(3).to_string())
        df.to_csv(RESULTS / "tta_compare.csv", encoding="utf-8-sig")
        det = pd.DataFrame({k: r["ext_detail"] for k, r in rows.items()})
        print(det.to_string())


if __name__ == "__main__":
    main()
