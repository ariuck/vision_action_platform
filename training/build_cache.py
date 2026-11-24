"""학습 데이터(원본 + 증강 7가지) Feature를 계산해 캐시로 저장한다.

    cd vision_action_platform && ai_service/.venv/bin/python training/build_cache.py
출력: training/cache/train_floor0.4.pkl (Git 제외)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import AI, build_cache  # noqa: E402

OUT = Path(__file__).resolve().parent / "cache"
OUT.mkdir(exist_ok=True)
cols = json.loads((AI / "models" / "rf_feature_columns.json").read_text())
frame_names = [c.split("_", 1)[1] for c in cols[:45]]
t0 = time.time()
df = build_cache(OUT / "train_floor0.4.pkl", 0.4, frame_names)
print(f"rows {len(df)}  ({df.groupby('aug').size().to_dict()})  {time.time() - t0:.0f}s")
