"""기준값(NO_ACTION_MOTION, MAIN_MIN_HEIGHT, MAIN_MIN_SEC)을 정하기 위한 측정 도구.

영상 여러 개를 YOLO로 추적해 사람마다 움직임 크기·키 비율·보인 시간을 CSV로 저장한다.
  python tools/calibrate_quality.py --out quality.csv 영상1.mp4 영상2.mp4 ...
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from services.model_loader import load_bundle  # noqa: E402
from services.pose_pipeline import track_to_frames, track_video  # noqa: E402
from services.track_quality import measure  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("videos", nargs="+")
    args = parser.parse_args()
    bundle = load_bundle()
    rows = []
    for i, path in enumerate(args.videos, 1):
        info, tracks = track_video(path, bundle)
        for tid, track in sorted(tracks.items()):
            m = measure(track_to_frames(track), info.fps, info.width, info.height)
            rows.append({"video": Path(path).name, "track_id": tid, **m})
        print(f"[{i}/{len(args.videos)}] {Path(path).name}: {len(tracks)} tracks", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
