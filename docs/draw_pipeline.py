"""전체 흐름도: 팀 촬영 데이터 → 특징·학습 → 웹 서비스 (한 장 요약).

출력: docs/images/pipeline.png
실행: (한글 폰트가 있는 환경) python docs/draw_pipeline.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent / "images" / "pipeline.png"
W, H = 1500, 1900
RED, GRAY = "#dc2626", "#4b5563"

KIND = {  # 채움, 테두리, 선 모양
    "config": ("#dbeafe", "#2563eb", "-"),
    "raw": ("#ede9fe", "#7c3aed", "-"),
    "code": ("#f3f4f6", "#374151", "-"),
    "check": ("#ffffff", "#6b7280", "--"),
    "csv": ("#dcfce7", "#16a34a", "-"),
    "model": ("#f3e8ff", "#9333ea", "-"),
    "result": ("#fef3c7", "#b45309", "-"),
}

# id: (중심 x, 중심 y, 너비, 높이, 종류, 제목, 요약)
N = {
    # 1. 데이터
    "classes": (250, 175, 380, 84, "config", "action_classes.csv", "팀이 정한 17개 이상행동\nA001 키오스크 두드리기 ~ A017 앞으로 넘어지기"),
    "mp4": (1250, 175, 380, 84, "raw", "팀 촬영 영상 255개", "5명 × 17동작 × 정면·측면·후면\n(얼굴이 나와 Git 제외)"),
    "kp_code": (750, 300, 560, 72, "code", "YOLO11n-pose Keypoint 추출", "Frame마다 관절 17개 · 파일명으로 정답(action_label)"),
    "kp": (750, 425, 640, 84, "csv", "Keypoint CSV 255개", "1행 = Frame 1개 · 관절 x·y·신뢰도 + action_label\n자동 QA + 확인 사진으로 255개 모두 확인"),
    "seg": (750, 548, 600, 72, "code", "동작 구간 자르기", "관절 움직임 크기로 시작·끝 (기준선 최소값 0.4)"),
    # 2. 특징·학습
    "lib": (1265, 690, 360, 84, "config", "ai_service/services · src", "서비스와 학습이 같이 쓰는\nFeature 계산 코드"),
    "feat": (750, 690, 560, 84, "code", "45-D Frame 특징", "정규화 좌표 34 + 관절 각도 4 + 팀 특징 7\n(상체 기울기·골반 높이·손목·발목 속도)"),
    "vec": (750, 830, 600, 84, "csv", "1,575 Feature (사람 1명)", "30 Step 시간표 1,350 + 결정적인 순간 225\n(최대·최소·평균·표준편차 + 가장 크게 움직인 순간)"),
    "aug": (235, 830, 330, 84, "code", "증강 7가지", "반전·빠르게·느리게·잡음\n±7도 기울임·앞뒤 서 있기"),
    "train": (750, 965, 480, 72, "code", "ExtraTrees 학습 (training/)", "500 Tree · Feature 10% · 원본 255 + 증강 = 2,040행"),
    "loso": (1265, 965, 360, 84, "check", "5명 교차 평가 (LOSO)", "한 명씩 빼고 시험 · 시드 3번 평균\n0.809 → 0.881"),
    "model": (750, 1095, 600, 84, "model", "rf_action_baseline.joblib",
              "+ rf_feature_columns.json · rf_training_info.json\nFastAPI가 시작할 때 한 번 불러옴"),
    # 3. 서비스
    "input": (235, 1265, 330, 84, "raw", "영상 입력", "영상 업로드 (MP4 등)\nIP Webcam 8초 녹화 + 미리보기"),
    "django": (750, 1265, 520, 84, "code", "Django · DRF", "로그인·권한 확인 → AnalysisJob 저장\n→ 202 + 작업 ID 바로 반환"),
    "redis": (1265, 1265, 360, 72, "code", "Redis", "작업 대기열"),
    "celery": (1265, 1415, 360, 96, "code", "Celery Worker", "카메라 녹화 · 영상 전달\nAI 결과를 Django ORM으로 저장"),
    "fastapi": (750, 1415, 560, 96, "code", "FastAPI  POST /api/v1/predict/", "YOLO 추적 → 사람(Track ID)별 1,575 Feature\n→ 배경·가짜 사람 제외 → 동작 없음 판단 → 예측"),
    "db": (1265, 1570, 360, 84, "csv", "PostgreSQL", "AnalysisJob (요청·상태)\nPersonPrediction (사람별 결과)"),
    "ui": (750, 1700, 640, 96, "result", "분석 화면 · 관리자 대시보드", "사람별 동작·신뢰도·묶음 확률·정답 일치\n성공률·처리 시간·동작별 정답률 차트"),
}

# (출발, 도착, 글자, 정답 흐름?, 경유점)
E = [
    ("classes", "kp_code", "A001~A017 → action_label", True, []),
    ("mp4", "kp_code", "MP4", False, []),
    ("kp_code", "kp", "", True, []),
    ("kp", "seg", "", True, []),
    ("seg", "feat", "구간 안 Frame", True, []),
    ("lib", "feat", "같은 함수", False, []),
    ("feat", "vec", "", True, []),
    ("vec", "aug", "학습 영상만", True, []),
    ("aug", "train", "2,040행", True, [(235, 965)]),
    ("vec", "train", "X · y", True, []),
    ("train", "loso", "", False, []),
    ("train", "model", "joblib.dump", False, []),
    ("model", "fastapi", "", False, [(425, 1095), (425, 1415)]),   # 입력 상자와 Django 상자 사이로
    ("input", "django", "분석 요청", False, []),
    ("django", "redis", "작업 등록", False, []),
    ("redis", "celery", "작업 꺼냄", False, []),
    ("celery", "fastapi", "영상 (multipart)", False, []),
    ("celery", "db", "ORM 저장", False, []),
    ("db", "ui", "상태·결과 조회 (DRF)", False, [(1265, 1700)]),
]


def clip(k, p):
    cx, cy, w, h, *_ = N[k]
    dx, dy = p[0] - cx, p[1] - cy
    if dx == 0 and dy == 0:
        return cx, cy
    s = min((w / 2) / abs(dx) if dx else 1e9, (h / 2) / abs(dy) if dy else 1e9)
    return cx + dx * s, cy + dy * s


def main():
    for name in ("Noto Sans CJK KR", "Noto Sans KR", "NanumGothic"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    ax.text(W / 2, 40, "무인 상점 이상행동 분석 — 전체 흐름", fontsize=22, weight="bold", ha="center", color="#1e3a8a")
    ax.text(W / 2, 76, "팀 촬영 영상 → Keypoint → 동작 구간 → 1,575 Feature → ExtraTrees → Django·Celery·FastAPI 웹 서비스",
            fontsize=12.5, ha="center", color="#4b5563")
    zones = [(100, 600, "1. 데이터 준비 (팀 촬영 · subject10_pose_action)", "#f8fafc", "#cbd5e1"),
             (608, 1150, "2. 특징 · 학습 (training/)", "#f0fdf4", "#bbf7d0"),
             (1158, 1770, "3. 웹 서비스 (Docker Compose 5개 컨테이너)", "#fffbeb", "#fde68a")]
    for y0, y1, title, fc, ec in zones:
        ax.add_patch(Rectangle((20, y0), W - 40, y1 - y0 - 8, fc=fc, ec=ec, lw=1.2, zorder=0))
        ax.text(34, y0 + 16, title, fontsize=12.5, weight="bold", color="#334155", va="center", zorder=1)

    for src, dst, label, red, way in E:
        first = way[0] if way else N[dst][:2]
        last = way[-1] if way else N[src][:2]
        pts = [clip(src, first)] + list(way) + [clip(dst, last)]
        color = RED if red else GRAY
        lw = 2.2 if red else 1.5
        ls = (0, (5, 3)) if red else "-"
        ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]], color=color, lw=lw, ls=ls, zorder=2)
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=15, lw=lw, color=color,
                                     linestyle=ls, zorder=2))
        if label:
            segs = list(zip(pts[:-1], pts[1:]))
            a, b = max(segs, key=lambda s: (s[0][0] - s[1][0]) ** 2 + (s[0][1] - s[1][1]) ** 2)
            ax.text((a[0] + b[0]) / 2 + 6, (a[1] + b[1]) / 2, label, fontsize=9.5, color=color, ha="left",
                    va="center", zorder=7, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))

    for k, (cx, cy, w, h, kind, title, desc) in N.items():
        x, y = cx - w / 2, cy - h / 2
        fc, ec, ls = KIND[kind]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8", fc=fc, ec=ec, lw=2,
                                    ls=ls, zorder=5))
        ax.text(cx, y + 20, title, fontsize=11.2, weight="bold", ha="center", va="center", color="#111827", zorder=6)
        ax.text(cx, y + 20 + (h - 20) / 2 + 2, desc, fontsize=9.4, ha="center", va="center", color="#374151",
                linespacing=1.35, zorder=6)

    ly = 1808
    items = [("config", "정의·공통 코드"), ("raw", "영상 입력"), ("code", "처리 단계"), ("check", "평가"),
             ("csv", "데이터·DB"), ("model", "학습 모델"), ("result", "사용자 화면")]
    x = 40
    for kind, text in items:
        fc, ec, ls = KIND[kind]
        ax.add_patch(FancyBboxPatch((x, ly - 9), 30, 18, boxstyle="round,pad=0,rounding_size=4", fc=fc, ec=ec,
                                    lw=1.6, ls=ls, zorder=5))
        ax.text(x + 38, ly, text, fontsize=10, va="center", zorder=6)
        x += 38 + len(text) * 11.5 + 30
    ax.plot([x, x + 34], [ly, ly], color=RED, lw=2.2, ls=(0, (5, 3)))
    ax.text(x + 42, ly, "action_label(정답) 흐름", fontsize=10, color=RED, va="center")
    ax.text(W / 2, 1855, "학습 때와 서비스 때 Feature를 같은 코드로 계산한다. 학습 코드와 서비스 계산을 비교해 최대 차이 0.000008을 확인했다.",
            fontsize=10.5, ha="center", va="center", color="#374151")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=100, facecolor="white")
    print("[SAVED]", OUT)


if __name__ == "__main__":
    main()
