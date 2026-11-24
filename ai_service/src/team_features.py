from __future__ import annotations

# ==================================================
# 3팀 — 무인 상점 이상행동 인식 : 팀 추가 Feature
# ==================================================
# 교재의 38-D Feature(정규화 좌표 34개 + 관절각도 4개)는
# 그대로 사용하고, 우리 팀 17개 동작을 설명하기 위한
# Feature를 따로 계산합니다.
#
# [자세 Feature] 이미지 한 장으로 계산
# 1. torso_tilt_deg       : 상체 기울기 (0° 서 있음, 90° 수평)
# 2. hip_ankle_dy         : 골반과 발목의 높이 차이 (상체 길이 단위)
# 3. bbox_aspect_ratio    : 사람 BBox 가로/세로 비율
# 4. wrist_above_shoulder : 손목이 어깨보다 얼마나 위에 있는지 (상체 길이 단위)
# 5. ankle_height_diff    : 두 발목의 높이 차이 (상체 길이 단위)
#
# [움직임 Feature] 영상에서 앞 Frame과 비교해 계산
# 6. wrist_speed : 손목 이동 속도 (상체 길이 / 초)
# 7. ankle_speed : 발목 이동 속도 (상체 길이 / 초)
# 8. hip_speed   : 골반 중심 이동 속도 (상체 길이 / 초)
#
# 모든 값을 "각도" 또는 "상체 길이로 나눈 비율"로 만들어
# 사람이 화면에서 크거나 작아도 값이 크게 변하지 않게 했습니다.
# 측면에서는 두 어깨가 겹쳐 어깨 폭이 아주 작아질 수 있어서
# 어깨 폭 대신 상체 길이를 기준 길이로 사용합니다.

import math

import numpy as np

from src.feature_utils import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    calculate_angle,
    midpoint,
)


# ==================================================
# [01] 팀 Feature 이름과 판정 기준
# ==================================================

POSE_FEATURE_NAMES = [
    "torso_tilt_deg",
    "hip_ankle_dy",
    "bbox_aspect_ratio",
    "wrist_above_shoulder",
    "ankle_height_diff",
]

MOTION_FEATURE_NAMES = [
    "wrist_speed",
    "ankle_speed",
    "hip_speed",
]

# 눕기    : 상체가 60° 이상 기울고, 다리도 바닥과 나란함
LIE_TILT_DEG = 60.0
LIE_ANKLE_DY = 0.6

# 숙이기  : 상체가 45° 이상 기울었지만 발은 골반보다 한참 아래
BEND_TILT_DEG = 45.0

# 앉기    : 상체는 서 있고, 다리가 바닥과 나란하거나 무릎이 크게 굽음
SIT_KNEE_DEG = 120.0

# 팔 올림 : 손목이 어깨보다 상체 길이의 0.2배 이상 위
ARM_RAISED = 0.2

# 발 들기 : 두 발목 높이 차이가 상체 길이의 0.5배 이상
LEG_RAISED = 0.5


# ==================================================
# [02] 어깨 중심 / 골반 중심 / 상체 길이
# ==================================================

def shoulder_center(keypoints):
    return midpoint(
        keypoints[LEFT_SHOULDER],
        keypoints[RIGHT_SHOULDER],
    )


def hip_center(keypoints):
    return midpoint(
        keypoints[LEFT_HIP],
        keypoints[RIGHT_HIP],
    )


def torso_length(keypoints) -> float:
    """
    어깨 중심과 골반 중심 사이 거리입니다.
    옆에서 봐도 어깨 폭처럼 0에 가까워지지 않습니다.
    """

    return float(
        np.linalg.norm(
            shoulder_center(keypoints)
            - hip_center(keypoints)
        )
    )


# ==================================================
# [03] torso_tilt_deg() - 상체 기울기
# ==================================================
# 골반 중심 → 어깨 중심 벡터와 화면 위쪽 (0, -1) 사이 각도
#   0°   : 상체가 똑바로 서 있음
#   90°  : 상체가 바닥과 수평
#   180° : 머리가 골반보다 아래

def torso_tilt_deg(keypoints) -> float:
    vector = (
        shoulder_center(keypoints)
        - hip_center(keypoints)
    )

    length = float(np.linalg.norm(vector))

    if length < 1e-8:
        return float("nan")

    cosine = float(
        np.clip(-vector[1] / length, -1.0, 1.0)
    )

    return math.degrees(math.acos(cosine))


# ==================================================
# [04] visible side - 잘 보이는 쪽 선택
# ==================================================
# 옆에서 촬영하면 카메라 반대쪽 팔다리는 가려져
# Confidence가 낮거나 위치가 틀릴 수 있습니다.

def visible_ankle(keypoints, confidence):
    if confidence[LEFT_ANKLE] >= confidence[RIGHT_ANKLE]:
        return "left", keypoints[LEFT_ANKLE], float(confidence[LEFT_ANKLE])

    return "right", keypoints[RIGHT_ANKLE], float(confidence[RIGHT_ANKLE])


def visible_knee_angle(keypoints, confidence) -> float:
    """
    골반-무릎-발목 Confidence 평균이 더 높은 쪽 무릎 각도
    """

    left = float(np.mean(confidence[[LEFT_HIP, LEFT_KNEE, LEFT_ANKLE]]))
    right = float(np.mean(confidence[[RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE]]))

    if left >= right:
        return calculate_angle(
            keypoints[LEFT_HIP], keypoints[LEFT_KNEE], keypoints[LEFT_ANKLE]
        )

    return calculate_angle(
        keypoints[RIGHT_HIP], keypoints[RIGHT_KNEE], keypoints[RIGHT_ANKLE]
    )


# ==================================================
# [05] hip_ankle_dy() - 골반과 발목의 높이 차이
# ==================================================
# (발목 y - 골반 y) / 상체 길이
#   약 1 이상 : 발목이 골반보다 한참 아래 (서기, 숙이기)
#   0 근처    : 발목과 골반 높이가 비슷함 (눕기, 다리 뻗고 앉기)

def hip_ankle_dy(keypoints, confidence) -> float:
    length = torso_length(keypoints)

    if length < 1e-8:
        return float("nan")

    _, ankle, _ = visible_ankle(keypoints, confidence)

    return float((ankle[1] - hip_center(keypoints)[1]) / length)


# ==================================================
# [06] bbox_aspect_ratio() - 사람 BBox 가로/세로 비율
# ==================================================

def bbox_aspect_ratio(box) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]

    height = y2 - y1

    if height < 1e-8:
        return float("nan")

    return (x2 - x1) / height


# ==================================================
# [07] wrist_above_shoulder() - 팔 올림
# ==================================================
# (어깨 y - 손목 y) / 상체 길이 를 양쪽에서 계산하고
# 더 높이 올린 쪽 값을 사용합니다.
#   양수 : 손목이 어깨보다 위 (두드리기·던지기·휘두르기 준비)
#   음수 : 손목이 어깨보다 아래
#
# 교재 24번 Arms Up의 "손목 y - 어깨 y"를
# 한 팔만 들어도 잡히도록 바꾼 것입니다.

def wrist_above_shoulder(keypoints, confidence, min_conf=0.5) -> float:
    length = torso_length(keypoints)

    if length < 1e-8:
        return float("nan")

    values = []

    for shoulder, wrist in (
        (LEFT_SHOULDER, LEFT_WRIST),
        (RIGHT_SHOULDER, RIGHT_WRIST),
    ):
        if min(confidence[shoulder], confidence[wrist]) >= min_conf:
            values.append(
                (keypoints[shoulder][1] - keypoints[wrist][1]) / length
            )

    if not values:
        return float("nan")

    return float(max(values))


# ==================================================
# [08] ankle_height_diff() - 두 발목 높이 차이
# ==================================================
# |왼발목 y - 오른발목 y| / 상체 길이
#   0 근처 : 두 발이 같은 높이 (서기, 앉기)
#   큼     : 한 발을 들어 올림 (차기)

def ankle_height_diff(keypoints, confidence, min_conf=0.5) -> float:
    length = torso_length(keypoints)

    if (
        length < 1e-8
        or min(confidence[LEFT_ANKLE], confidence[RIGHT_ANKLE]) < min_conf
    ):
        return float("nan")

    return float(
        abs(keypoints[LEFT_ANKLE][1] - keypoints[RIGHT_ANKLE][1]) / length
    )


# ==================================================
# [09] pose_features() - 자세 Feature 5개
# ==================================================

def pose_features(keypoints, confidence, box) -> dict:
    return {
        "torso_tilt_deg": torso_tilt_deg(keypoints),
        "hip_ankle_dy": hip_ankle_dy(keypoints, confidence),
        "bbox_aspect_ratio": bbox_aspect_ratio(box),
        "wrist_above_shoulder": wrist_above_shoulder(keypoints, confidence),
        "ankle_height_diff": ankle_height_diff(keypoints, confidence),
    }


# ==================================================
# [10] motion_features() - 움직임 Feature 3개
# ==================================================
# 앞 Frame과 같은 관절의 이동 거리를
# 상체 길이로 나누고, 시간(초)으로 나눕니다.
#   단위 : 1초 동안 상체 길이의 몇 배만큼 움직였는지
#
# 손목·발목은 두 쪽 중 더 빠른 쪽을 사용합니다.
# 앞 Frame이 없으면 NaN입니다.

def motion_features(keypoints, prev_keypoints, dt) -> dict:
    empty = {name: float("nan") for name in MOTION_FEATURE_NAMES}

    if prev_keypoints is None or dt <= 0:
        return empty

    length = torso_length(keypoints)

    if length < 1e-8:
        return empty

    def speed(ids):
        return max(
            float(np.linalg.norm(keypoints[i] - prev_keypoints[i]))
            for i in ids
        ) / length / dt

    hip_move = float(
        np.linalg.norm(hip_center(keypoints) - hip_center(prev_keypoints))
    )

    return {
        "wrist_speed": speed([LEFT_WRIST, RIGHT_WRIST]),
        "ankle_speed": speed([LEFT_ANKLE, RIGHT_ANKLE]),
        "hip_speed": hip_move / length / dt,
    }


# ==================================================
# [11] pose_state() - 자세 상태 판정 (서기 / 앉기 / 숙이기 / 눕기)
# ==================================================
# 규칙 순서:
#   1. 상체 >= 60° + 다리도 수평         → lie   (눕기)
#   2. 상체 >= 45° + 발이 골반보다 아래  → bend  (숙이기)
#   3. 상체 < 45° + (다리 수평 또는 무릎 < 120°) → sit (앉기)
#   4. 나머지                             → stand (서기)

def pose_state(features: dict, knee_angle: float) -> str:
    tilt = features["torso_tilt_deg"]
    dy = features["hip_ankle_dy"]

    if tilt >= LIE_TILT_DEG and abs(dy) < LIE_ANKLE_DY:
        return "lie"

    if tilt >= BEND_TILT_DEG and dy >= LIE_ANKLE_DY:
        return "bend"

    if tilt < BEND_TILT_DEG and (
        abs(dy) < LIE_ANKLE_DY
        or (not math.isnan(knee_angle) and knee_angle < SIT_KNEE_DEG)
    ):
        return "sit"

    return "stand"


# ==================================================
# [12] lie_down_rule() - 드러눕기 규칙 (기울기만 vs 조합)
# ==================================================
# Feature 하나만 볼 때와 두 개를 조합할 때를 비교하기 위해
# 두 결과를 함께 반환합니다.

def lie_down_rule(features: dict):
    rule_tilt_only = bool(features["torso_tilt_deg"] >= LIE_TILT_DEG)

    lie_down = bool(
        rule_tilt_only
        and abs(features["hip_ankle_dy"]) < LIE_ANKLE_DY
    )

    return rule_tilt_only, lie_down


# ==================================================
# [13] action_flags() - 팔 올림 / 발 들기
# ==================================================

def action_flags(features: dict):
    arm = features["wrist_above_shoulder"]
    leg = features["ankle_height_diff"]

    arm_raised = bool(not math.isnan(arm) and arm >= ARM_RAISED)
    leg_raised = bool(not math.isnan(leg) and leg >= LEG_RAISED)

    return arm_raised, leg_raised
