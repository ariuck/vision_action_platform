"""3팀 17개 이상행동 목록 (AI 서버의 config/action_classes.csv와 같은 내용).

화면의 '수행한 동작(선택)' 목록과 대시보드 차트 순서에 쓴다.
"""
ACTIONS = [
    ("A001", "kiosk_tap", "키오스크 두드리기", "타격·파손"),
    ("A002", "throw_one_hand", "한손 물건 던지기", "던지기"),
    ("A003", "front_kick", "앞차기", "차기"),
    ("A004", "grab_shake", "잡고 흔들기", "밀기·흔들기"),
    ("A005", "push", "밀치기", "밀기·흔들기"),
    ("A006", "swing", "휘두르기", "타격·파손"),
    ("A007", "sit_floor", "바닥에 앉기", "앉기"),
    ("A008", "sit_on_object", "위에 앉기", "앉기"),
    ("A009", "lie_forward", "앞으로 눕기", "눕기·쓰러짐"),
    ("A010", "lie_backward", "뒤로 눕기", "눕기·쓰러짐"),
    ("A011", "body_slam", "몸통 박치기", "밀기·흔들기"),
    ("A012", "crouch_take_item", "쪼그려 물건 빼내기", "앉기"),
    ("A013", "shotgun", "샷건", "타격·파손"),
    ("A014", "vomit", "구토", "숙이기"),
    ("A015", "fall_sideways", "옆으로 쓰러지기", "눕기·쓰러짐"),
    ("A016", "fall_backward", "뒤로 넘어지기", "눕기·쓰러짐"),
    ("A017", "fall_forward", "앞으로 넘어지기", "눕기·쓰러짐"),
]

ACTION_CHOICES = [(label, f"{aid} {name}") for aid, label, name, _ in ACTIONS]
ACTION_NAMES = {label: name for _, label, name, _ in ACTIONS}
