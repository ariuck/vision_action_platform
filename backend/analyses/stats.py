"""화면·대시보드에 보여 줄 통계 (Django ORM 집계만 사용, FastAPI는 DB에 접근하지 않음).

- 주인공 예측: 배경 인물을 뺀 예측 (is_main=True)
- 대표 예측: 한 작업에서 가장 오래 보인 주인공의 예측 → '수행한 동작'과 비교할 때 사용
- 정답 비교는 사용자가 '수행한 동작'을 골랐을 때만 한다. 고른 값은 사용자 입력이지 검증된 정답이 아니다.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timedelta

from django.db.models import Avg, Count, Q
from django.db.models.functions import TruncDate
from django.utils import timezone

from .actions import ACTIONS
from .models import AnalysisJob, PersonPrediction

NAME = {label: name for _, label, name, _ in ACTIONS}
GROUP_ORDER = list(dict.fromkeys(group for *_, group in ACTIONS))


def _rate(a: int, b: int) -> float | None:
    return round(a / b * 100, 1) if b else None


def representative(job: AnalysisJob) -> PersonPrediction | None:
    """한 작업의 대표 예측 = 주인공 중 가장 오래 보인 사람 (prefetch된 predictions 사용)"""
    mains = [p for p in job.predictions.all() if p.is_main]
    return max(mains, key=lambda p: p.frames) if mains else None


def labeled_accuracy(jobs) -> dict:
    """'수행한 동작'을 고른 완료 작업에서 대표 예측이 맞은 비율 + 동작별 표 + 자주 헷갈린 쌍"""
    per = defaultdict(lambda: {"n": 0, "ok": 0, "wrong": Counter()})
    pairs = Counter()
    for job in jobs:
        rep = representative(job)
        if not job.expected_action or rep is None:
            continue
        row = per[job.expected_action]
        row["n"] += 1
        if rep.action_label == job.expected_action:
            row["ok"] += 1
        else:
            wrong = "동작 없음" if rep.no_action else rep.action_name
            row["wrong"][wrong] += 1
            pairs[(NAME.get(job.expected_action, job.expected_action), wrong)] += 1
    n = sum(r["n"] for r in per.values())
    ok = sum(r["ok"] for r in per.values())
    table = [{"label": lbl, "name": NAME.get(lbl, lbl), "n": r["n"], "ok": r["ok"], "acc": _rate(r["ok"], r["n"]),
              "often": r["wrong"].most_common(1)[0][0] if r["wrong"] else ""}
             for lbl, r in sorted(per.items(), key=lambda kv: (kv[1]["ok"] / kv[1]["n"], -kv[1]["n"]))]
    return {"n": n, "ok": ok, "acc": _rate(ok, n), "table": table,
            "pairs": [{"truth": a, "pred": b, "count": c} for (a, b), c in pairs.most_common(5)]}


def user_summary(user) -> dict:
    jobs = AnalysisJob.objects.filter(user=user)
    total = jobs.count()
    done = jobs.filter(status=AnalysisJob.Status.SUCCESS)
    failed = jobs.filter(status=AnalysisJob.Status.FAILURE).count()
    preds = PersonPrediction.objects.filter(job__user=user, is_main=True, no_action=False)
    top = list(preds.values("action_label").annotate(n=Count("id")).order_by("-n")[:3])
    acc = labeled_accuracy(jobs.filter(status=AnalysisJob.Status.SUCCESS).exclude(expected_action="")
                           .prefetch_related("predictions"))
    return {
        "total": total,
        "success_rate": _rate(done.count(), done.count() + failed),
        "avg_sec": done.aggregate(v=Avg("total_processing_sec"))["v"],
        "avg_conf": preds.aggregate(v=Avg("confidence"))["v"],
        "top_actions": [{"name": NAME.get(t["action_label"], t["action_label"]), "n": t["n"]} for t in top],
        "labeled_n": acc["n"], "labeled_acc": acc["acc"],
    }


def dashboard_stats() -> dict:
    jobs = AnalysisJob.objects.all()
    total = jobs.count()
    by_status = dict(jobs.values_list("status").annotate(n=Count("id")))
    success, failure = by_status.get("SUCCESS", 0), by_status.get("FAILURE", 0)
    done = jobs.filter(status=AnalysisJob.Status.SUCCESS)
    mains = PersonPrediction.objects.filter(is_main=True)
    acted = mains.filter(no_action=False)
    times = done.aggregate(total=Avg("total_processing_sec"), ai=Avg("ai_processing_sec"))
    today = timezone.localdate()

    # 날짜별 (최근 14일, 완료·실패 따로 쌓기)
    start = today - timedelta(days=13)
    daily_rows = (jobs.filter(created_at__date__gte=start).annotate(d=TruncDate("created_at"))
                  .values("d").annotate(ok=Count("id", filter=Q(status="SUCCESS")),
                                        fail=Count("id", filter=Q(status="FAILURE")), n=Count("id")))
    daily = {r["d"]: r for r in daily_rows}
    days = [start + timedelta(days=i) for i in range(14)]

    # 동작별 건수·평균 신뢰도 (주인공·동작 있음)
    act = {r["action_label"]: r for r in acted.values("action_label").annotate(n=Count("id"), conf=Avg("confidence"))}
    # 묶음별 건수 (+ 동작 없음)
    grp = dict(acted.values_list("action_group").annotate(n=Count("id")))
    no_action = mains.filter(no_action=True).count()
    # 신뢰도 분포 (10% 단위)
    hist = [0] * 10
    for c in acted.values_list("confidence", flat=True):
        hist[min(int(c * 10), 9)] += 1
    # 실패 원인
    reasons = list(jobs.filter(status="FAILURE").values("error_code").annotate(n=Count("id")).order_by("-n"))
    # 입력 방식별
    by_input = dict(jobs.values_list("input_type").annotate(n=Count("id")))
    avg_sec = dict(done.values_list("input_type").annotate(s=Avg("total_processing_sec")))

    labeled = labeled_accuracy(done.exclude(expected_action="").prefetch_related("predictions"))
    return {
        "kpi": {
            "total": total, "today": jobs.filter(created_at__date=today).count(),
            "users": jobs.values("user").distinct().count(),
            "success_rate": _rate(success, success + failure), "failure": failure,
            "avg_total_sec": times["total"], "avg_ai_sec": times["ai"],
            "people": mains.count(), "avg_conf": acted.aggregate(v=Avg("confidence"))["v"],
            "uncertain_rate": _rate(acted.filter(uncertain=True).count(), acted.count()),
            "no_action": no_action, "background": PersonPrediction.objects.filter(is_main=False).count(),
            "labeled_n": labeled["n"], "labeled_acc": labeled["acc"],
        },
        "labeled": labeled,
        "chart": {
            "daily": {"labels": [d.strftime("%m-%d") for d in days],
                      "ok": [daily.get(d, {}).get("ok", 0) for d in days],
                      "fail": [daily.get(d, {}).get("fail", 0) for d in days]},
            "actions": {"labels": [f"{aid} {name}" for aid, _, name, _ in ACTIONS],
                        "values": [act.get(label, {}).get("n", 0) for _, label, _, _ in ACTIONS],
                        "conf": [round((act.get(label, {}).get("conf") or 0) * 100, 1) for _, label, _, _ in ACTIONS]},
            "groups": {"labels": GROUP_ORDER + ["동작 없음"], "values": [grp.get(g, 0) for g in GROUP_ORDER] + [no_action]},
            "status": {"labels": [s.label for s in AnalysisJob.Status],
                       "values": [by_status.get(s.value, 0) for s in AnalysisJob.Status]},
            "conf_hist": {"labels": [f"{i * 10}~{i * 10 + 10}%" for i in range(10)], "values": hist},
            "reasons": {"labels": [r["error_code"] or "(코드 없음)" for r in reasons], "values": [r["n"] for r in reasons]},
            "input": {"labels": [t.label for t in AnalysisJob.InputType],
                      "values": [by_input.get(t.value, 0) for t in AnalysisJob.InputType],
                      "avg_sec": [round(avg_sec.get(t.value) or 0, 1) for t in AnalysisJob.InputType]},
        },
    }
