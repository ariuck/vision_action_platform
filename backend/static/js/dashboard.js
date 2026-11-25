// Django가 ORM으로 집계해 넘겨준 숫자로 Chart.js 차트를 그린다.
(() => {
  const data = JSON.parse(document.getElementById("chart-data").textContent);
  const BLUE = "#2563eb";
  // 상태 색은 상태에만 쓴다 (대기·처리 중·완료·실패)
  const STATUS = ["#9ca3af", "#f59e0b", "#16a34a", "#dc2626"];
  // 묶음 색은 순서 고정 (마지막 '동작 없음'은 회색)
  const GROUPS = ["#4f46e5", "#0891b2", "#ea580c", "#7c3aed", "#0d9488", "#db2777", "#65a30d", "#94a3b8"];
  Chart.defaults.font.family = '"Noto Sans KR", system-ui, sans-serif';
  Chart.defaults.maintainAspectRatio = false;
  const noLegend = { plugins: { legend: { display: false } } };
  const intAxis = (axis) => ({ [axis]: { beginAtZero: true, ticks: { precision: 0 } } });

  new Chart(document.getElementById("dailyChart"), {
    type: "bar",
    data: {
      labels: data.daily.labels,
      datasets: [
        { label: "완료", data: data.daily.ok, backgroundColor: STATUS[2], borderRadius: 4 },
        { label: "실패", data: data.daily.fail, backgroundColor: STATUS[3], borderRadius: 4 },
      ],
    },
    options: { scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true, ticks: { precision: 0 } } } },
  });

  new Chart(document.getElementById("actionChart"), {
    type: "bar",
    data: { labels: data.actions.labels, datasets: [{ label: "예측 건수", data: data.actions.values, backgroundColor: BLUE, borderRadius: 4 }] },
    options: { indexAxis: "y", ...noLegend, scales: intAxis("x") },
  });

  new Chart(document.getElementById("confChart"), {
    type: "bar",
    data: { labels: data.actions.labels, datasets: [{ label: "평균 신뢰도(%)", data: data.actions.conf, backgroundColor: "#7c3aed", borderRadius: 4 }] },
    options: {
      indexAxis: "y", ...noLegend,
      scales: { x: { beginAtZero: true, max: 100, ticks: { callback: (v) => `${v}%` } } },
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `평균 ${c.raw}%` } } },
    },
  });

  new Chart(document.getElementById("statusChart"), {
    type: "doughnut",
    data: { labels: data.status.labels, datasets: [{ data: data.status.values, backgroundColor: STATUS, borderColor: "#fff", borderWidth: 2 }] },
    options: { plugins: { legend: { position: "right" } } },
  });

  new Chart(document.getElementById("groupChart"), {
    type: "doughnut",
    data: { labels: data.groups.labels, datasets: [{ data: data.groups.values, backgroundColor: GROUPS, borderColor: "#fff", borderWidth: 2 }] },
    options: { plugins: { legend: { position: "right" } } },
  });

  new Chart(document.getElementById("histChart"), {
    type: "bar",
    data: {
      labels: data.conf_hist.labels,
      datasets: [{
        label: "예측 수", data: data.conf_hist.values, borderRadius: 4,
        backgroundColor: data.conf_hist.labels.map((_, i) => (i < 2 ? "#f59e0b" : BLUE)),  // 20% 미만 = 판단 불확실
      }],
    },
    options: { ...noLegend, scales: intAxis("y") },
  });

  new Chart(document.getElementById("reasonChart"), {
    type: "bar",
    data: { labels: data.reasons.labels.length ? data.reasons.labels : ["실패 없음"],
            datasets: [{ label: "실패 수", data: data.reasons.values.length ? data.reasons.values : [0], backgroundColor: STATUS[3], borderRadius: 4 }] },
    options: { indexAxis: "y", ...noLegend, scales: intAxis("x") },
  });

  new Chart(document.getElementById("inputChart"), {
    type: "bar",
    data: { labels: data.input.labels, datasets: [{ label: "작업 수", data: data.input.values, backgroundColor: BLUE, borderRadius: 4 }] },
    options: { ...noLegend, scales: intAxis("y") },
  });

  new Chart(document.getElementById("timeChart"), {
    type: "bar",
    data: { labels: data.input.labels, datasets: [{ label: "평균 처리 시간(초)", data: data.input.avg_sec, backgroundColor: "#0891b2", borderRadius: 4 }] },
    options: { ...noLegend, scales: { y: { beginAtZero: true, ticks: { callback: (v) => `${v}초` } } } },
  });
})();
