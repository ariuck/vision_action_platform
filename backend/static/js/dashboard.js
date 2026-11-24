// Django가 ORM으로 집계해 넘겨준 숫자로 Chart.js 차트를 그린다.
(() => {
  const data = JSON.parse(document.getElementById("chart-data").textContent);
  const blue = "#2563eb";
  const statusColors = ["#9ca3af", "#f59e0b", "#16a34a", "#dc2626"];

  new Chart(document.getElementById("actionChart"), {
    type: "bar",
    data: { labels: data.actions.labels, datasets: [{ label: "예측 건수", data: data.actions.values, backgroundColor: blue }] },
    options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { ticks: { precision: 0 } } } },
  });
  new Chart(document.getElementById("statusChart"), {
    type: "doughnut",
    data: { labels: data.status.labels, datasets: [{ data: data.status.values, backgroundColor: statusColors }] },
  });
  new Chart(document.getElementById("inputChart"), {
    type: "bar",
    data: { labels: data.input.labels, datasets: [{ label: "작업 수", data: data.input.values, backgroundColor: blue }] },
    options: { plugins: { legend: { display: false } }, scales: { y: { ticks: { precision: 0 } } } },
  });
  new Chart(document.getElementById("timeChart"), {
    type: "bar",
    data: { labels: data.input.labels, datasets: [{ label: "평균 처리 시간(초)", data: data.input.avg_sec, backgroundColor: "#7c3aed" }] },
    options: { plugins: { legend: { display: false } } },
  });
})();
