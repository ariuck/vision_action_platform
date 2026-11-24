// 분석 요청 → 작업 ID 받기 → 상태를 주기적으로 조회 → 완료되면 사람별 결과 표시
(() => {
  const form = document.getElementById("analysis-form");
  const csrf = form.querySelector("[name=csrfmiddlewaretoken]").value;
  const startBtn = document.getElementById("start-btn");
  const current = document.getElementById("current");
  const POLL_MS = 2000;

  // 입력 방식 탭 전환
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
      document.querySelectorAll(".pane").forEach((p) => p.classList.toggle("hidden", p.dataset.pane !== tab.dataset.tab));
      form.input_type.value = tab.dataset.tab;
    });
  });

  const api = (url, options = {}) =>
    fetch(url, { credentials: "same-origin", ...options, headers: { "X-CSRFToken": csrf, ...(options.headers || {}) } });

  function setStep(status) {
    const done = status === "SUCCESS" || status === "FAILURE";
    current.querySelectorAll(".steps li").forEach((li) => {
      const step = li.dataset.step;
      li.className = (step === status || (step === "DONE" && done)) ? "on" : "";
      if (step === "DONE" && done) li.textContent = status === "SUCCESS" ? "완료" : "실패";
    });
  }

  function renderResult(job) {
    const box = document.getElementById("current-result");
    if (job.status === "FAILURE") {
      box.innerHTML = `<p class="error">실패: ${job.error_message || job.error_code}</p>`;
      return;
    }
    if (!job.predictions.length) {
      box.innerHTML = `<p>${job.message || "예측 결과가 없습니다."}</p>`;
      return;
    }
    const rows = job.predictions.map((p) => `
      <tr><td>${p.track_id}</td><td>${p.action_id} ${p.action_name}</td>
      <td>${p.confidence.toFixed(2)} ${p.uncertain ? '<span class="badge warn">판단 불확실</span>' : ""}</td>
      <td>${p.top3.map((t) => `${t.action_name} ${t.confidence.toFixed(2)}`).join(" / ")}</td>
      <td>${p.first_sec}~${p.last_sec}초</td></tr>`).join("");
    const expected = job.expected_action ? `<p class="muted">수행한 동작(사용자 선택): ${job.expected_action_display}</p>` : "";
    box.innerHTML = `${expected}<div class="table-wrap"><table class="table"><thead><tr><th>Track ID</th><th>예측 동작</th><th>Confidence</th>
      <th>후보 3개</th><th>보인 구간</th></tr></thead><tbody>${rows}</tbody></table></div>
      <p class="muted">영상 ${job.video_duration_sec ?? "-"}초 · AI 추론 ${job.ai_processing_sec ?? "-"}초 · 전체 ${job.total_processing_sec ?? "-"}초</p>`;
  }

  async function poll(id) {
    const res = await api(`/api/analyses/${id}/status/`);
    if (!res.ok) return;
    const s = await res.json();
    setStep(s.status);
    document.getElementById("current-message").textContent = s.message || s.status_display;
    if (!s.is_finished) {
      setTimeout(() => poll(id), POLL_MS);
      return;
    }
    const detail = await (await api(`/api/analyses/${id}/`)).json();
    renderResult(detail);
    startBtn.disabled = false;
    setTimeout(() => location.reload(), 4000); // 이력 표 갱신
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const data = new FormData(form);
    if (form.input_type.value === "CAMERA") data.delete("video_file");
    startBtn.disabled = true;
    current.hidden = false;
    document.getElementById("current-result").innerHTML = "";
    document.getElementById("current-message").textContent = "작업을 등록하는 중…";
    const res = await api("/api/analyses/", { method: "POST", body: data });
    const body = await res.json();
    if (!res.ok) {
      const msg = Object.values(body).flat().join(" ");
      document.getElementById("current-message").textContent = "";
      document.getElementById("current-result").innerHTML = `<p class="error">${msg}</p>`;
      startBtn.disabled = false;
      return;
    }
    document.getElementById("current-id").textContent = `#${body.id}`;
    setStep(body.status);
    poll(body.id);
  });

  // 이력: 상세 보기 · 삭제
  document.getElementById("history").addEventListener("click", async (e) => {
    const detailId = e.target.dataset.detail;
    const deleteId = e.target.dataset.delete;
    if (detailId) {
      current.hidden = false;
      document.getElementById("current-id").textContent = `#${detailId}`;
      const job = await (await api(`/api/analyses/${detailId}/`)).json();
      setStep(job.status);
      document.getElementById("current-message").textContent = job.message || "";
      renderResult(job);
      if (!["SUCCESS", "FAILURE"].includes(job.status)) poll(detailId);
      current.scrollIntoView({ behavior: "smooth" });
    }
    if (deleteId && confirm(`#${deleteId} 분석 기록과 결과를 삭제할까요? 되돌릴 수 없습니다.`)) {
      const res = await api(`/api/analyses/${deleteId}/`, { method: "DELETE" });
      if (res.ok) location.reload();
    }
  });
})();
