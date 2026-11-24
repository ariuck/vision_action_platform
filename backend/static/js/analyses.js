// 분석 화면
// - 영상 업로드: 고른 파일을 바로 미리보기
// - IP Webcam: 서버를 거친 카메라 영상 + 사람 뼈대 표시, 시작하면 3초 뒤 녹화
// - 분석 요청 → 작업 ID → 상태를 주기적으로 조회 → 단계(대기·영상 수집·AI 분석·완료)와 결과 표시
(() => {
  const $ = (id) => document.getElementById(id);
  const form = $("analysis-form");
  const csrf = form.querySelector("[name=csrfmiddlewaretoken]").value;
  const startBtn = $("start-btn");
  const POLL_MS = 1000;
  const STEPS = ["QUEUED", "CAPTURING", "ANALYZING", "DONE"];
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const api = (url, options = {}) =>
    fetch(url, { credentials: "same-origin", ...options, headers: { "X-CSRFToken": csrf, ...(options.headers || {}) } });
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  let mode = "UPLOAD";
  let currentStage = null;

  // ---------------- 탭 전환 ----------------
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      mode = tab.dataset.tab;
      document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
      document.querySelectorAll(".pane").forEach((p) => p.classList.toggle("hidden", p.dataset.pane !== mode));
      form.input_type.value = mode;
      $("preview-title").textContent = mode === "CAMERA" ? "IP Webcam 미리보기" : "영상 업로드 미리보기";
      $("capture-info").classList.toggle("hidden", mode !== "CAMERA");
      mode === "CAMERA" ? camera.start() : camera.stop();
    });
  });

  // ---------------- 업로드 미리보기 ----------------
  const fileInput = $("video-file");
  const uploadVideo = $("upload-preview");
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (uploadVideo.src) URL.revokeObjectURL(uploadVideo.src);
    uploadVideo.hidden = !file;
    $("upload-empty").hidden = !!file;
    if (file) uploadVideo.src = URL.createObjectURL(file);
  });

  // ---------------- IP Webcam 미리보기 + 뼈대 ----------------
  const camera = (() => {
    const img = $("cam-img");
    if (!img) return { start() {}, stop() {}, name: () => "" };
    const canvas = $("cam-overlay");
    const ctx = canvas.getContext("2d");
    const state = $("cam-state");
    const nameEl = $("camera-name");
    const BONES = [[5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];
    let running = false;
    let loopId = 0;

    const setState = (text) => { state.textContent = text; state.hidden = !text; };
    const clear = () => ctx.clearRect(0, 0, canvas.width, canvas.height);

    function draw(people) {
      canvas.width = img.clientWidth;
      canvas.height = img.clientHeight;
      clear();
      if (!$("show-skeleton").checked) return;
      const W = canvas.width, H = canvas.height;
      ctx.lineWidth = 3;
      ctx.strokeStyle = "#2dd4bf";
      ctx.fillStyle = "#ffffff";
      people.forEach((p) => {
        const k = p.keypoints;
        BONES.forEach(([a, b]) => {
          if (k[a][2] < 0.4 || k[b][2] < 0.4) return;
          ctx.beginPath();
          ctx.moveTo(k[a][0] * W, k[a][1] * H);
          ctx.lineTo(k[b][0] * W, k[b][1] * H);
          ctx.stroke();
        });
        k.slice(5).forEach(([x, y, c]) => {
          if (c < 0.4) return;
          ctx.beginPath();
          ctx.arc(x * W, y * H, 4, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        });
      });
    }

    async function statusLoop(id) {
      while (running && id === loopId) {
        try {
          const s = await (await api(`/cameras/${nameEl.value}/status/`)).json();
          if (s.live) setState("");
          else if (s.error === "CAMERA_UNREACHABLE") setState("카메라에 연결할 수 없습니다. 폰 앱과 Wi-Fi를 확인하세요.");
          else if (s.error === "CAMERA_DISCONNECTED") setState("카메라 연결이 끊겼습니다. 다시 연결 중…");
          else setState("카메라 연결 중…");
        } catch (e) { setState("서버 연결 확인 중…"); }
        await sleep(2000);
      }
    }

    async function poseLoop(id) {
      while (running && id === loopId) {
        // AI 분석 중에는 추론 서버를 비워 두려고 뼈대 요청을 쉰다
        if (!$("show-skeleton").checked || currentStage === "ANALYZING" || !state.hidden) {
          clear();
          await sleep(400);
          continue;
        }
        try {
          const res = await api(`/cameras/${nameEl.value}/pose/`);
          draw(res.ok ? (await res.json()).people : []);
        } catch (e) { clear(); }
        await sleep(120);
      }
    }

    return {
      name: () => nameEl.value,
      start() {
        if (running) return;
        if (!nameEl.value) {
          setState("등록된 카메라가 없습니다. 아래 '+ 카메라 추가'로 휴대폰 주소를 등록해 주세요.");
          return;
        }
        running = true;
        loopId += 1;
        setState("카메라 연결 중…");
        img.src = `/cameras/${nameEl.value}/stream/?t=${Date.now()}`;
        statusLoop(loopId);
        poseLoop(loopId);
      },
      stop() {
        running = false;
        loopId += 1;
        img.removeAttribute("src"); // 스트림 연결을 끊어야 서버도 카메라 연결을 닫는다
        clear();
      },
    };
  })();
  // ---------------- 카메라 등록·삭제 ----------------
  const camSelect = $("camera-name");
  const restartCamera = () => { camera.stop(); if (mode === "CAMERA") camera.start(); };
  camSelect.addEventListener("change", restartCamera);

  async function reloadCameras(select) {
    const res = await api("/api/cameras/");
    if (!res.ok) return;
    const { cameras } = await res.json();
    camSelect.innerHTML = cameras.map((c) => `<option value="${esc(c.name)}">${esc(c.name)} · ${esc(c.address)}</option>`).join("");
    if (select) camSelect.value = select;
    $("cam-del").hidden = !cameras.some((c) => c.deletable && c.name === camSelect.value);
    camSelect.onchange = () => { $("cam-del").hidden = !cameras.some((c) => c.deletable && c.name === camSelect.value); };
  }
  reloadCameras();

  $("cam-add-toggle").addEventListener("click", () => { $("cam-add").hidden = !$("cam-add").hidden; });
  $("cam-add").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); $("cam-add-btn").click(); } });
  $("cam-add-btn").addEventListener("click", async () => {
    const body = { name: $("cam-new-name").value.trim(), ip: $("cam-new-ip").value.trim(),
                   port: $("cam-new-port").value, path: $("cam-new-path").value.trim() };
    const res = await api("/api/cameras/", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json();
    if (!res.ok) {
      $("cam-add-error").textContent = (data.errors || [data.detail]).join(" ");
      return;
    }
    $("cam-add-error").textContent = "";
    $("cam-add").hidden = true;
    await reloadCameras(data.name);
    restartCamera();
  });
  $("cam-del").addEventListener("click", async () => {
    const name = camSelect.value;
    if (!name || !confirm(`'${name}' 카메라 등록을 삭제할까요?`)) return;
    const res = await api(`/api/cameras/${encodeURIComponent(name)}/`, { method: "DELETE" });
    if (res.ok) { await reloadCameras(); restartCamera(); }
  });

  // ---------------- 분석 상태 표시 ----------------
  function setFlow(stage, inputType) {
    currentStage = stage;
    const failed = stage === "FAILED";
    const reached = failed ? lastReached : STEPS.indexOf(stage);
    if (!failed) lastReached = reached;
    $("flow").querySelectorAll("li").forEach((li, i) => {
      const step = li.dataset.step;
      li.className = "";
      if (step === "CAPTURING" && inputType === "UPLOAD") li.classList.add("skip");
      else if (failed && i === reached + 1) li.classList.add("fail");
      else if (i < reached || (i === reached && stage === "DONE")) li.classList.add("done");
      else if (i === reached) li.classList.add("active");
    });
    const last = $("flow").lastElementChild.querySelector("span:last-child");
    last.textContent = failed ? "실패" : "완료";
  }
  let lastReached = -1;

  // 녹화 남은 시간 (서버 시계 기준)
  let recTimer = null;
  function showRecording(s) {
    const badge = $("rec-badge");
    if (!badge) return;
    if (s.stage !== "CAPTURING" || !s.capture_started_at) {
      badge.hidden = true;
      clearInterval(recTimer);
      recTimer = null;
      return;
    }
    const offset = Date.now() - Date.parse(s.server_time);
    const startAt = Date.parse(s.capture_started_at);
    badge.hidden = false;
    clearInterval(recTimer);
    const tick = () => {
      const left = Math.max(0, s.capture_seconds - (Date.now() - offset - startAt) / 1000);
      $("rec-left").textContent = `${left.toFixed(1)}초`;
    };
    tick();
    recTimer = setInterval(tick, 100);
  }

  function renderResult(job) {
    const box = $("current-result");
    if (job.status === "FAILURE") {
      box.innerHTML = `<p class="error">실패 (${esc(job.error_code)}): ${esc(job.error_message)}</p>`;
      return;
    }
    if (job.status !== "SUCCESS") {
      box.innerHTML = `<p class="muted center">분석이 끝나면 결과가 표시됩니다.</p>`;
      return;
    }
    if (!job.predictions.length) {
      box.innerHTML = `<p>${esc(job.message || "예측 결과가 없습니다.")}</p>`;
      return;
    }
    const rows = job.predictions.map((p) => `
      <tr><td class="center">${p.track_id}</td>
      <td><strong>${esc(p.action_name)}</strong><br><small class="muted">${esc(p.action_id)} ${esc(p.action_label)} · ${p.first_sec}~${p.last_sec}초</small></td>
      <td class="center"><strong>${Math.round(p.confidence * 100)}%</strong>${p.uncertain ? '<br><span class="badge warn">판단 불확실</span>' : ""}</td></tr>
      <tr class="sub"><td></td><td colspan="2"><small class="muted">후보: ${p.top3.map((t) => `${esc(t.action_name)} ${Math.round(t.confidence * 100)}%`).join(" · ")}</small></td></tr>`).join("");
    const expected = job.expected_action ? `<p class="muted">수행한 동작(직접 선택): ${esc(job.expected_action_display)}</p>` : "";
    box.innerHTML = `<div class="table-wrap"><table class="table result-table"><thead><tr><th>Track ID</th><th>예측 동작</th><th>신뢰도</th></tr></thead>
      <tbody>${rows}</tbody></table></div>${expected}
      <p class="muted">영상 ${job.video_duration_sec ?? "-"}초 · AI 추론 ${job.ai_processing_sec ?? "-"}초 · 전체 ${job.total_processing_sec ?? "-"}초</p>`;
  }

  async function poll(id) {
    while (true) {
      const res = await api(`/api/analyses/${id}/status/`);
      if (!res.ok) return;
      const s = await res.json();
      setFlow(s.stage, s.input_type);
      showRecording(s);
      $("current-message").textContent = s.is_finished
        ? (s.status === "SUCCESS" ? (s.message || "분석이 끝났습니다.") : "분석에 실패했습니다.")
        : { QUEUED: "작업 대기열에서 순서를 기다리는 중…", CAPTURING: "카메라 앞에서 동작해 주세요.", ANALYZING: "관절 추출과 동작 예측 중… (CPU라 20~30초 걸릴 수 있음)" }[s.stage] || s.stage_display;
      if (s.is_finished) break;
      await sleep(POLL_MS);
    }
    const job = await (await api(`/api/analyses/${id}/`)).json();
    renderResult(job);
    startBtn.disabled = false;
    refreshHistory();
  }

  async function countdown() {
    const el = $("countdown");
    if (!el) return;
    el.hidden = false;
    for (const n of [3, 2, 1]) {
      el.textContent = n;
      await sleep(1000);
    }
    el.hidden = true;
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (mode === "UPLOAD" && !fileInput.files.length) {
      $("current-result").innerHTML = `<p class="error">분석할 영상 파일을 골라 주세요.</p>`;
      return;
    }
    startBtn.disabled = true;
    $("current-result").innerHTML = `<p class="muted center">분석이 끝나면 결과가 표시됩니다.</p>`;
    $("current-id").textContent = "";
    lastReached = -1;
    setFlow("QUEUED", mode);
    if (mode === "CAMERA") {
      $("current-message").textContent = "카메라 앞에서 준비해 주세요.";
      await countdown();
    }
    const data = new FormData(form);
    if (mode === "CAMERA") data.delete("video_file");
    $("current-message").textContent = mode === "UPLOAD" ? "영상을 올리는 중…" : "작업을 등록하는 중…";
    const res = await api("/api/analyses/", { method: "POST", body: data });
    const body = await res.json();
    if (!res.ok) {
      $("current-message").textContent = "";
      $("current-result").innerHTML = `<p class="error">${esc(Object.values(body).flat().join(" "))}</p>`;
      currentStage = null;
      startBtn.disabled = false;
      return;
    }
    $("current-id").textContent = `#${body.id}`;
    poll(body.id);
  });

  // ---------------- 최근 이력 ----------------
  function historyRow(j) {
    const result = j.predictions.length
      ? j.predictions.map((p) => `<div>ID ${p.track_id} · ${esc(p.action_name)} · ${Math.round(p.confidence * 100)}%${p.uncertain ? ' <span class="badge warn">판단 불확실</span>' : ""}</div>`).join("")
      : j.status === "FAILURE" ? `<span class="error">${esc((j.error_message || "").slice(0, 60))}</span>`
        : j.status === "SUCCESS" ? esc(j.message) : "-";
    return `<tr><td>${j.id}</td><td>${esc(j.input_type_display)}</td>
      <td>${esc(j.title || "-")}${j.expected_action ? `<br><small class="muted">수행: ${esc(j.expected_action_display)}</small>` : ""}</td>
      <td><span class="badge ${j.status.toLowerCase()}">${esc(j.status_display)}</span></td><td>${result}</td>
      <td>${j.total_processing_sec ? `${j.total_processing_sec.toFixed(1)}초` : "-"}</td>
      <td class="nowrap"><button type="button" class="link" data-detail="${j.id}">상세</button>
      <button type="button" class="link danger" data-delete="${j.id}">삭제</button></td></tr>`;
  }

  async function refreshHistory() {
    const res = await api("/api/analyses/");
    if (!res.ok) return;
    const list = (await res.json()).results;
    $("history").querySelector("tbody").innerHTML = list.length
      ? list.map(historyRow).join("")
      : `<tr><td colspan="7" class="muted">아직 분석 기록이 없습니다.</td></tr>`;
  }

  $("history").addEventListener("click", async (e) => {
    const detailId = e.target.dataset.detail;
    const deleteId = e.target.dataset.delete;
    if (detailId) {
      const job = await (await api(`/api/analyses/${detailId}/`)).json();
      $("current-id").textContent = `#${detailId}`;
      lastReached = job.status === "FAILURE" ? 0 : lastReached;
      setFlow(job.stage, job.input_type);
      $("current-message").textContent = job.message || "";
      renderResult(job);
      if (!["SUCCESS", "FAILURE"].includes(job.status)) poll(detailId);
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
    if (deleteId && confirm(`#${deleteId} 분석 기록과 결과를 삭제할까요? 되돌릴 수 없습니다.`)) {
      const res = await api(`/api/analyses/${deleteId}/`, { method: "DELETE" });
      if (res.ok) refreshHistory();
    }
  });

  window.addEventListener("beforeunload", () => camera.stop());
})();
