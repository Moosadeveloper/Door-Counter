/* Doorlog - front end (plain JavaScript, no build step).
 *
 * Flow:  upload video -> draw the door line on the first frame -> analyse (background job,
 *        polled every second) -> results (counters, annotated video, event log,
 *        presence chart, per-person table).
 *
 * Backend API used: /api/health, /upload, /frame/{id}, /analyze/{id}, /status/{id},
 *                   /results/{id}, /video/{id}, /csv/{id}, /person/{id}/{pid}, DELETE /video/{id}
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const BASE_TITLE = document.title;
  const STORE_KEY = "doorlog.videoId";
  const YOLO_MISSING = "The person detector (ultralytics) is not installed. Run pip install -r requirements.txt " +
    "in the backend folder, then restart the server.";
  const MIN_LINE_FRAC = 0.03;            // door line must be at least 3 % of the frame diagonal

  const el = {
    alert: $("alert"), restart: $("restart"), health: $("health"),
    stepUpload: $("step-upload"), stepDoor: $("step-door"),
    stepProgress: $("step-progress"), stepResults: $("step-results"),
    drop: $("drop"), file: $("file"), uploadBar: $("uploadBar"), uploadFill: $("uploadFill"), uploadText: $("uploadText"),
    canvas: $("doorCanvas"), doorHint: $("doorHint"), videoMeta: $("videoMeta"),
    presetV: $("presetV"), presetH: $("presetH"), flipBtn: $("flipBtn"), clearBtn: $("clearBtn"),
    stride: $("stride"), useFaces: $("useFaces"), faceNote: $("faceNote"), analyzeBtn: $("analyzeBtn"),
    useCapacity: $("useCapacity"), capacityField: $("capacityField"), capacity: $("capacity"),
    progressBar: $("progressBar"), progressFill: $("progressFill"), progressText: $("progressText"),
  };

  const state = {
    videoId: null, meta: null, img: null,
    p1: null, p2: null,            // door line end points, normalised 0..1
    mode: "idle",                  // idle | placing | ready
    drag: null,                    // "p1" | "p2" | null
    flip: false,
    health: null, uploading: false, submitting: false,
    pollTimer: null, polling: false, pollFails: 0, startedAt: 0,
    results: null,
  };

  /* ------------------------------------------------------------------ helpers */
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const clamp01 = (v) => Math.min(1, Math.max(0, v));
  const plural = (n, one, many) => (n === 1 ? one : many);
  const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function fmt(sec) {
    sec = Math.max(0, Math.round(Number(sec) || 0));
    const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    const two = (n) => String(n).padStart(2, "0");
    return h ? `${h}:${two(m)}:${two(s)}` : `${m}:${two(s)}`;
  }
  function hmsToSec(text) {                       // "0:00:17" -> 17
    if (!text) return null;
    const parts = String(text).split(":").map(Number);
    if (parts.some(Number.isNaN)) return null;
    return parts.reduce((acc, v) => acc * 60 + v, 0);
  }

  function errText(body, fallback) {
    if (!body || body.detail == null) return fallback;
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
    return fallback;
  }

  async function api(path, opts) {
    let res;
    try {
      res = await fetch(path, opts);
    } catch (e) {
      throw new Error("Cannot reach the server. Make sure it is still running.");
    }
    const isJson = (res.headers.get("content-type") || "").includes("json");
    const body = isJson ? await res.json().catch(() => null) : null;
    if (!res.ok) {
      const err = new Error(errText(body, `The server answered with an error (${res.status}).`));
      err.status = res.status;
      throw err;
    }
    return body;
  }

  const store = {
    get() { try { return localStorage.getItem(STORE_KEY); } catch (e) { return null; } },
    set(v) { try { localStorage.setItem(STORE_KEY, v); } catch (e) { /* private mode */ } },
    clear() { try { localStorage.removeItem(STORE_KEY); } catch (e) { /* ignore */ } },
  };

  function showAlert(msg, kind) {
    el.alert.textContent = msg;
    el.alert.className = "alert" + (kind === "info" ? " info" : "");
    el.alert.hidden = false;
  }
  function clearAlert() { el.alert.hidden = true; el.alert.textContent = ""; }

  function showStage(name) {
    el.stepUpload.hidden = name !== "upload";
    el.stepDoor.hidden = name !== "door";
    el.stepProgress.hidden = name !== "progress";
    el.stepResults.hidden = name !== "results";
    el.restart.hidden = !(name === "door" || name === "results");
  }

  /* ------------------------------------------------------------------ health */
  async function loadHealth() {
    try {
      const h = await api("/api/health");
      state.health = h;
      const item = (ok, good, bad) => `<span class="${ok ? "" : "bad"}"><i></i>${ok ? good : bad}</span>`;
      el.health.innerHTML =
        item(h.yolo_installed, "Detector ready", "Detector not installed") +
        item(h.face_models, "Face models ready", "Face models missing") +
        item(h.ffmpeg, "Browser video ready", "Browser video limited");
      if (!h.face_models) {
        el.useFaces.checked = false;
        el.useFaces.disabled = true;
        el.faceNote.hidden = false;
        el.faceNote.innerHTML = "Face models are not installed yet. Run <code>python download_models.py</code> " +
          "in the backend folder to switch this on. Until then people are followed by movement only.";
      }
      if (!h.yolo_installed) showAlert(YOLO_MISSING);
    } catch (e) {
      state.health = null;
      el.health.innerHTML = '<span class="bad"><i></i>Server not reachable</span>';
      showAlert(e.message);
    }
    updateDoorUI();
  }

  /* ------------------------------------------------------------------ upload */
  function setUploading(on) {
    state.uploading = on;
    el.drop.classList.toggle("busy", on);
    el.file.disabled = on;
    el.uploadBar.hidden = !on;
    el.uploadText.hidden = !on;
    if (on) { el.uploadFill.style.width = "0%"; el.uploadText.textContent = "Uploading 0%"; }
  }

  function uploadFile(file) {
    if (!file || state.uploading) return;
    clearAlert();
    setUploading(true);
    const form = new FormData();
    form.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/upload");
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      const pct = Math.round((e.loaded / e.total) * 100);
      el.uploadFill.style.width = pct + "%";
      el.uploadText.textContent = pct >= 100 ? "Reading the video" : `Uploading ${pct}%`;
    };
    xhr.onerror = () => {
      setUploading(false);
      showAlert("Upload failed. Make sure the server is running and try again.");
    };
    xhr.onload = () => {
      setUploading(false);
      let body = null;
      try { body = JSON.parse(xhr.responseText); } catch (e) { /* not JSON */ }
      if (xhr.status >= 200 && xhr.status < 300 && body && body.video_id) {
        onUploaded(body);
      } else {
        showAlert(errText(body, `Upload failed (${xhr.status}).`));
      }
    };
    xhr.send(form);
  }

  function onUploaded(meta) {
    state.videoId = meta.video_id;
    state.meta = meta;
    store.set(meta.video_id);
    const img = new Image();
    img.onload = () => {
      state.img = img;
      el.canvas.width = img.naturalWidth;
      el.canvas.height = img.naturalHeight;
      resetLine();
      setLine(0.5, 0.03, 0.5, 0.97);        // start with a ready-to-use line so the door is never blank
      const dur = meta.duration_seconds != null ? `, ${fmt(meta.duration_seconds)} long` : "";
      el.videoMeta.textContent = `${meta.filename} - ${meta.width} x ${meta.height}, ${meta.fps} frames per second${dur}`;
      showStage("door");
      updateDoorUI();
      drawScene();
    };
    img.onerror = () => showAlert("The video was saved, but its first frame could not be loaded.");
    img.src = `/frame/${meta.video_id}?t=${Date.now()}`;
  }

  /* ------------------------------------------------------------------ door line */
  function resetLine() {
    state.p1 = state.p2 = null;
    state.mode = "idle";
    state.drag = null;
    state.flip = false;
  }

  function lineLengthPx() {
    if (!state.p1 || !state.p2) return 0;
    const W = el.canvas.width, H = el.canvas.height;
    return Math.hypot((state.p2.x - state.p1.x) * W, (state.p2.y - state.p1.y) * H);
  }
  const minLinePx = () => MIN_LINE_FRAC * Math.hypot(el.canvas.width, el.canvas.height);

  function updateDoorUI() {
    const ready = state.mode === "ready";
    el.flipBtn.disabled = !ready;
    el.clearBtn.disabled = state.mode === "idle";
    const yoloOk = !state.health || state.health.yolo_installed;
    el.analyzeBtn.disabled = !ready || state.submitting || !yoloOk;
    el.doorHint.textContent =
      state.mode === "idle" ? "Click one end of the door line, or drag across the doorway."
      : state.mode === "placing" ? "Now click the other end of the door line."
      : "Drag either yellow dot onto the doorway. The arrow shows which side counts as INSIDE - use \"Swap inside and outside\" if it points the wrong way.";
  }

  function pointerPos(e) {
    const r = el.canvas.getBoundingClientRect();
    return { x: clamp01((e.clientX - r.left) / r.width), y: clamp01((e.clientY - r.top) / r.height) };
  }
  const cssScale = () => el.canvas.width / (el.canvas.getBoundingClientRect().width || el.canvas.width);

  function hitHandle(pos) {
    const W = el.canvas.width, H = el.canvas.height, reach = 20 * cssScale();
    for (const key of ["p2", "p1"]) {
      const p = state[key];
      if (p && Math.hypot((p.x - pos.x) * W, (p.y - pos.y) * H) <= reach) return key;
    }
    return null;
  }

  el.canvas.addEventListener("pointerdown", (e) => {
    if (e.button !== undefined && e.button !== 0) return;
    const pos = pointerPos(e);
    if (state.mode === "placing" && !state.drag) {            // second click of a click-click line
      state.p2 = pos;
      state.mode = lineLengthPx() >= minLinePx() ? "ready" : "placing";
      if (state.mode === "placing") showAlert("That end is too close to the first one. Click further away.", "info");
      else clearAlert();
    } else {
      const handle = state.mode === "ready" ? hitHandle(pos) : null;
      if (handle) {
        state.drag = handle;
      } else {                                                 // start a new line
        state.p1 = pos; state.p2 = pos;
        state.mode = "placing";
        state.drag = "p2";
        state.flip = false;
      }
      el.canvas.setPointerCapture(e.pointerId);
    }
    updateDoorUI(); drawScene();
  });

  el.canvas.addEventListener("pointermove", (e) => {
    const pos = pointerPos(e);
    if (state.drag) {
      state[state.drag] = pos;
    } else if (state.mode === "placing") {
      state.p2 = pos;                                          // rubber band while hovering
    } else if (state.mode === "ready") {
      el.canvas.style.cursor = hitHandle(pos) ? "grab" : "crosshair";
      return;
    } else {
      return;
    }
    drawScene();
  });

  function endDrag(e) {
    if (!state.drag) return;
    const wasCreating = state.mode === "placing";
    state.drag = null;
    if (el.canvas.hasPointerCapture && el.canvas.hasPointerCapture(e.pointerId)) el.canvas.releasePointerCapture(e.pointerId);
    if (wasCreating) {
      if (lineLengthPx() >= minLinePx()) { state.mode = "ready"; clearAlert(); }
      else { state.p2 = state.p1; }                            // it was a click: wait for the second one
    }
    updateDoorUI(); drawScene();
  }
  el.canvas.addEventListener("pointerup", endDrag);
  el.canvas.addEventListener("pointercancel", endDrag);

  function setLine(x1, y1, x2, y2) {
    state.p1 = { x: x1, y: y1 }; state.p2 = { x: x2, y: y2 };
    state.mode = "ready"; state.drag = null; state.flip = false;
    clearAlert(); updateDoorUI(); drawScene();
  }
  el.useCapacity.addEventListener("change", () => {
    el.capacityField.hidden = !el.useCapacity.checked;
  });

  el.presetV.addEventListener("click", () => setLine(0.5, 0.03, 0.5, 0.97));   // inside = left
  el.presetH.addEventListener("click", () => setLine(0.03, 0.5, 0.97, 0.5));   // inside = bottom
  el.flipBtn.addEventListener("click", () => { state.flip = !state.flip; drawScene(); });
  el.clearBtn.addEventListener("click", () => { resetLine(); updateDoorUI(); drawScene(); });

  /* Same convention as backend/geometry.py: inside normal = sign * (-dy, dx) / length. */
  function drawScene() {
    const c = el.canvas, g = c.getContext("2d"), W = c.width, H = c.height;
    if (!state.img) return;
    g.clearRect(0, 0, W, H);
    g.drawImage(state.img, 0, 0, W, H);
    if (!state.p1) return;

    const k = cssScale();
    const a = { x: state.p1.x * W, y: state.p1.y * H }, b = { x: state.p2.x * W, y: state.p2.y * H };
    const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy);
    const YELLOW = "#f2c300", handleR = 9 * k;

    const dot = (p) => {
      g.beginPath(); g.arc(p.x, p.y, handleR, 0, Math.PI * 2);
      g.fillStyle = YELLOW; g.fill(); g.lineWidth = 3 * k; g.strokeStyle = "#000"; g.stroke();
    };
    if (len < 2) { dot(a); return; }

    const sign = state.flip ? -1 : 1;
    const n = { x: sign * -dy / len, y: sign * dx / len };
    const D = 0.12 * Math.min(W, H);

    const band = (dir, colour) => {                             // tinted area on each side of the line
      g.beginPath();
      g.moveTo(a.x, a.y); g.lineTo(b.x, b.y);
      g.lineTo(b.x + dir * n.x * D, b.y + dir * n.y * D);
      g.lineTo(a.x + dir * n.x * D, a.y + dir * n.y * D);
      g.closePath(); g.fillStyle = colour; g.fill();
    };
    band(1, "rgba(30,138,90,0.28)");
    band(-1, "rgba(210,89,31,0.28)");

    g.lineCap = "round";
    g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y);
    g.setLineDash([]); g.lineWidth = 7 * k; g.strokeStyle = "rgba(0,0,0,0.75)"; g.stroke();
    g.setLineDash([14 * k, 9 * k]); g.lineWidth = 3.5 * k; g.strokeStyle = YELLOW; g.stroke();
    g.setLineDash([]);

    const label = (text, cx, cy) => {
      g.font = `600 ${15 * k}px system-ui, sans-serif`;
      const w = g.measureText(text).width + 14 * k, h = 24 * k;
      const x = Math.min(W - w / 2 - 4, Math.max(w / 2 + 4, cx));
      const y = Math.min(H - h / 2 - 4, Math.max(h / 2 + 4, cy));
      g.fillStyle = "rgba(0,0,0,0.78)";
      g.fillRect(x - w / 2, y - h / 2, w, h);
      g.fillStyle = "#fff"; g.textAlign = "center"; g.textBaseline = "middle";
      g.fillText(text, x, y + 1 * k);
    };

    if (state.mode === "ready" || len >= minLinePx()) {
      const m = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      const tip = { x: m.x + n.x * D * 0.85, y: m.y + n.y * D * 0.85 };
      const head = 12 * k, ang = Math.atan2(tip.y - m.y, tip.x - m.x);
      const arrow = () => {
        g.beginPath(); g.moveTo(m.x, m.y); g.lineTo(tip.x, tip.y);
        g.moveTo(tip.x, tip.y); g.lineTo(tip.x - head * Math.cos(ang - 0.45), tip.y - head * Math.sin(ang - 0.45));
        g.moveTo(tip.x, tip.y); g.lineTo(tip.x - head * Math.cos(ang + 0.45), tip.y - head * Math.sin(ang + 0.45));
      };
      arrow(); g.lineWidth = 7 * k; g.strokeStyle = "rgba(0,0,0,0.75)"; g.stroke();
      arrow(); g.lineWidth = 3.5 * k; g.strokeStyle = YELLOW; g.stroke();
      label("INSIDE", tip.x + n.x * 22 * k, tip.y + n.y * 22 * k);
      label("OUTSIDE", m.x - n.x * D * 0.85 - n.x * 22 * k, m.y - n.y * D * 0.85 - n.y * 22 * k);
    }
    dot(a); dot(b);
  }
  window.addEventListener("resize", () => { if (!el.stepDoor.hidden) drawScene(); });

  /* ------------------------------------------------------------------ analyse */
  el.analyzeBtn.addEventListener("click", async () => {
    if (state.mode !== "ready" || state.submitting) return;
    if (lineLengthPx() < minLinePx()) { showAlert("The door line is too short. Draw it across the whole doorway."); return; }
    clearAlert();
    state.submitting = true; updateDoorUI();
    const q = new URLSearchParams({
      x1: state.p1.x.toFixed(5), y1: state.p1.y.toFixed(5),
      x2: state.p2.x.toFixed(5), y2: state.p2.y.toFixed(5),
      flip: String(state.flip), stride: el.stride.value,
      use_faces: String(el.useFaces.checked && !el.useFaces.disabled),
    });
    if (el.useCapacity.checked) {
      const cap = parseInt(el.capacity.value, 10);
      if (!Number.isFinite(cap) || cap < 1) { showAlert("Enter a capacity of at least 1, or turn the capacity warning off."); state.submitting = false; updateDoorUI(); return; }
      q.set("capacity", String(cap));
    }
    try {
      await api(`/analyze/${state.videoId}?${q}`, { method: "POST" });
      startPolling();
    } catch (e) {
      showAlert(e.message);
    } finally {
      state.submitting = false; updateDoorUI();
    }
  });

  function startPolling() {
    stopPolling();
    state.startedAt = Date.now(); state.pollFails = 0;
    setProgress(0, "Starting");
    showStage("progress");
    poll();
    state.pollTimer = setInterval(poll, 1000);
  }
  function stopPolling() {
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = null;
    document.title = BASE_TITLE;
  }

  function setProgress(p, stage) {
    const pct = Math.round(clamp01(p) * 100);
    el.progressFill.style.width = pct + "%";
    el.progressBar.setAttribute("aria-valuenow", String(pct));
    const elapsed = (Date.now() - state.startedAt) / 1000;
    let eta = "";
    if (pct >= 5 && pct < 100 && elapsed > 10) {
      const left = (elapsed * (100 - pct)) / pct;
      eta = left < 90 ? " - less than 2 minutes left" : ` - about ${Math.round(left / 60)} minutes left`;
    }
    el.progressText.textContent = `${stage || "Working"}: ${pct}%${eta}`;
    document.title = `${pct}% - ${BASE_TITLE}`;
  }

  async function poll() {
    if (state.polling || !state.videoId) return;
    state.polling = true;
    try {
      const st = await api(`/status/${state.videoId}`);
      state.pollFails = 0;
      if (st.state === "done") { stopPolling(); await loadResults(); }
      else if (st.state === "error") { stopPolling(); backToDoor(st.message || "The analysis failed."); }
      else if (st.state === "idle") { stopPolling(); backToDoor("The analysis did not start. Please try again."); }
      else setProgress(st.progress || 0, st.stage);
    } catch (e) {
      if (e.status === 404) { stopPolling(); store.clear(); restart("This video no longer exists on the server. Please upload it again."); }
      else if (++state.pollFails >= 8) { stopPolling(); backToDoor("Lost the connection to the server. Check that it is running."); }
    } finally {
      state.polling = false;
    }
  }

  function backToDoor(message) {
    if (state.img) { showStage("door"); updateDoorUI(); drawScene(); }
    else restart();
    if (message) showAlert(message);
  }

  /* ------------------------------------------------------------------ results */
  async function loadResults() {
    try {
      state.results = await api(`/results/${state.videoId}`);
      renderResults(state.results);
      showStage("results");
      clearAlert();
      window.scrollTo({ top: 0, behavior: reducedMotion() ? "auto" : "smooth" });
    } catch (e) {
      backToDoor(e.message);
    }
  }

  function renderResults(r) {
    const id = state.videoId, v = r.video || {};
    const dur = v.duration_seconds || 0;
    const records = [...(r.records || [])].sort((a, b) => (a.visits[0]?.start ?? 0) - (b.visits[0]?.start ?? 0));

    const warnings = [...(r.warnings || [])];
    if (v.browser_playable === false) {
      warnings.push("The annotated video could not be converted for browsers, so it may not play here. Use the download button below.");
    }

    const an = r.analytics || {};
    const capacityAlert = an.capacity != null && an.over_capacity_seconds > 0
      ? `<div class="warn">Over its capacity of ${an.capacity} for ${fmt(an.over_capacity_seconds)}` +
        (an.over_capacity_periods?.length
          ? ` (${an.over_capacity_periods.map((p) => `<button class="link-btn" type="button" data-t="${p.start}">${fmt(p.start)}\u2013${fmt(p.end)}</button>`).join(", ")})`
          : "") + `.</div>`
      : "";

    const stats = (an.peak_inside || an.avg_dwell_seconds)
      ? `<div class="table-wrap"><table><tbody>
          <tr><th>Busiest moment</th><td class="num">${an.peak_inside} inside</td>
            <td>at <button class="link-btn" type="button" data-t="${an.peak_at}">${fmt(an.peak_at)}</button></td></tr>
          <tr><th>Average time inside</th><td class="num">${fmt(an.avg_dwell_seconds)}</td>
            <td>longest stay: ${fmt(an.max_dwell_seconds)}</td></tr>
          ${an.busiest_window ? `<tr><th>Busiest arrivals</th>
            <td class="num">${an.busiest_window.entries} ${plural(an.busiest_window.entries, "entry", "entries")}</td>
            <td>between <button class="link-btn" type="button" data-t="${an.busiest_window.start}">${fmt(an.busiest_window.start)}</button>
              and ${fmt(an.busiest_window.end)}</td></tr>` : ""}
        </tbody></table></div>` : "";

    const counters = `
      <div class="counters" role="group" aria-label="Door counts">
        <div class="counter in"><b>${r.total_entered}</b><span>Entered</span><small>times someone walked in</small></div>
        <div class="counter out"><b>${r.total_exited}</b><span>Exited</span><small>times someone walked out</small></div>
        <div class="counter"><b>${r.currently_inside}</b><span>Inside at the end</span>
          <small>${r.already_inside ? `${r.already_inside} ${plural(r.already_inside, "was", "were")} already inside at the start` : "still inside when the video ends"}</small></div>
      </div>
      <p class="summary-line"><strong>${r.unique_people} different ${plural(r.unique_people, "person was", "people were")} seen</strong>
        in ${fmt(dur)} of video. ${r.unique_entered} came in through the door,
        ${r.already_inside} ${plural(r.already_inside, "was", "were")} already inside when it began, and
        ${r.passersby} only walked past.</p>
      <p class="fine">Entered and Exited count every crossing, so someone who leaves and comes back is counted twice.
        Each person keeps one ID (P1, P2, ...) for the whole video.</p>
      ${capacityAlert}
      ${warnings.map((w) => `<div class="warn">${esc(w)}</div>`).join("")}
      ${stats}`;

    const events = (r.events || []).map((e) => {
      const entered = e.type === "entry";
      const parts = [e.face_verified ? "Face confirmed" : "Followed by movement"];
      if (!entered) { const d = hmsToSec(e.dwell_time); if (d != null) parts.push(`stayed ${fmt(d)}`); }
      if (e.inferred) parts.push("crossing estimated, person was hidden");
      return `<li><button type="button" data-t="${e.seconds}">
        <span class="t">${fmt(e.seconds)}</span>
        <span class="what ${entered ? "in" : "out"}">${e.name ? esc(e.name) : `P${e.person_id}`} ${entered ? "entered" : "exited"}${e.flag === "unknown" ? " (unknown)" : e.flag === "face_hidden" ? " (face hidden)" : ""}</span>
        <span class="meta">${esc(parts.join(", "))}</span></button></li>`;
    }).join("");

    const rows = records.map((p) => {
      const first = p.visits[0], last = p.visits[p.visits.length - 1];
      const inside = p.status === "inside";
      const enterCell = first.start_kind === "door"
        ? `<button class="link-btn" type="button" data-t="${first.start}">${fmt(first.start)}</button>`
        : `Already inside`;
      const exitCell = inside ? "Still inside"
        : `<button class="link-btn" type="button" data-t="${last.end}">${fmt(last.end)}</button>`;
      const hasImg = p.has_face || p.has_snapshot;
      const thumb = hasImg
        ? `<img class="thumb" loading="lazy" src="/person/${id}/${p.person_id}" alt="${p.has_face ? "Face" : "Photo"} of person ${p.person_id}">`
        : `<span class="thumb-ph" aria-hidden="true">P${p.person_id}</span>`;
      const merged = (p.track_ids || []).length > 1
        ? `<br><small class="fine">${p.track_ids.length} tracker IDs joined into one person</small>` : "";
      return `<tr>
        <td><span class="who">${thumb}<span>${p.name ? `${esc(p.name)}<br><small class="fine">${esc(p.roll_no || "")} &middot; P${p.person_id}</small>`
          : p.flag === "unknown" ? `P${p.person_id}<br><small class="flag bad">Unknown - not registered</small>`
          : p.flag === "face_hidden" ? `P${p.person_id}<br><small class="flag warn">Face hidden</small>` : `P${p.person_id}`}</span></span></td>
        <td>${enterCell}</td><td>${exitCell}</td>
        <td class="num">${fmt(p.dwell_seconds)}</td>
        <td><span class="badge ${inside ? "in" : "out"}">${inside ? "Inside" : "Exited"}</span></td>
        <td class="num">${p.visits.length}</td>
        <td>${p.identity === "face" ? `Face (${p.face_samples} ${plural(p.face_samples, "photo", "photos")})` : "Movement only"}${merged}</td>
      </tr>`;
    }).join("");

    const playerHtml = `
      <video id="resVideo" controls preload="metadata" src="/video/${id}"></video>
      <p class="noplay" id="noplay" hidden>Your browser could not play this video. Use the download button below.</p>`;

    el.stepResults.innerHTML = `
      ${counters}

      <section class="section" aria-labelledby="h-video">
        <div class="player-grid">
          <div>
            <h2 id="h-video">Annotated video</h2>
            <div class="player">${playerHtml}</div>
          </div>
          <div class="events-wrap">
            <h3>Door log</h3>
            ${events ? `<ol class="events">${events}</ol>` : `<p class="empty">Nobody crossed the door line.</p>`}
          </div>
        </div>
      </section>

      <section class="section" aria-labelledby="h-chart">
        <h2 id="h-chart">Who was inside, and when</h2>
        ${records.length ? `
        <div class="chart-box" id="chartBox"></div>
        <p class="legend">
          <span><i style="background:#16704a"></i>Time inside</span>
          <span><i style="background:#16704a;opacity:.4;border:1px dashed #14201d"></i>Already inside when first seen</span>
          <span><i style="background:none;border-top:3px solid #14201d;height:0"></i>People inside (top graph)</span>
        </p>
        <p class="fine">Click anywhere on the chart to jump to that moment in the video.</p>`
        : `<p class="empty">Nobody was seen inside in this video.</p>`}
      </section>

      <section class="section" aria-labelledby="h-people">
        <h2 id="h-people">People</h2>
        ${records.length ? `<div class="table-wrap"><table>
          <thead><tr><th>Person</th><th>Entered at</th><th>Exited at</th><th class="num">Time inside</th>
            <th>Status</th><th class="num">Visits</th><th>Identified by</th></tr></thead>
          <tbody>${rows}</tbody></table></div>` : `<p class="empty">No people to list.</p>`}
        <div class="actions">
          <a class="secondary" href="/csv/${id}?kind=people" download>People (CSV)</a>
          <a class="secondary" href="/csv/${id}?kind=events" download>Door log (CSV)</a>
          <a class="secondary" href="/results/${id}" download="results.json">All data (JSON)</a>
          <a class="secondary" href="/video/${id}" download="annotated_video.mp4">Annotated video</a>
          <span class="gap"></span>
          <button class="link-btn" type="button" id="deleteBtn">Delete this video and its results</button>
        </div>
        <p class="fine">Analysed in ${esc(r.processing?.seconds ?? "?")} seconds
          ${r.processing?.face_engine ? "with face recognition" : "without face recognition"}.</p>
      </section>`;

    const video = $("resVideo");
    video.addEventListener("error", () => { $("noplay").hidden = false; });
    video.addEventListener("timeupdate", updatePlayhead);
    video.addEventListener("seeked", updatePlayhead);
    el.stepResults.addEventListener("click", onResultsClick);
    $("deleteBtn").addEventListener("click", deleteVideo);

    if (records.length) {
      drawChart();
      if ("ResizeObserver" in window) {
        state.chartObserver && state.chartObserver.disconnect();
        state.chartObserver = new ResizeObserver(() => requestAnimationFrame(drawChart));
        state.chartObserver.observe($("chartBox"));
      }
    }
  }

  function seekTo(t, play) {
    const video = $("resVideo");
    if (!video) return;
    video.currentTime = Math.max(0, t - 1);
    video.scrollIntoView({ block: "center", behavior: reducedMotion() ? "auto" : "smooth" });
    if (play) video.play().catch(() => { /* needs a user gesture on some browsers */ });
  }

  function onResultsClick(e) {
    const btn = e.target.closest("button[data-t]");
    if (btn) seekTo(parseFloat(btn.dataset.t), true);
  }

  async function deleteVideo() {
    if (!confirm("Delete the uploaded video, the annotated video and all results?")) return;
    try {
      await api(`/video/${state.videoId}`, { method: "DELETE" });
      restart("Deleted.");
    } catch (e) { showAlert(e.message); }
  }

  /* ---- presence chart: people inside over time (top) + one bar per person (below) ---- */
  const CH = { gutter: 52, right: 14, top: 6, occH: 72, gap: 26, rowH: 26, bar: 18, axisH: 26 };

  function niceStep(dur) {
    const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200];
    return steps.find((s) => dur / s <= 8) || 7200;
  }

  function drawChart() {
    const box = $("chartBox"), r = state.results;
    if (!box || !r) return;
    const width = Math.max(320, Math.floor(box.clientWidth));
    const dur = Math.max(1, (r.video && r.video.duration_seconds) || 1);
    const recs = [...r.records].sort((a, b) => (a.visits[0]?.start ?? 0) - (b.visits[0]?.start ?? 0));
    const plotW = width - CH.gutter - CH.right;
    const X = (t) => CH.gutter + (Math.min(dur, Math.max(0, t)) / dur) * plotW;

    const tl = r.timeline && r.timeline.length ? r.timeline : [{ t: 0, inside: 0 }];
    const maxIn = Math.max(1, ...tl.map((p) => p.inside));
    const occBottom = CH.top + CH.occH;
    const Y = (n) => occBottom - (n / maxIn) * (CH.occH - 8);
    const rowsTop = occBottom + CH.gap;
    const rowsEnd = rowsTop + recs.length * CH.rowH;
    const height = rowsEnd + CH.axisH;

    let out = `<svg id="chartSvg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img"
      aria-label="Chart of how many people were inside over time, and when each person was inside. The table below lists the same information.">`;

    // time grid + axis
    const step = niceStep(dur);
    for (let t = 0; t <= dur + 0.001; t += step) {
      const x = X(t);
      out += `<line x1="${x}" x2="${x}" y1="${CH.top}" y2="${rowsEnd}" stroke="#c3cbc7" stroke-width="1"/>`;
      out += `<text x="${x}" y="${rowsEnd + 17}" text-anchor="${t === 0 ? "start" : "middle"}">${fmt(t)}</text>`;
    }

    // occupancy step graph
    let d = `M ${X(tl[0].t)} ${Y(tl[0].inside)}`;
    for (let i = 1; i < tl.length; i++) d += ` H ${X(tl[i].t)} V ${Y(tl[i].inside)}`;
    d += ` H ${X(dur)}`;
    out += `<path d="${d} V ${Y(0)} H ${X(0)} Z" fill="#1e8a5a" fill-opacity="0.16"/>`;
    out += `<path d="${d}" fill="none" stroke="#14201d" stroke-width="2.5" stroke-linejoin="round"/>`;
    const yTicks = maxIn <= 6 ? Array.from({ length: maxIn + 1 }, (_, i) => i) : [0, maxIn];
    for (const n of yTicks) out += `<text x="${CH.gutter - 8}" y="${Y(n) + 4}" text-anchor="end">${n}</text>`;

    // one row per person
    recs.forEach((p, i) => {
      const ry = rowsTop + i * CH.rowH, by = ry + (CH.rowH - CH.bar) / 2;
      out += `<text class="lbl" x="${CH.gutter - 8}" y="${by + CH.bar - 5}" text-anchor="end">P${p.person_id}</text>`;
      p.visits.forEach((v) => {
        const open = v.end == null, x0 = X(v.start), x1 = X(open ? dur : v.end);
        const w = Math.max(3, x1 - x0), seen = v.start_kind === "seen_inside";
        const tip = `P${p.person_id} inside ${fmt(v.start)} to ${open ? "end of video" : fmt(v.end)} (${fmt(v.dwell_seconds)})`;
        const fill = seen ? 'fill="#16704a" fill-opacity="0.4" stroke="#14201d" stroke-dasharray="4 3"' : 'fill="#16704a"';
        const pts = open   // pointed right end = still inside
          ? `${x0},${by} ${x0 + w - 8},${by} ${x0 + w},${by + CH.bar / 2} ${x0 + w - 8},${by + CH.bar} ${x0},${by + CH.bar}`
          : null;
        out += pts
          ? `<polygon points="${pts}" ${fill}><title>${esc(tip)}</title></polygon>`
          : `<rect x="${x0}" y="${by}" width="${w}" height="${CH.bar}" rx="3" ${fill}><title>${esc(tip)}</title></rect>`;
        if (w > 46) out += `<text x="${x0 + 7}" y="${by + CH.bar - 5}" fill="${seen ? "#14201d" : "#fff"}" style="fill:${seen ? "#14201d" : "#fff"};font-weight:600" pointer-events="none">${fmt(v.dwell_seconds)}</text>`;
      });
    });

    out += `<line id="playhead" x1="${CH.gutter}" x2="${CH.gutter}" y1="${CH.top}" y2="${rowsEnd}" stroke="#d2591f" stroke-width="2.5"/>`;
    out += `</svg>`;
    box.innerHTML = out;

    const svg = $("chartSvg");
    svg.addEventListener("pointerdown", (e) => {
      const rect = svg.getBoundingClientRect();
      const t = ((e.clientX - rect.left - CH.gutter) / plotW) * dur;
      if (t < 0 || t > dur) return;
      const video = $("resVideo");
      if (video) { video.currentTime = t; video.scrollIntoView({ block: "center", behavior: reducedMotion() ? "auto" : "smooth" }); }
    });
    state.chart = { X, dur };
    updatePlayhead();
  }

  function updatePlayhead() {
    const ph = $("playhead"), video = $("resVideo");
    if (!ph || !video || !state.chart) return;
    const x = state.chart.X(video.currentTime || 0);
    ph.setAttribute("x1", x); ph.setAttribute("x2", x);
  }

  /* ------------------------------------------------------------------ restart / boot */
  function restart(message) {
    stopPolling();
    if (state.chartObserver) state.chartObserver.disconnect();
    el.stepResults.removeEventListener("click", onResultsClick);
    el.stepResults.innerHTML = "";
    store.clear();
    Object.assign(state, { videoId: null, meta: null, img: null, results: null, chart: null, submitting: false });
    resetLine();
    el.file.value = "";
    setUploading(false);
    showStage("upload");
    if (message) showAlert(message, "info");
    else if (state.health && !state.health.yolo_installed) showAlert(YOLO_MISSING);
    else clearAlert();
  }
  el.restart.addEventListener("click", () => restart());

  el.file.addEventListener("change", () => uploadFile(el.file.files[0]));
  ["dragenter", "dragover"].forEach((ev) => el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.remove("over"); }));
  el.drop.addEventListener("drop", (e) => uploadFile(e.dataTransfer.files[0]));

  async function resume() {                     // pick up a running or finished analysis after a page reload
    const id = store.get();
    if (!id) return;
    try {
      const st = await api(`/status/${id}`);
      if (st.state === "queued" || st.state === "running") { state.videoId = id; startPolling(); }
      else if (st.state === "done") { state.videoId = id; await loadResults(); }
      else store.clear();
    } catch (e) {
      if (e.status === 404) store.clear();
    }
  }

  showStage("upload");
  updateDoorUI();
  loadHealth().then(resume);
})();
