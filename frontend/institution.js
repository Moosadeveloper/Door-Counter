/* Doorlog - institution pages (plain JavaScript, no build step).
 *
 * Tabs:  Live camera | Attendance | People | Alerts | Video analysis (app.js)
 * API:   /api/people, /api/live/*, /api/attendance*, /api/log, /api/alerts*, /api/snapshot/*
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const today = () => {
    const d = new Date();
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
  };
  const snapUrl = (rel) => (rel ? `/api/snapshot/${rel.split("/").map(encodeURIComponent).join("/")}` : "");

  async function api(path, opts) {
    let res;
    try { res = await fetch(path, opts); } catch (e) { throw new Error("Cannot reach the server. Make sure it is still running."); }
    const isJson = (res.headers.get("content-type") || "").includes("json");
    const body = isJson ? await res.json().catch(() => null) : null;
    if (!res.ok) {
      let msg = `The server answered with an error (${res.status}).`;
      if (body && typeof body.detail === "string") msg = body.detail;
      else if (body && Array.isArray(body.detail)) msg = body.detail.map((d) => d.msg || "").join("; ");
      throw new Error(msg);
    }
    return body;
  }
  const postJson = (path, data) => api(path, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data || {}),
  });

  function msg(box, text, kind) {
    const el = $(box);
    if (!text) { el.hidden = true; el.textContent = ""; return; }
    el.className = "msg" + (kind ? " " + kind : "");
    el.textContent = text;
    el.hidden = false;
  }

  /* ================================================================ tabs */
  const TABS = ["live", "attendance", "people", "alerts", "video"];
  let current = null;
  const timers = {};

  function showTab(name, push) {
    if (!TABS.includes(name)) name = "live";
    current = name;
    for (const t of TABS) {
      $("tab-" + t).hidden = t !== name;
      const b = $("tabbtn-" + t);
      b.setAttribute("aria-selected", String(t === name));
      b.tabIndex = t === name ? 0 : -1;
    }
    if (push && location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
    clearInterval(timers.att); clearInterval(timers.al);
    if (name === "live") refreshLive();
    if (name === "attendance") { loadAttendance(); timers.att = setInterval(loadAttendance, 10000); }
    if (name === "people") loadPeople();
    if (name === "alerts") { loadAlerts(); timers.al = setInterval(loadAlerts, 15000); }
  }

  document.querySelector(".tabs-in").addEventListener("click", (e) => {
    const b = e.target.closest("[data-tab]");
    if (b) showTab(b.dataset.tab, true);
  });
  document.querySelector(".tabs-in").addEventListener("keydown", (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const i = TABS.indexOf(current) + (e.key === "ArrowRight" ? 1 : -1);
    const next = TABS[(i + TABS.length) % TABS.length];
    showTab(next, true);
    $("tabbtn-" + next).focus();
  });
  window.addEventListener("hashchange", () => showTab(location.hash.slice(1), false));

  /* ============================================================ alert badge */
  async function updateAlertBadge() {
    try {
      const list = await api(`/api/alerts?day=${today()}`);
      const open = list.filter((a) => !a.reviewed).length;
      const b = $("alertCount");
      b.hidden = !open;
      b.textContent = open;
      b.setAttribute("aria-label", `${open} new`);
    } catch (e) { /* server down: health bar already says so */ }
  }

  /* ================================================================ LIVE */
  const live = { running: false, feedTimer: null, line: null, flip: false, drag: null, img: null, lastFeedKey: "" };

  function selectedMode() { return document.querySelector('input[name="mode"]:checked').value; }

  /* ---- CCTV address builder (most CCTV in Pakistan: Hikvision, Dahua, CP Plus) ---- */
  const CC_KEY = "doorlog.cctv";
  const CC_FIELDS = ["srcKind", "ccBrand", "ccIp", "ccPort", "ccUser", "ccCh", "ccQ", "camName"];
  function loadCcSettings() {
    try {
      const saved = JSON.parse(localStorage.getItem(CC_KEY) || "{}");
      for (const k of CC_FIELDS) if (saved[k] != null && $(k)) $(k).value = saved[k];
    } catch (e) { /* ignore */ }
  }
  function saveCcSettings() {          // password is NOT saved in the browser
    try {
      const o = {};
      for (const k of CC_FIELDS) o[k] = $(k).value;
      localStorage.setItem(CC_KEY, JSON.stringify(o));
    } catch (e) { /* ignore */ }
  }
  function cctvUrl(mask) {
    const brand = $("ccBrand").value;
    if (brand === "custom") {
      const u = $("srcUrl").value.trim();
      return mask ? u.replace(/(:\/\/[^:/@\s]+:)\S*@/, "$1****@") : u;
    }
    const ip = $("ccIp").value.trim(), port = $("ccPort").value.trim() || "554";
    if (!ip) return "";
    const user = encodeURIComponent($("ccUser").value.trim());
    const pass = mask ? ($("ccPass").value ? "****" : "") : encodeURIComponent($("ccPass").value);
    const auth = user ? `${user}${pass ? ":" + pass : ""}@` : "";
    const ch = Math.max(1, parseInt($("ccCh").value, 10) || 1), sub = $("ccQ").value === "sub";
    const path = brand === "hikvision" ? `/Streaming/Channels/${ch}0${sub ? 2 : 1}`
      : `/cam/realmonitor?channel=${ch}&subtype=${sub ? 1 : 0}`;
    return `rtsp://${auth}${ip}:${port}${path}`;
  }
  function updateCcPreview() {
    const custom = $("ccBrand").value === "custom";
    $("ccParts").hidden = custom;
    $("ccCustomField").hidden = !custom;
    const u = cctvUrl(true);
    $("ccPreview").textContent = u ? "Address: " + u : "";
  }
  ["ccBrand", "ccIp", "ccPort", "ccUser", "ccPass", "ccCh", "ccQ", "srcUrl"].forEach((id) =>
    $(id).addEventListener("input", () => { updateCcPreview(); resetLine(); }));
  $("ccBrand").addEventListener("change", updateCcPreview);
  $("ccQ").addEventListener("change", updateCcPreview);

  $("ccTest").addEventListener("click", async () => {
    msg("liveMsg", "");
    const url = cctvUrl(false);
    if (!url) { msg("liveMsg", "Type the IP address of the CCTV / DVR first."); return; }
    $("ccTest").disabled = true;
    $("ccTest").textContent = "Connecting... (can take up to 20 seconds)";
    try {
      const res = await fetch(`/api/live/probe?source=${encodeURIComponent(url)}&t=${Date.now()}`);
      if (!res.ok) {
        const b = await res.json().catch(() => null);
        throw new Error((b && b.detail) || "Could not connect to the camera.");
      }
      $("ccTestImg").src = URL.createObjectURL(await res.blob());
      $("ccTestWrap").hidden = false;
      msg("liveMsg", "Connected. This is what the camera sees. Now press Start camera.", "ok");
      saveCcSettings();
    } catch (e) {
      $("ccTestWrap").hidden = true;
      msg("liveMsg", e.message);
    } finally {
      $("ccTest").disabled = false;
      $("ccTest").textContent = "Test connection";
    }
  });

  function updateSourceFields() {
    const k = $("srcKind").value;
    $("cctvBox").hidden = k !== "cctv";
    $("srcVideoField").hidden = k !== "video";
    if (k === "video") loadVideoList();
    if (k === "cctv") updateCcPreview();
  }
  async function loadVideoList() {
    try {
      const vids = await api("/api/videos");
      $("srcVideo").innerHTML = vids.length
        ? vids.map((v) => `<option value="${esc(v.video_id)}">${esc(v.video_id)} - ${esc(v.file)}</option>`).join("")
        : `<option value="">No uploaded videos yet</option>`;
    } catch (e) { msg("liveMsg", e.message); }
  }
  function sourceSpec() {
    const k = $("srcKind").value;
    if (k === "video") return { video_id: $("srcVideo").value || null };
    if (k === "cctv") return { source: cctvUrl(false) };
    return { source: k };
  }

  $("srcKind").addEventListener("change", () => { updateSourceFields(); resetLine(); });
  document.querySelectorAll('input[name="mode"]').forEach((r) => r.addEventListener("change", () => {
    $("doorBox").hidden = selectedMode() !== "door";
  }));

  /* ---- door line drawing (door mode) ---- */
  function resetLine() {
    live.line = null; live.img = null;
    $("lineWrap").hidden = true; $("lineHint").hidden = true; $("lineFlip").hidden = true;
  }
  $("probeBtn").addEventListener("click", async () => {
    msg("liveMsg", "");
    const spec = sourceSpec();
    if (spec.source === "") { msg("liveMsg", "Fill in the CCTV details first."); return; }
    const q = spec.video_id ? `video_id=${encodeURIComponent(spec.video_id)}` : `source=${encodeURIComponent(spec.source)}`;
    $("probeBtn").disabled = true;
    $("probeBtn").textContent = "Getting a picture...";
    try {
      const res = await fetch(`/api/live/probe?${q}&t=${Date.now()}`);
      if (!res.ok) {
        const b = await res.json().catch(() => null);
        throw new Error((b && b.detail) || "Could not get a picture from this camera.");
      }
      const blob = await res.blob();
      const img = new Image();
      img.onload = () => {
        live.img = img;
        const c = $("lineCanvas");
        c.width = img.naturalWidth; c.height = img.naturalHeight;
        if (!live.line) live.line = [0.5, 0.15, 0.5, 0.95];
        $("lineWrap").hidden = false; $("lineHint").hidden = false; $("lineFlip").hidden = false;
        drawLine();
      };
      img.src = URL.createObjectURL(blob);
    } catch (e) {
      msg("liveMsg", e.message);
    } finally {
      $("probeBtn").disabled = false;
      $("probeBtn").textContent = "Take a new picture";
    }
  });
  $("lineFlip").addEventListener("click", () => { live.flip = !live.flip; drawLine(); });

  function drawLine() {
    const c = $("lineCanvas"), ctx = c.getContext("2d");
    if (!live.img) return;
    ctx.drawImage(live.img, 0, 0);
    if (!live.line) return;
    const W = c.width, H = c.height, k = Math.max(1, W / 640);
    const [x1, y1, x2, y2] = live.line.map((v, i) => v * (i % 2 ? H : W));
    ctx.lineWidth = 4 * k; ctx.strokeStyle = "#f2c300"; ctx.setLineDash([14 * k, 10 * k]);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.setLineDash([]);
    // arrow toward INSIDE (same maths as the server: normal of the line, flipped if asked)
    const len = Math.hypot(x2 - x1, y2 - y1) || 1, sgn = live.flip ? -1 : 1;
    const nx = sgn * -(y2 - y1) / len, ny = sgn * (x2 - x1) / len;
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, L = 60 * k;
    const tx = mx + nx * L, ty = my + ny * L;
    ctx.beginPath(); ctx.moveTo(mx, my); ctx.lineTo(tx, ty); ctx.stroke();
    const a = Math.atan2(ty - my, tx - mx);
    ctx.fillStyle = "#f2c300";
    ctx.beginPath(); ctx.moveTo(tx, ty);
    ctx.lineTo(tx - 16 * k * Math.cos(a - 0.5), ty - 16 * k * Math.sin(a - 0.5));
    ctx.lineTo(tx - 16 * k * Math.cos(a + 0.5), ty - 16 * k * Math.sin(a + 0.5)); ctx.fill();
    ctx.font = `bold ${18 * k}px system-ui`; ctx.fillText("IN", tx + 8 * k, ty + 6 * k);
    for (const [x, y] of [[x1, y1], [x2, y2]]) {
      ctx.beginPath(); ctx.arc(x, y, 9 * k, 0, 7); ctx.fill();
      ctx.lineWidth = 2 * k; ctx.strokeStyle = "#14201d"; ctx.stroke(); ctx.strokeStyle = "#f2c300";
    }
  }
  function canvasPos(e) {
    const r = $("lineCanvas").getBoundingClientRect();
    return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))];
  }
  $("lineCanvas").addEventListener("pointerdown", (e) => {
    if (!live.line) return;
    const [x, y] = canvasPos(e), r = $("lineCanvas").getBoundingClientRect();
    const d1 = Math.hypot((x - live.line[0]) * r.width, (y - live.line[1]) * r.height);
    const d2 = Math.hypot((x - live.line[2]) * r.width, (y - live.line[3]) * r.height);
    live.drag = d1 < d2 ? 0 : 2;
    live.line[live.drag] = x; live.line[live.drag + 1] = y;
    $("lineCanvas").setPointerCapture(e.pointerId);
    drawLine();
  });
  $("lineCanvas").addEventListener("pointermove", (e) => {
    if (live.drag == null) return;
    const [x, y] = canvasPos(e);
    live.line[live.drag] = x; live.line[live.drag + 1] = y;
    drawLine();
  });
  const endDrag = () => { live.drag = null; };
  $("lineCanvas").addEventListener("pointerup", endDrag);
  $("lineCanvas").addEventListener("pointercancel", endDrag);

  /* ---- start / stop ---- */
  $("liveStart").addEventListener("click", async () => {
    msg("liveMsg", "");
    const mode = selectedMode();
    const body = { ...sourceSpec(), camera: $("camName").value.trim() || "Camera", mode, flip: live.flip };
    if (body.source === "") { msg("liveMsg", "Fill in the CCTV details (IP address, password, channel) first."); return; }
    if ($("srcKind").value === "video" && !body.video_id) { msg("liveMsg", "Upload a video on the Video analysis page first."); return; }
    if (mode === "door") {
      if (!live.line) { msg("liveMsg", "Door line mode: take a picture and draw the door line first."); return; }
      body.line = live.line;
    }
    $("liveStart").disabled = true;
    $("liveStart").textContent = "Starting... (loading the AI models)";
    try {
      await postJson("/api/live/start", body);
      saveCcSettings();
      setLiveOn(true);
    } catch (e) {
      msg("liveMsg", e.message);
      setLiveOn(false);
    } finally {
      $("liveStart").disabled = false;
      $("liveStart").textContent = "Start camera";
    }
  });
  $("liveStop").addEventListener("click", async () => {
    try { await postJson("/api/live/stop"); } catch (e) { msg("liveMsg", e.message); }
    setLiveOn(false);
  });

  function setLiveOn(on) {
    live.running = on;
    $("liveStart").hidden = on;
    $("liveStop").hidden = !on;
    $("liveEmpty").hidden = on;
    const img = $("liveImg");
    if (on) img.src = `/api/live/stream?t=${Date.now()}`;
    else { img.removeAttribute("src"); $("liveInfo").textContent = ""; }
    clearInterval(live.feedTimer);
    if (on) { pollLive(); live.feedTimer = setInterval(pollLive, 1500); }
  }

  async function refreshLive() {
    try {
      const st = await api("/api/live/status");
      const on = ["running", "starting", "reconnecting"].includes(st.state);
      if (on !== live.running) setLiveOn(on);
      if (!on) { pollFeedOnce(); updatePresentLine(); }
    } catch (e) { /* ignore */ }
  }

  async function pollLive() {
    try {
      const st = await api("/api/live/status");
      if (st.state === "error") { msg("liveMsg", "Camera stopped: " + st.error); setLiveOn(false); return; }
      if (st.state === "finished") { msg("liveMsg", "The video has finished.", "info"); setLiveOn(false); return; }
      if (st.state === "off" || st.state === "stopped") { setLiveOn(false); return; }
      if (st.state === "reconnecting") {
        msg("liveMsg", "CCTV connection lost - reconnecting automatically. Check the network cable / DVR if this stays.", "info");
        live.wasReconnecting = true;
      } else if (live.wasReconnecting && st.state === "running") {
        live.wasReconnecting = false;
        msg("liveMsg", "Camera reconnected.", "ok");
        $("liveImg").src = `/api/live/stream?t=${Date.now()}`;
      }
      $("liveInfo").textContent = `${st.camera} - ${st.mode} camera - ${st.source} - ${st.fps} frames/second - ` +
        `${st.people_on_screen} ${st.people_on_screen === 1 ? "person" : "people"} being followed`;
      await pollFeedOnce();
      updatePresentLine();
    } catch (e) { /* keep trying */ }
  }

  async function pollFeedOnce() {
    let items;
    try { items = await api("/api/live/feed?limit=40"); } catch (e) { return; }
    const key = items.length ? `${items.length}:${items[0].type}:${items[0].time}:${items[0].id ?? ""}` : "none";
    if (key === live.lastFeedKey) return;
    live.lastFeedKey = key;
    if (!items.length) {
      $("liveFeed").innerHTML = `<li class="empty">Nothing yet. Recognitions and alerts appear here.</li>`;
      return;
    }
    $("liveFeed").innerHTML = items.map(feedItem).join("");
    if (items.some((i) => i.type === "unknown" || i.type === "face_hidden")) updateAlertBadge();
  }

  const KIND_TEXT = { in: "IN", out: "OUT", unknown: "UNKNOWN", face_hidden: "FACE HIDDEN", recognized: "SEEN" };
  function feedItem(ev) {
    const img = ev.face_snapshot || ev.snapshot;
    const pic = img ? `<img src="${snapUrl(img)}" alt="" loading="lazy">` : `<span class="ph"></span>`;
    let what, meta;
    if (ev.type === "unknown" || ev.type === "face_hidden") {
      what = ev.type === "unknown" ? "Unknown person" : "Face hidden";
      meta = ev.note || "";
    } else {
      what = `${esc(ev.name)} <span class="muted">(${esc(ev.roll_no || "")})</span>`;
      meta = ev.type === "recognized" ? "recognised (already logged a moment ago)"
        : `${ev.type === "in" ? "checked in" : "checked out"}${ev.score != null ? ` - match ${Math.round(ev.score * 100)}%` : ""}`;
    }
    return `<li>${pic}<span class="t">${esc(ev.time)}</span>
      <span class="what"><span class="kind ${esc(ev.type)}">${KIND_TEXT[ev.type] || esc(ev.type)}</span>${what}</span>
      <span class="meta" style="grid-column:3">${esc(ev.camera || "")}${meta ? " - " + esc(meta) : ""}</span></li>`;
  }

  async function updatePresentLine() {
    try {
      const r = await api(`/api/attendance?day=${today()}`);
      $("presentLine").textContent = r.total
        ? `Today: ${r.present} of ${r.total} present, ${r.inside_now} inside now, ${r.alerts} ${r.alerts === 1 ? "alert" : "alerts"}.`
        : "Nobody is registered yet - add people on the People page.";
    } catch (e) { /* ignore */ }
  }

  /* ========================================================== ATTENDANCE */
  const att = { data: null };
  $("attDay").value = today();
  $("attDay").max = today();
  $("attDay").addEventListener("change", loadAttendance);
  $("attFilter").addEventListener("change", renderAttendance);
  $("attSearch").addEventListener("input", renderAttendance);

  async function loadAttendance() {
    const day = $("attDay").value || today();
    $("attCsv").href = `/api/attendance.csv?day=${day}`;
    try {
      const [rep, log] = await Promise.all([api(`/api/attendance?day=${day}`), api(`/api/log?day=${day}`)]);
      att.data = rep;
      msg("attMsg", "");
      renderAttendance();
      $("attLog").innerHTML = log.length ? log.map((l) => `<li>
          ${l.snapshot ? `<img src="${snapUrl(l.snapshot)}" alt="" loading="lazy">` : `<span class="ph"></span>`}
          <span class="t">${esc(l.time)}</span>
          <span class="what"><span class="kind ${l.kind}">${l.kind.toUpperCase()}</span>${esc(l.name)} <span class="muted">(${esc(l.roll_no || "")})</span></span>
          <span class="meta" style="grid-column:3">${esc(l.camera || "")}${l.manual ? " - marked by hand" : l.score != null ? ` - match ${Math.round(l.score * 100)}%` : ""}</span></li>`).join("")
        : `<li class="empty">No in / out events on this day.</li>`;
    } catch (e) { msg("attMsg", e.message); }
  }

  function renderAttendance() {
    const r = att.data;
    if (!r) return;
    const isToday = r.day === today();
    $("attCounters").innerHTML = `
      <div class="counter in"><b>${r.present}</b><span>Present</span><small>of ${r.total} registered</small></div>
      <div class="counter out"><b>${r.absent}</b><span>Absent</span><small>not seen ${isToday ? "yet today" : "that day"}</small></div>
      <div class="counter"><b>${r.inside_now}</b><span>Inside ${isToday ? "now" : "at the end"}</span><small>came in, not yet out</small></div>
      <div class="counter bad"><b>${r.alerts}</b><span>Alerts</span><small>unknown or hidden faces</small></div>`;
    const f = $("attFilter").value, q = $("attSearch").value.trim().toLowerCase();
    const rows = r.people.filter((p) =>
      (f === "all" || (f === "present" && p.status === "present") || (f === "absent" && p.status === "absent") ||
        (f === "inside" && p.inside)) &&
      (!q || p.name.toLowerCase().includes(q) || String(p.roll_no).toLowerCase().includes(q)));
    if (!r.total && !r.people.length) {
      $("attRows").innerHTML = `<tr><td colspan="8" class="empty">Nobody is registered yet. Go to the People page to add students and staff.</td></tr>`;
      return;
    }
    $("attRows").innerHTML = rows.length ? rows.map((p) => {
      const badge = p.status === "absent" ? `<span class="badge absent">Absent</span>`
        : p.inside ? `<span class="badge inside">Present - inside</span>` : `<span class="badge present">Present</span>`;
      const pic = p.snapshot ? `<img class="thumb" src="${snapUrl(p.snapshot)}" alt="" loading="lazy">`
        : `<span class="thumb-ph" aria-hidden="true">${esc(p.name.slice(0, 1))}</span>`;
      const act = isToday && !p.name.endsWith("(deleted)") ? `<div class="small-actions">
          <button type="button" data-mark="in" data-pid="${esc(p.person_id)}">Mark in</button>
          <button type="button" data-mark="out" data-pid="${esc(p.person_id)}">Mark out</button></div>` : "";
      return `<tr>
        <td><span class="who">${pic}<span>${esc(p.name)}<br><small class="fine">${esc(p.role)}${p.department ? " - " + esc(p.department) : ""}</small></span></span></td>
        <td>${esc(p.roll_no)}</td><td>${badge}${p.manual ? `<br><span class="tag-manual">marked by hand</span>` : ""}</td>
        <td>${esc(p.check_in || "-")}</td><td>${esc(p.check_out || (p.inside ? "still inside" : "-"))}</td>
        <td class="num">${esc(p.time_inside || "-")}</td><td>${esc(p.camera || "-")}</td><td>${act}</td></tr>`;
    }).join("") : `<tr><td colspan="8" class="empty">No one matches.</td></tr>`;
  }

  $("attRows").addEventListener("click", async (e) => {
    const b = e.target.closest("[data-mark]");
    if (!b) return;
    b.disabled = true;
    try {
      await postJson("/api/attendance/manual", { person_id: b.dataset.pid, kind: b.dataset.mark });
      await loadAttendance();
    } catch (err) { msg("attMsg", err.message); b.disabled = false; }
  });

  /* ============================================================== PEOPLE */
  const SLOTS = [
    { key: "front", title: "Front", tip: "Look straight at the camera", icon: "\u{1F642}" },
    { key: "left", title: "Left side", tip: "Turn the head to the LEFT ~30-45\u00B0", icon: "\u2190" },
    { key: "right", title: "Right side", tip: "Turn the head to the RIGHT ~30-45\u00B0", icon: "\u2192" },
  ];
  const reg = { files: {}, people: [], stream: null, camSlot: null, addTo: null };

  function renderSlots() {
    $("slots").innerHTML = SLOTS.map((s) => {
      const f = reg.files[s.key];
      const url = f ? URL.createObjectURL(f) : "";
      return `<div class="slot ${f ? "filled" : ""}">
        <div class="pic" ${url ? `style="background-image:url('${url}')"` : ""} aria-hidden="true">${url ? "" : s.icon}</div>
        <b>${s.title}${s.key === "front" ? " *" : ""}</b><small>${s.tip}</small>
        <div class="slot-btns">
          <label>Choose photo<input class="sr-only" type="file" accept="image/*" data-slot="${s.key}" aria-label="${s.title} photo"></label>
          <button type="button" data-cam="${s.key}">Use webcam</button>
          ${f ? `<button type="button" data-clear="${s.key}">Remove</button>` : ""}
        </div></div>`;
    }).join("");
  }
  $("slots").addEventListener("change", (e) => {
    const inp = e.target.closest("input[data-slot]");
    if (inp && inp.files[0]) { reg.files[inp.dataset.slot] = inp.files[0]; renderSlots(); }
  });
  $("slots").addEventListener("click", (e) => {
    const c = e.target.closest("[data-cam]"), x = e.target.closest("[data-clear]");
    if (c) openCam(c.dataset.cam);
    if (x) { delete reg.files[x.dataset.clear]; renderSlots(); }
  });

  async function openCam(slot) {
    reg.camSlot = slot;
    const s = SLOTS.find((q) => q.key === slot);
    $("camHint").textContent = `${s.title}: ${s.tip}. Then press Take photo.`;
    $("camBox").hidden = false;
    if (reg.stream) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      msg("pplMsg", "This browser cannot use the webcam here. Open the app as http://localhost:8000 or choose photo files instead.");
      $("camBox").hidden = true;
      return;
    }
    try {
      reg.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      $("camVideo").srcObject = reg.stream;
    } catch (e) {
      $("camBox").hidden = true;
      msg("pplMsg", "Could not open the webcam (" + e.message + "). If the live camera is using it, stop the live camera first, or choose photo files.");
    }
  }
  function closeCam() {
    if (reg.stream) reg.stream.getTracks().forEach((t) => t.stop());
    reg.stream = null;
    $("camVideo").srcObject = null;
    $("camBox").hidden = true;
  }
  $("camClose").addEventListener("click", closeCam);
  $("camSnap").addEventListener("click", () => {
    const v = $("camVideo");
    if (!v.videoWidth) return;
    const c = document.createElement("canvas");
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext("2d").drawImage(v, 0, 0);
    c.toBlob((blob) => {
      reg.files[reg.camSlot] = new File([blob], `${reg.camSlot}.jpg`, { type: "image/jpeg" });
      renderSlots();
      const next = SLOTS.find((s) => !reg.files[s.key]);        // move on to the next empty angle
      if (next) openCam(next.key); else closeCam();
    }, "image/jpeg", 0.92);
  });

  $("regForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    msg("pplMsg", "");
    $("regNotes").innerHTML = "";
    const name = $("regName").value.trim(), roll = $("regRoll").value.trim();
    if (!name) { msg("pplMsg", "Please enter the full name."); $("regName").focus(); return; }
    if (!roll) { msg("pplMsg", "Please enter the roll number / employee ID."); $("regRoll").focus(); return; }
    if (!reg.files.front) { msg("pplMsg", "Please add at least the Front photo."); return; }
    const fd = new FormData();
    fd.append("name", name); fd.append("roll_no", roll);
    fd.append("role", $("regRole").value); fd.append("department", $("regDept").value.trim());
    for (const s of SLOTS) if (reg.files[s.key]) fd.append(s.key, reg.files[s.key]);
    for (const f of $("regExtra").files) fd.append("extra", f);
    $("regSubmit").disabled = true;
    $("regSubmit").textContent = "Reading the faces...";
    try {
      const r = await api("/api/people", { method: "POST", body: fd });
      $("regNotes").innerHTML = `<div class="msg ok">${esc(r.person.name)} is registered with ${r.person.samples} face samples
        (${esc(r.person.angles.join(", "))}).</div>` + r.notes.map((n) => `<div class="msg info">${esc(n)}</div>`).join("");
      $("regForm").reset();
      reg.files = {};
      renderSlots();
      closeCam();
      loadPeople();
    } catch (err) {
      msg("pplMsg", err.message);
    } finally {
      $("regSubmit").disabled = false;
      $("regSubmit").textContent = "Register";
    }
  });

  async function loadPeople() {
    try {
      reg.people = await api("/api/people");
      renderPeople();
    } catch (e) { msg("pplMsg", e.message); }
  }
  $("pplSearch").addEventListener("input", renderPeople);

  function renderPeople() {
    const q = $("pplSearch").value.trim().toLowerCase();
    const list = reg.people.filter((p) => !q || p.name.toLowerCase().includes(q) || String(p.roll_no).toLowerCase().includes(q));
    $("pplCount").textContent = `(${reg.people.length})`;
    if (!reg.people.length) {
      $("pplList").innerHTML = `<p class="empty">Nobody registered yet. Use the form to add the first person.</p>`;
      return;
    }
    $("pplList").innerHTML = list.map((p) => {
      const faces = (p.photos || []).filter((ph) => ph.face).slice(0, 4)
        .map((ph) => `<img src="/api/people/${esc(p.id)}/image/${esc(ph.face)}" alt="${esc(ph.angle)} face" title="${esc(ph.angle)}" loading="lazy">`).join("");
      const angles = ["front", "left", "right"].map((a) =>
        `<span class="${(p.angles || []).includes(a) ? "" : "miss"}">${a}</span>`).join("");
      return `<div class="pcard">
        <div class="faces">${faces}</div>
        <h3>${esc(p.name)}</h3>
        <p>${esc(p.roll_no)} - ${esc(p.role)}${p.department ? " - " + esc(p.department) : ""}</p>
        <div class="angles">${angles}<span class="muted" style="background:none">${p.samples} samples</span></div>
        <div class="small-actions">
          <button type="button" data-add="${esc(p.id)}">Add photos</button>
          <button type="button" data-del="${esc(p.id)}" data-name="${esc(p.name)}">Remove</button>
        </div></div>`;
    }).join("") || `<p class="empty">No one matches.</p>`;
  }

  $("pplList").addEventListener("click", async (e) => {
    const add = e.target.closest("[data-add]"), del = e.target.closest("[data-del]");
    if (add) { reg.addTo = add.dataset.add; $("addPhotosInput").value = ""; $("addPhotosInput").click(); }
    if (del) {
      if (!confirm(`Remove ${del.dataset.name}? Their face data is deleted. Past attendance stays.`)) return;
      try { await api(`/api/people/${del.dataset.del}`, { method: "DELETE" }); loadPeople(); }
      catch (err) { msg("pplMsg", err.message); }
    }
  });
  $("addPhotosInput").addEventListener("change", async () => {
    const files = $("addPhotosInput").files;
    if (!files.length || !reg.addTo) return;
    const fd = new FormData();
    for (const f of files) fd.append("extra", f);
    msg("pplMsg", "Adding photos...", "info");
    try {
      const r = await api(`/api/people/${reg.addTo}/photos`, { method: "POST", body: fd });
      msg("pplMsg", `Added. ${r.person.name} now has ${r.person.samples} face samples. ${r.notes.join(" ")}`, "ok");
      loadPeople();
    } catch (err) { msg("pplMsg", err.message); }
  });

  /* ============================================================== ALERTS */
  $("alDay").value = today();
  $("alDay").max = today();
  $("alDay").addEventListener("change", loadAlerts);
  $("alOpenOnly").addEventListener("change", loadAlerts);

  async function loadAlerts() {
    try {
      let list = await api(`/api/alerts?day=${$("alDay").value || today()}`);
      if ($("alOpenOnly").checked) list = list.filter((a) => !a.reviewed);
      msg("alMsg", "");
      $("alList").innerHTML = list.length ? list.map((a) => `
        <article class="acard ${esc(a.kind)} ${a.reviewed ? "done" : ""}">
          <div class="pics">
            ${a.snapshot ? `<a href="${snapUrl(a.snapshot)}" target="_blank" rel="noopener"><img src="${snapUrl(a.snapshot)}" alt="Person" loading="lazy"></a>` : `<div class="noimg">no photo</div>`}
            ${a.face_snapshot ? `<img src="${snapUrl(a.face_snapshot)}" alt="Face" loading="lazy">` : `<div class="noimg">no face</div>`}
          </div>
          <div class="body">
            <div><span class="kind ${esc(a.kind)}">${a.kind === "unknown" ? "UNKNOWN PERSON" : "FACE HIDDEN"}</span>
              <b>${esc(a.time)}</b> <span class="muted">- ${esc(a.camera || "")}</span></div>
            <p>${esc(a.note)}</p>
            ${a.resolved_name ? `<p class="resolved">Later recognised as ${esc(a.resolved_name)}</p>` : ""}
            <label class="check"><input type="checkbox" data-review="${a.id}" ${a.reviewed ? "checked" : ""}><span>Checked by staff</span></label>
          </div></article>`).join("")
        : `<p class="empty">No alerts${$("alOpenOnly").checked ? " left to check" : ""} on this day.</p>`;
    } catch (e) { msg("alMsg", e.message); }
    updateAlertBadge();
  }
  $("alList").addEventListener("change", async (e) => {
    const c = e.target.closest("[data-review]");
    if (!c) return;
    try { await postJson(`/api/alerts/${c.dataset.review}/review`, { reviewed: c.checked }); loadAlerts(); }
    catch (err) { msg("alMsg", err.message); }
  });

  /* ================================================================ start */
  renderSlots();
  loadCcSettings();
  updateSourceFields();
  $("doorBox").hidden = selectedMode() !== "door";
  showTab(location.hash.slice(1) || "live", false);
  updateAlertBadge();
  setInterval(updateAlertBadge, 30000);
})();
