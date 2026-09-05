// Rep counter in the browser. Frames stay on the visitor's device: MediaPipe Tasks Vision runs
// as WebAssembly here, reads 33 body landmarks, we measure the angle at one joint and count.
// Mirrors AITrainer.py: same joints, same thresholds (curl 50 to 160 at the elbow, squat 70 to
// 165 at the knee), same two half rep rule, same 0.5 visibility cut.

const LOCAL = { wasm: "./vendor/wasm", bundle: "./vendor/vision_bundle.mjs", model: "./models/pose_landmarker_lite.task" };
const CDN = {
  wasm: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm",
  bundle: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs",
  model: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
};

// landmark indices from the MediaPipe pose model
const LM = { L_SHOULDER: 11, R_SHOULDER: 12, L_ELBOW: 13, R_ELBOW: 14, L_WRIST: 15, R_WRIST: 16, L_HIP: 23, R_HIP: 24, L_KNEE: 25, R_KNEE: 26, L_ANKLE: 27, R_ANKLE: 28 };
const EXERCISES = {
  curl: { label: "Bicep curl", joint: "Elbow", joints: [[LM.L_SHOULDER, LM.L_ELBOW, LM.L_WRIST], [LM.R_SHOULDER, LM.R_ELBOW, LM.R_WRIST]], range: [50, 160] },
  squat: { label: "Squat", joint: "Knee", joints: [[LM.L_HIP, LM.L_KNEE, LM.L_ANKLE], [LM.R_HIP, LM.R_KNEE, LM.R_ANKLE]], range: [70, 165] },
};
const VISIBILITY = 0.5;
// skeleton edges to draw (upper body, torso, legs)
const EDGES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28]];

/** Two halves make a rep: reach the top (percent >= 99), then return (percent <= 1). */
export class RepCounter {
  constructor(low, high) { this.low = low; this.high = high; this.count = 0; this.direction = 0; this.percent = 0; }
  update(angle) {
    if (angle == null) return this.count;
    const t = (angle - this.low) / (this.high - this.low);
    this.percent = Math.max(0, Math.min(100, (1 - t) * 100));
    if (this.percent >= 99 && this.direction === 0) { this.count += 0.5; this.direction = 1; }
    else if (this.percent <= 1 && this.direction === 1) { this.count += 0.5; this.direction = 0; }
    return this.count;
  }
  get reps() { return Math.floor(this.count); }
  reset() { this.count = 0; this.direction = 0; this.percent = 0; }
}

/** Angle at b, from a to c, in degrees 0 to 180. */
export function angleAt(a, b, c) {
  const ang = Math.abs((Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x)) * 180 / Math.PI);
  return ang > 180 ? 360 - ang : ang;
}

const $ = (id) => document.getElementById(id);
const el = {
  video: $("video"), overlay: $("overlay"), empty: $("stageEmpty"), status: $("stageStatus"), badge: $("stageBadge"),
  btnCamera: $("btnCamera"), btnFile: $("btnFile"), fileInput: $("fileInput"), btnReset: $("btnReset"),
  reps: $("repsValue"), romPct: $("romPct"), romFill: $("romFill"), angle: $("statAngle"), joint: $("statJoint"), fps: $("statFps"),
  formState: $("formState"), formText: $("formText"),
};
const ctx = el.overlay.getContext("2d");

let exercise = EXERCISES.curl;
let counter = new RepCounter(...exercise.range);
let landmarker = null, running = false, stream = null, rafId = 0, lastTs = -1, fpsSamples = [], mirror = true;

const setStatus = (text) => { el.status.textContent = text || ""; };
const setForm = (state, text) => { el.formState.dataset.state = state; el.formText.textContent = text; };

async function loadRuntime() {
  setStatus("Loading the pose model…");
  const tryImport = async (url) => (await import(url));
  let vision;
  try { vision = await tryImport(LOCAL.bundle); } catch { vision = await tryImport(CDN.bundle); }
  const { FilesetResolver, PoseLandmarker } = vision;
  const make = async (wasm, model) => {
    const files = await FilesetResolver.forVisionTasks(wasm);
    return PoseLandmarker.createFromOptions(files, {
      baseOptions: { modelAssetPath: model, delegate: "GPU" },
      runningMode: "VIDEO", numPoses: 1, minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
    });
  };
  try { landmarker = await make(LOCAL.wasm, LOCAL.model); }
  catch (e) { console.warn("local assets missing, using the CDN", e); landmarker = await make(CDN.wasm, CDN.model); }
  setStatus("");
}

function fitCanvas() {
  const w = el.video.videoWidth || el.video.clientWidth, h = el.video.videoHeight || el.video.clientHeight;
  if (w && (el.overlay.width !== w || el.overlay.height !== h)) { el.overlay.width = w; el.overlay.height = h; }
}

function pick(landmarks) {
  // both sides tried; the side with better visibility at all three points wins
  let best = null;
  for (const [a, b, c] of exercise.joints) {
    const pts = [landmarks[a], landmarks[b], landmarks[c]];
    const vis = Math.min(...pts.map((p) => p.visibility ?? 1));
    if (vis < VISIBILITY) continue;
    if (!best || vis > best.vis) best = { pts, vis, idx: [a, b, c] };
  }
  return best;
}

function draw(landmarks, sel, angle) {
  const W = el.overlay.width, H = el.overlay.height;
  ctx.clearRect(0, 0, W, H);
  ctx.save();
  if (mirror) { ctx.translate(W, 0); ctx.scale(-1, 1); }
  const P = (i) => ({ x: landmarks[i].x * W, y: landmarks[i].y * H, v: landmarks[i].visibility ?? 1 });
  ctx.lineWidth = Math.max(2, W / 320); ctx.lineCap = "round"; ctx.strokeStyle = "rgba(255,255,255,0.85)";
  for (const [i, j] of EDGES) { const a = P(i), b = P(j); if (a.v < VISIBILITY || b.v < VISIBILITY) continue; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }
  ctx.fillStyle = "rgba(255,255,255,0.95)";
  for (const [i] of EDGES.concat(EDGES.map(([, j]) => [j]))) { const p = P(i); if (p.v < VISIBILITY) continue; ctx.beginPath(); ctx.arc(p.x, p.y, ctx.lineWidth * 1.4, 0, Math.PI * 2); ctx.fill(); }
  if (sel) {
    const [a, b, c] = sel.idx.map(P);
    // the goniometer arc at the working joint, coloured by range of motion
    const pct = counter.percent;
    const color = pct < 25 ? "#3aa655" : pct < 50 ? "#e8c02b" : pct < 75 ? "#2f6fd6" : "#d64545";
    const r = Math.max(18, W / 22);
    const a1 = Math.atan2(a.y - b.y, a.x - b.x), a2 = Math.atan2(c.y - b.y, c.x - b.x);
    ctx.strokeStyle = color; ctx.lineWidth = Math.max(3, W / 240);
    ctx.beginPath(); ctx.moveTo(b.x, b.y); ctx.lineTo(a.x, a.y); ctx.moveTo(b.x, b.y); ctx.lineTo(c.x, c.y); ctx.stroke();
    let d = a2 - a1; while (d <= -Math.PI) d += Math.PI * 2; while (d > Math.PI) d -= Math.PI * 2;
    ctx.beginPath(); ctx.arc(b.x, b.y, r, a1, a1 + d, d < 0); ctx.stroke();
    ctx.restore(); ctx.save();
    // the number reads the right way round even when the picture is mirrored
    const lx = mirror ? W - b.x : b.x;
    ctx.font = `${Math.max(14, W / 34)}px "IBM Plex Mono", ui-monospace, monospace`; ctx.fillStyle = color; ctx.strokeStyle = "rgba(0,0,0,0.6)"; ctx.lineWidth = 4; ctx.lineJoin = "round";
    const label = `${Math.round(angle)}°`;
    ctx.strokeText(label, lx + r + 6, b.y - r * 0.4); ctx.fillText(label, lx + r + 6, b.y - r * 0.4);
  }
  ctx.restore();
}

function readout(angle, pct, tracking) {
  el.reps.textContent = String(counter.reps);
  el.romPct.textContent = `${Math.round(pct)}%`;
  el.romFill.style.width = `${pct}%`;
  el.angle.textContent = angle == null ? "—" : `${Math.round(angle)}°`;
  el.joint.textContent = exercise.joint;
  if (tracking) setForm("ok", pct >= 99 ? "Top of the rep" : pct <= 1 ? "Back to the start" : "Tracking");
}

function loop() {
  if (!running || !landmarker) return;
  rafId = requestAnimationFrame(loop);
  if (el.video.readyState < 2) return;
  fitCanvas();
  const now = performance.now();
  if (now <= lastTs) return; // VIDEO mode needs strictly increasing timestamps
  const res = landmarker.detectForVideo(el.video, now);
  fpsSamples.push(now); while (fpsSamples.length && now - fpsSamples[0] > 1000) fpsSamples.shift();
  el.fps.textContent = `${fpsSamples.length} fps`;
  lastTs = now;
  const lm = res.landmarks?.[0];
  if (!lm) { ctx.clearRect(0, 0, el.overlay.width, el.overlay.height); setForm("idle", "Waiting for a body in frame"); return; }
  const sel = pick(lm);
  if (!sel) { draw(lm, null, null); setForm("warn", `Show your whole ${exercise.joint.toLowerCase()} to the camera`); return; }
  const angle = angleAt(sel.pts[0], sel.pts[1], sel.pts[2]);
  counter.update(angle);
  draw(lm, sel, angle);
  readout(angle, counter.percent, true);
}

async function start(source) {
  stop(false);
  if (!landmarker) await loadRuntime();
  el.empty.hidden = true; el.badge.hidden = false;
  if (source === "camera") {
    setStatus("Asking for the camera…");
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    } catch (e) {
      el.empty.hidden = false; el.badge.hidden = true;
      setStatus(e.name === "NotAllowedError" ? "Camera permission was refused. Allow it in the browser, or load a video file instead." : "No camera was found. Load a video file instead.");
      return;
    }
    mirror = true; el.video.srcObject = stream; el.video.loop = false;
    el.btnCamera.textContent = "Turn off camera";
  } else {
    mirror = false; el.video.srcObject = null; el.video.src = source; el.video.loop = true;
  }
  el.video.classList.toggle("is-mirrored", mirror);
  await el.video.play();
  setStatus(""); running = true; lastTs = -1; loop();
}

function stop(showEmpty = true) {
  running = false; cancelAnimationFrame(rafId);
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  el.video.pause(); el.video.srcObject = null; el.video.removeAttribute("src");
  ctx.clearRect(0, 0, el.overlay.width, el.overlay.height);
  el.btnCamera.textContent = "Turn on camera"; el.fps.textContent = "—";
  if (showEmpty) { el.empty.hidden = false; el.badge.hidden = true; setForm("idle", "Waiting for a body in frame"); }
}

el.btnCamera.addEventListener("click", () => (stream ? stop() : start("camera")));
el.btnFile.addEventListener("click", () => el.fileInput.click());
el.fileInput.addEventListener("change", () => { const f = el.fileInput.files?.[0]; if (f) start(URL.createObjectURL(f)); });
el.btnReset.addEventListener("click", () => { counter.reset(); readout(null, 0, false); });
for (const b of document.querySelectorAll(".ex")) {
  b.addEventListener("click", () => {
    document.querySelectorAll(".ex").forEach((x) => x.classList.toggle("is-on", x === b));
    exercise = EXERCISES[b.dataset.exercise]; counter = new RepCounter(...exercise.range); readout(null, 0, false);
  });
}
readout(null, 0, false);
if (!navigator.mediaDevices?.getUserMedia) { el.btnCamera.disabled = true; setStatus("This browser has no camera access. Load a video file instead."); }
