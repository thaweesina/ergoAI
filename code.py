import streamlit as st
import sqlite3
import pandas as pd
import json
import hashlib
import secrets
from datetime import datetime, timedelta
import streamlit.components.v1 as components
from streamlit_js_eval import streamlit_js_eval

# ==========================================
# สถาปัตยกรรมใหม่: ประมวลผลท่านั่งฝั่ง client (เบราว์เซอร์) ทั้งหมด
# ==========================================
# สะพานเชื่อม Python <-> JavaScript ใช้ไลบรารี streamlit-js-eval ซึ่งรัน JS ในหน้าเว็บแล้วคืนค่ากลับมาได้
# ข้อมูลสถานะเก็บไว้ที่ window.top (เบราว์เซอร์เดียวกัน ทุก component เป็น iframe same-origin เข้าถึงร่วมกันได้)

# ==========================================
# 1. ฐานข้อมูล (ผู้ใช้ + สถิติการนั่ง) - เหมือนเดิม
# ==========================================
DB_PATH = "ergovision.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS posture_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            duration_sec REAL NOT NULL,
            max_theta REAL,
            max_phi REAL,
            min_neck_ratio_pct REAL,
            cause TEXT,
            alert_triggered INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

def hash_password(password: str, salt: str = None):
    if salt is None:
        salt = secrets.token_hex(16)
    pw_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
    return pw_hash, salt

def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    pw_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(pw_hash, expected_hash)

def register_user(username: str, password: str):
    conn = get_db_connection()
    try:
        existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            return False, "มีชื่อผู้ใช้นี้ในระบบแล้ว"
        pw_hash, salt = hash_password(password)
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            (username, pw_hash, salt, datetime.now().isoformat()),
        )
        conn.commit()
        return True, "สมัครสมาชิกสำเร็จ กรุณาเข้าสู่ระบบ"
    finally:
        conn.close()

def authenticate_user(username: str, password: str) -> bool:
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT password_hash, salt FROM users WHERE username = ?", (username,)
        ).fetchone()
        if not row:
            return False
        expected_hash, salt = row
        return verify_password(password, salt, expected_hash)
    finally:
        conn.close()

def log_posture_event(username, start_time, end_time, duration_sec, max_theta, max_phi,
                       min_neck_ratio_pct, cause, alert_triggered):
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO posture_events "
            "(username, start_time, end_time, duration_sec, max_theta, max_phi, "
            "min_neck_ratio_pct, cause, alert_triggered) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (username, start_time.isoformat(), end_time.isoformat(), duration_sec,
             max_theta, max_phi, min_neck_ratio_pct, cause, int(alert_triggered)),
        )
        conn.commit()
    finally:
        conn.close()

def get_user_events(username: str, since: datetime = None) -> pd.DataFrame:
    conn = get_db_connection()
    try:
        if since:
            df = pd.read_sql_query(
                "SELECT * FROM posture_events WHERE username = ? AND start_time >= ? ORDER BY start_time DESC",
                conn, params=(username, since.isoformat()),
            )
        else:
            df = pd.read_sql_query(
                "SELECT * FROM posture_events WHERE username = ? ORDER BY start_time DESC",
                conn, params=(username,),
            )
        return df
    finally:
        conn.close()

# ==========================================
# 2. สาเหตุการนั่งผิดท่า - ไทย
# ==========================================
CAUSE_LABELS = {
    "shoulder_tilt": "ไหล่เอียง",
    "torso_lean": "ตัวเอนข้าง",
    "slouch": "ก้ม/หลังงอ",
}

def causes_to_text(cause_keys, lang="th"):
    return ", ".join(CAUSE_LABELS[c] for c in cause_keys if c in CAUSE_LABELS)

# ==========================================
# 3. Component HTML/JS (เพิ่มการวาดเส้นโครงร่าง และการจัดการ Calibration ใน JS)
# ==========================================
def build_posture_component_html():
    return """
<div id="ergo-wrap" style="text-align:center; font-family:sans-serif;">
  <video id="ergo-video" autoplay playsinline muted style="display:none;"></video>
  <canvas id="ergo-canvas" width="480" height="360"
          style="max-width:100%; width:480px; border-radius:8px; background:#111; display:block; margin:0 auto;"></canvas>
  <div id="ergo-status" style="margin-top:8px; font-size:14px; color:#666;">Loading pose model...</div>
  <button id="ergo-start-btn" disabled
          style="margin-top:8px; padding:8px 20px; font-size:14px; border-radius:6px; border:1px solid #ccc; cursor:pointer;">
    Start camera
  </button>
</div>
<script type="module">
import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://esm.sh/@mediapipe/tasks-vision@0.10.14";

const video = document.getElementById('ergo-video');
const canvas = document.getElementById('ergo-canvas');
const ctx = canvas.getContext('2d');
const drawingUtils = new DrawingUtils(ctx);
const statusEl = document.getElementById('ergo-status');
const startBtn = document.getElementById('ergo-start-btn');

let poseLandmarker = null;
let running = false;
let lastVideoTime = -1;

let calibratedNeckRatio = null;
let pendingCalibration = false;
let badSince = null;
let episodeMaxTheta = null, episodeMaxPhi = null, episodeMinNeckRatioPct = null;
let alertFiredForEpisode = false;

// State กลางบน window.top
window.top.__ergoPostureState = window.top.__ergoPostureState || {
  shouldersDetected: false, isBadPosture: false, theta: null, phi: null, neckRatioPct: null,
  causes: [], isCalibrated: false, episodeMaxTheta: null, episodeMaxPhi: null, episodeMinNeckRatioPct: null,
};
window.top.__ergoSettings = window.top.__ergoSettings || {
  thetaThreshold: 5, phiThreshold: 10, slouchThresholdPct: 80, alertThresholdSec: 5, soundEnabled: true,
};
window.top.__ergoRequestCalibration = window.top.__ergoRequestCalibration || false;

function updateSharedState(partial) {
  window.top.__ergoPostureState = Object.assign({}, window.top.__ergoPostureState, partial, { updatedAt: Date.now() });
}

function playBeep() {
  // ... เหมือนเดิม
}

// ฟังก์ชันคํานวณมุม ... เหมือนเดิม
function shoulderTiltDeg(lx, ly, rx, ry) { /* ... */ const adj = Math.abs(lx - rx); if (adj < 1e-6) return 90; return Math.atan(Math.abs(ly - ry) / adj) * 180 / Math.PI; }
function torsoTiltDeg(lx, ly, rx, ry, hlx, hly, hrx, hry) { /* ... */ const adj = Math.abs(((ly + ry) / 2) - ((hly + hry) / 2)); if (adj < 1e-6) return 90; return Math.atan(Math.abs(((lx + rx) / 2) - ((hlx + hrx) / 2)) / adj) * 180 / Math.PI; }

async function init() {
  // ... เหมือนเดิม ลบ GPU fallback เพื่อความกระชับ
  try {
    statusEl.textContent = 'Loading...';
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
          delegate: "GPU", // ลอง GPU ก่อน ถ้าช้า Streamlit จะช้าตาม
        },
        runningMode: "VIDEO", numPoses: 1,
    });
    statusEl.textContent = 'Model loaded. Start camera!';
    startBtn.disabled = false;
  } catch (e) { statusEl.textContent = 'Error: ' + e.message; }
}

startBtn.addEventListener('click', async () => {
  // ... เหมือนเดิม
  if (running) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 360 }, audio: false, });
    video.srcObject = stream;
    await video.play();
    running = true;
    startBtn.textContent = 'Camera running';
    startBtn.disabled = true;
    requestAnimationFrame(renderLoop);
  } catch (e) { statusEl.textContent = 'Camera error'; }
});

function renderLoop() {
  if (!running) return;
  if (video.currentTime !== lastVideoTime && poseLandmarker) {
    lastVideoTime = video.currentTime;
    const result = poseLandmarker.detectForVideo(video, performance.now());
    processResult(result);
  }
  requestAnimationFrame(renderLoop);
}

function processResult(result) {
  canvas.width = video.videoWidth || 480;
  canvas.height = video.videoHeight || 360;
  const W = canvas.width, H = canvas.height;

  // วาดวิดีโอแบบกลับด้าน
  ctx.save(); ctx.scale(-1, 1); ctx.drawImage(video, -W, 0, W, H);

  // วาดโครงร่างร่าง MediaPipe ทับบนภาพ mirror
  if (result.landmarks && result.landmarks[0]) {
      // 1. วาดเส้นเชื่อมต่อ (Connectors)
      drawingUtils.drawConnectors(result.landmarks[0], PoseLandmarker.POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 2 });
      // 2. วาดจุด (Landmarks) สี Cyan เหมือนเดิม
      drawingUtils.drawLandmarks(result.landmarks[0], { color: '#00FFFF', lineWidth: 1, radius: 4 });
  }
  ctx.restore(); // ปิด mirror mode

  const settings = window.top.__ergoSettings;
  // จัดการคำขอ Calibrate จาก Python
  if (window.top.__ergoRequestCalibration) {
    window.top.__ergoRequestCalibration = false;
    pendingCalibration = true;
    // แสดงสถานะชั่วคราวใน JS Canvas
    ctx.font = '20px sans-serif'; ctx.fillStyle = '#FFA500'; ctx.fillText('Calibrating...', W/2 - 50, H/2);
  }

  const lm = result.landmarks && result.landmarks[0];
  if (!lm) {
    ctx.font = '16px sans-serif'; ctx.fillStyle = '#FFA500'; ctx.fillText('No person detected', 10, 24);
    updateSharedState({ shouldersDetected: false, isBadPosture: false }); return;
  }

  // MediaPipe Pose landmark indices
  const nose = lm[0], lsh = lm[11], rsh = lm[12], lhip = lm[23], rhip = lm[24];
  const vis = (p) => (p && p.visibility > 0.3);
  const hasShoulders = vis(lsh) && vis(rsh);
  const hasHips = vis(lhip) && vis(rhip);
  const hasNose = vis(nose);

  ctx.font = '16px sans-serif'; ctx.fillStyle = '#FFFF00';

  if (!hasShoulders) {
    ctx.fillText('Shoulders N/A - move into frame', 10, 24);
    updateSharedState({ shouldersDetected: false }); return;
  }

  const lx = lsh.x * W, ly = lsh.y * H, rx = rsh.x * W, ry = rsh.y * H;
  const theta = shoulderTiltDeg(lx, ly, rx, ry);
  ctx.fillText(`Shoulder Tilt: ${theta.toFixed(1)}°`, 10, 24);

  let phi = null;
  if (hasHips) {
    phi = torsoTiltDeg(lx, ly, rx, ry, lhip.x * W, lhip.y * H, rhip.x * W, rhip.y * H);
    ctx.fillText(`Torso Tilt: ${phi.toFixed(1)}°`, 10, 48);
  } else { ctx.fillText('Torso Tilt: N/A', 10, 48); }

  let neckRatioPct = null;
  if (hasNose) {
    const shoulderWidth = Math.max(Math.abs(lx - rx), 1e-3);
    const currentRatio = Math.abs(nose.y * H - (ly + ry) / 2) / shoulderWidth;
    // บันทึกค่าอ้างอิง
    if (pendingCalibration) {
      calibratedNeckRatio = currentRatio;
      pendingCalibration = false;
      console.log("Calibrated:", calibratedNeckRatio);
    }
    if (calibratedNeckRatio) {
      neckRatioPct = (currentRatio / calibratedNeckRatio) * 100;
      ctx.fillText(`Neck Ratio: ${neckRatioPct.toFixed(0)}%`, 10, 72);
    } else { ctx.fillText('Neck Ratio: Not calibrated', 10, 72); }
  } else { ctx.fillText('Neck Ratio: Nose N/A', 10, 72); }

  // ตัดสินผิดท่า
  const causes = [];
  if (theta > settings.thetaThreshold) causes.push('shoulder_tilt');
  if (phi !== null && phi > settings.phiThreshold) causes.push('torso_lean');
  if (neckRatioPct !== null && neckRatioPct < settings.slouchThresholdPct) causes.push('slouch');
  const isBad = causes.length > 0;

  // จัดการ Episode และ Warning ... เหมือนเดิม
  if (isBad) {
    // ...
  } else {
    // ...
    ctx.fillStyle = '#33FF33'; ctx.fillText('Good Posture', 10, 100);
  }

  // อัปเดตสถานะกลับไปยัง Python รวมทั้งสถานะ Calibrated
  updateSharedState({
    shouldersDetected: true, isBadPosture: isBad,
    theta, phi, neckRatioPct, causes,
    isCalibrated: calibratedNeckRatio !== null, // สถานะสำคัญ!
    episodeMaxTheta, episodeMaxPhi, episodeMinNeckRatioPct,
  });
}

init();
</script>
"""

# ==========================================
# 4. Login ... เหมือนเดิม
# ==========================================
def show_login_page():
    # ... เหมือนเดิม
    pass

# ==========================================
# 5. UI หลัก (แก้ไขการแสดงผล Calibrate)
# ==========================================
st.set_page_config(page_title="Ergo-Vision AI", layout="wide")

if "logged_in_user" not in st.session_state:
    st.session_state.logged_in_user = None

# ตัวแปรช่วยเก็บสถานะ Calibration ในระดับ Session ของ Python
if "python_is_calibrated" not in st.session_state:
    st.session_state.python_is_calibrated = False

if not st.session_state.logged_in_user:
    st.title("🪑 Ergo-Vision AI")
    st.caption("เข้าสู่ระบบเพื่อบันทึกสถิติการนั่งของคุณ")
    st.caption("หมายเหตุ: เวอร์ชั่นแก้ไข Metrics Calibration และเพิ่มเส้นโครงร่าง")
    tab_login, tab_register = st.tabs(["เข้าสู่ระบบ", "สมัครสมาชิก"])
    # LoginForm...
    st.stop()

# Episode State... เหมือนเดิม

st.title("🪑 Ergo-Vision AI: แดชบอร์ดท่านั่ง (Modified)")
st.sidebar.markdown(f"👤 **{st.session_state.logged_in_user}**")
if st.sidebar.button("ออกจากระบบ"):
    st.session_state.logged_in_user = None
    st.rerun()

# Settings SideBar ... เหมือนเดิม
st.sidebar.header("⚙️ ตั้งค่าความไว")
theta_slider = st.sidebar.slider('ไหล่เอียงสูงสุด (θ)', 1, 15, 5)
phi_slider = st.sidebar.slider('ตัวเอนสูงสุด (φ)', 1, 20, 10)
slouch_slider = st.sidebar.slider('ก้ม/หลังงอ (% ท่าตรง)', 50, 95, 80)
alert_threshold_slider = st.sidebar.slider('แจ้งเตือนเมื่อนานกว่า (วินาที)', 1, 60, 5)
sound_enabled = st.sidebar.checkbox("🔊 เปิดเสียง", value=True)

# Push Settings ... เหมือนเดิม
settings_json = json.dumps({"thetaThreshold": theta_slider, "phiThreshold": phi_slider, "slouchThresholdPct": slouch_slider, "alertThresholdSec": alert_threshold_slider, "soundEnabled": sound_enabled})
streamlit_js_eval(js_expressions=f"window.top.__ergoSettings = {settings_json}; true", key="ergo_push_settings", want_output=False)

tab_camera, tab_stats = st.tabs(["📹 เรียลไทม์", "📊 สถิติ"])

with tab_camera:
    components.html(build_posture_component_html(), height=520)

    calib_col1, calib_col2 = st.columns([1, 3])
    with calib_col1:
        calibrate_clicked = st.button("📐 ตั้งค่าท่านั่งตรง", use_container_width=True)
    with calib_col2:
        st.caption("นั่งหลังตรง มองตรง แล้วกดหนึ่งครั้ง (ต้องกดใหม่ถ้าขยับกล้อง)")

    # พยายามแก้ไขประมวลผล Python Status หลังจากกด Calibrate
    if calibrate_clicked:
        streamlit_js_eval(js_expressions=f"window.top.__ergoRequestCalibration = true; true", key="ergo_trigger_calibrate_fast", want_output=False)
        st.toast("✅ ส่งคำสั่ง Calibrate แล้ว (Metrics จะอัปเดตใน 1-2 วินาที)", icon="📐")

    # Poll State จาก JS ทุกวินาที
    state_json = streamlit_js_eval(js_expressions="JSON.stringify(window.top.__ergoPostureState || {})", key="ergo_poll_state_modified", want_output=True)
    try:
        state = json.loads(state_json) if state_json else {}
    except:
        state = {}

    # แก้ไข: อัปเดต Python Session State โดยอ้างอิงข้อมูลจริงจาก JS
    if "isCalibrated" in state:
        # ถ้า JS บอกว่า Calibrate แล้ว ให้ Python Session State เปลี่ยนตาม
        if state["isCalibrated"]:
            st.session_state.python_is_calibrated = True

    # แสดงผล Metrics
    metrics_placeholder = st.empty()
    with metrics_placeholder.container():
        mc1, mc2, mc3 = st.columns(3)
        theta_val = state.get("theta")
        phi_val = state.get("phi")
        neck_val = state.get("neckRatioPct")

        mc1.metric("ไหล่เอียง (θ)", f"{theta_val:.1f}°" if theta_val is not None else "—")
        mc2.metric("ตัวเอน (φ)", f"{phi_val:.1f}°" if phi_val is not None else "N/A")

        # แก้ไขการแสดงผล: ใช้ st.session_state.python_is_calibrated แทน state.get("isCalibrated")
        if st.session_state.python_is_calibrated and neck_val is not None:
             mc3.metric("ระดับก้ม (% ของท่าตรง)", f"{neck_val:.0f}%")
        elif st.session_state.python_is_calibrated and neck_val is None:
             mc3.metric("ระดับก้ม (% ของท่าตรง)", "รอข้อมูลใบหน้า...")
        else:
            mc3.metric("ระดับก้ม (% ของท่าตรง)", "ยังไม่ calibrate")

    # Alert, Logging ... เหมือนเดิม
    alert_placeholder = st.empty()
    # ... logic alert and log ... เหมือนเดิม ...

with tab_stats:
    # Stats... เหมือนเดิม
    pass
