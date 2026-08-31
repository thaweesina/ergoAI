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
# 1. ฐานข้อมูล (ผู้ใช้ + สถิติการนั่ง)
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
    for ddl in (
        "ALTER TABLE posture_events ADD COLUMN min_neck_ratio_pct REAL",
        "ALTER TABLE posture_events ADD COLUMN cause TEXT",
    ):
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass
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
# 2. สาเหตุการนั่งผิดท่า
# ==========================================
CAUSE_LABELS = {
    "shoulder_tilt": {"th": "ไหล่เอียง", "en": "Shoulder Tilt"},
    "torso_lean": {"th": "ตัวเอนข้าง", "en": "Torso Lean"},
    "slouch": {"th": "ก้ม/หลังงอ", "en": "Slouching"},
}

def causes_to_text(cause_keys, lang="th"):
    return ", ".join(CAUSE_LABELS[c][lang] for c in cause_keys if c in CAUSE_LABELS)

# ==========================================
# 3. Component HTML/JS (แก้บัคเส้นไม่ขึ้น และค่าค้าง)
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
import { PoseLandmarker, FilesetResolver } from "https://esm.sh/@mediapipe/tasks-vision@0.10.14";

const video = document.getElementById('ergo-video');
const canvas = document.getElementById('ergo-canvas');
const ctx = canvas.getContext('2d');
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
  try {
    const actx = new (window.AudioContext || window.webkitAudioContext)();
    const o = actx.createOscillator();
    const g = actx.createGain();
    o.connect(g);
    g.connect(actx.destination);
    o.type = 'sine';
    o.frequency.value = 880;
    g.gain.value = 0.25;
    o.start();
    setTimeout(() => { o.stop(); actx.close(); }, 500);
  } catch (e) { console.error('beep error', e); }
}

function shoulderTiltDeg(lx, ly, rx, ry) {
  const adj = Math.abs(lx - rx);
  if (adj < 1e-6) return 90;
  return Math.atan(Math.abs(ly - ry) / adj) * 180 / Math.PI;
}
function torsoTiltDeg(lx, ly, rx, ry, hlx, hly, hrx, hry) {
  const adj = Math.abs(((ly + ry) / 2) - ((hly + hry) / 2));
  if (adj < 1e-6) return 90;
  return Math.atan(Math.abs(((lx + rx) / 2) - ((hlx + hrx) / 2)) / adj) * 180 / Math.PI;
}

async function init() {
  try {
    statusEl.textContent = 'Loading pose model...';
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
    });
    statusEl.textContent = 'Model loaded. Click "Start camera" to begin.';
    startBtn.disabled = false;
  } catch (e) {
    statusEl.textContent = 'Failed to load pose model: ' + e.message;
  }
}

startBtn.addEventListener('click', async () => {
  if (running) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 480 }, height: { ideal: 360 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    running = true;
    startBtn.textContent = 'Camera running';
    startBtn.disabled = true;
    requestAnimationFrame(renderLoop);
  } catch (e) {
    statusEl.textContent = 'Camera error: ' + e.message;
  }
});

function renderLoop() {
  if (!running) return;
  if (video.currentTime !== lastVideoTime && poseLandmarker) {
    lastVideoTime = video.currentTime;
    try {
        const result = poseLandmarker.detectForVideo(video, performance.now());
        processResult(result);
    } catch(err) {
        console.error("Frame processing error:", err);
    }
  }
  requestAnimationFrame(renderLoop);
}

function processResult(result) {
  canvas.width = video.videoWidth || 480;
  canvas.height = video.videoHeight || 360;
  const W = canvas.width, H = canvas.height;

  // 1. วาดวิดีโอแบบกลับด้าน (Mirror)
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(video, -W, 0, W, H);
  ctx.restore(); // ปิด mirror mode เพื่อวาดเส้นทับไม่ให้พิกัดติดลบ

  const lm = result.landmarks && result.landmarks[0];
  const settings = window.top.__ergoSettings;
  
  if (window.top.__ergoRequestCalibration) {
    window.top.__ergoRequestCalibration = false;
    pendingCalibration = true;
    ctx.font = '20px sans-serif'; ctx.fillStyle = '#FFA500'; ctx.fillText('Calibrating...', W/2 - 50, H/2);
  }

  if (!lm) {
    ctx.font = '16px sans-serif'; ctx.fillStyle = '#FFA500';
    ctx.fillText('No person detected', 10, 24);
    updateSharedState({ shouldersDetected: false, isBadPosture: false, causes: [], theta: null, phi: null, neckRatioPct: null });
    return;
  }

  // 2. ฟังก์ชันช่วยวาดจุดและเส้นบนภาพที่กลับด้านแล้ว
  const pt = (index) => {
      const p = lm[index];
      // พลิกพิกัด X กลับเพื่อให้ตรงกับวิดีโอที่ mirror (1 - X)
      return (p && p.visibility > 0.3) ? { x: (1 - p.x) * W, y: p.y * H } : null;
  };
  const drawLine = (i, j) => {
      const p1 = pt(i), p2 = pt(j);
      if (p1 && p2) {
          ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
      }
  };

  // วาดเส้นโครงร่าง (Skeleton)
  ctx.strokeStyle = '#00FF00'; ctx.lineWidth = 2;
  drawLine(11, 12); // Shoulders
  drawLine(11, 23); drawLine(12, 24); // Torso
  drawLine(23, 24); // Hips
  drawLine(11, 13); drawLine(13, 15); // Left Arm
  drawLine(12, 14); drawLine(14, 16); // Right Arm
  drawLine(0, 11); drawLine(0, 12); // Neck to Shoulders

  // วาดจุด (Landmarks) สีฟ้า
  ctx.fillStyle = '#00FFFF';
  for(let i=0; i<lm.length; i++) {
      const p = pt(i);
      if(p) { ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, 2*Math.PI); ctx.fill(); }
  }

  // 3. ประมวลผลท่านั่ง
  const nose = lm[0], lsh = lm[11], rsh = lm[12], lhip = lm[23], rhip = lm[24];
  const vis = (p) => (p && p.visibility !== undefined) ? p.visibility : 1;
  const hasShoulders = lsh && rsh && vis(lsh) > 0.3 && vis(rsh) > 0.3;
  const hasHips = lhip && rhip && vis(lhip) > 0.3 && vis(rhip) > 0.3;
  const hasNose = nose && vis(nose) > 0.3;

  ctx.font = '16px sans-serif';

  if (!hasShoulders) {
    ctx.fillStyle = '#FFA500';
    ctx.fillText('Shoulders not detected - move into frame', 10, 24);
    updateSharedState({ shouldersDetected: false, isBadPosture: false, causes: [], theta: null, phi: null, neckRatioPct: null });
    return;
  }

  // คำนวณ (ใช้พิกัด raw จาก MediaPipe ไม่ติดปัญหา mirror)
  const lx = lsh.x * W, ly = lsh.y * H, rx = rsh.x * W, ry = rsh.y * H;
  const theta = shoulderTiltDeg(lx, ly, rx, ry);

  let phi = null;
  if (hasHips) {
    phi = torsoTiltDeg(lx, ly, rx, ry, lhip.x * W, lhip.y * H, rhip.x * W, rhip.y * H);
  }

  let neckRatioPct = null;
  if (hasNose) {
    const nx = nose.x * W, ny = nose.y * H;
    const shoulderMidY = (ly + ry) / 2;
    const shoulderWidth = Math.max(Math.abs(lx - rx), 1e-3);
    const currentRatio = Math.abs(ny - shoulderMidY) / shoulderWidth;
    if (pendingCalibration) {
      calibratedNeckRatio = currentRatio;
      pendingCalibration = false;
    }
    if (calibratedNeckRatio) {
      neckRatioPct = (currentRatio / calibratedNeckRatio) * 100;
    }
  }

  const slouchBad = neckRatioPct !== null && neckRatioPct < settings.slouchThresholdPct;
  const causes = [];
  if (theta > settings.thetaThreshold) causes.push('shoulder_tilt');
  if (phi !== null && phi > settings.phiThreshold) causes.push('torso_lean');
  if (slouchBad) causes.push('slouch');
  const isBad = causes.length > 0;

  ctx.fillStyle = '#FFFF00';
  ctx.fillText(`Shoulder Tilt: ${theta.toFixed(1)} deg`, 10, 24);
  ctx.fillText(phi !== null ? `Torso Tilt: ${phi.toFixed(1)} deg` : 'Torso Tilt: N/A (hips not visible)', 10, 48);
  ctx.fillText(neckRatioPct !== null ? `Neck Ratio: ${neckRatioPct.toFixed(0)}% of upright` : 'Neck Ratio: not calibrated', 10, 72);

  // 4. บันทึกสถิติและแจ้งเตือน (ส่วนนี้คือสิ่งที่หายไปในรอบที่แล้ว!)
  if (isBad) {
    if (!badSince) {
      badSince = Date.now();
      episodeMaxTheta = theta;
      episodeMaxPhi = phi;
      episodeMinNeckRatioPct = neckRatioPct;
      alertFiredForEpisode = false;
    } else {
      episodeMaxTheta = Math.max(episodeMaxTheta, theta);
      if (phi !== null) episodeMaxPhi = Math.max(episodeMaxPhi ?? 0, phi);
      if (neckRatioPct !== null) {
        episodeMinNeckRatioPct = (episodeMinNeckRatioPct === null) ? neckRatioPct : Math.min(episodeMinNeckRatioPct, neckRatioPct);
      }
    }
    const elapsed = (Date.now() - badSince) / 1000;
    const causeStr = causes.join(', ');
    if (elapsed >= settings.alertThresholdSec) {
      ctx.fillStyle = '#FF3333';
      ctx.fillText(`WARNING: ${causeStr} > ${Math.floor(elapsed)}s!`, 10, 100);
      if (!alertFiredForEpisode) {
        alertFiredForEpisode = true;
        if (settings.soundEnabled) playBeep();
      }
    } else {
      ctx.fillStyle = '#FFA500';
      ctx.fillText(`Warning: ${causeStr} (${Math.floor(elapsed)}/${settings.alertThresholdSec}s)`, 10, 100);
    }
  } else {
    badSince = null;
    episodeMaxTheta = null;
    episodeMaxPhi = null;
    episodeMinNeckRatioPct = null;
    alertFiredForEpisode = false;
    ctx.fillStyle = '#33FF33';
    ctx.fillText('Good Posture', 10, 100);
  }

  updateSharedState({
    shouldersDetected: true,
    isBadPosture: isBad,
    theta, phi, neckRatioPct,
    causes,
    isCalibrated: calibratedNeckRatio !== null,
    episodeMaxTheta, episodeMaxPhi, episodeMinNeckRatioPct,
  });
}

init();
</script>
"""

# ==========================================
# 4. หน้า Login / สมัครสมาชิก
# ==========================================
def show_login_page():
    st.title("🪑 Ergo-Vision AI")
    st.caption("เข้าสู่ระบบเพื่อบันทึกสถิติการนั่งของคุณ")

    tab_login, tab_register = st.tabs(["เข้าสู่ระบบ", "สมัครสมาชิก"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("ชื่อผู้ใช้")
            password = st.text_input("รหัสผ่าน", type="password")
            submitted = st.form_submit_button("เข้าสู่ระบบ", use_container_width=True)
            if submitted:
                if authenticate_user(username, password):
                    st.session_state.logged_in_user = username
                    st.rerun()
                else:
                    st.error("ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")

    with tab_register:
        with st.form("register_form"):
            new_username = st.text_input("ชื่อผู้ใช้ใหม่")
            new_password = st.text_input("รหัสผ่าน (อย่างน้อย 6 ตัวอักษร)", type="password")
            confirm_password = st.text_input("ยืนยันรหัสผ่าน", type="password")
            submitted = st.form_submit_button("สมัครสมาชิก", use_container_width=True)
            if submitted:
                if not new_username or not new_password:
                    st.error("กรุณากรอกชื่อผู้ใช้และรหัสผ่าน")
                elif new_password != confirm_password:
                    st.error("รหัสผ่านไม่ตรงกัน")
                elif len(new_password) < 6:
                    st.error("รหัสผ่านต้องมีอย่างน้อย 6 ตัวอักษร")
                else:
                    ok, msg = register_user(new_username, new_password)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

# ==========================================
# 5. หน้าตา UI หลัก
# ==========================================
st.set_page_config(page_title="Ergo-Vision AI", layout="wide")

if "logged_in_user" not in st.session_state:
    st.session_state.logged_in_user = None

if "python_is_calibrated" not in st.session_state:
    st.session_state.python_is_calibrated = False

if not st.session_state.logged_in_user:
    show_login_page()
    st.stop()

for key, default in [
    ("episode_start", None),
    ("alert_fired", False),
    ("episode_max_theta", None),
    ("episode_max_phi", None),
    ("episode_min_neck_ratio_pct", None),
    ("episode_causes", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("🪑 Ergo-Vision AI: แจ้งเตือนท่านั่ง Real-time")
st.markdown("ระบบตรวจจับท่านั่งทำงานผ่านกล้องและ MediaPipe ใช้งานได้เต็มรูปแบบ")

st.sidebar.markdown(f"👤 เข้าสู่ระบบในชื่อ: **{st.session_state.logged_in_user}**")
if st.sidebar.button("ออกจากระบบ"):
    st.session_state.logged_in_user = None
    st.session_state.episode_start = None
    st.session_state.python_is_calibrated = False
    st.rerun()

st.sidebar.header("⚙️ ตั้งค่าความไวการแจ้งเตือน")
theta_slider = st.sidebar.slider('ไหล่เอียงสูงสุด (θ)', 1, 15, 5)
phi_slider = st.sidebar.slider('ตัวเอนสูงสุด (φ)', 1, 20, 10, help="กล้องต้องเห็นช่วงเอว/สะโพกเพื่อตรวจจับส่วนนี้")
slouch_slider = st.sidebar.slider('ความไวการตรวจจับก้ม/หลังงอ (% ของท่านั่งตรง)', 50, 95, 80)
alert_threshold_slider = st.sidebar.slider('แจ้งเตือนเมื่อนั่งผิดท่านานกว่า (วินาที)', 1, 60, 5)
sound_enabled = st.sidebar.checkbox("🔊 เปิดเสียงแจ้งเตือน", value=True)

settings_json = json.dumps({
    "thetaThreshold": theta_slider,
    "phiThreshold": phi_slider,
    "slouchThresholdPct": slouch_slider,
    "alertThresholdSec": alert_threshold_slider,
    "soundEnabled": sound_enabled,
})
streamlit_js_eval(
    js_expressions=f"window.top.__ergoSettings = {settings_json}; true",
    key="ergo_push_settings",
    want_output=False,
)

tab_camera, tab_stats = st.tabs(["📹 เรียลไทม์", "📊 สถิติของฉัน"])

with tab_camera:
    components.html(build_posture_component_html(), height=520)

    calib_col1, calib_col2 = st.columns([1, 3])
    with calib_col1:
        calibrate_clicked = st.button("📐 ตั้งค่าท่านั่งตรง", use_container_width=True)
    with calib_col2:
        st.caption(
            "นั่งหลังตรง มองตรง แล้วกดปุ่มนี้หนึ่งครั้ง "
            "(ต้องกดใหม่ทุกครั้งที่ขยับเก้าอี้/กล้อง หรือรีเฟรชหน้า)"
        )
        
    if calibrate_clicked:
        streamlit_js_eval(
            js_expressions="window.top.__ergoRequestCalibration = true; true",
            key="ergo_trigger_calibrate_sync",
            want_output=False,
        )
        st.toast("✅ ส่งคำสั่งตั้งค่าท่านั่งตรงแล้ว (ตัวเลขระดับก้มจะอัปเดตในไม่ช้า)", icon="📐")

    alert_placeholder = st.empty()
    metrics_placeholder = st.empty()

    state_json = streamlit_js_eval(
        js_expressions="JSON.stringify(window.top.__ergoPostureState || {})",
        key="ergo_poll_state",
        want_output=True,
    )
    try:
        state = json.loads(state_json) if state_json else {}
    except:
        state = {}
        
    if state.get("isCalibrated"):
        st.session_state.python_is_calibrated = True

    with metrics_placeholder.container():
        mc1, mc2, mc3 = st.columns(3)
        theta_val = state.get("theta")
        phi_val = state.get("phi")
        neck_val = state.get("neckRatioPct")
        
        mc1.metric("มุมเอียงไหล่ (θ)", f"{theta_val:.1f}°" if theta_val is not None else "—")
        mc2.metric("มุมเอนตัว (φ)", f"{phi_val:.1f}°" if phi_val is not None else "N/A (ไม่เห็นสะโพก)")
        
        if st.session_state.python_is_calibrated and neck_val is not None:
            mc3.metric("ระดับก้ม (% ของท่าตรง)", f"{neck_val:.0f}%")
        elif st.session_state.python_is_calibrated:
            mc3.metric("ระดับก้ม (% ของท่าตรง)", "รอข้อมูลใบหน้า...")
        else:
            mc3.metric("ระดับก้ม (% ของท่าตรง)", "ยังไม่ calibrate")

    if state.get("isBadPosture"):
        if st.session_state.episode_start is None:
            st.session_state.episode_start = datetime.now()
            st.session_state.alert_fired = False

        elapsed = (datetime.now() - st.session_state.episode_start).total_seconds()
        st.session_state.episode_max_theta = state.get("episodeMaxTheta")
        st.session_state.episode_max_phi = state.get("episodeMaxPhi")
        st.session_state.episode_min_neck_ratio_pct = state.get("episodeMinNeckRatioPct")
        st.session_state.episode_causes = state.get("causes", [])
        cause_text = causes_to_text(state.get("causes", []), lang="th") or "ท่านั่งผิดปกติ"

        if elapsed >= alert_threshold_slider:
            st.session_state.alert_fired = True
            alert_placeholder.error(
                f"🚨 {cause_text} มานาน {int(elapsed)} วินาทีแล้ว! กรุณาปรับท่านั่งให้ถูกต้อง"
            )
        else:
            alert_placeholder.warning(
                f"⚠️ ท่านั่งเริ่มผิดปกติ: {cause_text} ({int(elapsed)}/{alert_threshold_slider} วินาที)"
            )
    else:
        if st.session_state.episode_start is not None:
            episode_end = datetime.now()
            duration = (episode_end - st.session_state.episode_start).total_seconds()
            if duration >= 1:
                log_posture_event(
                    st.session_state.logged_in_user,
                    st.session_state.episode_start,
                    episode_end,
                    duration,
                    st.session_state.episode_max_theta,
                    st.session_state.episode_max_phi,
                    st.session_state.episode_min_neck_ratio_pct,
                    causes_to_text(st.session_state.get("episode_causes", []), lang="th") or None,
                    st.session_state.alert_fired,
                )
            st.session_state.episode_start = None
            st.session_state.alert_fired = False
        alert_placeholder.success("✅ ท่านั่งถูกต้อง")

with tab_stats:
    st.subheader("📊 สถิติการนั่งของคุณ")

    period = st.selectbox("ช่วงเวลา", ["วันนี้", "7 วันล่าสุด", "30 วันล่าสุด", "ทั้งหมด"])
    since_map = {
        "วันนี้": datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
        "7 วันล่าสุด": datetime.now() - timedelta(days=7),
        "30 วันล่าสุด": datetime.now() - timedelta(days=30),
        "ทั้งหมด": None,
    }
    df = get_user_events(st.session_state.logged_in_user, since=since_map[period])

    if df.empty:
        st.info("ยังไม่มีข้อมูลสถิติในช่วงเวลานี้ (สถิติจะถูกบันทึกเมื่อคุณนั่งผิดท่าอย่างน้อย 1 วินาทีแล้วกลับมานั่งตรง)")
    else:
        df["start_time"] = pd.to_datetime(df["start_time"])
        total_bad_sec = df["duration_sec"].sum()
        num_incidents = len(df)
        num_alerts = int(df["alert_triggered"].sum())

        col1, col2, col3 = st.columns(3)
        col1.metric("เวลานั่งผิดท่ารวม", f"{total_bad_sec / 60:.1f} นาที")
        col2.metric("จำนวนครั้งที่นั่งผิดท่า", f"{num_incidents} ครั้ง")
        col3.metric("จำนวนครั้งที่แจ้งเตือน", f"{num_alerts} ครั้ง")

        df["date"] = df["start_time"].dt.date
        daily = df.groupby("date")["duration_sec"].sum() / 60
        st.markdown("**เวลานั่งผิดท่ารายวัน (นาที)**")
        st.bar_chart(daily)

        st.markdown("**รายละเอียดล่าสุด**")
        display_df = df[["start_time", "duration_sec", "max_theta", "max_phi",
                          "min_neck_ratio_pct", "cause", "alert_triggered"]].copy()
        display_df["duration_sec"] = pd.to_numeric(display_df["duration_sec"], errors="coerce").round(1)
        display_df["max_theta"] = pd.to_numeric(display_df["max_theta"], errors="coerce").round(1)
        display_df["max_phi"] = pd.to_numeric(display_df["max_phi"], errors="coerce").round(1)
        display_df["max_phi"] = display_df["max_phi"].apply(lambda v: f"{v}" if pd.notna(v) else "N/A")
        display_df["min_neck_ratio_pct"] = pd.to_numeric(display_df["min_neck_ratio_pct"], errors="coerce").round(0)
        display_df["min_neck_ratio_pct"] = display_df["min_neck_ratio_pct"].apply(
            lambda v: f"{v:.0f}%" if pd.notna(v) else "N/A"
        )
        display_df["cause"] = display_df["cause"].fillna("—")
        display_df["alert_triggered"] = display_df["alert_triggered"].map({1: "✅", 0: "—"})
        display_df.columns = ["เวลาเริ่ม", "ระยะเวลา (วินาที)", "ไหล่เอียงสูงสุด (°)", "ตัวเอนสูงสุด (°)",
                               "ก้มมากสุด (% ของท่าตรง)", "สาเหตุ", "แจ้งเตือน"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
