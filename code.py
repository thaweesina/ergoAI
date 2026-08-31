import streamlit as st
import pandas as pd
import json
import hashlib
import secrets
from datetime import datetime, timedelta
import streamlit.components.v1 as components
import os
import libsql_experimental as libsql  # <-- ใช้ libsql สำหรับ Turso

# ==========================================
# 0. สร้างไฟล์ Frontend สำหรับเชื่อมต่อข้อมูล 2 ทาง
# ==========================================
if not os.path.exists("ergo_frontend"):
    os.makedirs("ergo_frontend")

HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style> body { margin:0; padding:0; background: transparent; } </style>
</head>
<body>
  <div id="ergo-wrap" style="text-align:center; font-family:sans-serif;">
    <video id="ergo-video" autoplay playsinline muted style="display:none;"></video>
    <canvas id="ergo-canvas" width="480" height="360"
            style="max-width:100%; width:480px; border-radius:8px; background:#111; display:block; margin:0 auto;"></canvas>
    <div id="ergo-status" style="margin-top:8px; font-size:14px; color:#666;">กำลังโหลดโมเดล AI...</div>
    <button id="ergo-start-btn" disabled
            style="margin-top:8px; padding:8px 20px; font-size:14px; border-radius:6px; border:1px solid #ccc; cursor:pointer; background:#4CAF50; color:white;">
      เปิดกล้อง
    </button>
  </div>

  <script type="module">
    import { PoseLandmarker, FilesetResolver } from "https://esm.sh/@mediapipe/tasks-vision@0.10.14";

    const Streamlit = {
        setComponentReady: function() {
            window.parent.postMessage({ isStreamlitMessage: true, type: "streamlit:componentReady", apiVersion: 1 }, "*");
        },
        setFrameHeight: function(height) {
            window.parent.postMessage({ isStreamlitMessage: true, type: "streamlit:setFrameHeight", height: height }, "*");
        },
        setComponentValue: function(value) {
            window.parent.postMessage({ isStreamlitMessage: true, type: "streamlit:setComponentValue", value: value }, "*");
        }
    };

    let settings = { thetaThreshold: 5, slouchThresholdPct: 80, alertThresholdSec: 5, soundEnabled: true };
    let lastCalibrationId = 0;
    let pendingCalibration = false;

    window.addEventListener("message", function(event) {
        if (event.data && event.data.type === "streamlit:render") {
            const args = event.data.args;
            if (args.settings) settings = args.settings;
            if (args.calibrationId && args.calibrationId !== lastCalibrationId) {
                lastCalibrationId = args.calibrationId;
                pendingCalibration = true;
            }
            Streamlit.setFrameHeight(450);
        }
    });

    Streamlit.setComponentReady();
    Streamlit.setFrameHeight(450);

    const video = document.getElementById('ergo-video');
    const canvas = document.getElementById('ergo-canvas');
    const ctx = canvas.getContext('2d');
    const statusEl = document.getElementById('ergo-status');
    const startBtn = document.getElementById('ergo-start-btn');

    let poseLandmarker = null;
    let running = false;
    let lastVideoTime = -1;
    let calibratedNeckRatio = null;
    let lastSentTime = 0;

    let badSince = null;
    let episodeMaxTheta = null, episodeMinNeckRatioPct = null;
    let alertFiredForEpisode = false;
    let episodeCauses = new Set();

    function playBeep() {
      try {
        const actx = new (window.AudioContext || window.webkitAudioContext)();
        const o = actx.createOscillator();
        const g = actx.createGain();
        o.connect(g); g.connect(actx.destination);
        o.type = 'sine'; o.frequency.value = 880; g.gain.value = 0.25;
        o.start(); setTimeout(() => { o.stop(); actx.close(); }, 500);
      } catch (e) {}
    }

    function shoulderTiltDeg(lx, ly, rx, ry) {
      const adj = Math.abs(lx - rx);
      if (adj < 1e-6) return 90;
      return Math.atan(Math.abs(ly - ry) / adj) * 180 / Math.PI;
    }

    async function init() {
      try {
        const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
        poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task", delegate: "GPU" },
          runningMode: "VIDEO", numPoses: 1,
        });
        statusEl.textContent = 'พร้อมใช้งานแล้ว กดปุ่ม "เปิดกล้อง"';
        startBtn.disabled = false;
      } catch (e) { statusEl.textContent = 'Error: ' + e.message; }
    }

    startBtn.addEventListener('click', async () => {
      if (running) return;
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 360 }, audio: false });
        video.srcObject = stream;
        await video.play();
        running = true;
        startBtn.style.display = 'none';
        statusEl.style.display = 'none';
        requestAnimationFrame(renderLoop);
      } catch (e) { statusEl.textContent = 'Camera error: ' + e.message; }
    });

    function renderLoop() {
      if (!running) return;
      if (video.currentTime !== lastVideoTime && poseLandmarker) {
        lastVideoTime = video.currentTime;
        try {
            const result = poseLandmarker.detectForVideo(video, performance.now());
            processResult(result);
        } catch (e) { console.error(e); }
      }
      requestAnimationFrame(renderLoop);
    }

    function processResult(result) {
      canvas.width = video.videoWidth || 480;
      canvas.height = video.videoHeight || 360;
      const W = canvas.width, H = canvas.height;

      ctx.save(); ctx.scale(-1, 1); ctx.drawImage(video, -W, 0, W, H); ctx.restore();

      if (pendingCalibration) {
         ctx.font = '24px sans-serif'; ctx.fillStyle = '#FFA500'; ctx.fillText('Calibrating...', W/2 - 60, H/2);
      }

      const lm = result.landmarks && result.landmarks[0];
      if (!lm) {
         if (Date.now() - lastSentTime > 1000) {
             Streamlit.setComponentValue({ timestamp: Date.now(), theta: null, neckRatioPct: null, isBadPosture: false });
             lastSentTime = Date.now();
         }
         return;
      }

      const pt = (index) => {
          const p = lm[index];
          return (p && p.visibility > 0.3) ? { x: (1 - p.x) * W, y: p.y * H } : null;
      };
      const drawLine = (i, j) => {
          const p1 = pt(i), p2 = pt(j);
          if (p1 && p2) { ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke(); }
      };

      ctx.strokeStyle = '#00FF00'; ctx.lineWidth = 2;
      drawLine(11, 12); drawLine(11, 23); drawLine(12, 24); drawLine(23, 24);
      drawLine(11, 13); drawLine(13, 15); drawLine(12, 14); drawLine(14, 16);
      drawLine(0, 11); drawLine(0, 12);

      ctx.fillStyle = '#00FFFF';
      for(let i=0; i<lm.length; i++) {
          const p = pt(i);
          if(p) { ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, 2*Math.PI); ctx.fill(); }
      }

      const nose = lm[0], lsh = lm[11], rsh = lm[12];
      const vis = (p) => (p && p.visibility !== undefined) ? p.visibility : 1;
      const hasShoulders = lsh && rsh && vis(lsh) > 0.3 && vis(rsh) > 0.3;
      const hasNose = nose && vis(nose) > 0.3;

      ctx.font = '16px sans-serif';

      if (!hasShoulders) {
          if (Date.now() - lastSentTime > 1000) {
             Streamlit.setComponentValue({ timestamp: Date.now(), theta: null, neckRatioPct: null, isBadPosture: false });
             lastSentTime = Date.now();
          }
          return;
      }

      const lx = lsh.x * W, ly = lsh.y * H, rx = rsh.x * W, ry = rsh.y * H;
      const theta = shoulderTiltDeg(lx, ly, rx, ry);

      let neckRatioPct = null;
      if (hasNose) {
        const ny = nose.y * H;
        const shoulderMidY = (ly + ry) / 2;
        const shoulderWidth = Math.max(Math.abs(lx - rx), 1e-3);
        const currentRatio = Math.abs(ny - shoulderMidY) / shoulderWidth;
        if (pendingCalibration) { calibratedNeckRatio = currentRatio; pendingCalibration = false; }
        if (calibratedNeckRatio) neckRatioPct = (currentRatio / calibratedNeckRatio) * 100;
      }

      const causes = [];
      if (theta > settings.thetaThreshold) causes.push('shoulder_tilt');
      if (neckRatioPct !== null && neckRatioPct < settings.slouchThresholdPct) causes.push('slouch');
      const isBad = causes.length > 0;

      ctx.fillStyle = '#FFFF00';
      ctx.fillText(`Shoulder Tilt: ${theta.toFixed(1)} deg`, 10, 24);
      ctx.fillText(neckRatioPct !== null ? `Neck Ratio: ${neckRatioPct.toFixed(0)}%` : 'Neck Ratio: Not Calibrated', 10, 48);

      let episodeEnded = false;
      let epDur=0, epMaxT=0, epMinN=0, epCauses=[], epAlert=false;

      if (isBad) {
         if (!badSince) {
             badSince = Date.now();
             episodeMaxTheta = theta;
             episodeMinNeckRatioPct = neckRatioPct;
             alertFiredForEpisode = false;
             episodeCauses = new Set(causes);
         } else {
             episodeMaxTheta = Math.max(episodeMaxTheta, theta);
             if (neckRatioPct !== null) {
                 episodeMinNeckRatioPct = (episodeMinNeckRatioPct === null) ? neckRatioPct : Math.min(episodeMinNeckRatioPct, neckRatioPct);
             }
             causes.forEach(c => episodeCauses.add(c));
         }
         const elapsedSec = (Date.now() - badSince) / 1000;
         if (elapsedSec >= settings.alertThresholdSec) {
             ctx.fillStyle = '#FF3333';
             ctx.fillText(`WARNING: > ${Math.floor(elapsedSec)}s!`, 10, 72);
             if (!alertFiredForEpisode) {
                 alertFiredForEpisode = true;
                 if (settings.soundEnabled) playBeep();
             }
         } else {
             ctx.fillStyle = '#FFA500';
             ctx.fillText(`Warning... ${Math.floor(elapsedSec)}/${settings.alertThresholdSec}s`, 10, 72);
         }
      } else {
         ctx.fillStyle = '#33FF33';
         ctx.fillText('Good Posture', 10, 72);
         if (badSince) {
             const dur = (Date.now() - badSince) / 1000;
             if (dur >= 1.0) {
                 episodeEnded = true;
                 epDur = dur; epMaxT = episodeMaxTheta; epMinN = episodeMinNeckRatioPct;
                 epCauses = Array.from(episodeCauses); epAlert = alertFiredForEpisode;
             }
             badSince = null; episodeMaxTheta = null; episodeMinNeckRatioPct = null;
             alertFiredForEpisode = false; episodeCauses.clear();
         }
      }

      if (episodeEnded || Date.now() - lastSentTime > 1000) {
          const payload = {
              timestamp: Date.now(),
              theta: theta, neckRatioPct: neckRatioPct, isBadPosture: isBad,
              currentCauses: causes, isCalibrated: calibratedNeckRatio !== null,
              elapsedSec: badSince ? (Date.now() - badSince) / 1000 : 0
          };
          if (episodeEnded) {
              payload.episodeEnded = true; payload.durationSec = epDur;
              payload.maxTheta = epMaxT; payload.minNeckRatio = epMinN;
              payload.epCauses = epCauses; payload.alertTriggered = epAlert;
          }
          Streamlit.setComponentValue(payload);
          lastSentTime = Date.now();
      }
    }
    init();
  </script>
</body>
</html>
"""

with open("ergo_frontend/index.html", "w", encoding="utf-8") as f:
    f.write(HTML_CONTENT)

ergo_camera_component = components.declare_component("ergo_camera_component", path="ergo_frontend")

# ==========================================
# 1. ฐานข้อมูล (Turso Cloud SQLite)
# ==========================================
def get_db_connection():
    try:
        url = st.secrets["TURSO_URL"]
        token = st.secrets["TURSO_AUTH_TOKEN"]
        conn = libsql.connect(url, auth_token=token)
        return conn
    except Exception as e:
        st.error(f"ไม่สามารถเชื่อมต่อ Turso Database ได้: {e}")
        st.stop()

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

try:
    init_db()
except Exception:
    pass

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

def log_posture_event(username, duration_sec, max_theta, min_neck_ratio_pct, cause, alert_triggered):
    end_time = datetime.now()
    start_time = end_time - timedelta(seconds=duration_sec)
    
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO posture_events "
            "(username, start_time, end_time, duration_sec, max_theta, max_phi, "
            "min_neck_ratio_pct, cause, alert_triggered) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (username, start_time.isoformat(), end_time.isoformat(), duration_sec,
             max_theta, None, min_neck_ratio_pct, cause, int(alert_triggered)),
        )
        conn.commit()
    finally:
        conn.close()

def get_user_events(username: str, since: datetime = None) -> pd.DataFrame:
    conn = get_db_connection()
    try:
        if since:
            res = conn.execute(
                "SELECT * FROM posture_events WHERE username = ? AND start_time >= ? ORDER BY start_time DESC",
                (username, since.isoformat())
            )
        else:
            res = conn.execute(
                "SELECT * FROM posture_events WHERE username = ? ORDER BY start_time DESC",
                (username,)
            )
        
        rows = res.fetchall()
        columns = [column[0] for column in res.description]
        df = pd.DataFrame(rows, columns=columns)
        return df
    finally:
        conn.close()

# ==========================================
# 2. สาเหตุการนั่งผิดท่า
# ==========================================
CAUSE_LABELS = {
    "shoulder_tilt": {"th": "ไหล่เอียง", "en": "Shoulder Tilt"},
    "slouch": {"th": "ก้ม/หลังงอ", "en": "Slouching"},
}

def causes_to_text(cause_keys, lang="th"):
    return ", ".join(CAUSE_LABELS[c][lang] for c in cause_keys if c in CAUSE_LABELS)

# ==========================================
# 3. หน้า Login / UI หลัก
# ==========================================
st.set_page_config(page_title="Ergo-Vision AI", layout="wide")

if "logged_in_user" not in st.session_state:
    st.session_state.logged_in_user = None
if "calibration_id" not in st.session_state:
    st.session_state.calibration_id = 0
if "last_timestamp" not in st.session_state:
    st.session_state.last_timestamp = 0

if not st.session_state.logged_in_user:
    st.title("🪑 Ergo-Vision AI")
    st.caption("เข้าสู่ระบบเพื่อบันทึกสถิติการนั่งของคุณ")
    tab_login, tab_register = st.tabs(["เข้าสู่ระบบ", "สมัครสมาชิก"])
    with tab_login:
        with st.form("login_form"):
            username = st.text_input("ชื่อผู้ใช้")
            password = st.text_input("รหัสผ่าน", type="password")
            if st.form_submit_button("เข้าสู่ระบบ", use_container_width=True):
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
            if st.form_submit_button("สมัครสมาชิก", use_container_width=True):
                if new_password != confirm_password:
                    st.error("รหัสผ่านไม่ตรงกัน")
                elif len(new_password) < 6:
                    st.error("รหัสผ่านต้องมีอย่างน้อย 6 ตัวอักษร")
                else:
                    ok, msg = register_user(new_username, new_password)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
    st.stop()

st.title("🪑 Ergo-Vision AI: แจ้งเตือนท่านั่ง Real-time")

st.sidebar.markdown(f"👤 เข้าสู่ระบบในชื่อ: **{st.session_state.logged_in_user}**")
if st.sidebar.button("ออกจากระบบ"):
    st.session_state.logged_in_user = None
    st.rerun()

st.sidebar.header("⚙️ ตั้งค่าความไวการแจ้งเตือน")
theta_slider = st.sidebar.slider('ไหล่เอียงสูงสุด (θ)', 1, 15, 5)
slouch_slider = st.sidebar.slider('ความไวการตรวจจับก้ม (% ของท่าตรง)', 50, 95, 80)
alert_threshold_slider = st.sidebar.slider('แจ้งเตือนเมื่อนั่งผิดท่านานกว่า (วินาที)', 1, 60, 5)
sound_enabled = st.sidebar.checkbox("🔊 เปิดเสียงแจ้งเตือน", value=True)

tab_camera, tab_stats = st.tabs(["📹 เรียลไทม์", "📊 สถิติของฉัน"])

with tab_camera:
    calib_col1, calib_col2 = st.columns([1, 3])
    with calib_col1:
        if st.button("📐 ตั้งค่าท่านั่งตรง", use_container_width=True):
            st.session_state.calibration_id += 1
    with calib_col2:
        st.caption("นั่งหลังตรง มองตรง แล้วกดปุ่มนี้หนึ่งครั้งเพื่อ Calibrate ระดับก้ม")

    component_data = ergo_camera_component(
        settings={
            "thetaThreshold": theta_slider,
            "slouchThresholdPct": slouch_slider,
            "alertThresholdSec": alert_threshold_slider,
            "soundEnabled": sound_enabled
        },
        calibrationId=st.session_state.calibration_id,
        key="ergo_cam"
    )

    alert_placeholder = st.empty()
    metrics_placeholder = st.empty()

    if component_data:
        ts = component_data.get("timestamp")
        if ts != st.session_state.last_timestamp:
            st.session_state.last_timestamp = ts

            if component_data.get("episodeEnded"):
                log_posture_event(
                    st.session_state.logged_in_user,
                    component_data.get("durationSec", 0),
                    component_data.get("maxTheta"),
                    component_data.get("minNeckRatio"),
                    causes_to_text(component_data.get("epCauses", []), lang="th"),
                    component_data.get("alertTriggered", False)
                )
                st.toast("✅ บันทึกสถิติลง Turso Cloud Database สำเร็จ")

    if component_data:
        theta_val = component_data.get("theta")
        neck_val = component_data.get("neckRatioPct")
        is_calibrated = component_data.get("isCalibrated")
        is_bad = component_data.get("isBadPosture")
        elapsed = component_data.get("elapsedSec", 0)
        causes_now = component_data.get("currentCauses", [])
    else:
        theta_val, neck_val, is_calibrated, is_bad, elapsed, causes_now = None, None, False, False, 0, []

    with metrics_placeholder.container():
        mc1, mc2 = st.columns(2)
        mc1.metric("มุมเอียงไหล่ (θ)", f"{theta_val:.1f}°" if theta_val is not None else "—")
        if is_calibrated and neck_val is not None:
            mc2.metric("ระดับก้ม (% ของท่าตรง)", f"{neck_val:.0f}%")
        elif is_calibrated:
            mc2.metric("ระดับก้ม (% ของท่าตรง)", "รอข้อมูลใบหน้า...")
        else:
            mc2.metric("ระดับก้ม (% ของท่าตรง)", "ยังไม่ calibrate")

    if is_bad:
        cause_text = causes_to_text(causes_now, lang="th") or "ท่านั่งผิดปกติ"
        if elapsed >= alert_threshold_slider:
            alert_placeholder.error(f"🚨 {cause_text} มานาน {int(elapsed)} วินาทีแล้ว! กรุณาปรับท่านั่งให้ถูกต้อง")
        else:
            alert_placeholder.warning(f"⚠️ ท่านั่งเริ่มผิดปกติ: {cause_text} ({int(elapsed)}/{alert_threshold_slider} วินาที)")
    else:
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
        st.info("ยังไม่มีข้อมูลสถิติในช่วงเวลานี้ (สถิติจะถูกบันทึกเมื่อคุณนั่งผิดท่าอย่างน้อย 1 วินาที แล้วกลับมานั่งตรง)")
    else:
        df["start_time"] = pd.to_datetime(df["start_time"])
        total_bad_sec = df["duration_sec"].sum()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("เวลานั่งผิดท่ารวม", f"{total_bad_sec / 60:.1f} นาที")
        col2.metric("จำนวนครั้งที่นั่งผิดท่า", f"{len(df)} ครั้ง")
        col3.metric("จำนวนครั้งที่แจ้งเตือน", f"{int(df['alert_triggered'].sum())} ครั้ง")

        df["date"] = df["start_time"].dt.date
        daily = df.groupby("date")["duration_sec"].sum() / 60
        st.markdown("**เวลานั่งผิดท่ารายวัน (นาที)**")
        st.bar_chart(daily)

        st.markdown("**รายละเอียดล่าสุด**")
        display_df = df[["start_time", "duration_sec", "max_theta", "min_neck_ratio_pct", "cause", "alert_triggered"]].copy()
        
        display_df["duration_sec"] = pd.to_numeric(display_df["duration_sec"], errors="coerce").round(1)
        display_df["max_theta"] = pd.to_numeric(display_df["max_theta"], errors="coerce").round(1)
        
        display_df["min_neck_ratio_pct"] = pd.to_numeric(display_df["min_neck_ratio_pct"], errors="coerce").round(0)
        display_df["min_neck_ratio_pct"] = display_df["min_neck_ratio_pct"].apply(lambda v: f"{v:.0f}%" if pd.notna(v) else "—")
        
        display_df["cause"] = display_df["cause"].fillna("—")
        display_df["alert_triggered"] = display_df["alert_triggered"].map({1: "✅", 0: "—"})
        
        display_df.columns = ["เวลาเริ่ม", "ระยะเวลา (วินาที)", "ไหล่เอียงสูงสุด (°)", "ก้มมากสุด (% ของท่าตรง)", "สาเหตุ", "แจ้งเตือน"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
