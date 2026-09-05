import io
import os
from pathlib import Path
import tempfile
import time
import datetime
import sqlite3
from flask import Flask, Response, render_template, jsonify, request, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from deepface import DeepFace
except ImportError:
    DeepFace = None

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-this-secret")

DB_FILE = str(
    Path(tempfile.gettempdir()) / "facesense.db"
    if os.environ.get("VERCEL")
    else Path("facesense.db")
)

# --- DATABASE SETUP ---
def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_start TEXT,
                session_end TEXT,
                focused_s INTEGER,
                distracted_s INTEGER,
                drowsy_s INTEGER,
                slouch_events INTEGER,
                score INTEGER,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = ?", ("admin",))
        if not cur.fetchone():
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                ("admin", generate_password_hash("admin123"))
            )
        conn.commit()

init_db()

# --- HARDWARE CAMERA STATE ---
current_camera_index = 0
camera = None if os.environ.get("VERCEL") or cv2 is None else cv2.VideoCapture(current_camera_index)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if cv2 else None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") if cv2 else None
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml") if cv2 else None

current_mode = "emotion"
frame_counter = 0
current_label = "Scanning..."
min_confidence = 0

emotion_history = {
    "Angry": 0, "Disgust": 0, "Fear": 0,
    "Happy": 0, "Sad": 0, "Surprise": 0, "Neutral": 0
}

study_state = {
    "status": "Scanning",
    "posture": "Good",
    "focused_seconds": 0,
    "distracted_seconds": 0,
    "drowsy_seconds": 0,
    "slouch_events": 0,
    "alert": False,
    "alert_reason": None,
    "session_start": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

rolling_timeline = [{"time": f"-{14-i}m", "score": 100, "slouch": 0} for i in range(15)]
timeline_last_shift = time.time()
minute_focused = 0
minute_total = 0
minute_slouch = 0

closed_eyes_frames = 0
last_study_tick = time.time()
calibrated_y = None
calibrated_h = None


def generate_frames():
    global frame_counter, current_label, min_confidence, emotion_history
    global current_mode, closed_eyes_frames, study_state, last_study_tick
    global calibrated_y, calibrated_h, camera, clahe
    global rolling_timeline, timeline_last_shift, minute_focused, minute_total, minute_slouch

    while True:
        if camera is None or not camera.isOpened():
            time.sleep(0.1)
            continue

        success, frame = camera.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
        enhanced_frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
        frame_height, frame_width = frame.shape[:2]

        now = time.time()

        if now - timeline_last_shift >= 60.0:
            calc_score = int((minute_focused / minute_total) * 100) if minute_total > 0 else 100
            rolling_timeline.pop(0)
            rolling_timeline.append({
                "time": datetime.datetime.now().strftime("%H:%M"),
                "score": calc_score,
                "slouch": minute_slouch
            })
            minute_focused = 0
            minute_total = 0
            minute_slouch = 0
            timeline_last_shift = now

        if current_mode == "emotion":
            if frame_counter % 10 == 0:
                try:
                    if DeepFace is None:
                        raise RuntimeError("DeepFace is not installed")
                    result = DeepFace.analyze(
                        enhanced_frame, actions=["emotion"], detector_backend="opencv", enforce_detection=True
                    )
                    if result:
                        dom_emo = result[0]["dominant_emotion"].capitalize()
                        conf = int(result[0]["emotion"][dom_emo.lower()])
                        if conf >= min_confidence:
                            current_label = f"{dom_emo} ({conf}%)"
                            if dom_emo in emotion_history:
                                emotion_history[dom_emo] += 1
                        else:
                            current_label = f"Scanning... (Need {min_confidence}%)"
                except Exception:
                    current_label = "No face detected"

            cv2.putText(
                frame, current_label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 128), 2, cv2.LINE_AA
            )

        elif current_mode == "study":
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
            elapsed = now - last_study_tick

            if len(faces) == 0:
                if calibrated_y is not None and study_state["posture"] in ["Slouching", "Head Bowed"]:
                    status = "Posture: Head Bowed / Slouching"
                    color = (0, 140, 255)
                    study_state["status"] = "Posture Alert"
                    study_state["posture"] = "Head Bowed"
                    study_state["alert"] = True
                    study_state["alert_reason"] = "Posture"
                    if elapsed >= 1.0:
                        study_state["slouch_events"] += 1
                        minute_slouch += 1
                        minute_total += int(elapsed)
                        last_study_tick = now
                else:
                    status = "Distracted (Looking Away)"
                    color = (0, 165, 255)
                    study_state["status"] = "Distracted"
                    study_state["posture"] = "Unknown"
                    study_state["alert"] = False
                    study_state["alert_reason"] = None
                    if elapsed >= 1.0:
                        study_state["distracted_seconds"] += int(elapsed)
                        minute_total += int(elapsed)
                        last_study_tick = now
            else:
                (x, y, w, h) = faces[0]
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                if calibrated_y is None or calibrated_h is None:
                    calibrated_y = y
                    calibrated_h = h

                posture_issue = False
                if y > (calibrated_y + int(frame_height * 0.07)):
                    study_state["posture"] = "Slouching"
                    posture_issue = True
                elif h > (calibrated_h * 1.30):
                    study_state["posture"] = "Too Close"
                    posture_issue = True
                else:
                    study_state["posture"] = "Good"

                cv2.rectangle(frame, (x, y), (x + w, y + h), (56, 189, 248), 2)

                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(18, 18))
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
                    if ey > (h * 0.48):
                        study_state["posture"] = "Head Bowed"
                        posture_issue = True

                if len(eyes) == 0:
                    closed_eyes_frames += 1
                else:
                    closed_eyes_frames = 0

                if closed_eyes_frames > 15:
                    status = "DROWSY ALERT!"
                    color = (0, 0, 255)
                    study_state["status"] = "Drowsy"
                    study_state["alert"] = True
                    study_state["alert_reason"] = "Drowsy"
                    if elapsed >= 1.0:
                        study_state["drowsy_seconds"] += int(elapsed)
                        minute_total += int(elapsed)
                        last_study_tick = now
                elif posture_issue:
                    status = f"Posture: {study_state['posture']}"
                    color = (0, 140, 255)
                    study_state["status"] = "Posture Alert"
                    study_state["alert"] = True
                    study_state["alert_reason"] = "Posture"
                    if elapsed >= 1.0:
                        study_state["slouch_events"] += 1
                        minute_slouch += 1
                        minute_total += int(elapsed)
                        last_study_tick = now
                else:
                    status = "Focused"
                    color = (0, 255, 0)
                    study_state["status"] = "Focused"
                    study_state["alert"] = False
                    study_state["alert_reason"] = None
                    if elapsed >= 1.0:
                        study_state["focused_seconds"] += int(elapsed)
                        minute_focused += int(elapsed)
                        minute_total += int(elapsed)
                        last_study_tick = now

            cv2.putText(
                frame, f"Study: {status}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA
            )

        frame_counter += 1
        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


# --- AUTHENTICATION ROUTES ---
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        with get_db() as conn:
            user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            if user and check_password_hash(user["password_hash"], password):
                session["logged_in"] = True
                session["user_id"] = user["id"]
                session["username"] = user["username"]
                return redirect(url_for("index"))
            else:
                error = "Invalid credentials. Try default: admin / admin123"
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def index():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("index.html", username=session.get("username"))


@app.route("/video_feed")
def video_feed():
    if not session.get("logged_in"):
        return "Unauthorized", 401
    if camera is None or cv2 is None:
        return jsonify({"status": "unavailable", "message": "Camera streaming is available only when running locally."}), 503
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/switch_camera", methods=["POST"])
def switch_camera():
    global camera, current_camera_index
    if cv2 is None or os.environ.get("VERCEL"):
        return jsonify({"status": "unavailable", "message": "Camera switching is available only when running locally."}), 503
    new_index = int(request.json.get("index", 0))
    if new_index != current_camera_index:
        if camera is not None:
            camera.release()
        current_camera_index = new_index
        camera = cv2.VideoCapture(current_camera_index)
    return jsonify({"status": "success", "camera_index": current_camera_index})


@app.route("/set_mode", methods=["POST"])
def set_mode():
    global current_mode, last_study_tick
    current_mode = request.json.get("mode", "emotion")
    last_study_tick = time.time()
    return jsonify({"status": "success", "mode": current_mode})


@app.route("/emotion_data")
def emotion_data():
    return jsonify(emotion_history)


@app.route("/study_data")
def study_data():
    total = study_state["focused_seconds"] + study_state["distracted_seconds"] + study_state["drowsy_seconds"]
    score = int((study_state["focused_seconds"] / total) * 100) if total > 0 else 100
    return jsonify({
        "status": study_state["status"],
        "posture": study_state["posture"],
        "focused": study_state["focused_seconds"],
        "distracted": study_state["distracted_seconds"],
        "drowsy": study_state["drowsy_seconds"],
        "slouch_events": study_state["slouch_events"],
        "score": score,
        "alert": study_state["alert"],
        "alert_reason": study_state["alert_reason"],
        "timeline": rolling_timeline
    })


@app.route("/save_study_session", methods=["POST"])
def save_study_session():
    if not session.get("user_id"):
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    total = study_state["focused_seconds"] + study_state["distracted_seconds"] + study_state["drowsy_seconds"]
    score = int((study_state["focused_seconds"] / total) * 100) if total > 0 else 100
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_db() as conn:
        conn.execute("""
            INSERT INTO sessions (user_id, session_start, session_end, focused_s, distracted_s, drowsy_s, slouch_events, score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session["user_id"],
            study_state["session_start"],
            end_time,
            study_state["focused_seconds"],
            study_state["distracted_seconds"],
            study_state["drowsy_seconds"],
            study_state["slouch_events"],
            score
        ))
        conn.commit()
    return jsonify({"status": "success", "message": "Session stored permanently in SQLite"})


@app.route("/calibrate_posture", methods=["POST"])
def calibrate_posture():
    global calibrated_y, calibrated_h
    calibrated_y = None
    calibrated_h = None
    return jsonify({"status": "success"})


@app.route("/reset_study", methods=["POST"])
def reset_study():
    global last_study_tick
    study_state["focused_seconds"] = 0
    study_state["distracted_seconds"] = 0
    study_state["drowsy_seconds"] = 0
    study_state["slouch_events"] = 0
    study_state["alert"] = False
    study_state["alert_reason"] = None
    study_state["session_start"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    last_study_tick = time.time()
    return jsonify({"status": "success"})


# --- EXPORT REPORT AS PDF ---
@app.route("/export_session_pdf")
def export_session_pdf():
    total = study_state["focused_seconds"] + study_state["distracted_seconds"] + study_state["drowsy_seconds"]
    score = int((study_state["focused_seconds"] / total) * 100) if total > 0 else 100
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format seconds into mm:ss or hh:mm:ss
    def format_time(s):
        m, sec = divmod(s, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}h {m:02d}m {sec:02d}s" if h > 0 else f"{m:02d}m {sec:02d}s"

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=18
    )

    story = []
    story.append(Paragraph("Face Sense Pro — Study Session Report", title_style))
    story.append(Paragraph(f"Generated for user: <b>{session.get('username', 'Student')}</b> | Exported: {current_time}", subtitle_style))
    story.append(Spacer(1, 10))

    # Metric Table Content
    data = [
        ["Metric", "Details / Value"],
        ["Session Start", study_state["session_start"]],
        ["Report Timestamp", current_time],
        ["Total Session Time", format_time(total)],
        ["Focused Time", format_time(study_state["focused_seconds"])],
        ["Distracted Time", format_time(study_state["distracted_seconds"])],
        ["Drowsy / Closed Eyes Time", format_time(study_state["drowsy_seconds"])],
        ["Slouch / Head Bow Incidents", f"{study_state['slouch_events']} events"],
        ["Overall Productivity Score", f"{score}%"]
    ]

    t = Table(data, colWidths=[240, 280])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0284c7')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f1f5f9')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        # Highlight overall score row
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (1, -1), (1, -1), colors.HexColor('#059669') if score >= 70 else colors.HexColor('#dc2626'))
    ]))

    story.append(t)
    doc.build(story)

    buffer.seek(0)
    return Response(
        buffer.getvalue(),
        mimetype="application/pdf",
        headers={"Content-Disposition": "attachment; filename=study_session_report.pdf"}
    )


@app.route("/set_threshold", methods=["POST"])
def set_threshold():
    global min_confidence
    min_confidence = int(request.json.get("threshold", 0))
    return jsonify({"status": "success"})


@app.route("/reset_data", methods=["POST"])
def reset_data():
    global emotion_history
    for key in emotion_history:
        emotion_history[key] = 0
    return jsonify({"status": "success"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)