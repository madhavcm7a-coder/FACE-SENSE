import cv2
from flask import Flask, Response, render_template, jsonify, request, session, redirect, url_for
from deepface import DeepFace

app = Flask(__name__)
app.secret_key = "secure_password_key_123" # Required for login sessions

# If your camera is still black/glitchy, change the 0 to a 1 or 2 here
camera = cv2.VideoCapture(0)

frame_counter = 0
current_label = "Scanning..."
min_confidence = 0

emotion_history = {
    "Angry": 0, "Disgust": 0, "Fear": 0, 
    "Happy": 0, "Sad": 0, "Surprise": 0, "Neutral": 0
}

def generate_frames():
    global frame_counter, current_label, min_confidence, emotion_history
    
    while True:
        success, frame = camera.read()
        if not success:
            continue
        
        frame = cv2.flip(frame, 1)

        # Process AI every 10 frames to keep the video feed smooth
        if frame_counter % 10 == 0:
            try:
                # 1. Full Resolution Processing (no resizing)
                # 2. enforce_detection=True (stops guessing when no face is present)
                result = DeepFace.analyze(
                    frame, actions=["emotion"], detector_backend="opencv", enforce_detection=True
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
                # 3. Instant Clear if you step out of the frame
                current_label = "No face detected"

        # Draw the label on the frame
        cv2.putText(
            frame, current_label, (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 128), 2, cv2.LINE_AA
        )
        frame_counter += 1
        
        # Lower JPG quality slightly (80%) for faster web streaming
        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue
            
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

# --- AUTHENTICATION ROUTES ---
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form.get("username") == "admin" and request.form.get("password") == "admin123":
            session["logged_in"] = True
            return redirect(url_for("index"))
        else:
            error = "Invalid credentials. Try: admin / admin123"
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("login"))

# --- MAIN APP ROUTES (PROTECTED) ---
@app.route("/")
def index():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    if not session.get("logged_in"):
        return "Unauthorized", 401
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# --- DATA ROUTES ---
@app.route("/emotion_data")
def emotion_data():
    return jsonify(emotion_history)

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

# --- CRITICAL STARTUP BLOCK ---
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)