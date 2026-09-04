import cv2
from flask import Flask, Response, render_template, jsonify, request
from deepface import DeepFace

app = Flask(__name__)
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

frame_counter = 0
current_label = "Scanning..."
min_confidence = 0

# Store total emotion counts for the dashboard
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

        if frame_counter % 5 == 0:
            try:
                result = DeepFace.analyze(
                    frame, actions=["emotion"], detector_backend="opencv", enforce_detection=False
                )
                if result:
                    dom_emo = result[0]["dominant_emotion"].capitalize()
                    conf = int(result[0]["emotion"][dom_emo.lower()])
                    
                    # Only accept emotions that beat the slider's threshold
                    if conf >= min_confidence:
                        current_label = f"{dom_emo} ({conf}%)"
                        if dom_emo in emotion_history:
                            emotion_history[dom_emo] += 1
                    else:
                        current_label = f"Scanning... (Need {min_confidence}%)"
            except Exception:
                pass

        cv2.putText(
            frame, current_label, (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 128), 2, cv2.LINE_AA
        )

        frame_counter += 1
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# New endpoint: Sends live stats to the chart
@app.route("/emotion_data")
def emotion_data():
    return jsonify(emotion_history)

# New endpoint: Receives slider value from frontend
@app.route("/set_threshold", methods=["POST"])
def set_threshold():
    global min_confidence
    min_confidence = int(request.json.get("threshold", 0))
    return jsonify({"status": "success"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)