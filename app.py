from flask import Flask, render_template, Response
import cv2
import face_recognition
import pickle
import csv
import os
from datetime import datetime

app = Flask(__name__)
cap = None  # Webcam capture object

# Load known face encodings
with open("encodings.pickle", "rb") as f:
    known_encodings, known_names = pickle.load(f)

# Ensure attendance file exists
ATTENDANCE_FILE = "attendance.csv"
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time"])
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time"])

def mark_attendance(name):
    """Marks attendance in CSV."""
    with open(ATTENDANCE_FILE, "r+") as f:
        data = f.readlines()
        recorded_names = [line.split(",")[0] for line in data]

        if name not in recorded_names:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{name},{now}\n")

def generate_frames():
    """Captures webcam frames and performs face recognition."""
    global cap
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for encoding, location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, encoding)
            name = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]
                mark_attendance(name)

            y1, x2, y2, x1 = location
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/")
def index():
    """Loads the home page."""
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    """Starts video streaming."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stop")
def stop_camera():
    """Stops the webcam."""
    global cap
    if cap is not None:
        cap.release()
    return "Camera stopped"

if __name__ == "__main__":
    app.run(debug=True)