import face_recognition
import cv2
import pickle
import os
import csv
import numpy as np
from datetime import datetime


# Load face encodings
with open("encodings.pickle", "rb") as f:
    known_encodings, known_names = pickle.load(f)

# Ensure attendance file exists
ATTENDANCE_FILE = "attendance.csv"
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time"])

def mark_attendance(name):
    """Marks attendance in CSV file."""
    with open(ATTENDANCE_FILE, "r+") as f:
        recorded_names = [line.split(",")[0] for line in f.readlines()]

        if name not in recorded_names:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{name},{now}\n")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for encoding, location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, encoding)
        face_distances = face_recognition.face_distance(known_encodings, encoding)
        match_index = np.argmin(face_distances) if face_distances.size > 0 else -1

        if match_index != -1 and matches[match_index]:
            name = known_names[match_index]
            mark_attendance(name)

            # Draw box & label
            y1, x2, y2, x1 = [v * 4 for v in location]  # Scale up
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

