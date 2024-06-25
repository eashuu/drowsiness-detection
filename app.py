import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from playsound import playsound

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Constants
EAR_THRESHOLD = 0.3
EAR_CONSEC_FRAMES = 48
counter = 0
alarm_on = False

# Define function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to play alert sound
def play_alert_sound():
    playsound('alert.wav')


# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract the landmarks
            h, w, _ = frame.shape
            landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]

            # Indices for the left and right eye landmarks
            leftEye_indices = [33, 160, 158, 133, 153, 144]
            rightEye_indices = [362, 385, 387, 263, 373, 380]

            leftEye = [landmarks[i] for i in leftEye_indices]
            rightEye = [landmarks[i] for i in rightEye_indices]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < EAR_THRESHOLD:
                counter += 1
                if counter >= EAR_CONSEC_FRAMES:
                    if not alarm_on:
                        alarm_on = True
                        print("Drowsiness Alert!")
                        play_alert_sound()
            else:
                counter = 0
                alarm_on = False

            # Draw landmarks
            for (x, y) in leftEye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            for (x, y) in rightEye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow('Drowsiness Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
