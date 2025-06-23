import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from deepface import DeepFace
import joblib

# Load model and labels
model = load_model("sign_model.h5")
label_encoder = joblib.load("labels.pkl")
labels = label_encoder.classes_

# Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# OpenCV camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    output_frame = frame.copy()

    # Emotion detection
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        cv2.putText(output_frame, f"Emotion: {emotion}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    except:
        pass

    # Hand gesture detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(output_frame, handLms, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            x_list = [lm.x for lm in handLms.landmark]
            y_list = [lm.y for lm in handLms.landmark]
            x_min, x_max = int(min(x_list) * w), int(max(x_list) * w)
            y_min, y_max = int(min(y_list) * h), int(max(y_list) * h)

            roi = frame[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                continue
            roi = cv2.resize(roi, (64, 64)) / 255.0
            roi = roi.reshape(1, 64, 64, 3)

            pred = model.predict(roi)
            idx = np.argmax(pred)

            if idx < len(labels):
                gesture = labels[idx]
                cv2.rectangle(output_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(output_frame, f"Gesture: {gesture}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Real-Time Gesture + Emotion Detection", output_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
