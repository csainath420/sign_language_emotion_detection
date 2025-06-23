import tkinter as tk
from tkinter import messagebox
import cv2
import threading
import datetime
import os
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from tensorflow.keras.models import load_model
import joblib

# Load model and labels
model = load_model("sign_model.h5")
label_encoder = joblib.load("labels.pkl")
labels = label_encoder.classes_

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Logger setup
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/log_{datetime.datetime.now().strftime('%Y%m%d')}.txt"
log_file = open(log_filename, "a")  # 'a' means append mode


running = False  # For controlling the webcam thread

def detect():
    global running
    cap = cv2.VideoCapture(0)

    while running:
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
            log_file.write(f"[{datetime.datetime.now()}] Emotion: {emotion}\n")
        except:
            emotion = None

        # Hand detection
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
                    log_file.write(f"[{datetime.datetime.now()}] Gesture: {gesture}\n")

        cv2.imshow("Gesture + Emotion Detection", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    log_file.close()

def start_detection():
    global running
    running = True
    threading.Thread(target=detect).start()

def stop_detection():
    global running
    running = False
    messagebox.showinfo("Stopped", "Detection stopped and log saved.")

# GUI setup
root = tk.Tk()
root.title("Sign Language + Emotion Detector")
root.geometry("400x250")

tk.Label(root, text="Real-Time Detection System", font=("Arial", 16)).pack(pady=20)

tk.Button(root, text="Start Detection", font=("Arial", 12), bg="green", fg="white", command=start_detection).pack(pady=10)
tk.Button(root, text="Stop Detection", font=("Arial", 12), bg="red", fg="white", command=stop_detection).pack(pady=10)

tk.Label(root, text="Press 'q' to quit video feed", font=("Arial", 10)).pack(pady=10)

root.mainloop()
