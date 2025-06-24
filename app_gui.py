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
import speech_recognition as sr

model = load_model("sign_model.h5")
label_encoder = joblib.load("labels.pkl")
labels = label_encoder.classes_

# Mediapipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

running = False
log_file = None


def detect():
    global running, log_file
    cap = cv2.VideoCapture(0)

    last_emotion_time = datetime.datetime.now() - datetime.timedelta(seconds=5)
    last_gesture_time = datetime.datetime.now() - datetime.timedelta(seconds=5)

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        output_frame = frame.copy()
        now = datetime.datetime.now()

        # Emotion detection every 5 seconds
        if (now - last_emotion_time).total_seconds() >= 5:
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                last_emotion_time = now

                cv2.putText(output_frame, f"Emotion: {emotion}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                if log_file:
                    log_file.write(f"[{now}] Emotion: {emotion}\n")
            except Exception as e:
                print(f"[Emotion Detection Error] {e}")

        # Hand gesture detection every 5 seconds
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

                padding = 30
                x_min = max(0, x_min - padding)
                x_max = min(w, x_max + padding)
                y_min = max(0, y_min - padding)
                y_max = min(h, y_max + padding)

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

                    # Log gesture only every 5 seconds
                    if (now - last_gesture_time).total_seconds() >= 5:
                        last_gesture_time = now
                        if log_file:
                            log_file.write(f"[{now}] Gesture: {gesture}\n")

        cv2.imshow("Real-Time Detection", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def listen_to_voice():
    global running, log_file
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)

    while running:
        try:
            with mic as source:
                print("[Voice] Listening...")
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio)
                print(f"[Voice] Detected: {text}")
                if log_file:
                    log_file.write(f"[{datetime.datetime.now()}] Voice: {text}\n")
        except sr.WaitTimeoutError:
            continue
        except sr.UnknownValueError:
            print("[Voice] Could not understand audio.")
        except Exception as e:
            print(f"[Voice] Error: {e}")


def start_detection():
    global running, log_file
    running = True
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_file = open(log_filename, "w")
    threading.Thread(target=detect).start()
    threading.Thread(target=listen_to_voice).start()


def stop_detection():
    global running, log_file
    running = False
    if log_file:
        log_file.close()
        log_file = None
    messagebox.showinfo("Stopped", "Detection stopped and logs saved.")


root = tk.Tk()
root.title("Sign Language + Emotion + Voice Recognition")
root.geometry("450x300")

tk.Label(root, text="AI Multimodal Detection", font=("Arial", 18)).pack(pady=20)

tk.Button(root, text="Start Detection", font=("Arial", 14), bg="green", fg="white", command=start_detection).pack(pady=10)
tk.Button(root, text="Stop Detection", font=("Arial", 14), bg="red", fg="white", command=stop_detection).pack(pady=10)

tk.Label(root, text="Press 'q' to quit video feed", font=("Arial", 10)).pack(pady=10)

root.mainloop()
