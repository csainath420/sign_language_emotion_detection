import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load trained sign language model
model = load_model("sign_model.h5")

# Load gesture labels (from gestures folder names)
labels = sorted(os.listdir("gestures"))

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

print("[INFO] Press 'q' to quit the real-time detection")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Detect and draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract bounding box around hand
            h, w, _ = frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min = int(min(x_coords) * w) - 20
            y_min = int(min(y_coords) * h) - 20
            x_max = int(max(x_coords) * w) + 20
            y_max = int(max(y_coords) * h) + 20

            # Ensure ROI is within frame boundaries
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)

            roi = frame[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                continue

            try:
                # Preprocess ROI for prediction
                roi_resized = cv2.resize(roi, (64, 64))
                roi_normalized = roi_resized / 255.0
                roi_input = np.expand_dims(roi_normalized, axis=0)

                # Predict gesture
                prediction = model.predict(roi_input)
                class_id = np.argmax(prediction)
                confidence = prediction[0][class_id]

                # Display label
                label = labels[class_id]
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            except Exception as e:
                print(f"[WARNING] Error during prediction: {e}")
                continue

    # Show output
    cv2.imshow("Real-Time Sign Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
