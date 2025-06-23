import cv2
import os
import mediapipe as mp
import time

gesture_name = input("Enter gesture name: ")
total_images = 100
interval_seconds = 1

save_dir = f'gestures/{gesture_name}'
os.makedirs(save_dir, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
img_count = 0
last_capture_time = time.time()

print(f"\n[INFO] Collecting {total_images} images for '{gesture_name}' every {interval_seconds} second...\nPress 'q' to cancel early.")

while img_count < total_images:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Collecting '{gesture_name}': {img_count}/{total_images}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Auto Capture", frame)

    if time.time() - last_capture_time >= interval_seconds:
        img_path = f"{save_dir}/{gesture_name}_{img_count}.jpg"
        cv2.imwrite(img_path, frame)
        print(f"[Saved] {img_path}")
        img_count += 1
        last_capture_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Capture cancelled.")
        break

print("\n✅ Done! Image collection complete.")
cap.release()
cv2.destroyAllWindows()
