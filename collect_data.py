import cv2
import os
import mediapipe as mp

gesture_name = input("Enter gesture name: ")
save_dir = f'gestures/{gesture_name}'
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

img_count = 0
print("Press 'c' to capture image, 'q' to quit")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Capture - " + gesture_name, frame)
    key = cv2.waitKey(1)

    if key == ord('c'):
        cv2.imwrite(f"{save_dir}/{gesture_name}_{img_count}.jpg", frame)
        print(f"Saved {gesture_name}_{img_count}.jpg")
        img_count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
