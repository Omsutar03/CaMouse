import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Screen size
screen_w, screen_h = pyautogui.size()

# Open camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip horizontally for a mirror-like view
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Index fingertip landmark (id 8)
            lm = hand_landmarks.landmark[8]
            x, y = int(lm.x * w), int(lm.y * h)

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Map camera coords → screen coords
            screen_x = screen_w / w * x
            screen_y = screen_h / h * y

            # Move mouse
            pyautogui.moveTo(screen_x, screen_y, duration=0)

            # Optional: small circle on fingertip
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

    cv2.imshow("CaMouse - Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
