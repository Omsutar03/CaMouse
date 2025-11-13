import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# --- Configuration ---
pyautogui.PAUSE = 0.0 # Make pyautogui faster
screen_w, screen_h = pyautogui.size()

# Tracking Zone configuration (adjust these values)
FRAME_REDUCTION_X = 160 # Pixels to crop from left/right of camera frame
FRAME_REDUCTION_Y = 90 # Pixels to crop from top/bottom of camera frame

# Click detection threshold (normalized distance between thumb and index)
CLICK_THRESHOLD = 0.05 
CLICK_COOLDOWN = 10 # Number of frames to wait after a click

# Smoothing factor (lower = more smoothing, less responsive)
SMOOTHING_FACTOR = 0.3

# --- Initialization ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

# Variables for state
prev_mouse_x, prev_mouse_y = 0, 0
is_clicking = False
click_cooldown_counter = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Reset frame state
    mouse_moved = False
    
    # Update cooldown counter
    if click_cooldown_counter > 0:
        click_cooldown_counter -= 1

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        
        # 1. Get landmark coordinates (index tip=8, thumb tip=4)
        lm_index = hand_landmarks.landmark[8]
        lm_thumb = hand_landmarks.landmark[4]
        
        # Convert index fingertip to pixel coordinates
        x, y = int(lm_index.x * w), int(lm_index.y * h)
        
        # 2. Map coordinates (with Restricted Zone)
        
        # Define the active region for tracking
        active_w = w - 2 * FRAME_REDUCTION_X
        active_h = h - 2 * FRAME_REDUCTION_Y

        # Calculate coordinates relative to the active region
        rel_x = x - FRAME_REDUCTION_X
        rel_y = y - FRAME_REDUCTION_Y

        # Clamp relative coordinates to be within [0, active_size]
        clamped_x = np.clip(rel_x, 0, active_w)
        clamped_y = np.clip(rel_y, 0, active_h)

        # Map clamped coordinates to screen resolution
        target_x = screen_w / active_w * clamped_x
        target_y = screen_h / active_h * clamped_y

        # 3. Smoothing (Exponential Moving Average)
        if prev_mouse_x == 0 and prev_mouse_y == 0:
            # Initialize on first detection
            prev_mouse_x, prev_mouse_y = target_x, target_y
        else:
            # Apply smoothing
            target_x = int(prev_mouse_x + (target_x - prev_mouse_x) * SMOOTHING_FACTOR)
            target_y = int(prev_mouse_y + (target_y - prev_mouse_y) * SMOOTHING_FACTOR)
            prev_mouse_x, prev_mouse_y = target_x, target_y

        # 4. Move Mouse
        pyautogui.moveTo(target_x, target_y)
        
        # 5. Click Detection
        # Calculate Euclidean distance between thumb and index tips (normalized to 0-1)
        dist = np.sqrt((lm_index.x - lm_thumb.x)**2 + (lm_index.y - lm_thumb.y)**2)
        
        if dist < CLICK_THRESHOLD and not is_clicking and click_cooldown_counter == 0:
            # Trigger Click
            pyautogui.click()
            is_clicking = True
            click_cooldown_counter = CLICK_COOLDOWN # Start cooldown
            cv2.circle(frame, (x, y), 15, (0, 0, 255), -1) # Red circle for click
        elif dist >= CLICK_THRESHOLD and is_clicking:
            # Reset click state when gesture is released
            is_clicking = False
            
        # Draw Visuals
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.circle(frame, (x, y), 10, (0, 255, 0) if not is_clicking else (0, 0, 255), -1)
        
    # Draw the defined tracking zone
    cv2.rectangle(
        frame,
        (FRAME_REDUCTION_X, FRAME_REDUCTION_Y),
        (w - FRAME_REDUCTION_X, h - FRAME_REDUCTION_Y),
        (255, 0, 0), 2
    )

    cv2.imshow("CaMouse - Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()