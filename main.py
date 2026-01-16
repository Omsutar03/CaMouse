import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# --- Configuration ---
pyautogui.PAUSE = 0.0 # Make pyautogui faster
screen_w, screen_h = pyautogui.size()

# Tracking Zone configuration (adjust these values)
# Use percentages (0.2 means 20% of the camera frame is a dead-border)
MARGIN_X = 0.2 
MARGIN_Y = 0.2

# Smoothing factor (lower = more smoothing, less responsive)
SMOOTHING_FACTOR = 0.2

# Initializing prev_smoothing values
prev_smooth_x, prev_smooth_y = 0.5, 0.5

# Click detection threshold (normalized distance between thumb and index)
CLICK_THRESHOLD = 0.05
CLICK_COOLDOWN = 10 # Number of frames to wait after a click

# --- Initialization ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands = 1, min_detection_confidence = 0.7, min_tracking_confidence = 0.7)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)


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
    #print("frame size = ", h, w)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Reset frame state
    mouse_moved = False
    
    # Update cooldown counter
    if click_cooldown_counter > 0:
        click_cooldown_counter -= 1

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        
        # Get landmark coordinates (index tip=8, thumb tip=4)
        f_index_tip = hand_landmarks.landmark[8]
        f_index_pip = hand_landmarks.landmark[6]
        f_index_mcp = hand_landmarks.landmark[5]
        f_thumb_tip = hand_landmarks.landmark[4]
        
        # STEP 1: Get Raw High-Precision Coordinates (0.0 - 1.0)
        raw_x, raw_y = f_index_tip.x, f_index_tip.y
        
        # STEP 2: Exponential Moving Average (EMA) Smoothing
        # We smooth the raw sensor data BEFORE scaling it to the screen pixels.
        # This kills the "jitter" at the source.
        smooth_x = prev_smooth_x + (raw_x - prev_smooth_x) * SMOOTHING_FACTOR
        smooth_y = prev_smooth_y + (raw_y - prev_smooth_y) * SMOOTHING_FACTOR
        
        # Save current smooth values for the next frame's calculation
        prev_smooth_x, prev_smooth_y = smooth_x, smooth_y

        # STEP 3: Map to Screen using Linear Interpolation (np.interp)
        # This maps the 'active area' (e.g., 0.2 to 0.8) to the full screen (0 to screen_w)
        # It also handles 'clamping' automatically so the mouse doesn't go off-screen.
        target_x = np.interp(smooth_x, [MARGIN_X, 1 - MARGIN_X], [0, screen_w])
        target_y = np.interp(smooth_y, [MARGIN_Y, 1 - MARGIN_Y], [0, screen_h])

        # 4. Move Mouse
        pyautogui.moveTo(target_x, target_y)
        
        
        # Calculate Euclidean distance between thumb and index tips (normalized to 0-1)
        # dist = np.sqrt((f_index_tip.x - f_thumb_tip.x)**2 + (f_index_tip.y - f_thumb_tip.y)**2)
        # 5. Click Detection
        # 2. Calculate distances (Normalized)
        # Distance to the knuckle
        dist_mcp = np.sqrt((f_thumb_tip.x - f_index_mcp.x)**2 + (f_thumb_tip.y - f_index_mcp.y)**2)
        # Distance to the middle joint
        dist_pip = np.sqrt((f_thumb_tip.x - f_index_pip.x)**2 + (f_thumb_tip.y - f_index_pip.y)**2)

        # 3. Logic: If thumb is close to either joint, it's a click
        # We use 'min' to see which one is closer
        current_dist = min(dist_mcp, dist_pip)
        print(current_dist, CLICK_THRESHOLD)

        if current_dist < CLICK_THRESHOLD and not is_clicking and click_cooldown_counter == 0:
            pyautogui.click()
            is_clicking = True
            click_cooldown_counter = CLICK_COOLDOWN
            # Visual feedback: Change color of the circle or draw a line
            # cv2.line(frame, (int(f_thumb_tip.x * w), int(f_thumb_tip.y * h)), 
            #         (int(f_index_pip.x * w), int(f_index_pip.y * h)), (255, 0, 0), 5)
            cv2.circle(frame, (int(f_thumb_tip.x), int(f_thumb_tip.y)), 15, (0, 0, 255), -1)
            
        elif current_dist > (CLICK_THRESHOLD + 0.02): # Added a small "buffer" to prevent flickering
            is_clicking = False
        
        # if dist < CLICK_THRESHOLD and not is_clicking and click_cooldown_counter == 0:
        #     # Trigger Click
        #     pyautogui.click()
        #     is_clicking = True
        #     click_cooldown_counter = CLICK_COOLDOWN # Start cooldown
        #     cv2.circle(frame, (x, y), 15, (0, 0, 255), -1) # Red circle for click
        # elif dist >= CLICK_THRESHOLD and is_clicking:
        #     # Reset click state when gesture is released
        #     is_clicking = False
            
        # # Draw Visuals
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        #cv2.circle(frame, (x, y), 10, (0, 255, 0) if not is_clicking else (0, 0, 255), -1)

    cv2.imshow("CaMouse - Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()