"""
main.py — Real-Time Gesture Control for Spotify
-----------------------------------------------

This script uses MediaPipe and a trained KNN model to recognize hand gestures
in real-time. It detects 21 hand landmarks (63 x,y,z features) per frame and
classifies gestures such as:
 - 👍 Thumbs Up → Play
 - 🖐️ Open Palm → Pause
 - 🤘 Rock On → Like
 - ☝️ Index Swipe Up / Down → Volume Up / Down
 - 👉 / 👈 Swipe Right / Left → Next / Previous Track

The model was trained on 2115 manually collected samples, achieving 94.9% accuracy.
Each recognized gesture triggers a Spotify action via PyAutoGUI keyboard automation.
"""

import cv2
import mediapipe as mp
import pickle
import pyautogui
import time
import psutil
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# SPOTIFY DETECTION
# ============================================================================
def is_spotify_running():
    """
    Check if Spotify process is running on the system.
    Returns True if Spotify is found, False otherwise.
    """
    for process in psutil.process_iter(attrs=['name']):
        if 'spotify' in process.info['name'].lower():
            return True
    return False

# ============================================================================
# MODEL LOADING
# ============================================================================
# Load the trained gesture recognition model
# This model was trained using train_model.py and saved as a pickle file
with open('gesture_model.pkl', 'rb') as f:
    model = pickle.load(f)

# ============================================================================
# MEDIAPIPE INITIALIZATION
# ============================================================================
# Initialize MediaPipe Hands for real-time hand landmark detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configure MediaPipe for optimal performance
# static_image_mode=False: Optimized for video streams (faster)
# max_num_hands=1: Only detect one hand (simpler, faster)
# Lower confidence thresholds for faster processing
hands = mp_hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    static_image_mode=False
)

# ============================================================================
# WEBCAM SETUP
# ============================================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit(1)

# Optimize webcam settings for smooth performance
# Lower resolution (640x480) reduces processing load and prevents lag
cap.set(3, 640)  # width
cap.set(4, 480)  # height
cap.set(cv2.CAP_PROP_FPS, 30)  # Target 30 frames per second
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce latency

print("\n🚀 Running in Optimized Performance Mode (~25–30 FPS)")
print(" Performance Mode ON: Webcam running at 640x480 for smooth FPS\n")

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================
FRAME_DELAY = 0.03  # ~30 FPS delay for smoother frame processing
CONFIRM_FRAMES = 5  # Number of consecutive frames needed to confirm a gesture
COOLDOWN = 1.2  # Seconds between actions (prevents rapid repeated triggers)

# ============================================================================
# GESTURE STATE VARIABLES
# ============================================================================
current_state = "none"  # Track if music is playing or paused
last_action_time = 0  # Timestamp of last action (for cooldown)
gesture_buffer = []  # Buffer to store recent gesture predictions (for confirmation)

# ============================================================================
# USER INTERFACE
# ============================================================================
print("\n=== Gesture Control Active ===")
print("Please make sure Spotify is open before running this.")
print("\n📸 Tip: Restart your terminal before every gesture recording for smooth webcam")
print("🧹 This clears old camera buffers and prevents lag.\n")
print("Available gestures:")
print("  👍 Thumbs Up → Play")
print("  🖐️ Open Palm → Pause")
print("  👆 Index Swipe Up → Volume Up")
print("  👇 Index Swipe Down → Volume Down")
print("  ➡️ Swipe Right → Next Song")
print("  ⬅️ Swipe Left → Previous Song")
print("  🤘 Rock On → Like Song")
print("Press 'q' to quit\n")

# ============================================================================
# SPOTIFY CHECK
# ============================================================================
if not is_spotify_running():
    print("⚠️ Spotify not detected. Please open Spotify and then run this program.")
else:
    print("✅ Spotify detected! Ready to control music.\n")

# ============================================================================
# MAIN GESTURE DETECTION LOOP
# ============================================================================
while True:
    # Capture one frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for mirror effect (feels more natural)
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB (MediaPipe requires RGB format)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with MediaPipe to detect hand landmarks
    results = hands.process(rgb)

    # ========================================================================
    # GESTURE DETECTION AND CLASSIFICATION
    # ========================================================================
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on frame (visual feedback)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 21 landmarks and convert to 63 features (x, y, z for each)
            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])

            # Predict gesture using trained model
            gesture = model.predict([features])[0]
            
            # Display predicted gesture on video feed
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # ================================================================
            # GESTURE CONFIRMATION BUFFER
            # ================================================================
            # Add current prediction to buffer
            gesture_buffer.append(gesture)
            
            # Keep buffer size at CONFIRM_FRAMES (remove oldest if full)
            if len(gesture_buffer) > CONFIRM_FRAMES:
                gesture_buffer.pop(0)
            
            # Only confirm gesture if all frames in buffer show the same gesture
            # This prevents accidental triggers from brief gesture flickers
            if len(set(gesture_buffer)) == 1:
                confirmed_gesture = gesture_buffer[0]  # Gesture is consistent
            else:
                confirmed_gesture = None  # Gesture is inconsistent, don't act

            current_time = time.time()

            # ================================================================
            # GESTURE ACTION LOGIC
            # ================================================================
            # Only act on confirmed gestures and respect cooldown period
            if confirmed_gesture and (current_time - last_action_time > COOLDOWN):
                
                # Play gesture
                if confirmed_gesture == 'thumbs_up' and current_state != 'playing':
                    pyautogui.press('space')
                    print("▶️ Gesture confirmed: Play")
                    current_state = 'playing'
                    last_action_time = current_time

                # Pause gesture
                elif confirmed_gesture == 'open_palm' and current_state != 'paused':
                    pyautogui.press('space')
                    print("⏸️ Gesture confirmed: Pause")
                    current_state = 'paused'
                    last_action_time = current_time

                # Like song gesture
                elif confirmed_gesture == 'rock_on':
                    pyautogui.hotkey('Alt', 'Shift', 'B')
                    print("❤️ Gesture confirmed: Like Song")
                    last_action_time = current_time

                # Volume up gesture
                elif confirmed_gesture == 'index_swipe_up':
                    pyautogui.hotkey('ctrl', 'up')
                    print("🔊 Gesture confirmed: Volume Up")
                    last_action_time = current_time

                # Volume down gesture
                elif confirmed_gesture == 'index_swipe_down':
                    pyautogui.hotkey('ctrl', 'down')
                    print("🔉 Gesture confirmed: Volume Down")
                    last_action_time = current_time

                # Next song gesture
                elif confirmed_gesture == 'swipe_right':
                    pyautogui.hotkey('ctrl', 'right')
                    print("⏭️ Gesture confirmed: Next Song")
                    last_action_time = current_time

                # Previous song gesture
                elif confirmed_gesture == 'swipe_left':
                    pyautogui.hotkey('ctrl', 'left')
                    print("⏮️ Gesture confirmed: Previous Song")
                    last_action_time = current_time

    # ========================================================================
    # DISPLAY AND FRAME CONTROL
    # ========================================================================
    # Always display the frame (even when no hand is detected)
    cv2.imshow("Gesture Control - Spotify", frame)
    
    # Check for quit key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Frame delay for smoother FPS (prevents CPU overload)
    time.sleep(FRAME_DELAY)

# ============================================================================
# CLEANUP
# ============================================================================
cap.release()
cv2.destroyAllWindows()
hands.close()
print("\n🛑 Gesture Control stopped safely. See you next time!")
print("\n🎬 Demo Completed — Gesture Control System stopped safely.")
