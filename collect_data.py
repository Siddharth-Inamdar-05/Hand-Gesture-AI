"""
collect_data.py — Gesture Dataset Collection Tool
-------------------------------------------------

This script records hand gesture data from the webcam using MediaPipe. It extracts
21 3D hand landmarks (x, y, z) and saves them into gestures.csv. Each saved sample
contains 63 numerical values (21 landmarks × 3 coordinates) plus a label.

Press the following keys to save data:
 - 't' → thumbs_up
 - 'p' → open_palm
 - 'r' → swipe_right
 - 'l' → swipe_left
 - 'k' → rock_on
 - 's' → index_swipe_up
 - 'x' → index_swipe_down
Press 'q' to quit.

Workflow:
 1) Initialize webcam and MediaPipe Hands
 2) Process frames, extract landmarks if a hand is detected
 3) On key press, save 63 features + label to gestures.csv

📸 Performance Mode ON — capturing at 640x480 resolution for smoother frame rate.
"""

import cv2
import mediapipe as mp
import csv
import os

# ============================================================================
# MEDIAPIPE INITIALIZATION
# ============================================================================
# MediaPipe Hands detects and tracks hand landmarks in real-time
# It identifies 21 specific points on your hand (wrist, thumb joints, finger tips, etc.)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Hands detector
# static_image_mode=False: Optimized for video streams (faster processing)
# max_num_hands=1: Only detect one hand (simpler, faster)
# min_detection_confidence=0.5: Minimum confidence to detect a hand
# min_tracking_confidence=0.5: Minimum confidence to keep tracking a hand
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================================
# CSV FILE SETUP
# ============================================================================
CSV_FILE = "gestures.csv"

# Create CSV file with headers if it doesn't exist
# Headers: 63 columns for coordinates (21 landmarks × 3 coordinates) + 1 label column
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = []
        # Create column names: landmark_0_x, landmark_0_y, landmark_0_z, ... landmark_20_z
        for i in range(21):
            header.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
        header.append('label')  # Last column is the gesture label
        writer.writerow(header)

# ============================================================================
# WEBCAM SETUP
# ============================================================================
cap = cv2.VideoCapture(0)  # 0 = first available camera

# Lower resolution to 640x480 for faster processing and smoother recording
# Higher resolutions cause lag and aren't needed for gesture recognition
cap.set(3, 640)  # width
cap.set(4, 480)  # height

print("\n📸 Performance Mode ON: Capturing at 640x480 for smoother recording.")
print("🧹 Tip: Restart the terminal between gesture recordings to clear camera memory.\n")

# ============================================================================
# USER INSTRUCTIONS
# ============================================================================
print("\n=== Gesture Data Collection ===")
print("Press keys to save gestures:")
print("  't' → thumbs_up (👍 Play)")
print("  'p' → open_palm (🖐️ Pause)")
print("  'r' → swipe_right (⏭️ Next Song)")
print("  'l' → swipe_left (⏮️ Previous Song)")
print("  'k' → rock_on (🤘 Like Song)")
print("  's' → index_swipe_up (☝️ Swipe Up - Volume Up)")
print("  'x' → index_swipe_down (👇 Swipe Down - Volume Down)")
print("\n💡 Collect 30-40 samples per gesture for best accuracy")
print("💡 Make sure your hand is visible when pressing keys")
print("Press 'q' to quit\n")

# ============================================================================
# MAIN DATA COLLECTION LOOP
# ============================================================================
while True:
    # Capture one frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame horizontally for mirror effect (feels more natural)
    frame = cv2.flip(frame, 1)
    
    # Convert BGR (Blue-Green-Red) to RGB (Red-Green-Blue)
    # OpenCV uses BGR by default, but MediaPipe requires RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with MediaPipe to detect hand landmarks
    # Returns results object containing hand landmark data if hand is detected
    results = hands.process(rgb_frame)
    
    # Draw hand landmarks on frame (visual feedback)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
    
    # Display keyboard shortcuts on video feed
    cv2.putText(frame, "t:Play p:Pause r:Next l:Prev k:Like s:Vol+ x:Vol- q:Quit", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Show the video frame
    cv2.imshow('Gesture Data Collection', frame)
    
    # Check for key presses (wait 1ms, non-blocking)
    key = cv2.waitKey(1) & 0xFF
    
    # Quit if 'q' is pressed
    if key == ord('q'):
        break
    
    # ========================================================================
    # GESTURE SAVING LOGIC
    # ========================================================================
    # Each key press saves the current hand landmark positions to CSV
    # Only saves if hand is detected (results.multi_hand_landmarks exists)
    
    elif key == ord('t') and results.multi_hand_landmarks:
        # Extract 21 landmarks, each with x, y, z coordinates = 63 features
        hand_landmarks = results.multi_hand_landmarks[0]
        row = []
        for landmark in hand_landmarks.landmark:
            row.extend([landmark.x, landmark.y, landmark.z])
        row.append('thumbs_up')  # Add label
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print("Saved: thumbs_up (👍 Play)")
    
    elif key == ord('p') and results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        row = []
        for landmark in hand_landmarks.landmark:
            row.extend([landmark.x, landmark.y, landmark.z])
        row.append('open_palm')
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print("Saved: open_palm (🖐️ Pause)")
    
    elif key == ord('r') and results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        row = []
        for landmark in hand_landmarks.landmark:
            row.extend([landmark.x, landmark.y, landmark.z])
        row.append('swipe_right')
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print("Saved: swipe_right (⏭️ Next Song)")
    
    elif key == ord('l') and results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        row = []
        for landmark in hand_landmarks.landmark:
            row.extend([landmark.x, landmark.y, landmark.z])
        row.append('swipe_left')
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print("Saved: swipe_left (⏮️ Previous Song)")
    
    elif key == ord('k') and results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        row = []
        for landmark in hand_landmarks.landmark:
            row.extend([landmark.x, landmark.y, landmark.z])
        row.append('rock_on')
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print("Saved: rock_on (🤘 Like Song)")
    
    elif key == ord('s') and results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        row = []
        for landmark in hand_landmarks.landmark:
            row.extend([landmark.x, landmark.y, landmark.z])
        row.append('index_swipe_up')
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print("Saved: index_swipe_up (☝️ Swipe Up - Volume Up)")
    
    elif key == ord('x') and results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        row = []
        for landmark in hand_landmarks.landmark:
            row.extend([landmark.x, landmark.y, landmark.z])
        row.append('index_swipe_down')
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print("Saved: index_swipe_down (👇 Swipe Down - Volume Down)")

# ============================================================================
# CLEANUP
# ============================================================================
cap.release()
cv2.destroyAllWindows()
hands.close()
print(f"\n✅ Data collection complete! Data saved to {CSV_FILE}")
