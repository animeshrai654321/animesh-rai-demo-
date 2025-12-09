import sys
import time
from collections import deque
from typing import Tuple, Optional
import math
import threading

# Import and validate dependencies
def import_with_error_check():
    """Import all required libraries with error handling"""
    try:
        import cv2
    except ImportError:
        print("Error: OpenCV (cv2) is not installed.")
        print("Install with: python -m pip install opencv-python")
        sys.exit(1)
    
    try:
        import mediapipe as mp # type: ignore
    except ImportError:
        print("Error: MediaPipe (mediapipe) is not installed.")
        print("Install with: python -m pip install mediapipe")
        sys.exit(1)
    
    try:
        import numpy as np
    except ImportError:
        print("Error: NumPy (numpy) is not installed.")
        print("Install with: python -m pip install numpy")
        sys.exit(1)
    
    try:
        import pyautogui
    except ImportError:
        print("Error: PyAutoGUI (pyautogui) is not installed.")
        print("Install with: python -m pip install pyautogui")
        sys.exit(1)
    
    return cv2, mp, np, pyautogui

cv2, mp, np, pyautogui = import_with_error_check()

# Initialize MediaPipe with optimized settings for stability and speed
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Single hand for better performance
    min_detection_confidence=0.7,  # Balanced for reliability
    min_tracking_confidence=0.5,  # Lower for smoother tracking
    model_complexity=0  # Lighter model for speed
)

# Disable PyAutoGUI failsafe
pyautogui.FAILSAFE = False

# Configuration constants - OPTIMIZED FOR SMOOTH & FAST PERFORMANCE
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
GESTURE_COOLDOWN = 0.2  # Balanced cooldown
SMOOTHING_FACTOR = 0.7  # Optimized for smooth + responsive
SWIPE_THRESHOLD = 70  # Balanced swipe detection
SWIPE_DURATION = 0.4  # Balanced duration
FINGER_CLOSED_THRESHOLD = 0.08  # Reliable fist detection
FINGER_OPEN_THRESHOLD = 0.16  # Reliable palm detection
CLICK_COOLDOWN = 0.25  # Prevent accidental double clicks
PALM_HOLD_TIME = 0.2  # Quick but stable gesture detection

# Hand landmarks indices
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
PALM_CENTER = 9
THUMB_IP = 3
INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18

class HandGestureTracker:
    """Track all hand movements and gestures with optimized detection"""
    def __init__(self):
        self.hand_positions = deque(maxlen=15)  # Balanced buffer
        self.prev_mouse_x = SCREEN_WIDTH // 2
        self.prev_mouse_y = SCREEN_HEIGHT // 2
        self.last_swipe_time = 0
        self.last_click_time = 0
        self.last_palm_open_time = 0
        self.last_fist_close_time = 0
        self.finger_states = {f: False for f in range(5)}
        self.wrist_pos = None
        self.palm_pos = None
        self.all_fingers_extended = False
        self.all_fingers_closed = False
        self.prev_all_fingers_extended = False
        self.prev_all_fingers_closed = False
        self.swipe_direction = None
        self.swipe_start_time = 0
        self.palm_open_triggered = False
        self.fist_close_triggered = False
        self.gesture_stability_counter = 0
    
    def update_position(self, x: int, y: int):
        """Update hand position history"""
        self.hand_positions.append((x, y))
    
    def reset_hand_tracking(self):
        """Reset tracking data"""
        self.hand_positions.clear()
        self.wrist_pos = None
        self.palm_pos = None
        self.palm_open_triggered = False
        self.fist_close_triggered = False
        self.gesture_stability_counter = 0
        self.prev_all_fingers_extended = False
        self.prev_all_fingers_closed = False

tracker = HandGestureTracker()

def calculate_distance(point1: Tuple[float, float], 
                       point2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_hand_center(hand_landmarks) -> Tuple[int, int]:
    """Get the center position of the hand (wrist/palm area)"""
    wrist = hand_landmarks.landmark[WRIST]
    palm = hand_landmarks.landmark[PALM_CENTER]
    center_x = int((wrist.x + palm.x) / 2 * SCREEN_WIDTH)
    center_y = int((wrist.y + palm.y) / 2 * SCREEN_HEIGHT)
    return center_x, center_y

def get_finger_positions(hand_landmarks) -> dict:
    """Get positions of all fingers"""
    fingers = {
        'thumb': hand_landmarks.landmark[THUMB_TIP],
        'index': hand_landmarks.landmark[INDEX_TIP],
        'middle': hand_landmarks.landmark[MIDDLE_TIP],
        'ring': hand_landmarks.landmark[RING_TIP],
        'pinky': hand_landmarks.landmark[PINKY_TIP],
        'wrist': hand_landmarks.landmark[WRIST],
        'palm': hand_landmarks.landmark[PALM_CENTER]
    }
    return fingers

def check_all_fingers_extended(hand_landmarks) -> bool:
    """Check if all fingers are spread open - OPTIMIZED"""
    try:
        wrist = hand_landmarks.landmark[WRIST]
        fingers_tips = [
            hand_landmarks.landmark[THUMB_TIP],
            hand_landmarks.landmark[INDEX_TIP],
            hand_landmarks.landmark[MIDDLE_TIP],
            hand_landmarks.landmark[RING_TIP],
            hand_landmarks.landmark[PINKY_TIP]
        ]
        
        # Also check PIP joints for better detection
        fingers_pip = [
            hand_landmarks.landmark[THUMB_IP],
            hand_landmarks.landmark[INDEX_PIP],
            hand_landmarks.landmark[MIDDLE_PIP],
            hand_landmarks.landmark[RING_PIP],
            hand_landmarks.landmark[PINKY_PIP]
        ]
        
        distances_tips = [
            calculate_distance((tip.x, tip.y), (wrist.x, wrist.y))
            for tip in fingers_tips
        ]
        
        distances_pip = [
            calculate_distance((pip.x, pip.y), (wrist.x, wrist.y))
            for pip in fingers_pip
        ]
        
        return all(d > FINGER_OPEN_THRESHOLD for d in distances_tips) and \
               all(d > 0.12 for d in distances_pip)
    except Exception as e:
        return False

def check_all_fingers_closed(hand_landmarks) -> bool:
    """Check if all fingers are closed (fist) - OPTIMIZED"""
    try:
        wrist = hand_landmarks.landmark[WRIST]
        fingers_tips = [
            hand_landmarks.landmark[THUMB_TIP],
            hand_landmarks.landmark[INDEX_TIP],
            hand_landmarks.landmark[MIDDLE_TIP],
            hand_landmarks.landmark[RING_TIP],
            hand_landmarks.landmark[PINKY_TIP]
        ]
        
        distances = [
            calculate_distance((tip.x, tip.y), (wrist.x, wrist.y))
            for tip in fingers_tips
        ]
        
        return all(d < FINGER_CLOSED_THRESHOLD for d in distances)
    except Exception as e:
        return False

def detect_swipe_gesture(hand_landmarks) -> Optional[str]:
    """Detect swipe gestures (left, right, up, down) - BALANCED"""
    try:
        if len(tracker.hand_positions) < 10:
            return None
        
        # Get start and end positions
        start_pos = tracker.hand_positions[0]
        end_pos = tracker.hand_positions[-1]
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        current_time = time.time()
        
        # Check if it's a significant swipe
        if distance > SWIPE_THRESHOLD and (current_time - tracker.swipe_start_time) < SWIPE_DURATION:
            if abs(dx) > abs(dy):  # Horizontal swipe
                if dx > 0:
                    return "SWIPE_RIGHT"
                else:
                    return "SWIPE_LEFT"
            else:  # Vertical swipe
                if dy > 0:
                    return "SWIPE_DOWN"
                else:
                    return "SWIPE_UP"
        
        return None
    except Exception as e:
        return None

def move_mouse_smooth(hand_landmarks) -> None:
    """
    Move mouse based on index finger with ultra-smooth kalman-like filtering
    """
    try:
        # Use index finger for primary tracking
        index_finger = hand_landmarks.landmark[INDEX_TIP]
        
        # Calculate screen position
        raw_x = int(index_finger.x * SCREEN_WIDTH)
        raw_y = int(index_finger.y * SCREEN_HEIGHT)
        
        # Clamp to screen bounds
        raw_x = max(0, min(SCREEN_WIDTH - 1, raw_x))
        raw_y = max(0, min(SCREEN_HEIGHT - 1, raw_y))
        
        # Multi-stage smoothing for optimal performance
        # Stage 1: Exponential smoothing
        smooth_x = int(tracker.prev_mouse_x * SMOOTHING_FACTOR + raw_x * (1 - SMOOTHING_FACTOR))
        smooth_y = int(tracker.prev_mouse_y * SMOOTHING_FACTOR + raw_y * (1 - SMOOTHING_FACTOR))
        
        # Stage 2: Micro-jitter filter (ignore tiny movements)
        dx = abs(smooth_x - tracker.prev_mouse_x)
        dy = abs(smooth_y - tracker.prev_mouse_y)
        
        if dx < 3 and dy < 3:  # Ignore micro-movements
            smooth_x = tracker.prev_mouse_x
            smooth_y = tracker.prev_mouse_y
        
        # Move mouse only if position changed significantly
        if dx > 1 or dy > 1:
            pyautogui.moveTo(smooth_x, smooth_y, _pause=False)
        
        tracker.prev_mouse_x = smooth_x
        tracker.prev_mouse_y = smooth_y
        tracker.update_position(smooth_x, smooth_y)
        
    except Exception as e:
        pass

def handle_swipe_gesture(direction: str, current_time: float) -> None:
    """Handle swipe-based window switching - SMOOTH & FAST"""
    if current_time - tracker.last_swipe_time < 0.5:  # Prevent rapid-fire swipes
        return
    
    try:
        if direction == "SWIPE_RIGHT":
            print("➡️  Swiping Right - Next Window...")
            threading.Thread(target=lambda: pyautogui.hotkey('alt', 'tab'), daemon=True).start()
            tracker.last_swipe_time = current_time
        elif direction == "SWIPE_LEFT":
            print("⬅️  Swiping Left - Previous Window...")
            threading.Thread(target=lambda: pyautogui.hotkey('alt', 'shift', 'tab'), daemon=True).start()
            tracker.last_swipe_time = current_time
        elif direction == "SWIPE_UP":
            print("⬆️  Swiping Up - Maximize Window...")
            threading.Thread(target=lambda: pyautogui.hotkey('alt', 'F10'), daemon=True).start()
            tracker.last_swipe_time = current_time
        elif direction == "SWIPE_DOWN":
            print("⬇️  Swiping Down - Show Desktop...")
            threading.Thread(target=lambda: pyautogui.hotkey('win', 'd'), daemon=True).start()
            tracker.last_swipe_time = current_time
    except Exception as e:
        pass

def execute_click_async(button='left'):
    """Execute click in a separate thread for faster response"""
    def click():
        try:
            pyautogui.click(button=button)
        except:
            pass
    thread = threading.Thread(target=click, daemon=True)
    thread.start()

def handle_all_finger_gestures(hand_landmarks, current_time: float) -> None:
    """Handle gestures with all fingers - STABLE & RESPONSIVE"""
    try:
        # Gesture stability detection - require 2 consistent frames
        if tracker.all_fingers_extended == tracker.prev_all_fingers_extended:
            if tracker.all_fingers_extended:
                tracker.gesture_stability_counter += 1
        else:
            tracker.gesture_stability_counter = 0
        
        tracker.prev_all_fingers_extended = tracker.all_fingers_extended
        
        # Palm open - STABLE DETECTION
        if tracker.all_fingers_extended and not tracker.palm_open_triggered:
            if tracker.gesture_stability_counter >= 2:  # 2 consistent frames
                if current_time - tracker.last_palm_open_time > PALM_HOLD_TIME:
                    print("✋ Palm Open - LEFT CLICK!")
                    tracker.last_palm_open_time = current_time
                    tracker.palm_open_triggered = True
                    execute_click_async('left')
        
        # Reset palm trigger when fingers close
        if not tracker.all_fingers_extended:
            tracker.palm_open_triggered = False
            tracker.gesture_stability_counter = 0
        
        # Fist stability detection
        if tracker.all_fingers_closed == tracker.prev_all_fingers_closed:
            if tracker.all_fingers_closed:
                tracker.gesture_stability_counter += 1
        else:
            tracker.gesture_stability_counter = 0
        
        tracker.prev_all_fingers_closed = tracker.all_fingers_closed
        
        # Fist closed - STABLE DETECTION
        if tracker.all_fingers_closed and not tracker.fist_close_triggered:
            if tracker.gesture_stability_counter >= 2:  # 2 consistent frames
                if current_time - tracker.last_fist_close_time > PALM_HOLD_TIME:
                    print("✊ Fist Closed - RIGHT CLICK!")
                    tracker.last_fist_close_time = current_time
                    tracker.fist_close_triggered = True
                    execute_click_async('right')
        
        # Reset fist trigger when fingers open
        if not tracker.all_fingers_closed:
            tracker.fist_close_triggered = False
            tracker.gesture_stability_counter = 0
    except Exception as e:
        pass

def draw_hand_info(frame, hand_landmarks, handedness, frame_width: int, frame_height: int):
    """Draw detailed hand information on frame"""
    try:
        fingers = get_finger_positions(hand_landmarks)
        h, w = frame.shape[:2]
        
        # Draw larger hand skeleton with better visibility
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3)
        )
        
        # Draw finger labels
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        finger_keys = ['thumb', 'index', 'middle', 'ring', 'pinky']
        
        for name, key in zip(finger_names, finger_keys):
            finger = fingers[key]
            x = int(finger.x * w)
            y = int(finger.y * h)
            cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
            cv2.putText(frame, name, (x + 10, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw wrist
        wrist = fingers['wrist']
        wrist_x = int(wrist.x * w)
        wrist_y = int(wrist.y * h)
        cv2.circle(frame, (wrist_x, wrist_y), 10, (255, 0, 255), -1)
        cv2.putText(frame, "Wrist", (wrist_x + 10, wrist_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Draw mouse pointer
        mouse_x, mouse_y = pyautogui.position()
        cv2.circle(frame, (int(mouse_x * w / SCREEN_WIDTH), 
                          int(mouse_y * h / SCREEN_HEIGHT)), 8, (0, 0, 255), -1)
        
    except Exception as e:
        pass

def main():
    """Main function to run advanced hand gesture control - SMOOTH & OPTIMIZED"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access camera. Please check your camera connection.")
        return
    
    # Set camera properties for optimal balance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Standard 30 FPS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("\n" + "="*70)
    print("🖐️  ADVANCED HAND GESTURE DESKTOP CONTROL - OPTIMIZED")
    print("="*70)
    print("\n✨ ENABLED GESTURES:")
    print("  👆 Ultra-Smooth Mouse - Index finger controls cursor")
    print("  ➡️  Swipe Right - Switch to next window (Alt+Tab)")
    print("  ⬅️  Swipe Left - Switch to previous window")
    print("  ⬆️  Swipe Up - Maximize window")
    print("  ⬇️  Swipe Down - Show desktop")
    print("  ✋ Palm OPEN - LEFT CLICK anywhere")
    print("  ✊ Fist CLOSED - RIGHT CLICK anywhere")
    print("  🎯 Smart Filtering - Eliminates jitter & false positives")
    print("\n💡 TIPS:")
    print("  • Keep hand within camera view")
    print("  • Make deliberate, clear gestures")
    print("  • Hold gestures briefly for detection")
    print("  • Avoid rapid repeated gestures")
    print("\n📍 Press 'q' to quit | 'r' to reset tracking")
    print("="*70 + "\n")
    
    frame_count = 0
    fps_start_time = time.time()
    fps = 0.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame from camera")
                break
            
            frame_count += 1
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Calculate FPS
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
            
            # Draw info
            cv2.putText(frame, f"Resolution: {w}x{h} | FPS: {fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Mouse: ({tracker.prev_mouse_x}, {tracker.prev_mouse_y})", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                                      results.multi_handedness):
                    # Draw detailed hand info
                    draw_hand_info(frame, hand_landmarks, handedness, w, h)
                    
                    # Track hand movement
                    hand_center_x, hand_center_y = get_hand_center(hand_landmarks)
                    tracker.update_position(hand_center_x, hand_center_y)
                    
                    # Check finger states - FAST
                    tracker.all_fingers_extended = check_all_fingers_extended(hand_landmarks)
                    tracker.all_fingers_closed = check_all_fingers_closed(hand_landmarks)
                    
                    current_time = time.time()
                    if tracker.swipe_start_time == 0:
                        tracker.swipe_start_time = current_time
                    
                    # Detect swipe gesture - FASTER
                    swipe_dir = detect_swipe_gesture(hand_landmarks)
                    if swipe_dir:
                        handle_swipe_gesture(swipe_dir, current_time)
                    
                    # Handle all finger gestures - INSTANT
                    handle_all_finger_gestures(hand_landmarks, current_time)
                    
                    # Move mouse with adaptive smooth tracking - ULTRA RESPONSIVE
                    move_mouse_smooth(hand_landmarks)
                    
                    # Display status
                    if tracker.all_fingers_extended:
                        cv2.putText(frame, "✋ PALM OPEN - LEFT CLICK", (w - 400, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    elif tracker.all_fingers_closed:
                        cv2.putText(frame, "✊ FIST CLOSED - RIGHT CLICK", (w - 400, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                tracker.reset_hand_tracking()
                tracker.swipe_start_time = 0
            
            # Display help
            cv2.putText(frame, "Q: Quit | R: Reset | ULTRA FAST MODE", (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Show frame
            cv2.imshow("Advanced Hand Gesture Control - ULTRA FAST", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Exiting gracefully...")
                break
            elif key == ord('r'):
                print("🔄 Resetting tracking...")
                tracker.reset_hand_tracking()
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("✅ Resources cleaned up. Goodbye!")

if __name__ == "__main__":
    main()