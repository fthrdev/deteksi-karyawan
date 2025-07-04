import cv2
import mediapipe as mp
import time
import pyttsx3
import numpy as np
from datetime import datetime
import os
import threading

# Constants
WARNING_THRESHOLD = 3  # seconds
ATTENDANCE_LOG = "attendance_log.csv"
MAX_FRAMES_FOR_SMOOTHING = 10
SOUND_EFFECTS = {
    'warning': 'warning.wav',
    'detected': 'detected.wav'
}

# Initialize text-to-speech engine with enhanced settings
engine = pyttsx3.init()
engine.setProperty('rate', 140)  # speech speed
engine.setProperty('volume', 0.9)  # volume level
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # change voice (1 for female)

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Face detection variables
face_detected = False
last_seen = time.time()
warning_issued = False
detection_history = []
attendance_recorded = False

# Performance metrics
fps_counter = 0
fps_last_time = time.time()
fps = 0

# Initialize camera with better settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Create attendance log if it doesn't exist
if not os.path.exists(ATTENDANCE_LOG):
    with open(ATTENDANCE_LOG, 'w') as f:
        f.write("Timestamp,Status,Duration\n")

def log_attendance(status, duration=0):
    """Log employee attendance status with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ATTENDANCE_LOG, 'a') as f:
        f.write(f"{timestamp},{status},{duration}\n")
    print(f"Attendance logged: {status} at {timestamp}")

def play_sound_async(sound_type):
    """Play sound effect in a separate thread"""
    def play():
        if sound_type in SOUND_EFFECTS and os.path.exists(SOUND_EFFECTS[sound_type]):
            os.system(f"aplay {SOUND_EFFECTS[sound_type]} 2>/dev/null")  # Linux
            # For Windows: os.system(f"start {SOUND_EFFECTS[sound_type]}")
    threading.Thread(target=play).start()

def smooth_detection(current_detection):
    """Apply smoothing to detection results using moving average"""
    detection_history.append(current_detection)
    if len(detection_history) > MAX_FRAMES_FOR_SMOOTHING:
        detection_history.pop(0)
    return sum(detection_history) / len(detection_history) > 0.5

def draw_fancy_box(img, pt1, pt2, color, thickness, r, d):
    """Draw rounded rectangle around face"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

def calculate_fps():
    """Calculate and display FPS"""
    global fps_counter, fps_last_time, fps
    fps_counter += 1
    if time.time() - fps_last_time >= 1.0:
        fps = fps_counter
        fps_counter = 0
        fps_last_time = time.time()
    return fps

# Main processing loop
with mp_face_detection.FaceDetection(
    min_detection_confidence=0.7,  # Higher confidence threshold
    model_selection=1  # Use accurate model
) as face_detection:
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            break
        
        # Calculate FPS
        current_fps = calculate_fps()
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB and process
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_detection.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Check for face detections
        current_detection = 0
        if results.detections:
            current_detection = 1
            for detection in results.detections:
                # Get face bounding box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Draw fancy box around face
                draw_fancy_box(image, (bbox[0], bbox[1]), 
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]), 
                              (0, 255, 0), 2, 20, 20)
                
                # Display confidence score
                cv2.putText(image, f"{detection.score[0]:.2f}", 
                            (bbox[0], bbox[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Apply smoothing to detection results
        smoothed_detection = smooth_detection(current_detection)
        
        # Update face detection status
        if smoothed_detection:
            if not face_detected:
                play_sound_async('detected')
                if not attendance_recorded:
                    log_attendance("ARRIVED")
                    attendance_recorded = True
            face_detected = True
            last_seen = time.time()
            warning_issued = False
        else:
            face_detected = False
        
        # Check if face has been missing
        if not face_detected:
            absence_duration = time.time() - last_seen
            if absence_duration > WARNING_THRESHOLD:
                if not warning_issued:
                    print("Employee not detected!")
                    engine.say("Warning! Employee not detected")
                    engine.runAndWait()
                    play_sound_async('warning')
                    warning_issued = True
                    log_attendance("ABSENT", round(absence_duration))
                    attendance_recorded = False
        else:
            absence_duration = 0
        
        # Add HUD overlay
        hud_color = (0, 255, 0) if face_detected else (0, 0, 255)
        
        # Status text with more info
        status_text = "EMPLOYEE DETECTED" if face_detected else f"DETECTING... ({int(WARNING_THRESHOLD - (time.time() - last_seen))}s)"
        cv2.putText(image, status_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, hud_color, 2)
        
        # Display absence duration if applicable
        if not face_detected and (time.time() - last_seen) > 1:
            cv2.putText(image, f"Absent: {int(time.time() - last_seen)}s", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display FPS
        cv2.putText(image, f"FPS: {current_fps}", (20, image.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display current time
        current_time = datetime.now().strftime("%H:%M:%S")
        cv2.putText(image, current_time, (image.shape[1] - 150, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add semi-transparent overlay
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (image.shape[1], 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        
        # Show the frame
        cv2.imshow('Employee Monitoring System', image)
        
        # Exit on 'q' key
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()