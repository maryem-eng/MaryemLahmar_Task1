import cv2  # OpenCV for video capture and image processing
import mediapipe as mp  # Mediapipe for face and eye landmark detection
import numpy as np  # Numpy for numerical operations
from scipy.spatial import distance as dist  # For calculating distances
import winsound  # For alarm sound on Windows

# Constants for Eye Aspect Ratio (EAR) thresholds and fatigue detection
EAR_THRESHOLD_BLINK = 0.25  # Threshold for blink detection
EAR_THRESHOLD_FATIGUE = 0.15  # Threshold for fatigue detection
CONSEC_FRAMES_FATIGUE = 15  # Number of consecutive frames to confirm fatigue
CONSEC_FRAMES_BLINK = 2  # Number of consecutive frames to confirm a blink

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh  # Mediapipe face mesh solution
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, static_image_mode=False, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils  # Drawing utility for visualization

# Indices for eye landmarks in Mediapipe Face Mesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]  # Indices for left eye
RIGHT_EYE = [362, 385, 387, 263, 373, 380]  # Indices for right eye

def eye_aspect_ratio(eye_landmarks):
    """Calculate the Eye Aspect Ratio (EAR) to detect eye state."""
    # Compute distances between vertical eye landmarks
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    # Compute distance between horizontal eye landmarks
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    # Return the EAR
    return (A + B) / (2.0 * C)

def sound_alarm():
    """Sound an alarm when fatigue is detected."""
    winsound.Beep(1000, 1000)  # Play a beep sound at 1000 Hz for 1 second

# Initialize counters for blinks and fatigue
blink_count = 0  # Total number of blinks detected
frame_counter_blink = 0  # Frames to confirm a blink
frame_counter_fatigue = 0  # Frames to confirm fatigue

# Start video capture
cap = cv2.VideoCapture(0)  # Open default camera
print("Press 'q' to quit.")  # Instruction for quitting the application

while cap.isOpened():
    ret, frame = cap.read()  # Capture a frame from the camera
    if not ret:  # If frame is not read correctly, break the loop
        break
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally (mirroring)

    h, w, _ = frame.shape  # Get frame dimensions
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for Mediapipe

    # Process the frame with Mediapipe Face Mesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:  # If face landmarks are detected
        for face_landmarks in results.multi_face_landmarks:
            # Extract landmark coordinates and scale to frame size
            landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark], dtype=np.float32)
            
            # Extract landmarks for left and right eyes
            left_eye = landmarks[LEFT_EYE].astype(int)
            right_eye = landmarks[RIGHT_EYE].astype(int)

            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0  # Average EAR of both eyes

            # Blink detection logic
            if avg_ear < EAR_THRESHOLD_BLINK:  # If EAR falls below blink threshold
                frame_counter_blink += 1  # Increment blink frame counter
            else:
                if frame_counter_blink >= CONSEC_FRAMES_BLINK:  # If sufficient frames detected
                    blink_count += 1  # Increment blink count
                    print(f"Blinks detected: {blink_count}")
                frame_counter_blink = 0  # Reset blink frame counter

            # Fatigue detection logic
            if avg_ear < EAR_THRESHOLD_FATIGUE:  # If EAR falls below fatigue threshold
                frame_counter_fatigue += 1  # Increment fatigue frame counter
                if frame_counter_fatigue >= CONSEC_FRAMES_FATIGUE:  # If sufficient frames detected
                    cv2.putText(frame, "FATIGUE DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    sound_alarm()  # Sound alarm
            else:
                frame_counter_fatigue = 0  # Reset fatigue frame counter

            # Display EAR and blink count on the frame
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Blinks: {blink_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the processed frame
    cv2.imshow("Eye Blink and Fatigue Detection", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
