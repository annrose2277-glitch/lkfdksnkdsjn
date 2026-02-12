import cv2
import time
import pyttsx3
import threading
from ultralytics import YOLO
import numpy as np

# --- Constants and Configuration ---

# An object is a "warning" if it covers at least this much of the screen.
PROXIMITY_THRESHOLD = 0.40

# Time in seconds for the audio warning interval.
WARNING_INTERVAL = 5

# How long the visual warning label stays on screen.
DISPLAY_DURATION = 5

# Confidence threshold for object detection.
CONFIDENCE_THRESHOLD = 0.5

# --- TTS Engine Setup ---

def speak(engine, text):
    """Function to run TTS in a separate thread to prevent video lag."""
    engine.say(text)
    engine.runAndWait()

# Initialize the TTS engine
try:
    tts_engine = pyttsx3.init()
except Exception as e:
    print(f"Warning: Could not initialize pyttsx3 TTS engine: {e}")
    tts_engine = None


# --- Main Script ---

def main():
    """
    Main function to run the real-time assistive object detection with audio feedback.
    """
    print("Initializing YOLO model...")
    try:
        model = YOLO("yolo26n.pt")
    except Exception as e:
        print(f"Error initializing YOLO model: {e}")
        print("Please ensure you have run 'pip install ultralytics pyttsx3' and that 'yolo26n.pt' is accessible.")
        return

    print("Starting webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Get frame dimensions for proximity calculation.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = frame_width * frame_height

    # --- State Management Variables ---
    last_warning_time = 0
    current_warning_object = None # Stores info about the object to be displayed
    warning_display_start_time = 0

    print("Starting detection loop... Press 'q' to quit.")
    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to grab frame.")
            break

        # "Selfie" Mode: Flip the frame horizontally.
        processed_frame = cv2.flip(frame, 1)

        # --- YOLO Detection ---
        results = model.predict(
            processed_frame,
            stream=True,
            verbose=False,
            conf=CONFIDENCE_THRESHOLD
        )

        # --- Proximity and Warning Logic ---
        dominant_object_in_frame = None
        max_dominant_area = 0

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = model.names[int(box.cls[0])]

                # Calculate screen coverage.
                box_area = (x2 - x1) * (y2 - y1)
                coverage_ratio = box_area / frame_area

                # Draw standard boxes for all detected objects.
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                
                # Check if object meets the proximity threshold to be a "warning".
                if coverage_ratio >= PROXIMITY_THRESHOLD:
                    # If it's the largest warning object found so far in this frame, store it.
                    if box_area > max_dominant_area:
                        max_dominant_area = box_area
                        dominant_object_in_frame = {"name": class_name, "box": (x1, y1, x2, y2)}

        current_time = time.time()
        
        # If a dominant warning object was found, update the global state.
        if dominant_object_in_frame:
            current_warning_object = dominant_object_in_frame
            warning_display_start_time = current_time # Reset display timer

            # --- Audio Warning Trigger ---
            # If enough time has passed since the last warning, issue a new one.
            if (current_time - last_warning_time) > WARNING_INTERVAL:
                last_warning_time = current_time
                message = f"Warning, {current_warning_object['name']} is close"
                print(message) # Print to terminal
                if tts_engine:
                    threading.Thread(target=speak, args=(tts_engine, message)).start()
        
        # --- Visual Warning Label Display ---
        # If there is an active warning object and its display timer has not expired.
        if current_warning_object and (current_time - warning_display_start_time) < DISPLAY_DURATION:
            # --- Label Styling ---
            bx1, by1, bx2, by2 = current_warning_object["box"]
            class_name = current_warning_object["name"]

            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1.2
            thickness = 2
            text_color = (255, 255, 255) # White
            bg_color = (0, 0, 0)         # Black

            (text_w, text_h), baseline = cv2.getTextSize(class_name, font_face, font_scale, thickness)
            
            label_x, label_y = bx1, by2 
            
            bg_x1, bg_y1 = label_x, label_y - text_h - baseline
            bg_x2, bg_y2 = label_x + text_w, label_y + baseline
            
            cv2.rectangle(processed_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
            cv2.putText(processed_frame, class_name, (label_x, label_y), font_face, font_scale, text_color, thickness)
        else:
            # Clear the warning if the timer has expired.
            current_warning_object = None

        cv2.imshow("Assistive Object Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Script finished.")

if __name__ == "__main__":
    main()

