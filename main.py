import cv2
import time
import pyttsx3
import threading
from ultralytics import YOLO
import numpy as np

# --- Constants and Configuration ---

# An object is a "warning" if it covers at least this much of the screen.
PROXIMITY_THRESHOLD = 0.40


# Confidence threshold for object detection.
CONFIDENCE_THRESHOLD = 0.5

# --- TTS Engine Setup ---
tts_busy = False

def speak(engine, text):
    """Function to run TTS in a separate thread to prevent video lag."""
    global tts_busy
    tts_busy = True
    try:
        engine.say(text)
        engine.runAndWait()
    finally:
        tts_busy = False

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

                box_color = (255, 0, 255) # Default: Magenta

                # Check if object meets the proximity threshold to be a "warning".
                if coverage_ratio >= PROXIMITY_THRESHOLD:
                    box_color = (0, 0, 255) # Red for any warning object
                    # If it's the largest warning object found so far in this frame, store it as dominant.
                    if box_area > max_dominant_area:
                        max_dominant_area = box_area
                        dominant_object_in_frame = {"name": class_name, "box": (x1, y1, x2, y2)}
                
                # Draw the bounding box with the determined color
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), box_color, 2)

                # --- Draw Label on EVERY Bounding Box ---
                label = f"{class_name}"
                
                # Get text size to create a background for the label
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                
                # Position for the background rectangle
                label_bg_y1 = y1 - text_h - 10
                # If the label would go off the top of the screen, place it inside the box instead
                if label_bg_y1 < 0:
                    label_bg_y1 = y1 + 2

                # Draw background rectangle
                cv2.rectangle(processed_frame, (x1, label_bg_y1), (x1 + text_w, label_bg_y1 + text_h + baseline), box_color, -1)
                # Draw text label
                cv2.putText(processed_frame, label, (x1, label_bg_y1 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # If a dominant warning object was found, trigger an audio warning.
        if dominant_object_in_frame:
            # If the TTS engine is not busy, issue a new warning.
            if not tts_busy:
                message = f"Warning, {dominant_object_in_frame['name']} is close"
                print(message) # Print to terminal
                if tts_engine:
                    threading.Thread(target=speak, args=(tts_engine, message)).start()

        cv2.imshow("Assistive Object Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Script finished.")

if __name__ == "__main__":
    main()

