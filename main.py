
import cv2
import time
from ultralytics import YOLO
import numpy as np

# --- Constants and Configuration ---
# The model will identify an object as "close" if its bounding box
# covers at least this much of the total frame area.
PROXIMITY_THRESHOLD = 0.40

# How long the textual identification message should stay on screen (in seconds).
DISPLAY_DURATION = 5

# Confidence threshold for object detection. Detections below this are ignored.
CONFIDENCE_THRESHOLD = 0.5

# --- Main Script ---

def main():
    """
    Main function to run the real-time assistive object detection.
    """
    print("Initializing YOLO26n model...")
    # Initialize the YOLOv8-Nano model, chosen for its high speed.
    # The model is initialized only once, outside the main loop, for optimization.
    try:
        model = YOLO("yolo26n.pt")
    except Exception as e:
        print(f"Error initializing YOLO model: {e}")
        print("Please ensure you have run 'pip install ultralytics' and that the model file is accessible.")
        return

    print("Starting webcam...")
    # Use cv2.VideoCapture(0) to access the default webcam.
    cap = cv2.VideoCapture(0)

    # Gracefully handle cases where the webcam is not available.
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Get webcam frame dimensions for area calculations.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = frame_width * frame_height

    # --- State Management Variables ---
    # These variables manage the display of the centered text message.
    current_message = None
    message_start_time = 0

    print("Starting detection loop... Press 'q' to quit.")
    while True:
        # Read a frame from the webcam.
        success, frame = cap.read()
        if not success:
            print("Error: Failed to grab frame.")
            break

        # --- YOLO Detection ---
        # Run the model on the current frame.
        # stream=True is efficient for video feeds.
        # verbose=False reduces console spam for a cleaner output.
        results = model.predict(
            frame,
            stream=True,
            verbose=False,
            conf=CONFIDENCE_THRESHOLD
        )

        # Variable to track the most dominant "close" object in the current frame.
        dominant_object_found = None
        max_dominant_area = 0

        # Process the results from the model.
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # --- Bounding Box Processing ---
                # Get coordinates, class, and confidence.
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                
                # --- Proximity Calculation ---
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                
                # The "40% coverage calculation":
                # We determine how much of the screen the object occupies by dividing
                # the object's bounding box area by the total frame area.
                coverage_ratio = box_area / frame_area

                # Draw the standard bounding box for every detected object.
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                label = f"{class_name} {box.conf[0]:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                # --- "Close/Dominant" Object Logic ---
                # Check if the object is close enough AND is the most dominant one so far.
                if coverage_ratio >= PROXIMITY_THRESHOLD:
                    if box_area > max_dominant_area:
                        max_dominant_area = box_area
                        dominant_object_found = class_name

        # If a dominant object was found in this frame, update the message state.
        if dominant_object_found:
            current_message = dominant_object_found
            message_start_time = time.time()

        # --- "5-Second" Timer and Display Logic ---
        # Check if a message should be actively displayed.
        if current_message and (time.time() - message_start_time) < DISPLAY_DURATION:
            # --- Visuals: Centered Text Overlay ---
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 2
            thickness = 3
            text_size = cv2.getTextSize(current_message, font, font_scale, thickness)[0]
            
            # Calculate position to center the text
            text_x = (frame_width - text_size[0]) // 2
            text_y = (frame_height + text_size[1]) // 2

            # Create a black background rectangle for better readability.
            bg_x1 = text_x - 20
            bg_y1 = text_y - text_size[1] - 20
            bg_x2 = text_x + text_size[0] + 20
            bg_y2 = text_y + 20
            
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            
            # Draw the white text on top of the black background.
            cv2.putText(frame, current_message, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        else:
            # If the timer has expired, clear the message.
            current_message = None
            
        # Display the processed frame.
        cv2.imshow("Assistive Object Detection", frame)

        # Break the loop if the 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --- Cleanup ---
    # Release the webcam and destroy all OpenCV windows.
    cap.release()
    cv2.destroyAllWindows()
    print("Script finished.")

if __name__ == "__main__":
    main()
