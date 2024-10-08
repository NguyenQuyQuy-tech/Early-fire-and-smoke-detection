import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (make sure to specify the right model file)
model = YOLO('yolov8m.pt')  # Replace with your retrained model path if necessary

# Function to detect fire and smoke
def detect_fire_and_smoke(video_source=0):
    # Open a connection to the video source
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        # Read a frame from the video source
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Perform inference
        results = model(frame)

        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])

                # Draw bounding box and label
                if cls == 0:  # Assuming class 0 is 'fire' and class 1 is 'smoke'
                    label = f'Fire: {conf:.2f}'
                elif cls == 1:
                    label = f'Smoke: {conf:.2f}'
                else:
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with detections
        cv2.imshow('Fire and Smoke Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_fire_and_smoke()
