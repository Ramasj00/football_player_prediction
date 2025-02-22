import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model_path = "runs/detect/train/weights/best.pt"
model = YOLO(model_path)

# Open video file
video_path = "portugal_vs_spain.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output video writer
output_path = "cr7_tracked.mp4"
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
try:
    # Process video frame by frame
    while cap.isOpened():
        frame_count = 0
        ret, frame = cap.read()
        if not ret:
            break  # Stop if video ends

        # Run YOLO detection on frame
        results = model(frame)

        # Draw bounding boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = box.conf[0].item()  # Confidence score
                class_id = int(box.cls[0])  # Class ID
                
                # Only draw if it's CR7 (class_id should match "CR7" in your dataset)
                if confidence > 0.5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, f"CR7 {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Write frame to output video
        out.write(frame)
        frame_count += 1
        if frame_count % 100 == 0:  # Print progress every 100 frames
            print(f"Processed {frame_count} frames")

        # Display (optional)
        cv2.imshow("Tracking CR7", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break  # Press 'q' to exit

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Force close any remaining windows
    for i in range(4):
        cv2.waitKey(1)
except Exception as e:
    print(e)



