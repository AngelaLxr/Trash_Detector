import cv2
from ultralytics import YOLO

# Load your trained model (update the path if needed)
model = YOLO("runs/detect/train7/weights/best.pt")

# Open webcam (0 = default cam)
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # Run detection
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].item()
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # Draw label + confidence
            text = f"{label} {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Show the frame
    cv2.imshow("YOLOv8 Webcam Detection", frame)

    # Quit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
