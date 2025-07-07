import cv2
import easyocr
from ultralytics import YOLO

# Load YOLOv8 model (can be 'yolov8n.pt' or your custom-trained model)
model = YOLO('yolov8n.pt')

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Start video capture (0 = default webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = frame[y1:y2, x1:x2]

        # OCR on cropped region
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        ocr_result = reader.readtext(gray)
        plate_text = ocr_result[0][1] if ocr_result else "Unknown"

        # Draw rectangle and text
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, plate_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Show result in a window
    cv2.imshow("Real-time Number Plate Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()