import torch
from ultralytics import YOLO
import cv2

model_path = 'yolo11l.pt'
model = YOLO(model_path)

image_path = 'images/highway.jpg'
image = cv2.imread(image_path)

results = model.predict(source=image, show=True, conf=0.25)

def display_results(results):
    for result in results:
        for detection in result.boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
            label = int(detection.cls)
            confidence = float(detection.conf) 
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{label} ({confidence:.2f})', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('YOLOv11 Inference', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

display_results(results)