from ultralytics import YOLO
import cv2 as cv
import numpy as np


model = YOLO("S:/AI/projects/car & plate detector5/weights/best.pt")
cap = cv.VideoCapture("") # path the video

while True:
    ret, frame = cv.read()
    if not ret:
        break

    results = model.predict(source=frame, save=False)

    for r in results:
        boxes = r.boxes
        filtered_boxes = [box for box in boxes if box.cls == 0]
        r.boxes = filtered_boxes

    annotated_frame = results[0].plot()
    cv.imshow("Car Detection", annotated_frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()