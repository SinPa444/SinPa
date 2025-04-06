from ultralytics import YOLO
import cv2 as cv
import numpy as np


model = YOLO("S:/AI/projects/car & plate detector5/weights/best.pt")
img = cv.imread("S:/AI/Datasets/New folder/IMG_20250218_113330.jpg") # path the image
results = model.predict(source=img, save=False)

for r in results:
    boxes = r.boxes
    filtered_boxes = [box for box in boxes if box.cls == 0]
    r.boxes = filtered_boxes

annotated_images = results[0].plot()
annotated_images_resized = cv.resize(annotated_images, (640, 640))
cv.imshow("Car Detection", annotated_images_resized)

cv.waitKey(0)
cv.destroyAllWindows()