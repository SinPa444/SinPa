from ultralytics import YOLO
import cv2 as cv


model = YOLO("S:/AI/projects/car & plate detector5/weights/best.pt")
img = cv.imread("S:/AI/Datasets/New folder/IMG_20250218_154311.jpg") # path the image
results = model.predict(source=img, save=False)

for r in results:
    boxes = r.boxes
    filtered_boxes = [box for box in boxes if box.cls == 1]
    r.boxes = filtered_boxes

annotated_image = results[0].plot()
annotated_image_resized = cv.resize(annotated_image, (640, 640))
cv.imshow("Plate Detection", annotated_image_resized)

cv.waitKey(0)
cv.destroyAllWindows()