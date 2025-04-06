from ultralytics import YOLO
import cv2 as cv


model = YOLO("") # when training finished path your best.pt
img = cv.imread("") # path the image
results = model.predict(source=img, save=False)

for r in results:
    boxes = r.boxes
    filtered_boxes = [box for box in boxes if box.cls == 1]
    r.boxes = filtered_boxes

annotated_image = results[0].plot()
annotated_image_resized = cv.resize(annotated_image, (1920, 1080)) # you can change the Resolution
cv.imshow("Plate Detection", annotated_image_resized)

cv.waitKey(0)
cv.destroyAllWindows()
