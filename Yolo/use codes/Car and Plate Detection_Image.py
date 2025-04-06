from ultralytics import YOLO
import cv2 as cv


model = YOLO("") # when training finished path your best.pt
img = cv.imread("") # path the image
results = model.predict(source=img, save=False)
annotated_images = results[0].plot()
annotated_images_resized = cv.resize(annotated_images, (1920, 1080))
cv.imshow("Detection", annotated_images_resized)


cv.waitKey(0)
cv.destroyAllWindows()
