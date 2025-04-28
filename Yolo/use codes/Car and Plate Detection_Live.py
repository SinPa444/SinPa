from ultralytics import YOLO
import cv2 as cv


model = YOLO("") # when training finished path your best.pt
cap = cv.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, save=False)
    annotated_frame = results[0].plot()
    cv.imshow("Detection", annotated_frame)
    if cv.waitkey(1) & 0xFF == ord('q'):
        break

cap.relese()
cv.destroyAllWindows()
