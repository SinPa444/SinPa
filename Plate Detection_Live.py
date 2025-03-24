from ultralytics import YOLO
import cv2 as cv


model = YOLO("S:/AI/projects/car & plate detector5/weights/best.pt")
cap = cv.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("connection lost!")
        break

    results = model.predict(source=frame, save=False)

    for r in results:
        boxes = r.boxes
        filtered_boxes = [box for box in boxes if box.cls == 1]
        r.boxes = filtered_boxes

    annotated_frame = results[0].plot()
    cv.imshow("Plate Detection", annotated_frame)

    if cv.waitkey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()