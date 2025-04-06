from ultralytics import YOLO
import cv2 as cv


model = YOLO("S:/AI/projects/car & plate detector5/weights/best.pt")
cap = cv.VideoCapture("S:/AI/Datasets/Videos for test/VID_20250322_180642.mp4") # path the video

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, save=False)
    annotated_frame = results[0].plot()
    annotated_frame_resized = cv.resize(annotated_frame, (1080, 1920))
    cv.imshow("Detection", annotated_frame_resized)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()