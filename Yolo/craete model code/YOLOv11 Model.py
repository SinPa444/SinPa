from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("") # path your YOLO Model
    model.train(
        data="", # path your Data.yaml
        epochs=150,
        imgsz=640,
        batch=4,
        patience=20,
        device=0,
        workers=2,
        optimizer="AdamW",
        lr0=0.001,
        save_period=10,
        verbose=True,
        project="S:/AI/projects",
        name='car & plate detector',
        cache=True,
    )
    print("train finished")

