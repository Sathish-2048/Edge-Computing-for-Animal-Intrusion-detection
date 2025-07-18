from ultralytics import YOLO

# Load YOLOv8n model
model = YOLO("yolov8n.pt")

# Train with computed class weights
results = model.train(
    data="data.yaml",
    epochs=50,
    patience=10,
    imgsz=320,
    save=True,
)
