from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model

# Train the model
results = model.train(data="datasets/microvoc/data.yaml", epochs=100, imgsz=640, workers=0, freeze=10)
