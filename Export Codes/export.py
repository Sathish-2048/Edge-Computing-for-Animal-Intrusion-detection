from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("C:\\Users\\Sathish\\Animal intrusion.v3i.yolov8\\runs\detect\\train2\weights\\best.pt")

# Export the model to NCNN format
model.export(format="tflite")  # creates 'yolo11n_ncnn_model'

# Load the exported NCNN model
#ncnn_model = YOLO("yolov8n_ncnn_model")

# Run inference
#results = ncnn_model("https://tse1.mm.bing.net/th?id=OIP.P22fbFTVcQylpkAuiARTsQHaE8&pid=Api&P=0&h=180")