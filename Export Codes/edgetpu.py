from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("C:\\Users\\Sathish\\Animal intrusion.v3i.yolov8\\runs\\detect\\train2\\weights\\best.pt")
# Export the model to TFLite Edge TPU format
model.export(format="edgetpu")  # creates 'yolo11n_full_integer_quant_edgetpu.tflite'

# Load the exported TFLite Edge TPU model
#edgetpu_model = YOLO("yolo11n_full_integer_quant_edgetpu.tflite")

# Run inference
#results = edgetpu_model("https://ultralytics.com/images/bus.jpg")