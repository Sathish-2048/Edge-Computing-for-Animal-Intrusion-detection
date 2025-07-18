import os
import torch
import onnx
from ultralytics import YOLO
import tf2onnx
import tensorflow as tf

# Step 1: Load YOLOv8 model
pt_model_path = r"C:\Users\Sathish\Animal intrusion.v3i.yolov8\runs\detect\train2\weights\best.pt"
print("Loading YOLOv8 model...")
model = YOLO(pt_model_path)

# Step 2: Export to ONNX
print("Exporting to ONNX...")
onnx_path = "best.onnx"
model.export(format="onnx", opset=12)

# Step 3: Convert ONNX to TensorFlow SavedModel
print("Converting ONNX to TensorFlow SavedModel...")
saved_model_dir = "saved_model"
os.system(f"python -m tf2onnx.convert --opset 12 --onnx {onnx_path} --output {saved_model_dir} --saved-model-output")

# Step 4: Convert TensorFlow SavedModel to TFLite
print("Converting TensorFlow SavedModel to TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optional quantization
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = "best.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ Conversion complete! TFLite model saved at: {tflite_model_path}")
print("📌 Now transfer it to your Raspberry Pi and compile using: edgetpu_compiler best.tflite")
