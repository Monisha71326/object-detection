import cv2
import gradio as gr
import numpy as np
from yolov8 import YOLOv8

# Initialize yolov8 object detector
model_path = "models/yolov8m.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)

def detect_objects(input_image):
    # Convert RGB (Gradio gives RGB) to BGR (OpenCV format)
    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    # Detect Objects
    boxes, scores, class_ids = yolov8_detector(img_bgr)

    # Draw detections
    combined_img = yolov8_detector.draw_detections(img_bgr)

    # Convert back to RGB for Gradio display
    output_img = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)
    return output_img

demo = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
    outputs=gr.Image(type="numpy", label="Detected Objects"),
    title="YOLOv8 Object Detection",
    description="Upload an image to detect objects using YOLOv8 (ONNX model)."
)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
