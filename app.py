import cv2
import gradio as gr
import numpy as np
import os
import urllib.request
from yolov8 import YOLOv8

model_path = "models/yolov8n.onnx"

# Download model if it doesn't exist
if not os.path.exists(model_path):
    os.makedirs("models", exist_ok=True)
    print("Downloading YOLOv8 model...")
   url = "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8m.onnx"
    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded successfully.")

# Initialize yolov8 object detector
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)

def detect_image(input_image):
    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    yolov8_detector(img_bgr)
    combined_img = yolov8_detector.draw_detections(img_bgr)
    return cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)

def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yolov8_detector(frame)
        combined = yolov8_detector.draw_detections(frame)
        out.write(combined)

    cap.release()
    out.release()
    return output_path

image_tab = gr.Interface(
    fn=detect_image,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
    outputs=gr.Image(type="numpy", label="Detected Objects"),
)

webcam_tab = gr.Interface(
    fn=detect_image,
    inputs=gr.Image(type="numpy", sources=["webcam"], label="Capture from Webcam"),
    outputs=gr.Image(type="numpy", label="Detected Objects"),
)

video_tab = gr.Interface(
    fn=detect_video,
    inputs=gr.Video(label="Upload a Video"),
    outputs=gr.Video(label="Detected Objects Video"),
)

demo = gr.TabbedInterface(
    [image_tab, webcam_tab, video_tab],
    tab_names=["Image", "Webcam", "Video"],
    title="YOLOv8 Object Detection"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
