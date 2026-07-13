import cv2
import gradio as gr
import numpy as np
import os
import threading
import urllib.request
from yolov8 import YOLOv8

model_path = "models/yolov8n.onnx"
model_ready = False
yolov8_detector = None

def load_model():
    global yolov8_detector, model_ready
    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        print("Downloading YOLOv8 model...")
        url = "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx"
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded successfully.")
    yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)
    model_ready = True
    print("Model ready.")

threading.Thread(target=load_model, daemon=True).start()

def detect_image(input_image):
    if not model_ready:
        return input_image
    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    yolov8_detector(img_bgr)
    combined_img = yolov8_detector.draw_detections(img_bgr)
    return cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)

def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = 1.0
    if width > 640:
        scale = 640 / width
    new_width = int(width * scale)
    new_height = int(height * scale)

    output_path = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    frame_skip = 2
    max_frames = int(fps * 15)
    frame_count = 0
    last_detected = None

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (new_width, new_height))
        if frame_count % frame_skip == 0:
            yolov8_detector(frame_resized)
            combined = yolov8_detector.draw_detections(frame_resized)
            last_detected = combined
        else:
            combined = last_detected if last_detected is not None else frame_resized
        out.write(combined)
        frame_count += 1

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
    inputs=gr.Image(type="numpy", sources=["webcam"], streaming=True, label="Live Webcam"),
    outputs=gr.Image(type="numpy", label="Detected Objects"),
    live=True,
)

video_tab = gr.Interface(
    fn=detect_video,
    inputs=gr.Video(label="Upload a Video (first 15s processed)"),
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
