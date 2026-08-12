import cv2
import gradio as gr
import numpy as np
import os
import threading
import urllib.request
from yolov8 import YOLOv8


# ---------------- MODEL SETUP ---------------- #

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

    yolov8_detector = YOLOv8(
        model_path,
        conf_thres=0.2,
        iou_thres=0.3
    )

    model_ready = True

    print("Model ready.")


# Load model in background
threading.Thread(
    target=load_model,
    daemon=True
).start()


# ---------------- IMAGE DETECTION ---------------- #

def detect_image(input_image):

    if not model_ready:
        return input_image

    img_bgr = cv2.cvtColor(
        input_image,
        cv2.COLOR_RGB2BGR
    )

    yolov8_detector(img_bgr)

    combined_img = yolov8_detector.draw_detections(
        img_bgr
    )

    return cv2.cvtColor(
        combined_img,
        cv2.COLOR_BGR2RGB
    )


# ---------------- VIDEO DETECTION ---------------- #

def detect_video(video_path):

    if not model_ready:
        return video_path

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 24

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    # Resize large videos
    scale = 1.0

    if width > 640:
        scale = 640 / width

    new_width = int(width * scale)
    new_height = int(height * scale)

    output_path = "output.mp4"

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (new_width, new_height)
    )

    frame_skip = 2

    # Process maximum 15 seconds
    max_frames = int(fps * 15)

    frame_count = 0

    last_detected = None

    while cap.isOpened() and frame_count < max_frames:

        ret, frame = cap.read()

        if not ret:
            break

        frame_resized = cv2.resize(
            frame,
            (new_width, new_height)
        )

        if frame_count % frame_skip == 0:

            yolov8_detector(frame_resized)

            combined = yolov8_detector.draw_detections(
                frame_resized
            )

            last_detected = combined

        else:

            combined = (
                last_detected
                if last_detected is not None
                else frame_resized
            )

        out.write(combined)

        frame_count += 1

    cap.release()
    out.release()

    return output_path


# ---------------- CUSTOM DESIGN ---------------- #

custom_css = """
:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --bg-dark: #0f0f1a;
    --card-bg: #1a1a2e;
    --text-light: #e5e5f0;
    --accent: #22d3ee;
}

.gradio-container {
    background: linear-gradient(
        160deg,
        #0f0f1a 0%,
        #16213e 100%
    ) !important;

    font-family: 'Segoe UI', 'Poppins', sans-serif !important;
}

#app-header {
    text-align: center;
    padding: 28px 20px 18px 20px;
}

#app-header h1 {
    font-size: 2.1rem;
    font-weight: 800;

    background: linear-gradient(
        90deg,
        #6366f1,
        #22d3ee
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    margin-bottom: 6px;
}

#app-header p {
    color: #a1a1c2;
    font-size: 0.95rem;
}

.gr-tabs {
    border-radius: 16px !important;
}

.tabitem,
.tabs {
    background: transparent !important;
}

button.primary {
    background: linear-gradient(
        90deg,
        #6366f1,
        #4f46e5
    ) !important;

    border: none !important;
    color: white !important;

    font-weight: 600 !important;

    border-radius: 10px !important;

    box-shadow: 0 4px 14px rgba(
        99,
        102,
        241,
        0.35
    );
}

button.primary:hover {
    box-shadow: 0 6px 20px rgba(
        99,
        102,
        241,
        0.55
    );

    transform: translateY(-1px);
}

.block {
    border-radius: 16px !important;

    background: #1a1a2e !important;

    border: 1px solid #2a2a45 !important;
}

footer {
    display: none !important;
}
"""


# ---------------- HEADER ---------------- #

header_html = """
<div id="app-header">

    <h1>🎯 Real-Time Object Detection</h1>

    <p>
        YOLOv8-powered detection for images and
        short video clips — upload and see it work instantly.
    </p>

</div>
"""


# ---------------- GRADIO THEME ---------------- #

# IMPORTANT:
# Do NOT use font=["Poppins", "Segoe UI", "sans-serif"]
# because it causes the Gradio 6.x error:
# AttributeError: 'str' object has no attribute 'name'

theme = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
).set(
    body_background_fill="#0f0f1a",
    block_background_fill="#1a1a2e",
    block_border_color="#2a2a45",
    body_text_color="#e5e5f0",
    input_background_fill="#12121f",
)


# ---------------- GRADIO APP ---------------- #

with gr.Blocks(
    css=custom_css,
    theme=theme,
    title="Object Detection • YOLOv8"
) as demo:

    gr.HTML(header_html)

    # ---------------- TABS ---------------- #

    with gr.Tabs():

        # ---------------- IMAGE TAB ---------------- #

        with gr.TabItem("🖼️ Image"):

            with gr.Row():

                with gr.Column():

                    image_input = gr.Image(
                        label="Upload an Image",
                        type="numpy"
                    )

                    image_btn = gr.Button(
                        "Detect Objects",
                        variant="primary"
                    )

                with gr.Column():

                    image_output = gr.Image(
                        label="Detected Objects"
                    )

            image_btn.click(
                fn=detect_image,
                inputs=image_input,
                outputs=image_output
            )

            gr.Examples(
                examples=[],
                inputs=image_input
            )

        # ---------------- VIDEO TAB ---------------- #

        with gr.TabItem("🎬 Video"):

            with gr.Row():

                with gr.Column():

                    video_input = gr.Video(
                        label="Upload a Video (first 15s processed)"
                    )

                    video_btn = gr.Button(
                        "Detect Objects",
                        variant="primary"
                    )

                with gr.Column():

                    video_output = gr.Video(
                        label="Detected Objects Video"
                    )

            video_btn.click(
                fn=detect_video,
                inputs=video_input,
                outputs=video_output
            )

    # ---------------- FOOTER ---------------- #

    gr.HTML(
        """
        <p style="
            text-align:center;
            color:#6b6b8c;
            font-size:0.8rem;
            margin-top:20px;
        ">
            Built with YOLOv8 + Gradio · by Monisha D
        </p>
        """
    )


# ---------------- RENDER START ---------------- #

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(
            os.environ.get("PORT", 7860)
        )
    )
