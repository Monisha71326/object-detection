import cv2
import gradio as gr
import numpy as np
import os
import threading
import urllib.request
import tempfile

from yolov8 import YOLOv8


# =========================================================
# MODEL SETUP
# =========================================================

MODEL_PATH = "models/yolov8n.onnx"

model_ready = False
yolov8_detector = None


def load_model():
    global yolov8_detector, model_ready

    try:
        # Create models folder
        os.makedirs("models", exist_ok=True)

        # Download model if it does not exist
        if not os.path.exists(MODEL_PATH):

            print("Downloading YOLOv8 model...")

            url = (
                "https://huggingface.co/Kalray/yolov8/"
                "resolve/main/yolov8n.onnx"
            )

            urllib.request.urlretrieve(
                url,
                MODEL_PATH
            )

            print("Model downloaded successfully.")

        # Load YOLOv8 ONNX model
        yolov8_detector = YOLOv8(
            MODEL_PATH,
            conf_thres=0.2,
            iou_thres=0.3
        )

        model_ready = True

        print("YOLOv8 model ready.")

    except Exception as e:
        print("Model loading error:", e)
        model_ready = False


# Load model in background
threading.Thread(
    target=load_model,
    daemon=True
).start()


# =========================================================
# IMAGE DETECTION
# =========================================================

def detect_image(input_image):

    if input_image is None:
        return None

    if not model_ready:
        print("Model is still loading...")
        return input_image

    try:

        # RGB -> BGR
        img_bgr = cv2.cvtColor(
            input_image,
            cv2.COLOR_RGB2BGR
        )

        # Run detection
        yolov8_detector(img_bgr)

        # Draw detections
        combined_img = yolov8_detector.draw_detections(
            img_bgr
        )

        # BGR -> RGB
        result = cv2.cvtColor(
            combined_img,
            cv2.COLOR_BGR2RGB
        )

        return result

    except Exception as e:

        print("Image detection error:", e)

        return input_image


# =========================================================
# VIDEO DETECTION
# =========================================================

def detect_video(video_path):

    if video_path is None:
        return None

    if not model_ready:
        print("Model is still loading...")
        return video_path

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open video.")
        return None

    try:

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 24

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        if width <= 0 or height <= 0:
            cap.release()
            return None

        # ---------------------------------------------
        # Resize large videos
        # ---------------------------------------------

        scale = 1.0

        if width > 640:
            scale = 640 / width

        new_width = int(width * scale)
        new_height = int(height * scale)

        # Make dimensions even
        new_width = new_width - (new_width % 2)
        new_height = new_height - (new_height % 2)

        # ---------------------------------------------
        # Temporary output file
        # ---------------------------------------------

        temp_file = tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False
        )

        output_path = temp_file.name
        temp_file.close()

        # ---------------------------------------------
        # Video writer
        # ---------------------------------------------

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (
                new_width,
                new_height
            )
        )

        # Process every 2nd frame
        frame_skip = 2

        # Maximum 15 seconds
        max_frames = int(
            fps * 15
        )

        frame_count = 0

        last_detected = None

        # ---------------------------------------------
        # Process video
        # ---------------------------------------------

        while (
            cap.isOpened()
            and frame_count < max_frames
        ):

            ret, frame = cap.read()

            if not ret:
                break

            frame_resized = cv2.resize(
                frame,
                (
                    new_width,
                    new_height
                )
            )

            # Run YOLO every 2nd frame
            if frame_count % frame_skip == 0:

                yolov8_detector(
                    frame_resized
                )

                combined = (
                    yolov8_detector.draw_detections(
                        frame_resized
                    )
                )

                last_detected = combined

            else:

                if last_detected is not None:
                    combined = last_detected
                else:
                    combined = frame_resized

            out.write(combined)

            frame_count += 1

        cap.release()
        out.release()

        return output_path

    except Exception as e:

        print("Video detection error:", e)

        cap.release()

        return None


# =========================================================
# CUSTOM CSS
# =========================================================

custom_css = """

:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --bg-dark: #0f0f1a;
    --card-bg: #1a1a2e;
    --text-light: #e5e5f0;
    --accent: #22d3ee;
}


/* Main background */

.gradio-container {

    background:
        linear-gradient(
            160deg,
            #0f0f1a 0%,
            #16213e 100%
        ) !important;

    font-family:
        'Segoe UI',
        'Poppins',
        sans-serif !important;
}


/* Header */

#app-header {

    text-align: center;

    padding:
        28px
        20px
        18px
        20px;
}


#app-header h1 {

    font-size: 2.1rem;

    font-weight: 800;

    background:
        linear-gradient(
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


/* Tabs */

.gr-tabs {

    border-radius: 16px !important;
}


.tabitem,
.tabs {

    background:
        transparent !important;
}


/* Primary button */

button.primary {

    background:
        linear-gradient(
            90deg,
            #6366f1,
            #4f46e5
        ) !important;

    border: none !important;

    color: white !important;

    font-weight: 600 !important;

    border-radius: 10px !important;

    box-shadow:
        0 4px 14px
        rgba(
            99,
            102,
            241,
            0.35
        );
}


button.primary:hover {

    box-shadow:
        0 6px 20px
        rgba(
            99,
            102,
            241,
            0.55
        );

    transform:
        translateY(-1px);
}


/* Share button */

#share-result-btn {

    margin-top: 10px;

    border-radius: 10px !important;

    font-weight: 600 !important;
}


/* Cards */

.block {

    border-radius: 16px !important;

    background:
        #1a1a2e !important;

    border:
        1px solid
        #2a2a45 !important;
}


/* Footer */

footer {

    display: none !important;
}

"""


# =========================================================
# HEADER
# =========================================================

header_html = """

<div id="app-header">

    <h1>
        🎯 Real-Time Object Detection
    </h1>

    <p>
        YOLOv8-powered detection for images and
        short video clips — upload and see it work instantly.
    </p>

</div>

"""


# =========================================================
# GRADIO THEME
# =========================================================

theme = gr.themes.Base(

    primary_hue="indigo",

    secondary_hue="cyan",

    neutral_hue="slate"

).set(

    body_background_fill="#0f0f1a",

    block_background_fill="#1a1a2e",

    block_border_color="#2a2a45",

    body_text_color="#e5e5f0",

    input_background_fill="#12121f"

)


# =========================================================
# GRADIO APP
# =========================================================

with gr.Blocks(
    title="Object Detection • YOLOv8"
) as demo:

    gr.HTML(
        header_html
    )


    # =====================================================
    # TABS
    # =====================================================

    with gr.Tabs():


        # =================================================
        # IMAGE TAB
        # =================================================

        with gr.TabItem("🖼️ Image"):

            with gr.Row():


                # -----------------------------------------
                # INPUT
                # -----------------------------------------

                with gr.Column():

                    image_input = gr.Image(
                        label="Upload an Image",
                        type="numpy",
                        buttons=[
                            "fullscreen"
                        ]
                    )

                    image_btn = gr.Button(
                        "Detect Objects",
                        variant="primary"
                    )


                # -----------------------------------------
                # OUTPUT
                # -----------------------------------------

                with gr.Column():

                    image_output = gr.Image(
                        label="Detected Objects",
                        elem_id="detected-image",
                        buttons=[
                            "download",
                            "fullscreen"
                        ]
                    )

                    # Custom working share button
                    share_btn = gr.Button(
                        "🔗 Share Result",
                        variant="secondary",
                        elem_id="share-result-btn"
                    )


            # ---------------------------------------------
            # DETECT IMAGE
            # ---------------------------------------------

            image_btn.click(
                fn=detect_image,
                inputs=image_input,
                outputs=image_output
            )


            # ---------------------------------------------
            # CUSTOM IMAGE SHARE
            # ---------------------------------------------

            share_btn.click(

                fn=None,

                inputs=None,

                outputs=None,

                js="""

                async () => {

                    try {

                        /*
                         * Find detected image
                         */

                        const container =
                            document.querySelector(
                                '#detected-image'
                            );

                        if (!container) {

                            alert(
                                'Please detect an image first.'
                            );

                            return;
                        }


                        const img =
                            container.querySelector(
                                'img'
                            );


                        if (!img || !img.src) {

                            alert(
                                'Please detect an image first.'
                            );

                            return;
                        }


                        /*
                         * Get image URL
                         */

                        const imageUrl =
                            img.src;


                        /*
                         * Try browser native
                         * file sharing
                         */

                        if (
                            navigator.share &&
                            navigator.canShare
                        ) {

                            try {

                                const response =
                                    await fetch(
                                        imageUrl
                                    );

                                const blob =
                                    await response.blob();


                                const file =
                                    new File(
                                        [
                                            blob
                                        ],
                                        'yolov8-detection.png',
                                        {
                                            type:
                                                blob.type ||
                                                'image/png'
                                        }
                                    );


                                if (
                                    navigator.canShare(
                                        {
                                            files: [
                                                file
                                            ]
                                        }
                                    )
                                ) {

                                    await navigator.share(
                                        {
                                            title:
                                                'YOLOv8 Object Detection',

                                            text:
                                                'Detected using my YOLOv8 Object Detection app.',

                                            files: [
                                                file
                                            ]
                                        }
                                    );

                                    return;
                                }

                            } catch (fileError) {

                                console.log(
                                    'File share unavailable:',
                                    fileError
                                );
                            }
                        }


                        /*
                         * Desktop / unsupported browser
                         * fallback
                         */

                        if (
                            navigator.clipboard &&
                            navigator.clipboard.writeText
                        ) {

                            await navigator.clipboard.writeText(
                                window.location.href
                            );

                            alert(
                                'App link copied successfully!'
                            );

                        } else {

                            alert(
                                'Sharing is not supported in this browser.'
                            );
                        }

                    } catch (error) {

                        console.error(
                            'Share error:',
                            error
                        );

                        alert(
                            'Unable to share. Please use the Download button.'
                        );
                    }

                }

                """
            )


            # ---------------------------------------------
            # EXAMPLES
            # ---------------------------------------------

            gr.Examples(
                examples=[],
                inputs=image_input
            )


        # =================================================
        # VIDEO TAB
        # =================================================

        with gr.TabItem("🎬 Video"):

            with gr.Row():


                # -----------------------------------------
                # VIDEO INPUT
                # -----------------------------------------

                with gr.Column():

                    video_input = gr.Video(
                        label="Upload a Video (first 15s processed)"
                    )

                    video_btn = gr.Button(
                        "Detect Objects",
                        variant="primary"
                    )


                # -----------------------------------------
                # VIDEO OUTPUT
                # -----------------------------------------

                with gr.Column():

                    video_output = gr.Video(
                        label="Detected Objects Video"
                    )


            # ---------------------------------------------
            # VIDEO DETECTION
            # ---------------------------------------------

            video_btn.click(
                fn=detect_video,
                inputs=video_input,
                outputs=video_output
            )


    # =====================================================
    # FOOTER
    # =====================================================

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


# =========================================================
# RENDER START
# =========================================================

if __name__ == "__main__":

    demo.launch(

        server_name="0.0.0.0",

        server_port=int(
            os.environ.get(
                "PORT",
                7860
            )
        ),

        css=custom_css,

        theme=theme

    )
