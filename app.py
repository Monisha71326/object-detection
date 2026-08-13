import cv2
import gradio as gr
import numpy as np
import pandas as pd
import os
import csv
import json
import time
import threading
import urllib.request
import tempfile
import zipfile
from datetime import datetime
from collections import Counter

from yolov8 import YOLOv8


# =========================================================
# MODEL SETUP
# =========================================================

MODEL_VARIANTS = {
    "Nano (fastest)": {
        "path": "models/yolov8n.onnx",
        "url": "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx",
    },
    "Small (balanced)": {
        "path": "models/yolov8s.onnx",
        "url": "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8s.onnx",
    },
}

current_variant = "Nano (fastest)"
model_ready = False
model_loading_status = "Initializing..."
yolov8_detector = None

# Session-wide state
detection_history = []          # last N annotated images
cumulative_counts = Counter()   # running tally across the whole session


def load_model(variant_name="Nano (fastest)"):
    global yolov8_detector, model_ready, model_loading_status, current_variant

    model_ready = False
    current_variant = variant_name
    info = MODEL_VARIANTS[variant_name]
    model_path = info["path"]

    try:
        os.makedirs("models", exist_ok=True)

        if not os.path.exists(model_path):
            model_loading_status = f"Downloading {variant_name} model..."
            print(model_loading_status)
            urllib.request.urlretrieve(info["url"], model_path)
            print("Model downloaded successfully.")

        model_loading_status = f"Loading {variant_name} model into memory..."

        yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)

        model_ready = True
        model_loading_status = f"{variant_name} ready ✅"
        print("YOLOv8 model ready:", variant_name)

    except Exception as e:
        model_loading_status = f"Model loading error: {e}"
        print(model_loading_status)
        model_ready = False


threading.Thread(target=load_model, args=(current_variant,), daemon=True).start()


def switch_model(variant_name):
    threading.Thread(target=load_model, args=(variant_name,), daemon=True).start()
    return f"🟡 Switching to {variant_name}..."


def get_class_names():
    if yolov8_detector is None:
        return []
    for attr in ("class_names", "classes", "CLASS_NAMES"):
        names = getattr(yolov8_detector, attr, None)
        if names:
            return list(names)
    return []


def model_status_text():
    if model_ready:
        return f"🟢 {current_variant} ready"
    return f"🟡 {model_loading_status}"


# =========================================================
# CORE DETECTION (shared by image / webcam / batch)
# =========================================================

def run_detection(img_rgb, conf_thres, iou_thres, class_filter):
    """Returns (annotated_rgb_image, counts_dict)"""

    yolov8_detector.conf_threshold = conf_thres
    yolov8_detector.iou_threshold = iou_thres

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    boxes, scores, class_ids = yolov8_detector(img_bgr)

    all_names = get_class_names()

    if class_filter and all_names:
        keep_idx = [
            i for i, cid in enumerate(class_ids)
            if 0 <= cid < len(all_names) and all_names[cid] in class_filter
        ]
        if len(keep_idx) != len(class_ids):
            yolov8_detector.boxes = boxes[keep_idx] if len(keep_idx) else np.array([])
            yolov8_detector.scores = scores[keep_idx] if len(keep_idx) else np.array([])
            yolov8_detector.class_ids = class_ids[keep_idx] if len(keep_idx) else np.array([])
            class_ids = yolov8_detector.class_ids

    combined_img = yolov8_detector.draw_detections(img_bgr)
    result = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)

    counts = Counter()
    for cid in class_ids:
        label = all_names[cid] if 0 <= cid < len(all_names) else f"class_{cid}"
        counts[label] += 1

    return result, counts


def write_reports(counts, conf_thres, iou_thres):
    """Writes JSON + CSV report, returns (json_path, csv_path)"""
    total = sum(counts.values())

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_objects": total,
        "counts": dict(counts),
        "confidence_threshold": conf_thres,
        "iou_threshold": iou_thres,
        "model": current_variant,
    }

    json_path = os.path.join(tempfile.gettempdir(), "detection_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    csv_path = os.path.join(tempfile.gettempdir(), "detection_report.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "count"])
        for label, c in counts.most_common():
            writer.writerow([label, c])

    return json_path, csv_path


# =========================================================
# IMAGE DETECTION
# =========================================================

def detect_image(input_image, conf_thres, iou_thres, class_filter):

    if input_image is None:
        return None, "Upload an image to get started.", None, None

    if not model_ready:
        return input_image, f"⏳ {model_loading_status}", None, None

    try:
        result, counts = run_detection(input_image, conf_thres, iou_thres, class_filter)

        cumulative_counts.update(counts)
        total = sum(counts.values())

        if total == 0:
            stats_md = "**No objects detected.** Try lowering the confidence threshold."
        else:
            lines = [f"### 🔎 Detected {total} object(s)\n"]
            for label, c in counts.most_common():
                lines.append(f"- **{label}**: {c}")
            stats_md = "\n".join(lines)

        detection_history.insert(0, result)
        detection_history[:] = detection_history[:12]

        json_path, csv_path = write_reports(counts, conf_thres, iou_thres)

        return result, stats_md, json_path, csv_path

    except Exception as e:
        print("Image detection error:", e)
        return input_image, f"⚠️ Error: {e}", None, None


# =========================================================
# BATCH IMAGE DETECTION
# =========================================================

def detect_batch(files, conf_thres, iou_thres, class_filter, progress=gr.Progress()):

    if not files:
        return [], "Upload one or more images.", None

    if not model_ready:
        return [], f"⏳ {model_loading_status}", None

    results = []
    total_counts = Counter()
    tmp_dir = tempfile.mkdtemp()

    for i, file in enumerate(files):
        progress((i + 1) / len(files), desc=f"Processing {i + 1}/{len(files)}")
        try:
            img_bgr = cv2.imread(file.name)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            annotated, counts = run_detection(img_rgb, conf_thres, iou_thres, class_filter)
            total_counts.update(counts)
            cumulative_counts.update(counts)

            results.append(annotated)

            out_path = os.path.join(tmp_dir, f"detected_{i}.png")
            cv2.imwrite(out_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"Batch item {i} error:", e)

    zip_path = os.path.join(tempfile.gettempdir(), "batch_detections.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for fname in os.listdir(tmp_dir):
            zf.write(os.path.join(tmp_dir, fname), fname)

    total = sum(total_counts.values())
    lines = [f"### 🔎 {len(results)} image(s) processed · {total} object(s) total\n"]
    for label, c in total_counts.most_common():
        lines.append(f"- **{label}**: {c}")

    detection_history[:0] = results
    detection_history[:] = detection_history[:12]

    return results, "\n".join(lines), zip_path


# =========================================================
# VIDEO DETECTION
# =========================================================

def detect_video(video_path, conf_thres, iou_thres, progress=gr.Progress()):

    if video_path is None:
        return None, "Upload a video to get started."

    if not model_ready:
        return video_path, f"⏳ {model_loading_status}"

    yolov8_detector.conf_threshold = conf_thres
    yolov8_detector.iou_threshold = iou_thres

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, "⚠️ Could not open video."

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 24

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            cap.release()
            return None, "⚠️ Invalid video dimensions."

        scale = 1.0
        if width > 640:
            scale = 640 / width

        new_width = int(width * scale)
        new_height = int(height * scale)
        new_width = new_width - (new_width % 2)
        new_height = new_height - (new_height % 2)

        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output_path = temp_file.name
        temp_file.close()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

        frame_skip = 2
        max_frames = int(fps * 15)
        frame_count = 0
        last_detected = None
        video_counts = Counter()
        all_names = get_class_names()

        start_time = time.time()

        while cap.isOpened() and frame_count < max_frames:

            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, (new_width, new_height))

            if frame_count % frame_skip == 0:
                boxes, scores, class_ids = yolov8_detector(frame_resized)
                combined = yolov8_detector.draw_detections(frame_resized)
                last_detected = combined
                for cid in class_ids:
                    label = all_names[cid] if 0 <= cid < len(all_names) else f"class_{cid}"
                    video_counts[label] += 1
            else:
                combined = last_detected if last_detected is not None else frame_resized

            out.write(combined)
            frame_count += 1

            progress(
                min(frame_count / max_frames, 1.0),
                desc=f"Processing frame {frame_count}/{max_frames}"
            )

        cap.release()
        out.release()

        cumulative_counts.update(video_counts)

        elapsed = time.time() - start_time
        proc_fps = frame_count / elapsed if elapsed > 0 else 0

        top_classes = ", ".join(f"{k} ({v})" for k, v in video_counts.most_common(5))

        status = (
            f"✅ Processed {frame_count} frames in {elapsed:.1f}s (~{proc_fps:.1f} fps)\n\n"
            f"Top classes: {top_classes or 'none'}"
        )

        return output_path, status

    except Exception as e:
        print("Video detection error:", e)
        cap.release()
        return None, f"⚠️ Error: {e}"


def clear_history():
    detection_history.clear()
    return []


def get_analytics_df():
    if not cumulative_counts:
        return pd.DataFrame({"class": [], "count": []})
    df = pd.DataFrame(cumulative_counts.most_common(15), columns=["class", "count"])
    return df


def reset_analytics():
    cumulative_counts.clear()
    return get_analytics_df()


# =========================================================
# CUSTOM CSS — trendy glassmorphism + gradient theme
# =========================================================

custom_css = """

:root {
    --primary: #7c5cff;
    --primary-dark: #5b3df0;
    --bg-dark: #0b0b16;
    --card-bg: rgba(26, 26, 46, 0.6);
    --text-light: #eaeaf6;
    --accent: #22d3ee;
    --accent-2: #ff6ec7;
}

.gradio-container {
    background:
        radial-gradient(circle at 15% 10%, rgba(124,92,255,0.20), transparent 40%),
        radial-gradient(circle at 85% 0%, rgba(34,211,238,0.18), transparent 45%),
        linear-gradient(160deg, #0b0b16 0%, #14142a 100%) !important;
    font-family: 'Segoe UI', 'Poppins', sans-serif !important;
    transition: background 0.4s ease-in-out;
}

.light-mode.gradio-container {
    background:
        radial-gradient(circle at 15% 10%, rgba(124,92,255,0.10), transparent 40%),
        linear-gradient(160deg, #f5f5fb 0%, #eceafc 100%) !important;
}

#app-header { text-align: center; padding: 32px 20px 14px 20px; }

#app-header h1 {
    font-size: 2.3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #7c5cff, #22d3ee, #ff6ec7);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 6s linear infinite;
    margin-bottom: 6px;
}

@keyframes shine { to { background-position: 200% center; } }

#app-header p { color: #a1a1c2; font-size: 0.95rem; }

#status-badge {
    display: inline-block;
    margin-top: 10px;
    padding: 6px 16px;
    border-radius: 999px;
    background: rgba(124, 92, 255, 0.15);
    border: 1px solid rgba(124, 92, 255, 0.4);
    font-size: 0.85rem;
    color: #cfcfe8;
}

.gr-tabs { border-radius: 16px !important; }
.tabitem, .tabs { background: transparent !important; }

button.primary {
    background: linear-gradient(90deg, #7c5cff, #5b3df0) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 18px rgba(124, 92, 255, 0.4);
    transition: all 0.2s ease-in-out !important;
}

button.primary:hover {
    box-shadow: 0 8px 24px rgba(124, 92, 255, 0.6);
    transform: translateY(-2px) scale(1.01);
}

button.secondary {
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease-in-out !important;
}

button.secondary:hover { transform: translateY(-2px); }

#share-result-btn { margin-top: 10px; }

.block {
    border-radius: 18px !important;
    background: var(--card-bg) !important;
    border: 1px solid rgba(124, 92, 255, 0.18) !important;
    backdrop-filter: blur(10px);
    transition: border 0.2s ease-in-out;
}

.block:hover { border: 1px solid rgba(124, 92, 255, 0.4) !important; }

#stats-panel { padding: 14px 18px; }

footer { display: none !important; }

"""


header_html = """
<div id="app-header">
    <h1>🎯 Real-Time Object Detection</h1>
    <p>YOLOv8-powered detection for images, video, batch uploads and webcam — trendy, fast, and packed with insights.</p>
    <div id="status-badge">⏳ Loading model...</div>
</div>
"""


theme = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate"
).set(
    body_background_fill="#0b0b16",
    block_background_fill="#1a1a2e",
    block_border_color="#2a2a45",
    body_text_color="#eaeaf6",
    input_background_fill="#12121f"
)


# =========================================================
# GRADIO APP
# =========================================================

with gr.Blocks(title="Object Detection • YOLOv8") as demo:

    gr.HTML(header_html)

    with gr.Row():
        status_display = gr.Markdown(model_status_text())
        theme_toggle_btn = gr.Button("🌗 Toggle Light/Dark", size="sm", variant="secondary")
        sound_toggle = gr.Checkbox(label="🔊 Sound alert on detection", value=False)

    demo.load(fn=model_status_text, outputs=status_display, every=2)

    theme_toggle_btn.click(
        fn=None, inputs=None, outputs=None,
        js="""
        () => {
            document.querySelector('.gradio-container').classList.toggle('light-mode');
        }
        """
    )

    with gr.Tabs():

        # =================================================
        # IMAGE TAB
        # =================================================
        with gr.TabItem("🖼️ Image"):

            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="Upload an Image", type="numpy",
                        buttons=["fullscreen"]
                    )

                    with gr.Accordion("⚙️ Detection Settings", open=False):
                        conf_slider = gr.Slider(0.05, 0.9, value=0.2, step=0.05, label="Confidence Threshold")
                        iou_slider = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="IoU Threshold")
                        class_filter = gr.CheckboxGroup(choices=[], value=[], label="Filter Classes (leave empty = all)")
                        refresh_classes_btn = gr.Button("🔄 Load Class List", size="sm")
                        auto_detect = gr.Checkbox(label="⚡ Auto-detect on upload", value=False)

                    image_btn = gr.Button("Detect Objects", variant="primary")

                with gr.Column(scale=1):
                    image_output = gr.Image(
                        label="Detected Objects", elem_id="detected-image",
                        buttons=["download", "fullscreen"]
                    )
                    stats_output = gr.Markdown(elem_id="stats-panel")

                    with gr.Row():
                        share_btn = gr.Button("🔗 Share Result", variant="secondary", elem_id="share-result-btn")
                    with gr.Row():
                        report_json = gr.File(label="📄 JSON Report")
                        report_csv = gr.File(label="📊 CSV Report")

            image_btn.click(
                fn=detect_image,
                inputs=[image_input, conf_slider, iou_slider, class_filter],
                outputs=[image_output, stats_output, report_json, report_csv]
            ).then(
                fn=None, inputs=sound_toggle, outputs=None,
                js="""
                (playSound) => {
                    if (playSound) {
                        const ctx = new (window.AudioContext || window.webkitAudioContext)();
                        const osc = ctx.createOscillator();
                        osc.type = 'sine';
                        osc.frequency.setValueAtTime(880, ctx.currentTime);
                        osc.connect(ctx.destination);
                        osc.start();
                        osc.stop(ctx.currentTime + 0.15);
                    }
                }
                """
            )

            image_input.change(
                fn=lambda auto, *args: detect_image(*args) if auto else (None, "", None, None),
                inputs=[auto_detect, image_input, conf_slider, iou_slider, class_filter],
                outputs=[image_output, stats_output, report_json, report_csv]
            )

            refresh_classes_btn.click(
                fn=lambda: gr.update(choices=get_class_names()),
                outputs=class_filter
            )

            share_btn.click(
                fn=None, inputs=None, outputs=None,
                js="""
                async () => {
                    try {
                        const container = document.querySelector('#detected-image');
                        if (!container) { alert('Please detect an image first.'); return; }
                        const img = container.querySelector('img');
                        if (!img || !img.src) { alert('Please detect an image first.'); return; }
                        const imageUrl = img.src;

                        if (navigator.share && navigator.canShare) {
                            try {
                                const response = await fetch(imageUrl);
                                const blob = await response.blob();
                                const file = new File([blob], 'yolov8-detection.png', { type: blob.type || 'image/png' });
                                if (navigator.canShare({ files: [file] })) {
                                    await navigator.share({
                                        title: 'YOLOv8 Object Detection',
                                        text: 'Detected using my YOLOv8 Object Detection app.',
                                        files: [file]
                                    });
                                    return;
                                }
                            } catch (fileError) {
                                console.log('File share unavailable:', fileError);
                            }
                        }

                        if (navigator.clipboard && navigator.clipboard.writeText) {
                            await navigator.clipboard.writeText(window.location.href);
                            alert('App link copied successfully!');
                        } else {
                            alert('Sharing is not supported in this browser.');
                        }
                    } catch (error) {
                        console.error('Share error:', error);
                        alert('Unable to share. Please use the Download button.');
                    }
                }
                """
            )

            gr.Examples(examples=[], inputs=image_input)

        # =================================================
        # BATCH TAB
        # =================================================
        with gr.TabItem("🗂️ Batch"):

            gr.Markdown("Upload multiple images and detect objects across all of them in one go.")

            with gr.Row():
                with gr.Column(scale=1):
                    batch_input = gr.Files(label="Upload Images", file_types=["image"])
                    with gr.Accordion("⚙️ Detection Settings", open=False):
                        b_conf_slider = gr.Slider(0.05, 0.9, value=0.2, step=0.05, label="Confidence Threshold")
                        b_iou_slider = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="IoU Threshold")
                        b_class_filter = gr.CheckboxGroup(choices=[], value=[], label="Filter Classes (leave empty = all)")
                    batch_btn = gr.Button("Detect All", variant="primary")

                with gr.Column(scale=1):
                    batch_gallery = gr.Gallery(label="Results", columns=3, height="auto")
                    batch_stats = gr.Markdown()
                    batch_zip = gr.File(label="📦 Download All (ZIP)")

            batch_btn.click(
                fn=detect_batch,
                inputs=[batch_input, b_conf_slider, b_iou_slider, b_class_filter],
                outputs=[batch_gallery, batch_stats, batch_zip]
            )

        # =================================================
        # VIDEO TAB
        # =================================================
        with gr.TabItem("🎬 Video"):

            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Upload a Video (first 15s processed)")
                    with gr.Accordion("⚙️ Detection Settings", open=False):
                        v_conf_slider = gr.Slider(0.05, 0.9, value=0.2, step=0.05, label="Confidence Threshold")
                        v_iou_slider = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="IoU Threshold")
                    video_btn = gr.Button("Detect Objects", variant="primary")
                    video_status = gr.Markdown()

                with gr.Column():
                    video_output = gr.Video(label="Detected Objects Video")

            video_btn.click(
                fn=detect_video,
                inputs=[video_input, v_conf_slider, v_iou_slider],
                outputs=[video_output, video_status]
            )

        # =================================================
        # WEBCAM TAB
        # =================================================
        with gr.TabItem("📷 Webcam"):

            gr.Markdown("Capture a snapshot from your webcam and run detection instantly.")

            with gr.Row():
                with gr.Column():
                    webcam_input = gr.Image(label="Webcam", sources=["webcam"], type="numpy")
                    with gr.Accordion("⚙️ Detection Settings", open=False):
                        w_conf_slider = gr.Slider(0.05, 0.9, value=0.2, step=0.05, label="Confidence Threshold")
                        w_iou_slider = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="IoU Threshold")
                        w_class_filter = gr.CheckboxGroup(choices=[], value=[], label="Filter Classes (leave empty = all)")
                    webcam_btn = gr.Button("Detect Objects", variant="primary")

                with gr.Column():
                    webcam_output = gr.Image(label="Detected Objects")
                    webcam_stats = gr.Markdown()
                    with gr.Row():
                        webcam_json = gr.File(label="📄 JSON Report")
                        webcam_csv = gr.File(label="📊 CSV Report")

            webcam_btn.click(
                fn=detect_image,
                inputs=[webcam_input, w_conf_slider, w_iou_slider, w_class_filter],
                outputs=[webcam_output, webcam_stats, webcam_json, webcam_csv]
            )

        # =================================================
        # ANALYTICS TAB
        # =================================================
        with gr.TabItem("📈 Analytics"):

            gr.Markdown("Cumulative detection stats across your whole session (image + batch + video).")

            analytics_plot = gr.BarPlot(
                value=get_analytics_df(),
                x="class", y="count",
                title="Most Detected Classes",
                height=350
            )

            with gr.Row():
                refresh_analytics_btn = gr.Button("🔄 Refresh", variant="secondary")
                reset_analytics_btn = gr.Button("🗑️ Reset Stats", variant="secondary")

            refresh_analytics_btn.click(fn=get_analytics_df, outputs=analytics_plot)
            reset_analytics_btn.click(fn=reset_analytics, outputs=analytics_plot)

        # =================================================
        # HISTORY TAB
        # =================================================
        with gr.TabItem("🕒 History"):

            gr.Markdown("Your last 12 detections in this session.")

            history_gallery = gr.Gallery(label="Detection History", columns=4, height="auto")
            with gr.Row():
                refresh_history_btn = gr.Button("🔄 Refresh", variant="secondary")
                clear_history_btn = gr.Button("🗑️ Clear History", variant="secondary")

            refresh_history_btn.click(fn=lambda: detection_history, outputs=history_gallery)
            clear_history_btn.click(fn=clear_history, outputs=history_gallery)

        # =================================================
        # MODEL TAB
        # =================================================
        with gr.TabItem("⚙️ Model"):

            gr.Markdown("Switch between model sizes. Larger models are more accurate but slower.")

            model_dropdown = gr.Dropdown(
                choices=list(MODEL_VARIANTS.keys()),
                value=current_variant,
                label="Model Variant"
            )
            switch_btn = gr.Button("Switch Model", variant="primary")
            switch_status = gr.Markdown()

            switch_btn.click(fn=switch_model, inputs=model_dropdown, outputs=switch_status)

    gr.HTML(
        """
        <p style="text-align:center;color:#6b6b8c;font-size:0.8rem;margin-top:20px;">
            Built with YOLOv8 + Gradio · by Monisha D
        </p>
        """
    )


if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        css=custom_css,
        theme=theme
    )
