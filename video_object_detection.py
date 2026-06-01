import yt_dlp
import cv2
from yolov8 import YOLOv8
import os

videoUrl = 'https://youtu.be/tg00YEETFzg'
output_file = "video.mp4"

# Download video
if not os.path.exists(output_file):
    ydl_opts = {
        'format': 'bv*+ba/b',
        'outtmpl': output_file,
        'merge_output_format': 'mp4',
        'noplaylist': True,
        'extractor_args': {
            'youtube': {
                'player_client': ['android']
            }
        }
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([videoUrl])

# Open local file instead of stream
cap = cv2.VideoCapture(output_file)

if not cap.isOpened():
    print("❌ Could not open downloaded video.")
    exit()

yolov8_detector = YOLOv8("models\\yolov8m.onnx", conf_thres=0.5, iou_thres=0.5)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    boxes, scores, class_ids = yolov8_detector(frame)
    frame = yolov8_detector.draw_detections(frame)
    cv2.imshow("Detected Objects", frame)

    if cv2.waitKey(1) == ord('q'):
        break

print(f"✅ Processed {frame_count} frames")
cap.release()
cv2.destroyAllWindows() 