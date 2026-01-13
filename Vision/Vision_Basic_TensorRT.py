import time
import argparse
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


def open_camera(index: int, width: int, height: int, fps: int):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index={index}. "
                           f"Check /dev/video* and permissions.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))
    return cap


def main():
    parser = argparse.ArgumentParser(description="Real-time YOLO11n TensorRT FP16 webcam inference on Jetson")
    parser.add_argument("--engine", type=str, default="yolo11n.engine", help="Path to TensorRT engine file")
    parser.add_argument("--cam", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (must match engine export imgsz)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--width", type=int, default=1280, help="Camera capture width")
    parser.add_argument("--height", type=int, default=720, help="Camera capture height")
    parser.add_argument("--fps", type=int, default=30, help="Camera capture FPS request")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference (cuda:0 recommended)")
    parser.add_argument("--save-dir", type=str, default="captures", help="Directory to save captured images")
    args = parser.parse_args()

    # 저장 디렉토리 생성
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load TensorRT engine via Ultralytics
    model = YOLO(args.engine)

    cap = open_camera(args.cam, args.width, args.height, args.fps)

    # FPS smoothing
    fps_window = deque(maxlen=30)
    last_t = time.time()

    print(f"[INFO] Press 'q' to quit, 'SPACE' to save image to {save_dir}/")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to read frame from camera.")
            break

        # Run inference
        results = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False
        )

        r = results[0]
        annotated = r.plot()

        # FPS calculation
        now = time.time()
        dt = now - last_t
        last_t = now
        if dt > 0:
            fps_window.append(1.0 / dt)
        fps_avg = sum(fps_window) / len(fps_window) if fps_window else 0.0

        cv2.putText(
            annotated,
            f"FPS: {fps_avg:.1f}  Engine: {args.engine}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("YOLO11n TensorRT (FP16) - Webcam", annotated)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break
        elif key == ord(" "):  # SPACE 키
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = save_dir / f"capture_{timestamp}.jpg"
            cv2.imwrite(str(filename), annotated)
            print(f"[SAVED] {filename}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
