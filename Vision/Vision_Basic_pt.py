import time
import argparse
from collections import deque

import cv2
from ultralytics import YOLO


def open_camera(index: int, width: int, height: int, fps: int):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index={index}. Check /dev/video* and permissions.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))
    return cap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Path to .pt model")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    model = YOLO(args.model)

    cap = open_camera(args.cam, args.width, args.height, args.fps)

    fps_window = deque(maxlen=30)
    last_t = time.time()

    print("[INFO] Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False
        )

        annotated = results[0].plot()

        now = time.time()
        dt = now - last_t
        last_t = now
        if dt > 0:
            fps_window.append(1.0 / dt)
        fps_avg = sum(fps_window) / len(fps_window) if fps_window else 0.0

        cv2.putText(annotated, f"FPS: {fps_avg:.1f}  Model: {args.model}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("YOLO11n (PyTorch .pt) - Webcam", annotated)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
