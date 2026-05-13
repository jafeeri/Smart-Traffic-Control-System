"""
detect_vehicles_video.py
========================
Standalone YOLOv7 vehicle detector for a traffic video.

Reads a traffic video file frame-by-frame, runs YOLOv7 on each frame,
draws GREEN bounding boxes around detected vehicles (car / bus / truck /
motorbike / bicycle), and overlays a live counter at the top of the frame
showing how many vehicles are currently visible plus the peak count seen
so far.

Important: this script does NOT touch any other file in the project. The
control simulator and stability analysis remain unchanged. This is purely
a perception demo for the viva, to show the YOLO sensor working live on
a real video clip.

USAGE
-----
    python detect_vehicles_video.py <path_to_video.mp4>
    python detect_vehicles_video.py traffic.mp4 --save out.mp4
    python detect_vehicles_video.py traffic.mp4 --conf 0.4 --nms 0.4
    python detect_vehicles_video.py traffic.mp4 --no-show --save out.mp4

Press Q in the display window to quit early.

REQUIREMENTS
------------
Already in requirements.txt:
    opencv-python
    numpy

Project files used (must be present in the project root):
    yolov7.weights
    yolov7.cfg
    coco.names
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CONF = 0.5
DEFAULT_NMS  = 0.45
INPUT_SIZE   = 416         # YOLO input resolution (square)

# COCO class names that we treat as "vehicles" for this traffic demo.
VEHICLE_CLASSES = {"car", "bus", "truck", "motorbike", "motorcycle", "bicycle"}

# Colours (BGR — OpenCV convention)
GREEN  = (0, 255, 0)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
YELLOW = (0, 255, 255)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="YOLOv7 vehicle detector for a traffic video clip"
    )
    p.add_argument("video", help="Path to input video (mp4 / avi / mov)")
    p.add_argument("--save", default=None,
                   help="Optional output video path (mp4)")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF,
                   help=f"Confidence threshold (default {DEFAULT_CONF})")
    p.add_argument("--nms", type=float, default=DEFAULT_NMS,
                   help=f"NMS IoU threshold (default {DEFAULT_NMS})")
    p.add_argument("--weights", default="yolov7.weights",
                   help="YOLO weights file (default yolov7.weights)")
    p.add_argument("--cfg", default="yolov7.cfg",
                   help="YOLO config file (default yolov7.cfg)")
    p.add_argument("--names", default="coco.names",
                   help="Class names file (default coco.names)")
    p.add_argument("--no-show", action="store_true",
                   help="Don't open a display window (use with --save)")
    p.add_argument("--gpu", action="store_true",
                   help="Try CUDA backend (requires opencv-python built with CUDA; "
                        "the default pip wheel is CPU-only - leave this OFF).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# YOLO loading
# ---------------------------------------------------------------------------
def load_yolo(weights, cfg, names, use_gpu=False):
    for f in (weights, cfg, names):
        if not os.path.exists(f):
            sys.exit(f"ERROR: required file not found: {f}")

    print(f"Loading YOLOv7 weights ...")
    net = cv2.dnn.readNet(weights, cfg)

    # The pip-installed `opencv-python` wheel is CPU-only. We default to CPU
    # to avoid the run-time CUDA assertion error from net.forward(). The user
    # can opt in to CUDA with --gpu if they have a custom OpenCV build.
    if use_gpu:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Backend: CUDA (requires opencv-python built with CUDA support).")
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Backend: CPU.")

    with open(names) as f:
        classes = [c.strip() for c in f.readlines() if c.strip()]
    out_layers = net.getUnconnectedOutLayersNames()
    return net, classes, out_layers


# ---------------------------------------------------------------------------
# Per-frame detection
# ---------------------------------------------------------------------------
def detect_frame(frame, net, classes, out_layers, conf_th, nms_th):
    """Returns lists (boxes, confidences, labels) AFTER NMS, vehicles only."""
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0,
                                 (INPUT_SIZE, INPUT_SIZE),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)

    boxes, confidences, class_ids = [], [], []
    for layer_out in outs:
        for det in layer_out:
            scores = det[5:]
            cid = int(np.argmax(scores))
            confidence = float(scores[cid])
            label = classes[cid] if cid < len(classes) else None
            if confidence < conf_th or label not in VEHICLE_CLASSES:
                continue
            cx = int(det[0] * w)
            cy = int(det[1] * h)
            bw = int(det[2] * w)
            bh = int(det[3] * h)
            x = int(cx - bw / 2)
            y = int(cy - bh / 2)
            boxes.append([x, y, bw, bh])
            confidences.append(confidence)
            class_ids.append(cid)

    # Non-Maximum Suppression
    keep = cv2.dnn.NMSBoxes(boxes, confidences, conf_th, nms_th)
    if len(keep) == 0:
        return [], [], []

    # NMSBoxes returns either a 2-D array or a list-of-lists depending on
    # OpenCV version. Normalise.
    keep = np.array(keep).flatten()

    return ([boxes[i] for i in keep],
            [confidences[i] for i in keep],
            [classes[class_ids[i]] for i in keep])


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_overlay(frame, boxes, labels, confidences, peak, fps_text=""):
    h, w = frame.shape[:2]

    # Bounding boxes + labels
    for (x, y, bw, bh), label, conf in zip(boxes, labels, confidences):
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), GREEN, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(th + 4, y - 4)
        # solid label background for readability
        cv2.rectangle(frame, (x, ty - th - 4), (x + tw + 4, ty + 2), GREEN, -1)
        cv2.putText(frame, text, (x + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)

    # Top banner with counters (semi-transparent black)
    banner_h = 70
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    # cyan top edge
    cv2.line(frame, (0, banner_h), (w, banner_h), (255, 212, 0), 2)

    # Live count (big, yellow)
    n = len(boxes)
    cv2.putText(frame, f"VEHICLES IN FRAME: {n}",
                (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2, cv2.LINE_AA)
    # Peak count (smaller)
    cv2.putText(frame, f"PEAK SO FAR: {peak}",
                (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    # FPS on the right
    if fps_text:
        cv2.putText(frame, fps_text, (w - 170, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
    # Title small
    cv2.putText(frame, "YOLOv7  Vehicle Sensor  (project demo)",
                (w - 360, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open video file: {args.video}")

    src_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1

    print(f"Video: {args.video}  ({src_w}x{src_h} @ {src_fps:.1f} fps, "
          f"{n_total} frames)")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, src_fps, (src_w, src_h))
        if not writer.isOpened():
            sys.exit(f"ERROR: cannot open output writer for {args.save}")
        print(f"Saving annotated output to: {args.save}")

    net, classes, out_layers = load_yolo(args.weights, args.cfg, args.names, use_gpu=args.gpu)
    print("Detector ready.\n")

    frame_idx = 0
    peak = 0
    smoothed_fps = 0.0
    t_prev = time.time()
    print("Processing — press Q in the window to quit early.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            boxes, confs, labels = detect_frame(
                frame, net, classes, out_layers, args.conf, args.nms
            )
            n = len(boxes)
            if n > peak:
                peak = n

            # FPS smoothing
            t_now = time.time()
            dt = t_now - t_prev
            t_prev = t_now
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            smoothed_fps = (
                0.9 * smoothed_fps + 0.1 * inst_fps
                if smoothed_fps > 0 else inst_fps
            )

            draw_overlay(
                frame, boxes, labels, confs, peak,
                fps_text=f"FPS: {smoothed_fps:5.1f}",
            )

            if writer is not None:
                writer.write(frame)

            if not args.no_show:
                cv2.imshow(
                    "YOLOv7 Vehicle Detection  -  press Q to quit",
                    frame,
                )
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"  frame {frame_idx:>5}   "
                      f"in-frame={n:>3}   peak={peak:>3}   "
                      f"fps={smoothed_fps:5.1f}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print(f"\nDone. Frames processed: {frame_idx}.  "
          f"Peak vehicles in any frame: {peak}.")


if __name__ == "__main__":
    main()
