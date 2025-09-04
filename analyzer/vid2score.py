#!/usr/bin/env python3
"""Video/stream → CSV score writer.

This script is meant to be small and readable.  It looks at every frame of a
video (file or RTSP/HTTP URL), runs YOLO object detection, and writes a line of
`score.csv` for each object it tracks.

Each CSV row carries timestamp, object id/class, position in polar coords,
speed, confidence, and a handful of goofy "character" features like color and
edge density.  The SuperCollider renderer turns those numbers into synth
parameters.
"""

import argparse
import csv
import math

import cv2
import numpy as np
from ultralytics import YOLO


def glitch_meter(gray: np.ndarray) -> float:
    """Estimate how broken a frame looks.

    Uses a Sobel edge detector and averages the absolute value.  Lots of harsh
    horizontal lines push the value toward 1.0.
    """

    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    return float(np.clip(np.mean(np.abs(sobelx)) / 64.0, 0.0, 1.0))


def to_polar(cx: float, cy: float, area: float, W: int, H: int) -> tuple[float, float, float]:
    """Convert pixel coordinates and area into rough polar coordinates.

    Returns azimuth (degrees left/right of center), elevation (degrees up/down)
    and a fake distance metric based on bounding‑box area.
    """

    az = (cx / W) * 360.0 - 180.0
    el = (1 - cy / H) * 60.0 - 30.0
    dist = max(0.05, 1.0 - area / (W * H))
    return az, el, dist


def hsv_stats(bgr_roi: np.ndarray) -> tuple[float, float, float]:
    """Return average hue/saturation/value for a region of interest."""

    if bgr_roi.size == 0:
        return 0.0, 0.0, 0.0
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    h = float(np.mean(hsv[:, :, 0]) * 2.0)  # 0..360°
    s = float(np.mean(hsv[:, :, 1]) / 255.0)  # 0..1
    v = float(np.mean(hsv[:, :, 2]) / 255.0)  # 0..1
    return h, s, v


def edge_density(gray_roi: np.ndarray) -> float:
    """How edgy is the ROI? (0..1)"""

    if gray_roi.size == 0:
        return 0.0
    return float(np.clip(cv2.Laplacian(gray_roi, cv2.CV_32F).var() / 1000.0, 0.0, 1.0))


def main(args: argparse.Namespace) -> None:
    """Open the video, run tracking, and write the score."""

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"Cannot open {args.video}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    model = YOLO(args.model)

    prev_center: dict[int, tuple[float, float]] = {}
    t = 0.0
    out = open(args.out, "w", newline="")
    wr = csv.writer(out)
    wr.writerow([
        "t",
        "stream",
        "oid",
        "cls",
        "az",
        "el",
        "dist",
        "spd",
        "conf",
        "glitch",
        "hue",
        "sat",
        "val",
        "edge",
    ])
    stream_id = args.stream_id

    # Iterate over detection results from YOLO's built-in tracker.
    for r in model.track(
        source=args.video,
        stream=True,
        imgsz=args.imgsz,
        conf=args.conf,
        verbose=False,
    ):
        t += 1.0 / fps
        frame = getattr(r, "orig_img", None)
        gval = 0.0
        if frame is not None:
            # rough glitch indicator based on horizontal edges
            gval = glitch_meter(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        boxes = r.boxes
        if boxes is None:
            continue

        for b in boxes:
            if b.id is None:
                continue
            oid = int(b.id.item())
            cls = int(b.cls.item())
            conf = float(b.conf.item())

            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            area = max(1.0, (x2 - x1) * (y2 - y1))
            az, el, dist = to_polar(cx, cy, area, W, H)

            # speed in normalized pixels per second
            spd = 0.0
            if oid in prev_center:
                px, py = prev_center[oid]
                dx, dy = (cx - px) / W, (cy - py) / H
                spd = math.sqrt(dx * dx + dy * dy) * fps
            prev_center[oid] = (cx, cy)

            # crop region of interest for color/edge features
            x1i, y1i = int(max(0, x1)), int(max(0, y1))
            x2i, y2i = int(min(W, x2)), int(min(H, y2))
            roi = frame[y1i:y2i, x1i:x2i] if frame is not None else np.array([])
            hue, sat, val = hsv_stats(roi)
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.size else np.array([])
            edge = edge_density(gray_roi)

            wr.writerow(
                [
                    round(t, 3),
                    stream_id,
                    oid,
                    cls,
                    round(az, 2),
                    round(el, 2),
                    round(dist, 3),
                    round(spd, 3),
                    round(conf, 3),
                    round(gval, 3),
                    round(hue, 1),
                    round(sat, 3),
                    round(val, 3),
                    round(edge, 3),
                ]
            )
    out.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video", help="path or RTSP/HTTP URL")
    p.add_argument("--out", default="score.csv")
    p.add_argument("--model", default="yolov8n.pt")  # start tiny; TRT later
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--stream_id", default="camA")
    args = p.parse_args()
    main(args)

