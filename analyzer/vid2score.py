#!/usr/bin/env python3
"""Video/stream → CSV score writer.

This script is meant to be small and readable.  It looks at every frame of a
video (file or RTSP/HTTP URL), runs YOLO object detection, and writes a line of
`score.csv` for each object it tracks.

Each CSV row carries timestamp, object id/class, position in polar coords,
speed, confidence, and a handful of goofy "character" features like color,
edge density, and a DIY "shape" score.  The SuperCollider renderer turns those
numbers into synth parameters.
"""

import argparse
import csv
import json
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


def shape_magic(bgr_roi: np.ndarray) -> float:
    """Cook up a mystical shape score (0..1) from color, brightness and edges.

    This is the part where we pretend to be wizards of vision.  The algorithm is
    intentionally loose and noisy—perfect for feeding a synth:

    1. Convert the region of interest (ROI) to grayscale and run a Canny edge
       detector.  This gives us a cheap silhouette of whatever the detector
       thinks is there.
    2. Find the biggest contour in that edge map.  Its area and perimeter feed a
       **compactness** metric: 1.0 means "round as a punk's mohawk," while lower
       values hint at jagged or skinny shapes.
    3. Mix in average **brightness** (from the V channel of HSV) so darker
       objects don't hog the spotlight.
    4. Subtract hue variance because wildly colored blobs are usually messy
       shapes.  Less color chaos → higher score.

    The final number rolls all that into a single 0..1 value—noisy, biased, and
    ready for sonic mayhem.
    """

    # Bail out early if the ROI is empty; no pixels, no party.
    if bgr_roi.size == 0:
        return 0.0

    # Step 1: carve out edges so we know where the shape lives.
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Step 2: grab the fattest contour and measure its girth.
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return 0.0
    compact = 4.0 * math.pi * area / (peri * peri)  # 1.0 = perfect circle

    # Step 3: weigh by brightness so ghostly dark shapes stay quiet.
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    bright = float(np.mean(hsv[:, :, 2]) / 255.0)

    # Step 4: penalize rainbow chaos; uniform color keeps the score hot.
    hue_std = float(np.std(hsv[:, :, 0]) / 180.0)  # normalize 0..1

    # Mash everything together, clip to 0..1 for sanity.
    score = compact * bright * (1.0 - hue_std)
    return float(np.clip(score, 0.0, 1.0))


def main(args: argparse.Namespace) -> None:
    """Open the video, run tracking, and write the score."""

    field_order = [
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
        "shape",
    ]
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"Cannot open {args.video}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    model = YOLO(args.model)

    prev_center: dict[int, tuple[float, float]] = {}
    t = 0.0
    out = open(args.out, "w", newline="")
    writer = None
    if args.format == "csv":
        writer = csv.writer(out)
        writer.writerow(field_order)
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

            # crop region of interest for color/edge/shape features
            x1i, y1i = int(max(0, x1)), int(max(0, y1))
            x2i, y2i = int(min(W, x2)), int(min(H, y2))
            roi = frame[y1i:y2i, x1i:x2i] if frame is not None else np.array([])
            hue, sat, val = hsv_stats(roi)
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.size else np.array([])
            edge = edge_density(gray_roi)
            shape = shape_magic(roi)

            row = {
                "t": round(t, 3),
                "stream": stream_id,
                "oid": oid,
                "cls": cls,
                "az": round(az, 2),
                "el": round(el, 2),
                "dist": round(dist, 3),
                "spd": round(spd, 3),
                "conf": round(conf, 3),
                "glitch": round(gval, 3),
                "hue": round(hue, 1),
                "sat": round(sat, 3),
                "val": round(val, 3),
                "edge": round(edge, 3),
                "shape": round(shape, 3),
            }
            if writer is None:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                writer.writerow([row[key] for key in field_order])
    out.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video", help="path or RTSP/HTTP URL")
    p.add_argument("--out", default="score.csv")
    p.add_argument("--model", default="yolov8n.pt")  # start tiny; TRT later
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--stream_id", default="camA")
    p.add_argument("--format", choices=("csv", "jsonl"), default="csv")
    args = p.parse_args()
    main(args)
