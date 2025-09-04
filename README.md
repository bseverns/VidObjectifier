# Object-Based Soundscape (Jetson → Score → Render)

*A small machine watches images and writes music about them.*  
Jetson Nano (or any CUDA Jetson) analyzes video streams, emits a time-stamped **score** (`CSV`/`JSONL`), and a **SuperCollider** renderer plays a dense, multi-timbral sound field (default **20 voices**). Use prerecorded videos or live feeds from TouchDesigner via RTSP/NDI bridge.

```
[TD sources / files] → [Jetson: detector+characterizer] → score.csv/jsonl
                                                      ↓
                                            [Renderer: SuperCollider]
                                                      ↓
                                  (stereo / 8-out ring / binaural print)
```

## Why this exists
- **Low cost, high grit.** Free tools (SuperCollider, JACK, Ardour/REAPER eval).
- **Multi-timbral by design.** Each tracked object gets a “character vector” that maps to synthesis.
- **Live or offline.** Works from files, or live streams pre-mix and/or post-mix.

---

## Repo layout

```
.
├── analyzer/
│   └── vid2score.py           # file/stream → score.csv/jsonl
├── renderer/
│   ├── render.scd             # SuperCollider renderer (stereo; 20-voice budget)
│   ├── render_ring8.scd       # 8‑channel ring renderer (no external VBAP plugin)
│   ├── mapping.scd            # base class→timbre map + routing
│   ├── mapping_steel_mill.scd # preset 1: steel, presses, belts
│   ├── mapping_neon_glass.scd # preset 2: glass, hiss, sheen
│   └── mapping_rust_choir.scd # preset 3: drones, rust, vocals-of-metal
├── config/
│   └── timbre_map.yaml        # optional declarative map (reference)
├── examples/
│   ├── input.mp4              # (put your file here)
│   └── score_example.csv      # tiny header-only example
└── README.md                  # this file
```

Clone and drop your own media into `examples/` as you like.

---

## Prerequisites

### Jetson side (Nano is fine)
- JetPack flashed and updated, active cooling recommended.
- Python 3.8+, `pip`, `ffmpeg`.
- Python packages: `ultralytics`, `opencv-python`, `numpy`.

### Renderer side (any Linux/macOS machine)
- **SuperCollider** (free)
- **JACK2** + **QjackCtl** (for easy device routing)
- Optional: **Ardour** (free/open) or **REAPER** (uncrippled evaluation) to record multichannel stems.

---

## Install (Jetson)

```bash
sudo apt update && sudo apt install -y ffmpeg python3-pip
pip3 install --upgrade pip
pip3 install ultralytics opencv-python numpy

# (optional but helpful)
sudo nvpmodel -m 0 && sudo jetson_clocks
```

---

## Analyzer: video/stream → score

The analyzer writes a newline-timed **score** with spatial + “character” features.

**`analyzer/vid2score.py`**

```python
#!/usr/bin/env python3
# File: analyzer/vid2score.py
import argparse, csv, math, cv2, numpy as np
from ultralytics import YOLO

def glitch_meter(gray):
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    return float(np.clip(np.mean(np.abs(sobelx))/64.0, 0.0, 1.0))

def to_polar(cx, cy, area, W, H):
    az  = (cx/W)*360.0 - 180.0
    el  = (1 - cy/H)*60.0 - 30.0
    dist = max(0.05, 1.0 - area/(W*H))
    return az, el, dist

def hsv_stats(bgr_roi):
    if bgr_roi.size == 0: return 0.0,0.0,0.0
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    h = float(np.mean(hsv[:,:,0]) * 2.0)       # 0..360°
    s = float(np.mean(hsv[:,:,1]) / 255.0)     # 0..1
    v = float(np.mean(hsv[:,:,2]) / 255.0)     # 0..1
    return h,s,v

def edge_density(gray_roi):
    if gray_roi.size == 0: return 0.0
    return float(np.clip(cv2.Laplacian(gray_roi, cv2.CV_32F).var()/1000.0, 0.0, 1.0))

def main(args):
    cap  = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"Cannot open {args.video}"
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    model = YOLO(args.model)

    prev_center = {}
    t = 0.0
    out = open(args.out, "w", newline="")
    wr = csv.writer(out)
    wr.writerow(["t","stream","oid","cls","az","el","dist","spd","conf","glitch","hue","sat","val","edge"])
    stream_id = args.stream_id

    for r in model.track(source=args.video, stream=True, imgsz=args.imgsz, conf=args.conf, verbose=False):
        t += 1.0/fps
        frame = getattr(r, "orig_img", None)
        gval = 0.0
        if frame is not None:
            gval = glitch_meter(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        boxes = r.boxes
        if boxes is None: continue

        for b in boxes:
            if b.id is None: continue
            oid  = int(b.id.item())
            cls  = int(b.cls.item())
            conf = float(b.conf.item())

            x1,y1,x2,y2 = [float(v) for v in b.xyxy[0].tolist()]
            cx, cy = (x1+x2)/2, (y1+y2)/2
            area   = max(1.0, (x2-x1)*(y2-y1))
            az, el, dist = to_polar(cx, cy, area, W, H)

            spd = 0.0
            if oid in prev_center:
                px, py = prev_center[oid]
                dx, dy = (cx-px)/W, (cy-py)/H
                spd = math.sqrt(dx*dx + dy*dy) * fps
            prev_center[oid] = (cx, cy)

            x1i, y1i = int(max(0, x1)), int(max(0, y1))
            x2i, y2i = int(min(W, x2)), int(min(H, y2))
            roi = frame[y1i:y2i, x1i:x2i] if frame is not None else np.array([])
            hue,sat,val = hsv_stats(roi)
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.size else np.array([])
            edge = edge_density(gray_roi)

            wr.writerow([round(t,3), stream_id, oid, cls,
                         round(az,2), round(el,2), round(dist,3),
                         round(spd,3), round(conf,3), round(gval,3),
                         round(hue,1), round(sat,3), round(val,3), round(edge,3)])
    out.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video", help="path or RTSP/HTTP URL")
    p.add_argument("--out", default="score.csv")
    p.add_argument("--model", default="yolov8n.pt")  # start tiny; TRT later
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--stream_id", default="camA")
    args = p.parse_args(); main(args)
```

### Run on a file
```bash
cd analyzer
python3 vid2score.py ../examples/input.mp4 --out ../examples/score_example.csv --stream_id camA
```

### Run on a live stream (TouchDesigner tips)
- In TD, for each **pre-mix source**, add **Stream Out TOP** (RTSP) *or* **NDI Out TOP**.
- On the Jetson, pass the RTSP/HTTP URL as `video`. (NDI works too if you add a local NDI receiver that exposes frames/URL to OpenCV.)
```bash
python3 vid2score.py "rtsp://<ip>:<port>/<name>" --out score_camA.csv --stream_id camA
```

> **Best practice:** sniff each **input stream** for object-level timbres (clarity), and also sniff the **post-mix** once to extract macro “mood” controls.

---

## Renderer: multi-timbral, 20-voice budget (SuperCollider)

### Files you can run
- `renderer/render.scd` — stereo renderer (Pan2), **20-voice** budget.
- `renderer/render_ring8.scd` — 8‑channel ring renderer without external VBAP plugins (works on plain SC+plugins).
- `renderer/mapping*.scd` — three presets in addition to the base `mapping.scd`:
  - **Steel Mill** (fm metal + modal hits)
  - **Neon Glass** (band‑noise + sheen)
  - **Rust Choir** (slow drones + foldback grit)

Open SuperCollider, start JACK, run one of the renderers, then call `~playScore` with your CSV path.

---

## Voice policy (musical & safe)

- **Global cap:** `~MAX_VOICES = 20` (default).
- **Per-stream soft cap:** `~PER_STREAM = 4`.
- **Priority:** births > largest area > fastest.
- **Hysteresis:** renderer keeps a voice alive a few seconds after last update so tails breathe.

---

## TouchDesigner bridging (quick)

- **Pre-mix:** On each TD source, add **Stream Out TOP** (RTSP). Give unique names (camA, camB…).  
- On Jetson, run one analyzer per stream (or round-robin a single process) and set `--stream_id` accordingly.  
- **Post-mix:** Add another Stream Out TOP for the **final mix** and analyze to extract *macro* controls (you can add a second pass later that writes `macro.csv`).

---

## Practical notes

- Start with `yolov8n.pt`. If you need more FPS: export to **TensorRT** or step up to an Orin.
- Keep detection `imgsz` at `640` or `512` for Nano comfort.
- If footage has heavy stripes/flicker, the `glitch` metric will climb—map it to **bit depth**, **clock skew**, or **ratchet rate** for that beloved broken-machine energy.

---

## Roadmap
- [ ] Optional **DeepStream** backend.
- [ ] VBAP/FOA decoders for 8-out and binaural prints.
- [ ] One-shot event bus on object birth/death.
- [ ] JSON mapping (loadable at runtime) instead of `mapping.scd`.
- [ ] Simple OSC live mode (Jetson → Renderer in real time).

---

## License
MIT

---

### Quickstart (TL;DR)

```bash
# 1) Analyze a file
cd analyzer
python3 vid2score.py ../examples/input.mp4 --out ../examples/score_example.csv --stream_id camA

# 2) Render it (SuperCollider)
# open renderer/render.scd OR renderer/render_ring8.scd and run; adjust path in ~playScore
```
