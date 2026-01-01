# VidObjectifier — camera to noise box

*A tiny GPU watches video and scribbles a score; SuperCollider turns the scribble into a noisy choir.*

This repo is half diary, half toolkit.  It's for anyone who wants to poke at the
idea of **objects in a camera frame becoming musical voices**.  The code is kept
small and loud on purpose so you can read it like liner notes.

---

## In plain English

There are two main moving parts (plus a lo-fi Processing sketch if you want to stay in Java land):

1. **Analyzer** (`analyzer/vid2score.py`)
   - Python script that runs on a Jetson Nano (or any CUDA Jetson).
   - Looks at a video file or live stream, spots objects with YOLO, and writes a
     timestamped **score** as `CSV` or `JSONL`.
2. **Renderer** (`renderer/render.scd` or `renderer/render_ring8.scd`)
   - SuperCollider patch that reads the score and gives every object a voice.
   - Default budget is **20 voices**, so things stay musical instead of mush.
3. **Processing sketch** (`processing/VidObjectifierProcessing.pde`)
   - Java/Processing rewrite that spots moving blobs with OpenCV and hands each one a sine voice.

Picture a security camera feeding a garage band.

```
[video file / TouchDesigner] → [Jetson: detector + characterizer] → score.csv
                                                     ↓
                                            [SuperCollider renderer]
                                                     ↓
                                     (stereo / 8‑channel ring / binaural)
```

---

## Why bother?

- **Low cost, high grit.** Everything here runs on free tools, save the Jetson
  Nano which can be had for ~$225 right now. Not my favorite situation, but 
  when life gives you beautiful Jetsons, you run things like: SuperCollider, 
  JACK, and a Jetson that you probably already cooked noodles on.
- **Every object gets a personality.** The analyzer measures position, speed,
  color, **shape**, and even a janky "glitch" metric.  The renderer maps those
  numbers to timbre.
- **Live or offline.** Feed it prerecorded footage, or sling live frames from
  TouchDesigner over RTSP/NDI.

---

## What's in the box?

```
.
├── analyzer/
│   └── vid2score.py           # video/stream → score.csv
├── renderer/
│   ├── render.scd             # SuperCollider renderer (stereo)
│   ├── render_ring8.scd       # 8‑channel ring renderer
│   ├── mapping.scd            # generated class→timbre map (see config/timbre_map.yaml)
│   ├── mapping_steel_mill.scd # preset 1: steel, presses, belts
│   ├── mapping_neon_glass.scd # preset 2: glass, hiss, sheen
│   └── mapping_rust_choir.scd # preset 3: drones, rust, vocals-of-metal
├── config/
│   └── timbre_map.yaml        # the real mapping source; generator writes mapping.scd
├── examples/
│   ├── input.mp4              # drop your video here
│   └── score_example.csv      # tiny header-only example
├── processing/
│   ├── VidObjectifierProcessing.pde # webcam → blob tracker → sine choir
│   └── README.md                    # how to run and hack the gremlin
└── README.md                  # you are here
```

Clone the repo, throw your own media into `examples/`, and hack away.

---

## Getting set up

### On the Jetson (Nano is fine)

```bash
sudo apt update && sudo apt install -y ffmpeg python3-pip
pip3 install --upgrade pip
pip3 install ultralytics opencv-python numpy

# optional but makes the Nano run spicy hot
sudo nvpmodel -m 0 && sudo jetson_clocks
```

### On the render machine (Linux or macOS)

- [SuperCollider](https://supercollider.github.io/) — free, friendly synth
  language.
- [JACK2](https://jackaudio.org/) + QjackCtl — for routing audio devices.
- Optional: **Ardour** (open-source) or **REAPER** (generous demo) to record
  multichannel stems.

---

## Running the analyzer

The script writes a newline‑timed score with spatial, color, and shape features.  The
score is just text; open it in a spreadsheet if that makes you smile.

```bash
cd analyzer
python3 vid2score.py ../examples/input.mp4 --out ../examples/score_example.csv --stream_id camA
```

Want live video from TouchDesigner?  Add a **Stream Out TOP** (RTSP) or **NDI
Out TOP** in TD and point the script at the URL:

```bash
python3 vid2score.py "rtsp://<ip>:<port>/<name>" --out score_camA.csv --stream_id camA
```

Best practice is to analyze each pre‑mix stream for object‑level timbres, then
analyze the post‑mix once to pull out macro "mood" controls.

---

## Running the renderer

Open SuperCollider, start JACK, and load one of:

- `renderer/render.scd` — stereo, Pan2 based.
- `renderer/render_ring8.scd` — 8‑channel ring without needing VBAP.

Each mapping file in `renderer/` tweaks the personality of the voices.  Swap in
`mapping_steel_mill.scd`, `mapping_neon_glass.scd`, or `mapping_rust_choir.scd`
for different vibes.  They all respect the same 20‑voice budget.

If you want to tweak the default mapping, edit `config/timbre_map.yaml` and run:

```bash
python3 renderer/generate_mapping.py
```

That writes a fresh `renderer/mapping.scd` and keeps the YAML as the single source
of truth. It's like tuning your synth with a wrench instead of random vibes.

Once a renderer is running, call `~playScore` with the path to your CSV:

```supercollider
~playScore.("/full/path/to/examples/score_example.csv");
```

The renderer keeps a voice alive a few seconds after the last update so tails
breathe instead of choking.

---

## Processing quickie

If you're allergic to command lines or want everything in one Java file, peek at `processing/VidObjectifierProcessing.pde`. Open it in the Processing IDE, install the **Minim** and **OpenCV for Processing** libraries, and run. It tracks moving blobs, spits out color/motion/shape stats, and turns each blob into a sine voice. Cheap, loud, educational.

---

## Voice policy (musical & safe)

- **Global cap:** `~MAX_VOICES = 20`.
- **Per stream soft cap:** `~PER_STREAM = 4`.
- **Priority:** new voices beat loud voices which beat fast voices.  Something
  has to win.
- **Hysteresis:** letting notes hang for a moment keeps things human.

---

## TouchDesigner quick tips

- On each TD source, add **Stream Out TOP** (RTSP) and give it a unique name
  (`camA`, `camB`, ...).
- Run one analyzer per stream on the Jetson and set `--stream_id` to match.
- To sniff the final mix, add another Stream Out TOP and analyze it into a
  second score (e.g. `macro.csv`).

---

## Roadmap

- [ ] Optional DeepStream backend.
- [ ] VBAP/FOA decoders for 8‑out and binaural prints.
- [ ] One‑shot event bus on object birth/death.
- [ ] JSON mapping loaded at runtime instead of hard‑coding `mapping.scd`.
- [ ] Simple OSC live mode (Jetson → renderer in real time).

---

## License

MIT, because art should travel.

---

### TL;DR quickstart

```bash
# 1) Analyze a file
cd analyzer
python3 vid2score.py ../examples/input.mp4 --out ../examples/score_example.csv --stream_id camA

# 2) Render it (in SuperCollider)
# open renderer/render.scd OR renderer/render_ring8.scd and run; adjust path in ~playScore
```
