# Processing Noise Gremlin 2.0

This sketch grew up a bit. Instead of chasing the single brightest pixel, it now
uses **OpenCV** to spot moving blobs, measures a fistful of features, and gives
each blob its own sine voice. It's still scrappy Java, but it mirrors the
Python analyzer's spirit.

## Why this exists

Not everyone wants to juggle Python + SuperCollider. This Processing version
shows the whole pipeline—detection, character stats, and sound—in one file.
Read it like a manual for building your own noisy camera band.

## Setup

1. Install [Processing](https://processing.org/).
2. From the IDE choose **Sketch → Import Library → Add Library** and grab
   **Minim** and **OpenCV for Processing**.
3. Drop `VidObjectifierProcessing.pde` into a folder named
   `VidObjectifierProcessing` and open it.
4. Smash the **Run** button. A webcam window pops up and starts yodeling.

## What it's doing now

- Pulls frames from your default camera.
- Runs a cheap motion detector (blur + threshold + contours).
- For each contour it computes:
  - azimuth/elevation/dist of the blob in frame-space,
  - motion speed,
  - a glitch score from horizontal edges,
  - average hue, saturation, and brightness,
  - edge density and a cockeyed "shape" score.
- Each blob gets a sine oscillator. Azimuth maps to pitch, speed maps to volume.
- Stats spew to the console so you can log or plot them.

## Hacks invited

- Swap the sine waves for wilder synths.
- Dump the printed stats into a CSV and drive SuperCollider live.
- Replace the contour detector with a legit DNN if you're feeling fancy.

## Disclaimer

This is still a teaching toy. Expect glitches, rough edges, and occasional
screams. That's the point.
