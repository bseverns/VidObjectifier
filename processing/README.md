# Processing Noise Gremlin

Welcome to the **lo-fi, all-in-one** version of VidObjectifier. This sketch watches your webcam and spits out a single sine wave whose pitch and volume chase the brightest blob in view. It's the punk little cousin of the Python analyzer and SuperCollider renderer.

## Why this exists

Some folks just want to crack open Processing, hit run, and get weird. No Python, no SuperCollider, just straight Java vibes. This sketch keeps it simple so you can poke at the ideas quickly.

## How to run it

1. Install [Processing](https://processing.org/).
2. In Processing, go to **Sketch → Import Library → Add Library** and install **Minim**.
3. Drop the `VidObjectifierProcessing.pde` file into its own folder named `VidObjectifierProcessing` (Processing loves matching names).
4. Open the sketch and smash the **Run** button.
5. Point a flashlight, your phone screen, or a shiny object at the camera and hear the pitch go wild.

## What it's doing

- Grabs frames from your default webcam.
- Searches each frame for the brightest pixel.
- Draws a red dot on that pixel so you can see what's driving the sound.
- Maps horizontal position → frequency and brightness → amplitude.
- A lone sine oscillator drones along based on those values.

## Where to hack

- Swap the sine wave for something gnarlier in the `Oscil` setup.
- Use Processing's `Movie` class instead of `Capture` to analyze prerecorded video.
- Add more oscillators and let each bright blob own a voice.

## Disclaimer

This is not high art. It's a teaching tool and a playground. If you crash it, break it, or make it scream, you're doing it right.
