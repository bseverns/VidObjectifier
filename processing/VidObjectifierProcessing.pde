// VidObjectifierProcessing.pde
// A punk-rock Processing sketch that watches your webcam and screams in sine waves.
// Think of it as the scrappy cousin to the Python+SuperCollider chain.
//
// Dependencies:
//   - Processing (https://processing.org/)
//   - Video library (comes with Processing)
//   - Minim audio library (install via Sketch → Import Library → Add Library)
//
// This file is littered with comments because students deserve to know what's going on.
// Run it from the Processing IDE or via processing-java.

import processing.video.*;      // Grab frames from webcam or video file
import ddf.minim.*;             // Friendly audio toolkit
import ddf.minim.ugens.*;       // Oscillator objects live here

Capture cam;        // Video capture object
Minim minim;        // Minim engine
AudioOutput out;    // Where sound is pushed
Oscil osc;          // A single sine oscillator — our lone noisy voice

// setup() runs once when the sketch starts.
void setup() {
  size(640, 480);                       // Set window size

  // Fire up the webcam. Processing will nag you if none is found.
  cam = new Capture(this, 640, 480);    // width, height
  cam.start();                          // Start grabbing frames

  // Kick the audio engine into gear and patch a silent oscillator to it.
  minim = new Minim(this);
  out = minim.getLineOut();             // Stereo out
  osc = new Oscil(440, 0.0, Waves.SINE); // 440Hz but silent at start
  osc.patch(out);                       // Connect osc → speakers
}

// draw() loops forever. Each pass is a frame.
void draw() {
  background(0);                        // Black canvas each frame

  // Pull in a new frame if available.
  if (cam.available()) {
    cam.read();
  }
  cam.loadPixels();                     // Lets us read pixel array
  image(cam, 0, 0);                     // Paint the camera frame to the window

  // Find the brightest pixel in the frame.
  int brightestIndex = 0;
  float brightestValue = 0;
  for (int i = 0; i < cam.pixels.length; i++) {
    float b = brightness(cam.pixels[i]);
    if (b > brightestValue) {
      brightestValue = b;
      brightestIndex = i;
    }
  }

  // Convert 1D pixel index back to 2D position.
  int x = brightestIndex % cam.width;
  int y = brightestIndex / cam.width;

  // Draw a red circle where the brightest pixel lives.
  noStroke();
  fill(255, 0, 0);
  ellipse(x, y, 20, 20);

  // Map the x position to pitch and brightness to volume.
  float freq = map(x, 0, cam.width, 100, 1000);      // 100Hz → 1kHz
  float amp = map(brightestValue, 0, 255, 0, 0.5);   // Stay under clipping
  osc.setFrequency(freq);
  osc.setAmplitude(amp);
}

// When the window closes, unpatch and clean up.
void stop() {
  osc.unpatch(out);
  out.close();
  minim.stop();
  super.stop();
}
