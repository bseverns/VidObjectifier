// VidObjectifierProcessing.pde
// Webcam → quasi-analyzer with noisy voices.
//
// The Python script in ../analyzer/vid2score.py runs YOLO and spits out a CSV
// score packed with color, motion, and shape features.  This sketch can't run
// YOLO (we're in Java/Processing land), but it tries to hit the same beats with
// vanilla OpenCV tools.  Each detected blob gets a voice and a pile of nerdy
// stats.
//
// Dependencies
// ------------
// * Processing + the "Video" library (built in)
// * [OpenCV for Processing](https://github.com/atduskgreg/opencv-processing)
// * [Minim](http://code.compartmental.net/tools/minim/)
// Drop this file in a folder named "VidObjectifierProcessing" and run.
//
// The code is bloated with comments on purpose.  Treat it like a zine.

import processing.video.*;      // camera input
import gab.opencv.*;            // OpenCV wrapper
import ddf.minim.*;             // audio engine
import ddf.minim.ugens.*;       // oscillator voices
import java.util.*;

Capture cam;
OpenCV opencv;                  // for contour/edge work

Minim minim;
AudioOutput out;

// keep a synth voice per object id
HashMap<Integer, Oscil> voices = new HashMap<Integer, Oscil>();
HashMap<Integer, PVector> prevCenter = new HashMap<Integer, PVector>();
int nextId = 0;

void setup() {
  size(640, 480);

  cam = new Capture(this, width, height);
  cam.start();

  opencv = new OpenCV(this, width, height);

  minim = new Minim(this);
  out = minim.getLineOut();
}

void draw() {
  background(0);
  if (cam.available()) cam.read();
  image(cam, 0, 0);

  opencv.loadImage(cam);
  opencv.gray();

  // glitch metric = mean abs horizontal edges
  float glitch = glitchMeter(opencv.getGray());

  // cheap motion mask → contours
  opencv.blur(5);
  opencv.threshold(32);
  ArrayList<Contour> contours = opencv.findContours();

  HashMap<Integer, PVector> seen = new HashMap<Integer, PVector>();

  for (Contour c : contours) {
    if (c.area() < 500) continue; // ignore specks
    Rectangle r = c.getBoundingBox();
    float cx = r.x + r.width/2.0;
    float cy = r.y + r.height/2.0;
    float area = r.width * r.height;

    int id = assignId(cx, cy);
    seen.put(id, new PVector(cx, cy));

    float[] polar = toPolar(cx, cy, area, width, height);
    float az = polar[0], el = polar[1], dist = polar[2];

    float spd = 0;
    if (prevCenter.containsKey(id)) {
      PVector p = prevCenter.get(id);
      spd = PVector.dist(p, new PVector(cx, cy)) / max(width, height) * frameRate;
    }

    // region of interest for color/shape nerdiness
    PImage roi = cam.get(r.x, r.y, r.width, r.height);
    float[] hsv = hsvStats(roi);
    float edge = edgeDensity(roi);
    float shape = shapeMagic(roi);

    Oscil voice = voices.get(id);
    if (voice == null) {
      voice = new Oscil(440, 0, Waves.SINE);
      voice.patch(out);
      voices.put(id, voice);
    }
    // map azimuth → frequency, speed → amplitude
    voice.setFrequency(map(az, -180, 180, 100, 1000));
    voice.setAmplitude(constrain(spd, 0, 0.5));

    noFill();
    stroke(0, 255, 0);
    rect(r.x, r.y, r.width, r.height);
    fill(255);
    text("id " + id, r.x, r.y - 4);

    // spit stats to console so you can log/plot them later
    println(frameCount, id, nf(az, 0, 2), nf(el,0,2), nf(dist,0,3),
            nf(spd,0,3), nf(glitch,0,3),
            nf(hsv[0],0,1), nf(hsv[1],0,3), nf(hsv[2],0,3),
            nf(edge,0,3), nf(shape,0,3));
  }

  // ditch voices for vanished objects
  for (Integer id : voices.keySet()) {
    if (!seen.containsKey(id)) {
      voices.get(id).unpatch(out);
    }
  }
  voices.keySet().retainAll(seen.keySet());
  prevCenter = seen;
}

int assignId(float cx, float cy) {
  int best = -1; float bestDist = 25;
  for (Map.Entry<Integer, PVector> e : prevCenter.entrySet()) {
    float d = dist(cx, cy, e.getValue().x, e.getValue().y);
    if (d < bestDist) { bestDist = d; best = e.getKey(); }
  }
  if (best == -1) best = nextId++;
  return best;
}

float glitchMeter(PImage gray) {
  OpenCV gx = new OpenCV(this, gray);
  gx.sobel(OpenCV.HORIZONTAL, 3);
  gx.abs();
  float sum = 0;
  gx.getOutput().loadPixels();
  for (int p : gx.getOutput().pixels) sum += brightness(p);
  return constrain(sum / (gx.getOutput().pixels.length * 64.0), 0, 1);
}

float[] toPolar(float cx, float cy, float area, float W, float H) {
  float az = cx / W * 360 - 180;
  float el = (1 - cy / H) * 60 - 30;
  float dist = max(0.05, 1.0 - area / (W * H));
  return new float[]{az, el, dist};
}

float[] hsvStats(PImage roi) {
  if (roi.width == 0 || roi.height == 0) return new float[]{0,0,0};
  roi.loadPixels();
  float h=0,s=0,v=0;
  for (int c : roi.pixels) {
    h += hue(c);
    s += saturation(c)/255.0;
    v += brightness(c)/255.0;
  }
  int n = roi.pixels.length;
  return new float[]{(h/n), (s/n), (v/n)};
}

float edgeDensity(PImage roi) {
  if (roi.width==0) return 0;
  OpenCV e = new OpenCV(this, roi);
  e.gray();
  e.laplace(OpenCV.CV_32F);
  float sum = 0;
  e.getOutput().loadPixels();
  for (int p : e.getOutput().pixels) sum += abs(brightness(p));
  return constrain(sum/(e.getOutput().pixels.length*1000.0),0,1);
}

float shapeMagic(PImage roi) {
  if (roi.width==0) return 0;
  OpenCV s = new OpenCV(this, roi);
  s.gray();
  s.canny(50,150);
  ArrayList<Contour> cs = s.findContours();
  if (cs.size()==0) return 0;
  Contour big = cs.get(0);
  for (Contour c : cs) if (c.area()>big.area()) big = c;
  float area = big.area();
  float peri = big.perimeter();
  if (peri==0) return 0;
  float compact = 4*PI*area/(peri*peri);
  PImage hsv = roi.copy();
  hsv.loadPixels();
  float val = 0, hueSum=0, hueSq=0;
  for(int px : hsv.pixels){
    float v = brightness(px)/255.0;
    float h = hue(px);
    val += v; hueSum += h; hueSq += h*h;
  }
  int n = hsv.pixels.length;
  float bright = val/n;
  float hueStd = sqrt(hueSq/n - (hueSum/n)*(hueSum/n))/180.0;
  float score = compact * bright * (1 - hueStd);
  return constrain(score,0,1);
}

void stop() {
  for (Oscil o : voices.values()) o.unpatch(out);
  out.close();
  minim.stop();
  super.stop();
}
