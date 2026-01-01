(
// AUTO-GENERATED FILE — DO NOT HAND-EDIT.
//
// This file is written by renderer/generate_mapping.py from config/timbre_map.yaml.
// Edit the YAML, rerun the generator, and keep your mapping honest.

// class → family (SuperCollider symbols)
~classToFamily = (
    0: \servo,
    39: \hiss,
    41: \punch
);

// default family when a class isn't in the map
~defaultFamily = \servo;

// family → synth engine + parameter ranges (low..high)
~families = (
    servo: (engine: \fm, ranges: (ratio: [1.0, 3.0], index: [0.2, 7.0], cutoff: [800, 8000])),
    hiss: (engine: \noise, ranges: (center: [2000, 12000], q: [1, 12], rate: [3, 40])),
    punch: (engine: \modal, ranges: (baseHz: [60, 220], damp: [0.1, 0.9], hit: [0.0, 1.0])),
    arc: (engine: \fold, ranges: (bias: [0, 1], fold: [0, 6], spray: [0, 1]))
);

// routing: feature → parameter scaler (srcMin, srcMax → dstMin, dstMax)
~routing = (
    az_to_pan: [-180, 180, -1, 1],
    dist_to_cutoff: [1, 0, 500, 8000],
    spd_to_index: [0, 1, 0.3, 6.0],
    flow_to_rate: [0, 1, 2, 40],
    edge_to_q: [0, 1, 0.7, 12],
    sat_to_drive: [0, 1, 0, 18],
    strobe_to_hit: [0, 1, 0, 1],
    glitch_to_bit: [0, 1, 16, 4]
);

// voice budgets (keep things musical, not mushy)
~MAX_VOICES = 20;      // global cap
~PER_STREAM = 4;       // soft cap per stream
)
