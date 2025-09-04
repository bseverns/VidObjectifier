(
// Mapping file: decides which detection classes trigger which sonic "families"
// and how features from the CSV map onto synth parameters.

// class → family (only a few example classes for now)
~classToFamily = (
    0: \servo,     // person
    39: \hiss,     // bottle
    41: \punch     // cup
);

// family → synth engine + parameter ranges (low..high)
~families = (
    servo: (engine: \fm,   ranges: (ratio: [1.0, 3.0], index: [0.2, 6.0], cutoff: [800, 8000])),
    hiss:  (engine: \noise, ranges: (center: [2000, 12000], q: [1.0, 12.0], rate: [2.0, 40.0])),
    punch: (engine: \modal, ranges: (baseHz: [60, 220], damp: [0.1, 0.9], hit: [0.0, 1.0])),
    arc:   (engine: \fold, ranges: (bias: [0, 1], fold: [0, 6], spray: [0,1]))
);

// routing: feature → parameter scaler (srcMin, srcMax → dstMin, dstMax)
~routing = (
    az_to_pan:       [-180, 180,  -1, 1],          // left/right
    dist_to_cutoff:  [1, 0,  500, 8000],           // near = bright
    spd_to_index:    [0, 1,  0.3, 6.0],
    edge_to_q:       [0, 1,  0.7, 12.0],
    sat_to_drive:    [0, 1,  0.0, 18.0],           // dB into soft clip
    val_to_reverb:   [0, 1,  0.1, 0.8],
    glitch_to_bits:  [0, 1,  16, 4]
);

// voice budgets
~MAX_VOICES = 20;      // global cap
~PER_STREAM = 4;       // soft cap per stream
)
