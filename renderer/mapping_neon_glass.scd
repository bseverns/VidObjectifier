(
// NEON GLASS: airy band-noise, shimmering highs, less thud
~classToFamily = (0: \hiss, 39: \hiss, 41: \servo); // bias to airy textures

~families = (
    servo: (engine: \fm,   ranges: (ratio: [1.0, 2.5], index: [0.1, 3.0], cutoff: [2000, 16000])),
    hiss:  (engine: \noise, ranges: (center: [5000, 16000], q: [0.8, 8.0], rate: [3.0, 24.0])),
    punch: (engine: \modal, ranges: (baseHz: [80, 180], damp: [0.2, 0.9], hit: [0.0, 0.5]))
);

~routing = (
    az_to_pan:       [-180, 180,  -1, 1],
    dist_to_cutoff:  [1, 0,  3000, 16000],
    spd_to_index:    [0, 1,  0.1, 3.0],
    edge_to_q:       [0, 1,  0.7, 8.0],
    sat_to_drive:    [0, 1,  0.0, 12.0],
    val_to_reverb:   [0, 1,  0.2, 0.9],
    glitch_to_bits:  [0, 1,  16, 6]
);
~MAX_VOICES = 20; ~PER_STREAM = 4;
)
