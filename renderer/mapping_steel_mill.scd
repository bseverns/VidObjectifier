(
// STEEL MILL: metallic FM + tight modal thuds, aggressive drive
~classToFamily = (0: \servo, 39: \hiss, 41: \punch);

~families = (
    servo: (engine: \fm,   ranges: (ratio: [2.0, 5.0], index: [1.0, 10.0], cutoff: [1500, 14000])),
    hiss:  (engine: \noise, ranges: (center: [3500, 16000], q: [2.0, 18.0], rate: [6.0, 60.0])),
    punch: (engine: \modal, ranges: (baseHz: [70, 280],  damp: [0.05, 0.5], hit: [0.0, 1.0]))
);

~routing = (
    az_to_pan:       [-180, 180,  -1, 1],
    dist_to_cutoff:  [1, 0,  1200, 14000],
    spd_to_index:    [0, 1,  0.8, 10.0],
    edge_to_q:       [0, 1,  1.2, 18.0],
    sat_to_drive:    [0, 1,  6.0, 24.0],
    val_to_reverb:   [0, 1,  0.05, 0.4],
    glitch_to_bits:  [0, 1,  12, 4]
);
~MAX_VOICES = 20; ~PER_STREAM = 4;
)
