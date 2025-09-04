(
// RUST CHOIR: slow drones, foldback grit, moody reverbs
~classToFamily = (0: \arc, 39: \servo, 41: \modal);

~families = (
    servo: (engine: \fm,   ranges: (ratio: [0.5, 1.5], index: [0.1, 2.0], cutoff: [400, 6000])),
    noise: (engine: \noise, ranges: (center: [800, 6000], q: [0.7, 6.0], rate: [0.5, 8.0])),
    modal: (engine: \modal, ranges: (baseHz: [40, 140], damp: [0.4, 0.95], hit: [0.0, 0.3])),
    arc:   (engine: \fold,  ranges: (bias: [0, 1], fold: [2, 10], spray: [0,1]))
);

~routing = (
    az_to_pan:       [-180, 180,  -1, 1],
    dist_to_cutoff:  [1, 0,  300, 6000],
    spd_to_index:    [0, 1,  0.0, 2.0],
    edge_to_q:       [0, 1,  0.7, 6.0],
    sat_to_drive:    [0, 1,  0.0, 18.0],
    val_to_reverb:   [0, 1,  0.4, 0.95],
    glitch_to_bits:  [0, 1,  12, 5]
);
~MAX_VOICES = 20; ~PER_STREAM = 4;
)
