// Renderer: turn score.csv into sound.
// Run this in SuperCollider after booting the server.  Up to 20 voices will
// spawn, one per tracked object.  For 8‑channel output use render_ring8.scd.
s.options.numOutputBusChannels = 2;
s.boot;

(
~scaler = { |v, a,b, c,d| (v.clip(a,b)-a)/(b-a) * (d-c) + c }; // simple value mapper
~voices = IdentityDictionary.new;  // active synths keyed by "stream:oid"
~perStreamCounts = ();             // how many voices per stream this tick

// ===== engines (the actual synths) =====
SynthDef(\engine_fm, { |out=0, pan=0, ratio=1, index=1, cutoff=4000, drive=0, amp=0.15|
    var carHz = Lag.kr(60, 0.2) * ratio;               // base pitch ~60 Hz
    var mod = SinOsc.ar(carHz * ratio, 0, carHz*index); // FM modulator
    var sig = SinOsc.ar(carHz + mod);
    sig = RLPF.ar(sig, cutoff.clip(100,18000), 0.3);    // simple low‑pass
    sig = (sig * (1 + drive.dbamp)).tanh;               // soft clip
    Out.ar(out, Pan2.ar(sig, pan, amp));
}).add;

SynthDef(\engine_noise, { |out=0, pan=0, center=4000, q=4, rate=8, drive=0, amp=0.15|
    var src = PinkNoise.ar(0.4) + LFNoise2.ar(rate, 0.4);
    var bw = (center / q).clip(50, 8000);
    var sig = BPF.ar(src, center.clip(100,16000), bw/center);
    sig = (sig * (1 + drive.dbamp)).tanh;
    Out.ar(out, Pan2.ar(sig, pan, amp));
}).add;

SynthDef(\engine_modal, { |out=0, pan=0, baseHz=120, damp=0.4, hit=0.0, drive=0, amp=0.2|
    // tiny three‑resonator drum
    var exc = Decay2.kr(TDuty.kr(Impulse.kr(hit*20 + 0.1), 0, 1), 0.001, 0.02);
    var sig = Klank.ar(Ref([[baseHz, baseHz*1.25, baseHz*1.5], [1,0.8,0.7], [damp,damp*0.7,damp*0.5]]), WhiteNoise.ar*exc);
    sig = (sig * (1 + drive.dbamp)).tanh;
    Out.ar(out, Pan2.ar(sig, pan, amp));
}).add;

// ===== load mapping file (class→timbre) =====
this.executeFile((thisProcess.nowExecutingPath.dirname +/+ "mapping.scd"));

// ===== helper functions =====
~spawn = { |key, family, pan=0, amp=0.18|
    // instantiate a synth for a new object
    var engine = ~families[family].engine;
    var name = switch(engine, \fm, { \engine_fm }, \noise, { \engine_noise }, \modal, { \engine_modal }, \fold, { \engine_noise });
    var syn = Synth(name, [\pan, pan, \amp, amp]);
    ~voices[key] = (s: syn, t: thisThread.seconds);
};

~setParams = { |syn, fam, feats|
    // map features from the CSV into synth parameters
    var r = ~families[fam].ranges;
    var p = ();
    p[\pan]   = ~scaler.(feats[\az], *~routing[\az_to_pan]);
    p[\drive] = ~scaler.(feats[\sat], *~routing[\sat_to_drive]);
    switch(~families[fam].engine,
        \fm, {
            p[\ratio]  = ~scaler.(feats[\edge], 0,1, r[\ratio][0], r[\ratio][1]);
            p[\index]  = ~scaler.(feats[\spd],  0,1, *~routing[\spd_to_index][2..3]);
            p[\cutoff] = ~scaler.(feats[\dist], *~routing[\dist_to_cutoff]);
        },
        \noise, {
            p[\center] = ~scaler.(feats[\hue]/360, 0,1, r[\center][0], r[\center][1]);
            p[\q]      = ~scaler.(feats[\edge], 0,1, r[\q][0], r[\q][1]);
            p[\rate]   = ~scaler.(feats[\flow] ? feats[\spd], 0,1, r[\rate][0], r[\rate][1]);
        },
        \modal, {
            p[\baseHz] = ~scaler.(feats[\dist], 1,0, r[\baseHz][0], r[\baseHz][1]);
            p[\damp]   = ~scaler.(feats[\edge], 0,1, r[\damp][0], r[\damp][1]);
            p[\hit]    = feats[\strobe] ? 0.0;
        }
    );
    syn.set(*p.asPairs);
};

~eligible = { |stream|
    // enforce global/per‑stream voice limits
    var c = (~perStreamCounts[stream] ? 0);
    ( ~voices.size < ~MAX_VOICES ) && ( c < ~PER_STREAM );
};

~tickCounts = {
    // recompute per‑stream counts each control tick
    ~perStreamCounts = ();
    ~voices.keysValuesDo { |key, rec|
        var stream = key.split($:).first;
        ~perStreamCounts[stream] = (~perStreamCounts[stream] ? 0) + 1;
    };
};

~playScore = { |scorePath|
    // read the CSV in a routine so timing survives
    var f = File(scorePath, "r"); var header = f.getLine;
    var t0 = thisThread.seconds;
    Routine({
        var line;
        while({ (line = f.getLine).notNil }, {
            var cols = line.split($,);
            var t,stream,oid,cls,az,el,dist,spd,conf,glitch,hue,sat,val,edge;
            # t,stream,oid,cls,az,el,dist,spd,conf,glitch,hue,sat,val,edge = cols;
            t = t.asFloat;
            (t - (thisThread.seconds - t0)).max(0).wait;

            var fam = (~classToFamily[cls.asInteger] ? \servo);
            var key = "%:%".format(stream, oid);
            var feats = (
                az: az.asFloat, el: el.asFloat, dist: dist.asFloat, spd: spd.asFloat,
                glitch: glitch.asFloat, hue: hue.asFloat, sat: sat.asFloat, val: val.asFloat,
                edge: edge.asFloat
            );

            ~tickCounts.();

            if(~voices[key].isNil and: { ~eligible.(stream) }, {
                ~spawn.(key, fam, ~scaler.(feats[\az], *~routing[\az_to_pan]));
            });

            if(~voices[key].notNil, {
                var rec = ~voices[key]; rec[\t] = thisThread.seconds;
                ~setParams.(rec[\s], fam, feats);
            });
        });
        f.close;

        3.wait;                                // let tails ring
        ~voices.do { |rec| rec[\s].free };
        ~voices.clear;
    }).play(AppClock);
};
)

// Example usage (edit the path):
// ~playScore.(thisProcess.nowExecutingPath.dirname +/+ "../examples/score_example.csv");
