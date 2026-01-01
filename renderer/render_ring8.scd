// Renderer for 8 speakers in a ring.
// Same idea as render.scd but instead of stereo panning we drop energy into a
// circular array of eight outputs.
s.options.numOutputBusChannels = 8;
s.boot;

(
~scaler = { |v, a,b, c,d| (v.clip(a,b)-a)/(b-a) * (d-c) + c };
~voices = IdentityDictionary.new;
~perStreamCounts = ();

~ringPan = { |sig, az|
    // azimuth -180..180 → ring index 0..7, linear interp between adjacent
    var n = 8;
    var ang = (az.wrap(-180, 180) + 180) / 360 * n; // 0..8
    var i0 = ang.floor % n;
    var i1 = (i0 + 1) % n;
    var frac = ang - ang.floor;
    Array.fill(n, { |i| (i==i0).if({(1-frac)*sig}, {(i==i1).if({frac*sig},{0})}) });
};

// engines (same as stereo but no Pan2)
SynthDef(\engine_fm8, { |ratio=1, index=1, cutoff=4000, drive=0, az=0, amp=0.12|
    var carHz = Lag.kr(60, 0.2) * ratio;
    var mod = SinOsc.ar(carHz * ratio, 0, carHz*index);
    var sig = SinOsc.ar(carHz + mod);
    sig = RLPF.ar(sig, cutoff.clip(100,18000), 0.3);
    sig = (sig * (1 + drive.dbamp)).tanh * amp;
    Out.ar(0, ~ringPan.(sig, az));
}).add;

SynthDef(\engine_noise8, { |center=4000, q=4, rate=8, drive=0, az=0, amp=0.12|
    var src = PinkNoise.ar(0.4) + LFNoise2.ar(rate, 0.4);
    var bw = (center / q).clip(50, 8000);
    var sig = BPF.ar(src, center.clip(100,16000), bw/center);
    sig = (sig * (1 + drive.dbamp)).tanh * amp;
    Out.ar(0, ~ringPan.(sig, az));
}).add;

SynthDef(\engine_modal8, { |baseHz=120, damp=0.4, hit=0.0, drive=0, az=0, amp=0.16|
    var exc = Decay2.kr(TDuty.kr(Impulse.kr(hit*20 + 0.1), 0, 1), 0.001, 0.02);
    var sig = Klank.ar(Ref([[baseHz, baseHz*1.25, baseHz*1.5], [1,0.8,0.7], [damp,damp*0.7,damp*0.5]]), WhiteNoise.ar*exc);
    sig = (sig * (1 + drive.dbamp)).tanh * amp;
    Out.ar(0, ~ringPan.(sig, az));
}).add;

this.executeFile((thisProcess.nowExecutingPath.dirname +/+ "mapping.scd"));

~spawn = { |key, family, az=0, amp=0.14|
    var engine = ~families[family].engine;
    var name = switch(engine, \fm, { \engine_fm8 }, \noise, { \engine_noise8 }, \modal, { \engine_modal8 }, \fold, { \engine_noise8 });
    var syn = Synth(name, [\az, az, \amp, amp]);
    ~voices[key] = (s: syn, t: thisThread.seconds);
};

~setParams = { |syn, fam, feats|
    var r = ~families[fam].ranges;
    var p = ();
    p[\az]    = feats[\az];
    p[\drive]  = ~scaler.(feats[\sat], *~routing[\sat_to_drive]);
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
    var c = (~perStreamCounts[stream] ? 0);
    ( ~voices.size < ~MAX_VOICES ) && ( c < ~PER_STREAM );
};

~tickCounts = {
    ~perStreamCounts = ();
    ~voices.keysValuesDo { |key, rec|
        var stream = key.split($:).first;
        ~perStreamCounts[stream] = (~perStreamCounts[stream] ? 0) + 1;
    };
};

~playScore = { |scorePath|
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
                ~spawn.(key, fam, feats[\az]);
            });

            if(~voices[key].notNil, {
                var rec = ~voices[key]; rec[\t] = thisThread.seconds;
                ~setParams.(rec[\s], fam, feats);
            });
        });
        f.close; 3.wait;
        ~voices.do { |rec| rec[\s].free }; ~voices.clear;
    }).play(AppClock);
};
)

// Example usage:
// ~playScore.(thisProcess.nowExecutingPath.dirname +/+ "../examples/score_example_template.csv");
