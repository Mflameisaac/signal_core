//! Dynamic range processing (compression, limiting) over a generic
//! real-valued time series.
//!
//! Both functions use a log-domain envelope follower: the signal's
//! instantaneous level is tracked in dB with separate attack/release time
//! constants, then a gain curve is applied sample-by-sample. Nothing here
//! assumes the series is audio — `threshold_db`/`ceiling_db` are just gain
//! break points relative to the series' own amplitude scale.

fn to_db(level: f32) -> f32 {
    20.0 * level.abs().max(1e-10).log10()
}

fn from_db(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

/// Time-constant coefficient for a one-pole envelope follower: after
/// `time_ms` milliseconds the envelope has moved ~63% of the way to a step
/// change in its target.
fn time_coeff(time_ms: f32, sample_rate_hz: f32) -> f32 {
    if time_ms <= 0.0 {
        return 0.0;
    }
    (-1.0 / (sample_rate_hz * time_ms / 1000.0)).exp()
}

/// A feed-forward dynamic range compressor.
///
/// Samples louder than `threshold_db` are attenuated by `ratio` (e.g. `4.0`
/// means every 4dB over the threshold becomes 1dB of output); `attack_ms`
/// and `release_ms` control how fast the gain reduction engages and
/// recovers; `makeup_gain_db` is a flat gain applied to the output to
/// compensate for the average level lost to compression.
pub fn compress(
    samples: &[f32],
    sample_rate_hz: f32,
    threshold_db: f32,
    ratio: f32,
    attack_ms: f32,
    release_ms: f32,
    makeup_gain_db: f32,
) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let attack_coeff = time_coeff(attack_ms, sample_rate_hz);
    let release_coeff = time_coeff(release_ms, sample_rate_hz);
    let makeup_gain = from_db(makeup_gain_db);

    let mut envelope_db = to_db(samples[0]);
    let mut out = Vec::with_capacity(samples.len());
    for &x in samples {
        let level_db = to_db(x);
        let coeff = if level_db > envelope_db {
            attack_coeff
        } else {
            release_coeff
        };
        envelope_db = coeff * envelope_db + (1.0 - coeff) * level_db;

        let gain_reduction_db = if envelope_db > threshold_db {
            let excess = envelope_db - threshold_db;
            let compressed = threshold_db + excess / ratio;
            compressed - envelope_db
        } else {
            0.0
        };

        out.push(x * from_db(gain_reduction_db) * makeup_gain);
    }
    out
}

/// A brickwall peak limiter: guarantees no output sample exceeds
/// `ceiling_db` in magnitude.
///
/// Gain reduction engages instantly (so the ceiling is never crossed) and
/// recovers over `release_ms` once the signal drops back below the ceiling,
/// to avoid audible pumping.
pub fn limit(samples: &[f32], sample_rate_hz: f32, ceiling_db: f32, release_ms: f32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let ceiling = from_db(ceiling_db);
    let release_coeff = time_coeff(release_ms, sample_rate_hz);

    let mut gain = 1.0f32;
    let mut out = Vec::with_capacity(samples.len());
    for &x in samples {
        let peak = x.abs();
        let target_gain = if peak > ceiling { ceiling / peak } else { 1.0 };
        gain = if target_gain < gain {
            target_gain
        } else {
            release_coeff * gain + (1.0 - release_coeff) * target_gain
        };
        out.push(x * gain);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine_wave(freq_hz: f32, sample_rate_hz: f32, amplitude: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| amplitude * (2.0 * PI * freq_hz * i as f32 / sample_rate_hz).sin())
            .collect()
    }

    fn peak(samples: &[f32]) -> f32 {
        samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()))
    }

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    #[test]
    fn compress_reduces_level_of_signal_above_threshold() {
        let sample_rate = 8000.0;
        let loud = sine_wave(200.0, sample_rate, 1.0, 4000);
        let out = compress(&loud, sample_rate, -12.0, 4.0, 1.0, 50.0, 0.0);
        // steady loud tone well above threshold should end up quieter than input
        assert!(rms(&out[1000..]) < rms(&loud[1000..]));
    }

    #[test]
    fn compress_leaves_signal_below_threshold_unchanged() {
        let sample_rate = 8000.0;
        let quiet = sine_wave(200.0, sample_rate, 0.05, 4000); // ~ -26 dBFS peak
        let out = compress(&quiet, sample_rate, -12.0, 4.0, 1.0, 50.0, 0.0);
        for (a, b) in quiet.iter().zip(out.iter()).skip(1000) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn compress_makeup_gain_raises_level() {
        let sample_rate = 8000.0;
        let quiet = sine_wave(200.0, sample_rate, 0.05, 4000);
        let unity = compress(&quiet, sample_rate, -12.0, 4.0, 1.0, 50.0, 0.0);
        let boosted = compress(&quiet, sample_rate, -12.0, 4.0, 1.0, 50.0, 6.0);
        assert!(rms(&boosted[1000..]) > rms(&unity[1000..]) * 1.5);
    }

    #[test]
    fn compress_handles_empty_input() {
        assert!(compress(&[], 8000.0, -12.0, 4.0, 1.0, 50.0, 0.0).is_empty());
    }

    #[test]
    fn limit_never_exceeds_ceiling() {
        let sample_rate = 8000.0;
        let loud = sine_wave(200.0, sample_rate, 1.0, 4000); // 0 dBFS tone
        let ceiling_db = -3.0;
        let out = limit(&loud, sample_rate, ceiling_db, 50.0);
        let ceiling_linear = from_db(ceiling_db);
        assert!(
            peak(&out) <= ceiling_linear + 1e-4,
            "peak {} exceeds ceiling {}",
            peak(&out),
            ceiling_linear
        );
    }

    #[test]
    fn limit_leaves_quiet_signal_unchanged() {
        let sample_rate = 8000.0;
        let quiet = sine_wave(200.0, sample_rate, 0.1, 4000);
        let out = limit(&quiet, sample_rate, -3.0, 50.0);
        for (a, b) in quiet.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn limit_handles_empty_input() {
        assert!(limit(&[], 8000.0, -3.0, 50.0).is_empty());
    }
}
