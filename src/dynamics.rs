//! Dynamic-range processing: an attack/release envelope follower, and the
//! compressor/limiter built on top of it.
//!
//! Both a compressor (turn loud parts down, gently, above a threshold) and
//! a limiter (never let the signal cross a ceiling) are the same shape of
//! algorithm — track a signal's level, decide how much gain reduction that
//! level calls for, smooth the gain change over time so it doesn't click —
//! just with different ratios/speeds/lookahead. This module keeps that
//! shared shape explicit via [`EnvelopeFollower`] rather than duplicating
//! the smoothing logic in each function.

/// Attack/release-smoothed level tracker: the shared primitive a compressor
/// or limiter uses to decide *how fast* its gain reduction should move, as
/// opposed to *how much* (that's the gain computer in [`compress`]/[`limit`]
/// itself). A lower `time_seconds` means the envelope reaches a new target
/// faster.
struct EnvelopeFollower {
    coeff_up: f32,
    coeff_down: f32,
    value: f32,
}

impl EnvelopeFollower {
    /// `initial_value` matters: [`compress`] tracks gain reduction in dB
    /// (unity = `0.0`), while [`limit`] tracks gain as a linear multiplier
    /// (unity = `1.0`) — starting from the wrong "no reduction yet" value
    /// would make a signal that never needs gain reduction ramp audibly
    /// from silence/full-attenuation instead of passing through unchanged.
    fn new(attack_seconds: f32, release_seconds: f32, sample_rate_hz: f32, initial_value: f32) -> Self {
        EnvelopeFollower {
            coeff_up: time_to_coeff(attack_seconds, sample_rate_hz),
            coeff_down: time_to_coeff(release_seconds, sample_rate_hz),
            value: initial_value,
        }
    }

    /// Moves the envelope toward `target`, using the attack coefficient
    /// when `target` is below the current value (gain reduction engaging)
    /// and the release coefficient when it's above (gain reduction easing
    /// off) — i.e. "attack"/"release" here track *gain*, not raw level.
    fn step_toward(&mut self, target: f32) -> f32 {
        let coeff = if target < self.value {
            self.coeff_up
        } else {
            self.coeff_down
        };
        self.value += (target - self.value) * coeff;
        self.value
    }
}

/// One-pole smoothing coefficient for a given time constant: after
/// `time_seconds`, the envelope has closed ~63% of the gap to its target.
/// `0` (or negative) means "instant" — closes the gap completely every
/// sample.
fn time_to_coeff(time_seconds: f32, sample_rate_hz: f32) -> f32 {
    if time_seconds <= 0.0 {
        1.0
    } else {
        1.0 - (-1.0 / (sample_rate_hz * time_seconds)).exp()
    }
}

fn linear_to_db(x: f32) -> f32 {
    20.0 * x.max(1e-8).log10()
}

fn db_to_linear(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

/// Downward compressor: above `threshold_db`, gain is reduced so the signal
/// only rises `1/ratio` as much as it otherwise would past the threshold
/// (a `ratio` of `1.0` is a no-op; higher squashes harder). `attack_ms`/
/// `release_ms` control how fast the gain reduction engages/releases;
/// `makeup_db` is a flat gain added back afterward (compressing quietens
/// the loud parts, so a mastering chain typically follows with makeup gain
/// to bring the overall level back up).
pub fn compress(
    samples: &[f32],
    sample_rate_hz: f32,
    threshold_db: f32,
    ratio: f32,
    attack_ms: f32,
    release_ms: f32,
    makeup_db: f32,
) -> Vec<f32> {
    let mut envelope = EnvelopeFollower::new(attack_ms / 1000.0, release_ms / 1000.0, sample_rate_hz, 0.0);
    let makeup = db_to_linear(makeup_db);

    samples
        .iter()
        .map(|&x| {
            let level_db = linear_to_db(x.abs());
            let over_db = level_db - threshold_db;
            let target_gain_db = if over_db > 0.0 {
                -(over_db * (1.0 - 1.0 / ratio))
            } else {
                0.0
            };
            let smoothed_gain_db = envelope.step_toward(target_gain_db);
            x * db_to_linear(smoothed_gain_db) * makeup
        })
        .collect()
}

/// Lookahead peak limiter: guarantees the output never exceeds `ceiling_db`
/// (given as dBFS, e.g. `-1.0`), by scanning a short window *ahead* of each
/// sample for the loudest upcoming peak and reducing gain in advance —
/// otherwise a sudden transient would clip before a purely-reactive gain
/// reduction could respond. Since this runs offline over a whole in-memory
/// buffer (not real-time), lookahead is "free": no output delay bookkeeping
/// needed, unlike a real-time plugin. Gain recovers back toward unity at
/// `release_ms` after a peak has passed; reduction itself is treated as
/// instant, which is what makes this a limiter rather than a compressor.
pub fn limit(samples: &[f32], sample_rate_hz: f32, ceiling_db: f32, release_ms: f32) -> Vec<f32> {
    const LOOKAHEAD_MS: f32 = 5.0;
    let lookahead_samples = ((LOOKAHEAD_MS / 1000.0) * sample_rate_hz).round() as usize;
    let ceiling = db_to_linear(ceiling_db);
    let n = samples.len();

    let mut envelope = EnvelopeFollower::new(0.0, release_ms / 1000.0, sample_rate_hz, 1.0);
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let window_end = (i + lookahead_samples + 1).min(n);
        let mut required_gain = 1.0f32;
        for &s in &samples[i..window_end] {
            let peak = s.abs();
            if peak > 1e-8 {
                required_gain = required_gain.min(ceiling / peak).min(1.0);
            }
        }
        let gain = envelope.step_toward(required_gain);
        out.push(samples[i] * gain);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constant(value: f32, n: usize) -> Vec<f32> {
        vec![value; n]
    }

    #[test]
    fn compress_reduces_gain_above_threshold_by_roughly_expected_amount() {
        let sample_rate = 48000.0;
        // -6dBFS input, well above a -20dB threshold, fast attack/release so
        // the envelope settles within the test signal's length.
        let input = constant(db_to_linear(-6.0), 4800);
        let out = compress(&input, sample_rate, -20.0, 4.0, 1.0, 1.0, 0.0);

        // Settled region: skip the attack ramp.
        let settled = linear_to_db(out[4000].abs());
        // Expected: threshold + (level - threshold)/ratio = -20 + (14)/4 = -16.5 dBFS
        assert!(
            (settled - (-16.5)).abs() < 1.0,
            "expected compressed level near -16.5dBFS, got {settled}dBFS"
        );
    }

    #[test]
    fn compress_below_threshold_is_unchanged() {
        let sample_rate = 48000.0;
        let input = constant(db_to_linear(-40.0), 2000);
        let out = compress(&input, sample_rate, -20.0, 4.0, 1.0, 1.0, 0.0);
        assert!(
            (out[1500] - input[1500]).abs() < 1e-4,
            "signal below threshold should pass through unchanged"
        );
    }

    #[test]
    fn compress_makeup_gain_is_applied() {
        let sample_rate = 48000.0;
        let input = constant(db_to_linear(-40.0), 2000);
        let out = compress(&input, sample_rate, -20.0, 4.0, 1.0, 1.0, 6.0);
        let expected = input[1500] * db_to_linear(6.0);
        assert!(
            (out[1500] - expected).abs() < 1e-4,
            "makeup gain should apply even when below threshold, got {} expected {}",
            out[1500],
            expected
        );
    }

    #[test]
    fn limit_never_exceeds_ceiling_on_a_sudden_transient() {
        let sample_rate = 48000.0;
        let mut input = vec![0.1f32; 2000];
        // A sudden, sharp transient well above the ceiling.
        input[1000] = 1.5;
        input[1001] = -1.4;

        let out = limit(&input, sample_rate, -1.0, 50.0);
        let ceiling = db_to_linear(-1.0);
        for &s in &out {
            assert!(
                s.abs() <= ceiling + 1e-4,
                "limiter output {s} exceeded ceiling {ceiling}"
            );
        }
    }

    #[test]
    fn limit_leaves_quiet_signal_unchanged() {
        let sample_rate = 48000.0;
        let input = constant(0.1, 1000);
        let out = limit(&input, sample_rate, -1.0, 50.0);
        for (&a, &b) in input.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-4, "quiet signal should pass through unchanged");
        }
    }

    #[test]
    fn limit_anticipates_an_upcoming_peak_via_lookahead() {
        let sample_rate = 48000.0;
        let mut input = vec![0.5f32; 200];
        input[100] = 2.0; // one sharp spike
        let out = limit(&input, sample_rate, -1.0, 20.0);
        // Because of lookahead, gain should already be reduced a few
        // samples *before* the spike, not just at/after it.
        assert!(
            out[95] < input[95],
            "gain reduction should begin before the peak due to lookahead"
        );
    }
}
