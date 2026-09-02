//! Second-order (biquad) IIR filters via the RBJ Audio EQ Cookbook formulas.
//!
//! The single-pole filters in [`crate::filter`] are cheap but shallow
//! (6dB/octave, no gain control) — fine for a simple highpass/lowpass, not
//! enough for parametric EQ (boost/cut a band) or steeper rolloffs. A
//! biquad is the standard building block for both: this module provides the
//! coefficient math and a stateful [`Biquad`] that processes one sample at a
//! time, plus [`apply_biquad`] to run it over a whole series in the same
//! functional style the rest of the crate uses.

/// A single second-order IIR section (Direct Form I), with its own filter
/// state (previous two input/output samples). Reuse one `Biquad` per
/// independent signal (e.g. one per channel) — sharing a single instance
/// across unrelated series would mix their filter history together.
#[derive(Clone, Copy, Debug)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Biquad {
    /// Builds a biquad directly from its (already `a0`-normalized)
    /// difference-equation coefficients. Exposed at crate visibility so
    /// [`crate::loudness`] can construct the ITU-R BS.1770 K-weighting
    /// filters, whose coefficients come from a different derivation than
    /// the RBJ constructors below.
    pub(crate) fn from_coefficients(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Self {
        Biquad {
            b0,
            b1,
            b2,
            a1,
            a2,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Processes one sample, updating internal state.
    pub fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// Runs a biquad over a whole series, returning a new `Vec` the same length
/// as the input. `biquad` keeps whatever state it already had — pass a
/// freshly-constructed one for an independent series.
pub fn apply_biquad(samples: &[f32], biquad: &mut Biquad) -> Vec<f32> {
    samples.iter().map(|&x| biquad.process(x)).collect()
}

/// Shared RBJ prep: `(w0.cos(), alpha)` for a given center/corner frequency
/// and Q. `alpha = sin(w0) / (2*Q)` controls bandwidth/slope; a higher Q is
/// a narrower peaking band or a steeper shelf/cutoff.
fn cookbook_prep(freq_hz: f32, q: f32, sample_rate_hz: f32) -> (f64, f64) {
    let w0 = 2.0 * std::f64::consts::PI * freq_hz as f64 / sample_rate_hz as f64;
    let alpha = w0.sin() / (2.0 * q as f64);
    (w0.cos(), alpha)
}

/// Parametric (peaking) EQ: boosts or cuts a band centered on `freq_hz` by
/// `gain_db`, with bandwidth controlled by `q` (higher = narrower).
pub fn peaking_eq(freq_hz: f32, gain_db: f32, q: f32, sample_rate_hz: f32) -> Biquad {
    let (cosw0, alpha) = cookbook_prep(freq_hz, q, sample_rate_hz);
    let a = 10f64.powf(gain_db as f64 / 40.0);

    let a0 = 1.0 + alpha / a;
    let b0 = (1.0 + alpha * a) / a0;
    let b1 = (-2.0 * cosw0) / a0;
    let b2 = (1.0 - alpha * a) / a0;
    let a1 = (-2.0 * cosw0) / a0;
    let a2 = (1.0 - alpha / a) / a0;

    Biquad::from_coefficients(b0 as f32, b1 as f32, b2 as f32, a1 as f32, a2 as f32)
}

/// Low-shelf: boosts or cuts everything below `freq_hz` by `gain_db`.
pub fn low_shelf(freq_hz: f32, gain_db: f32, q: f32, sample_rate_hz: f32) -> Biquad {
    let (cosw0, alpha) = cookbook_prep(freq_hz, q, sample_rate_hz);
    let a = 10f64.powf(gain_db as f64 / 40.0);
    let sqrt_a_2alpha = 2.0 * a.sqrt() * alpha;

    let a0 = (a + 1.0) + (a - 1.0) * cosw0 + sqrt_a_2alpha;
    let b0 = a * ((a + 1.0) - (a - 1.0) * cosw0 + sqrt_a_2alpha) / a0;
    let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cosw0) / a0;
    let b2 = a * ((a + 1.0) - (a - 1.0) * cosw0 - sqrt_a_2alpha) / a0;
    let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cosw0) / a0;
    let a2 = ((a + 1.0) + (a - 1.0) * cosw0 - sqrt_a_2alpha) / a0;

    Biquad::from_coefficients(b0 as f32, b1 as f32, b2 as f32, a1 as f32, a2 as f32)
}

/// High-shelf: boosts or cuts everything above `freq_hz` by `gain_db`.
pub fn high_shelf(freq_hz: f32, gain_db: f32, q: f32, sample_rate_hz: f32) -> Biquad {
    let (cosw0, alpha) = cookbook_prep(freq_hz, q, sample_rate_hz);
    let a = 10f64.powf(gain_db as f64 / 40.0);
    let sqrt_a_2alpha = 2.0 * a.sqrt() * alpha;

    let a0 = (a + 1.0) - (a - 1.0) * cosw0 + sqrt_a_2alpha;
    let b0 = a * ((a + 1.0) + (a - 1.0) * cosw0 + sqrt_a_2alpha) / a0;
    let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cosw0) / a0;
    let b2 = a * ((a + 1.0) + (a - 1.0) * cosw0 - sqrt_a_2alpha) / a0;
    let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cosw0) / a0;
    let a2 = ((a + 1.0) - (a - 1.0) * cosw0 - sqrt_a_2alpha) / a0;

    Biquad::from_coefficients(b0 as f32, b1 as f32, b2 as f32, a1 as f32, a2 as f32)
}

/// 2-pole low-pass (steeper rolloff than [`crate::filter::low_pass_filter`]'s
/// single-pole version). `q` of `0.7071` (≈1/√2) gives a maximally-flat
/// (Butterworth) response.
pub fn low_pass(freq_hz: f32, q: f32, sample_rate_hz: f32) -> Biquad {
    let (cosw0, alpha) = cookbook_prep(freq_hz, q, sample_rate_hz);
    let a0 = 1.0 + alpha;
    let b0 = ((1.0 - cosw0) / 2.0) / a0;
    let b1 = (1.0 - cosw0) / a0;
    let b2 = ((1.0 - cosw0) / 2.0) / a0;
    let a1 = (-2.0 * cosw0) / a0;
    let a2 = (1.0 - alpha) / a0;

    Biquad::from_coefficients(b0 as f32, b1 as f32, b2 as f32, a1 as f32, a2 as f32)
}

/// 2-pole high-pass (steeper rolloff than [`crate::filter::high_pass_filter`]'s
/// single-pole version). `q` of `0.7071` (≈1/√2) gives a maximally-flat
/// (Butterworth) response.
pub fn high_pass(freq_hz: f32, q: f32, sample_rate_hz: f32) -> Biquad {
    let (cosw0, alpha) = cookbook_prep(freq_hz, q, sample_rate_hz);
    let a0 = 1.0 + alpha;
    let b0 = ((1.0 + cosw0) / 2.0) / a0;
    let b1 = (-(1.0 + cosw0)) / a0;
    let b2 = ((1.0 + cosw0) / 2.0) / a0;
    let a1 = (-2.0 * cosw0) / a0;
    let a2 = (1.0 - alpha) / a0;

    Biquad::from_coefficients(b0 as f32, b1 as f32, b2 as f32, a1 as f32, a2 as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine_wave(freq_hz: f32, sample_rate_hz: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sample_rate_hz).sin())
            .collect()
    }

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    #[test]
    fn peaking_eq_boosts_the_targeted_frequency_more_than_a_distant_one() {
        let sample_rate = 48000.0;
        let n = 8192;
        let target = sine_wave(1000.0, sample_rate, n);
        let distant = sine_wave(60.0, sample_rate, n);

        let mut f1 = peaking_eq(1000.0, 12.0, 1.0, sample_rate);
        let boosted_target = apply_biquad(&target, &mut f1);
        let mut f2 = peaking_eq(1000.0, 12.0, 1.0, sample_rate);
        let boosted_distant = apply_biquad(&distant, &mut f2);

        let target_gain = rms(&boosted_target[1000..]) / rms(&target[1000..]);
        let distant_gain = rms(&boosted_distant[1000..]) / rms(&distant[1000..]);

        assert!(
            target_gain > 3.0,
            "1kHz tone should be boosted close to +12dB (~4x), got {target_gain}x"
        );
        assert!(
            distant_gain < 1.2,
            "60Hz tone should be mostly unaffected by a 1kHz peaking boost, got {distant_gain}x"
        );
    }

    #[test]
    fn peaking_eq_cut_reduces_the_targeted_frequency() {
        let sample_rate = 48000.0;
        let n = 8192;
        let target = sine_wave(350.0, sample_rate, n);
        let mut f = peaking_eq(350.0, -6.0, 1.5, sample_rate);
        let cut = apply_biquad(&target, &mut f);
        let gain = rms(&cut[1000..]) / rms(&target[1000..]);
        assert!(gain < 0.7, "expected a cut close to -6dB (~0.5x), got {gain}x");
    }

    #[test]
    fn low_shelf_boosts_low_frequencies_not_high() {
        let sample_rate = 48000.0;
        let n = 8192;
        let low = sine_wave(80.0, sample_rate, n);
        let high = sine_wave(8000.0, sample_rate, n);

        let mut f1 = low_shelf(200.0, 6.0, 0.7, sample_rate);
        let low_out = apply_biquad(&low, &mut f1);
        let mut f2 = low_shelf(200.0, 6.0, 0.7, sample_rate);
        let high_out = apply_biquad(&high, &mut f2);

        let low_gain = rms(&low_out[1000..]) / rms(&low[1000..]);
        let high_gain = rms(&high_out[1000..]) / rms(&high[1000..]);

        assert!(low_gain > 1.5, "low tone should be boosted, got {low_gain}x");
        assert!(high_gain < 1.2, "high tone should be near-unaffected, got {high_gain}x");
    }

    #[test]
    fn high_shelf_boosts_high_frequencies_not_low() {
        let sample_rate = 48000.0;
        let n = 8192;
        let low = sine_wave(80.0, sample_rate, n);
        let high = sine_wave(8000.0, sample_rate, n);

        let mut f1 = high_shelf(4000.0, 6.0, 0.7, sample_rate);
        let low_out = apply_biquad(&low, &mut f1);
        let mut f2 = high_shelf(4000.0, 6.0, 0.7, sample_rate);
        let high_out = apply_biquad(&high, &mut f2);

        let low_gain = rms(&low_out[1000..]) / rms(&low[1000..]);
        let high_gain = rms(&high_out[1000..]) / rms(&high[1000..]);

        assert!(high_gain > 1.5, "high tone should be boosted, got {high_gain}x");
        assert!(low_gain < 1.2, "low tone should be near-unaffected, got {low_gain}x");
    }

    #[test]
    fn biquad_low_pass_and_high_pass_are_steeper_than_single_pole() {
        // Same intent as crate::filter's single-pole test, just confirming
        // the 2-pole versions still behave directionally correctly.
        let sample_rate = 8000.0;
        let n = 4096;
        let low_tone = sine_wave(20.0, sample_rate, n);
        let high_tone = sine_wave(2000.0, sample_rate, n);
        let cutoff = 100.0;

        let mut lp1 = low_pass(cutoff, 0.7071, sample_rate);
        let low_through_lp = apply_biquad(&low_tone, &mut lp1);
        let mut lp2 = low_pass(cutoff, 0.7071, sample_rate);
        let high_through_lp = apply_biquad(&high_tone, &mut lp2);
        assert!(rms(&low_through_lp[500..]) / rms(&low_tone[500..]) > 0.9);
        assert!(rms(&high_through_lp[500..]) / rms(&high_tone[500..]) < 0.1);

        let mut hp1 = high_pass(cutoff, 0.7071, sample_rate);
        let low_through_hp = apply_biquad(&low_tone, &mut hp1);
        let mut hp2 = high_pass(cutoff, 0.7071, sample_rate);
        let high_through_hp = apply_biquad(&high_tone, &mut hp2);
        assert!(rms(&low_through_hp[500..]) / rms(&low_tone[500..]) < 0.1);
        assert!(rms(&high_through_hp[500..]) / rms(&high_tone[500..]) > 0.9);
    }
}
