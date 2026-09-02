//! Second-order IIR ("biquad") filters over a generic real-valued time series.
//!
//! Coefficients follow the RBJ Audio EQ Cookbook formulas. A [`Biquad`] is a
//! stateful filter instance — [`apply_biquad`] takes `&mut Biquad` and
//! carries its internal delay line across calls, so callers can stream a
//! signal through in chunks and get the same result as one long call.

/// A stateful second-order IIR filter (Direct Form I).
///
/// Built by one of the constructor functions in this module ([`high_pass`],
/// [`low_pass`], [`peaking_eq`], [`low_shelf`], [`high_shelf`]) and driven by
/// [`apply_biquad`]. Coefficients are stored already normalized so that
/// `a0 == 1`.
#[derive(Debug, Clone, Copy)]
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
    fn from_raw(b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) -> Self {
        Biquad {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn step(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

fn omega(freq_hz: f32, sample_rate_hz: f32) -> f32 {
    2.0 * std::f32::consts::PI * freq_hz / sample_rate_hz
}

/// A high-pass filter: attenuates frequencies below `freq_hz`.
///
/// `q` is the filter Q (`0.7071` gives a maximally-flat / Butterworth
/// response); `sample_rate_hz` is the series' sample rate.
pub fn high_pass(freq_hz: f32, q: f32, sample_rate_hz: f32) -> Biquad {
    let w0 = omega(freq_hz, sample_rate_hz);
    let (sin_w0, cos_w0) = w0.sin_cos();
    let alpha = sin_w0 / (2.0 * q);

    let b0 = (1.0 + cos_w0) / 2.0;
    let b1 = -(1.0 + cos_w0);
    let b2 = (1.0 + cos_w0) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;
    Biquad::from_raw(b0, b1, b2, a0, a1, a2)
}

/// A low-pass filter: attenuates frequencies above `freq_hz`.
pub fn low_pass(freq_hz: f32, q: f32, sample_rate_hz: f32) -> Biquad {
    let w0 = omega(freq_hz, sample_rate_hz);
    let (sin_w0, cos_w0) = w0.sin_cos();
    let alpha = sin_w0 / (2.0 * q);

    let b0 = (1.0 - cos_w0) / 2.0;
    let b1 = 1.0 - cos_w0;
    let b2 = (1.0 - cos_w0) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;
    Biquad::from_raw(b0, b1, b2, a0, a1, a2)
}

/// A peaking EQ filter: boosts (positive `gain_db`) or cuts (negative
/// `gain_db`) a band centered at `freq_hz`, with bandwidth controlled by `q`.
pub fn peaking_eq(freq_hz: f32, gain_db: f32, q: f32, sample_rate_hz: f32) -> Biquad {
    let w0 = omega(freq_hz, sample_rate_hz);
    let (sin_w0, cos_w0) = w0.sin_cos();
    let alpha = sin_w0 / (2.0 * q);
    let a = 10f32.powf(gain_db / 40.0);

    let b0 = 1.0 + alpha * a;
    let b1 = -2.0 * cos_w0;
    let b2 = 1.0 - alpha * a;
    let a0 = 1.0 + alpha / a;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha / a;
    Biquad::from_raw(b0, b1, b2, a0, a1, a2)
}

/// A low shelf filter: boosts or cuts everything below `freq_hz` by
/// `gain_db`, with the shelf's transition steepness controlled by `q`
/// (`0.7071` gives a classic shelf slope).
pub fn low_shelf(freq_hz: f32, gain_db: f32, q: f32, sample_rate_hz: f32) -> Biquad {
    let w0 = omega(freq_hz, sample_rate_hz);
    let (sin_w0, cos_w0) = w0.sin_cos();
    let a = 10f32.powf(gain_db / 40.0);
    let alpha = sin_w0 / (2.0 * q);
    let sqrt_a = a.sqrt();
    let two_sqrt_a_alpha = 2.0 * sqrt_a * alpha;

    let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
    let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
    let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
    let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
    let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
    let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha;
    Biquad::from_raw(b0, b1, b2, a0, a1, a2)
}

/// A high shelf filter: boosts or cuts everything above `freq_hz` by
/// `gain_db`, with the shelf's transition steepness controlled by `q`.
pub fn high_shelf(freq_hz: f32, gain_db: f32, q: f32, sample_rate_hz: f32) -> Biquad {
    let w0 = omega(freq_hz, sample_rate_hz);
    let (sin_w0, cos_w0) = w0.sin_cos();
    let a = 10f32.powf(gain_db / 40.0);
    let alpha = sin_w0 / (2.0 * q);
    let sqrt_a = a.sqrt();
    let two_sqrt_a_alpha = 2.0 * sqrt_a * alpha;

    let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
    let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
    let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
    let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
    let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
    let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha;
    Biquad::from_raw(b0, b1, b2, a0, a1, a2)
}

/// Run `samples` through `filter`, in order, carrying its delay line forward.
///
/// Calling this repeatedly with the same `filter` and consecutive chunks of
/// a longer signal produces the same output as one call with the whole
/// signal — the filter's internal state (`x1`/`x2`/`y1`/`y2`) persists
/// across calls.
pub fn apply_biquad(samples: &[f32], filter: &mut Biquad) -> Vec<f32> {
    samples.iter().map(|&x| filter.step(x)).collect()
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
    fn low_pass_attenuates_high_frequency_more_than_low_frequency() {
        let sample_rate = 8000.0;
        let n = 4096;
        let low_tone = sine_wave(50.0, sample_rate, n);
        let high_tone = sine_wave(2000.0, sample_rate, n);

        let mut lp_low = low_pass(200.0, 0.7071, sample_rate);
        let mut lp_high = low_pass(200.0, 0.7071, sample_rate);
        let low_out = apply_biquad(&low_tone, &mut lp_low);
        let high_out = apply_biquad(&high_tone, &mut lp_high);

        let low_retention = rms(&low_out[500..]) / rms(&low_tone[500..]);
        let high_retention = rms(&high_out[500..]) / rms(&high_tone[500..]);

        assert!(low_retention > 0.9, "got {low_retention}");
        assert!(high_retention < 0.2, "got {high_retention}");
    }

    #[test]
    fn high_pass_attenuates_low_frequency_more_than_high_frequency() {
        let sample_rate = 8000.0;
        let n = 4096;
        let low_tone = sine_wave(50.0, sample_rate, n);
        let high_tone = sine_wave(2000.0, sample_rate, n);

        let mut hp_low = high_pass(200.0, 0.7071, sample_rate);
        let mut hp_high = high_pass(200.0, 0.7071, sample_rate);
        let low_out = apply_biquad(&low_tone, &mut hp_low);
        let high_out = apply_biquad(&high_tone, &mut hp_high);

        let low_retention = rms(&low_out[500..]) / rms(&low_tone[500..]);
        let high_retention = rms(&high_out[500..]) / rms(&high_tone[500..]);

        assert!(high_retention > 0.9, "got {high_retention}");
        assert!(low_retention < 0.2, "got {low_retention}");
    }

    #[test]
    fn peaking_eq_boosts_energy_at_center_frequency() {
        let sample_rate = 8000.0;
        let n = 4096;
        let tone = sine_wave(1000.0, sample_rate, n);

        let mut boost = peaking_eq(1000.0, 12.0, 1.0, sample_rate);
        let boosted = apply_biquad(&tone, &mut boost);

        let gain = rms(&boosted[500..]) / rms(&tone[500..]);
        assert!(gain > 1.5, "expected boosted output, got gain {gain}");
    }

    #[test]
    fn peaking_eq_cuts_energy_at_center_frequency() {
        let sample_rate = 8000.0;
        let n = 4096;
        let tone = sine_wave(1000.0, sample_rate, n);

        let mut cut = peaking_eq(1000.0, -12.0, 1.0, sample_rate);
        let cutted = apply_biquad(&tone, &mut cut);

        let gain = rms(&cutted[500..]) / rms(&tone[500..]);
        assert!(gain < 0.7, "expected cut output, got gain {gain}");
    }

    #[test]
    fn low_shelf_boosts_low_frequencies_and_leaves_high_frequencies_alone() {
        let sample_rate = 8000.0;
        let n = 4096;
        let low_tone = sine_wave(50.0, sample_rate, n);
        let high_tone = sine_wave(3000.0, sample_rate, n);

        let mut shelf_low = low_shelf(300.0, 12.0, 0.7071, sample_rate);
        let mut shelf_high = low_shelf(300.0, 12.0, 0.7071, sample_rate);
        let low_out = apply_biquad(&low_tone, &mut shelf_low);
        let high_out = apply_biquad(&high_tone, &mut shelf_high);

        let low_gain = rms(&low_out[500..]) / rms(&low_tone[500..]);
        let high_gain = rms(&high_out[500..]) / rms(&high_tone[500..]);

        assert!(low_gain > 1.5, "expected boosted low tone, got {low_gain}");
        assert!(
            (high_gain - 1.0).abs() < 0.2,
            "expected roughly unchanged high tone, got {high_gain}"
        );
    }

    #[test]
    fn high_shelf_boosts_high_frequencies_and_leaves_low_frequencies_alone() {
        let sample_rate = 8000.0;
        let n = 4096;
        let low_tone = sine_wave(50.0, sample_rate, n);
        let high_tone = sine_wave(3000.0, sample_rate, n);

        let mut shelf_low = high_shelf(1000.0, 12.0, 0.7071, sample_rate);
        let mut shelf_high = high_shelf(1000.0, 12.0, 0.7071, sample_rate);
        let low_out = apply_biquad(&low_tone, &mut shelf_low);
        let high_out = apply_biquad(&high_tone, &mut shelf_high);

        let low_gain = rms(&low_out[500..]) / rms(&low_tone[500..]);
        let high_gain = rms(&high_out[500..]) / rms(&high_tone[500..]);

        assert!(
            (low_gain - 1.0).abs() < 0.2,
            "expected roughly unchanged low tone, got {low_gain}"
        );
        assert!(high_gain > 1.5, "expected boosted high tone, got {high_gain}");
    }

    #[test]
    fn apply_biquad_state_persists_across_calls() {
        let sample_rate = 8000.0;
        let tone = sine_wave(200.0, sample_rate, 2000);

        let mut one_shot_filter = low_pass(500.0, 0.7071, sample_rate);
        let one_shot = apply_biquad(&tone, &mut one_shot_filter);

        let mut chunked_filter = low_pass(500.0, 0.7071, sample_rate);
        let mut chunked = apply_biquad(&tone[..1000], &mut chunked_filter);
        chunked.extend(apply_biquad(&tone[1000..], &mut chunked_filter));

        for (a, b) in one_shot.iter().zip(chunked.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn apply_biquad_handles_empty_input() {
        let mut filter = low_pass(1000.0, 0.7071, 8000.0);
        assert!(apply_biquad(&[], &mut filter).is_empty());
    }
}
