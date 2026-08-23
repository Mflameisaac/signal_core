//! Smoothing and frequency-selective filters over a generic real-valued series.

/// Centered moving average with a window of `window_size` samples.
///
/// At the edges, the window is clamped to the available samples (i.e. the
/// average is taken over however many samples fall within `window_size / 2`
/// of the current index), so the output is always the same length as the
/// input. `window_size` of `0` or `1` returns the input unchanged.
pub fn moving_average(samples: &[f32], window_size: usize) -> Vec<f32> {
    if window_size <= 1 || samples.is_empty() {
        return samples.to_vec();
    }
    let half = window_size / 2;
    let n = samples.len();
    (0..n)
        .map(|i| {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(n);
            let slice = &samples[start..end];
            slice.iter().sum::<f32>() / slice.len() as f32
        })
        .collect()
}

/// Single-pole (RC-equivalent) low-pass filter.
///
/// `sample_rate_hz` is the series' sample rate; `cutoff_hz` is the -3dB
/// corner frequency. Frequencies well below `cutoff_hz` pass through
/// largely unattenuated; frequencies well above it are attenuated.
pub fn low_pass_filter(samples: &[f32], sample_rate_hz: f32, cutoff_hz: f32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let dt = 1.0 / sample_rate_hz;
    let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
    let alpha = dt / (rc + dt);

    let mut out = Vec::with_capacity(samples.len());
    let mut prev = samples[0];
    out.push(prev);
    for &x in &samples[1..] {
        prev += alpha * (x - prev);
        out.push(prev);
    }
    out
}

/// Single-pole (RC-equivalent) high-pass filter.
///
/// Complementary to [`low_pass_filter`]: attenuates frequencies below
/// `cutoff_hz`, passes frequencies above it.
pub fn high_pass_filter(samples: &[f32], sample_rate_hz: f32, cutoff_hz: f32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let dt = 1.0 / sample_rate_hz;
    let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
    let alpha = rc / (rc + dt);

    let mut out = Vec::with_capacity(samples.len());
    let mut prev_y = 0.0f32;
    let mut prev_x = samples[0];
    out.push(prev_y);
    for &x in &samples[1..] {
        prev_y = alpha * (prev_y + x - prev_x);
        prev_x = x;
        out.push(prev_y);
    }
    out
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
    fn moving_average_of_constant_signal_is_unchanged() {
        let samples = vec![3.0f32; 20];
        let out = moving_average(&samples, 5);
        for &v in &out {
            assert!((v - 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn moving_average_window_of_one_is_identity() {
        let samples = vec![1.0, 5.0, -3.0, 2.0];
        assert_eq!(moving_average(&samples, 1), samples);
        assert_eq!(moving_average(&samples, 0), samples);
    }

    #[test]
    fn moving_average_smooths_a_spike() {
        let mut samples = vec![0.0f32; 21];
        samples[10] = 21.0; // single spike, centered so its window is never clamped
        let out = moving_average(&samples, 21);
        // index 10's window covers the whole 21-sample series exactly once,
        // so its output is the spike divided by the full window length.
        assert!(
            (out[10] - 1.0).abs() < 1e-5,
            "expected spike smoothed to 1.0, got {}",
            out[10]
        );
        // every output sample should be far smaller than the raw spike
        assert!(out.iter().all(|&v| v < 2.0));
        // but the spike's influence should still be visible (nonzero) at every index
        assert!(out.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn moving_average_same_length_as_input() {
        let samples = vec![1.0, 2.0, 3.0];
        assert_eq!(moving_average(&samples, 2).len(), samples.len());
    }

    #[test]
    fn low_pass_attenuates_high_frequency_more_than_low_frequency() {
        let sample_rate = 8000.0;
        let n = 4096;
        let low_tone = sine_wave(20.0, sample_rate, n);
        let high_tone = sine_wave(2000.0, sample_rate, n);

        let cutoff = 100.0;
        let low_out = low_pass_filter(&low_tone, sample_rate, cutoff);
        let high_out = low_pass_filter(&high_tone, sample_rate, cutoff);

        // skip filter warm-up region
        let low_rms_in = rms(&low_tone[500..]);
        let low_rms_out = rms(&low_out[500..]);
        let high_rms_in = rms(&high_tone[500..]);
        let high_rms_out = rms(&high_out[500..]);

        let low_retention = low_rms_out / low_rms_in;
        let high_retention = high_rms_out / high_rms_in;

        assert!(
            low_retention > 0.9,
            "low-frequency tone should mostly pass through, retained {low_retention}"
        );
        assert!(
            high_retention < 0.2,
            "high-frequency tone should be heavily attenuated, retained {high_retention}"
        );
    }

    #[test]
    fn high_pass_attenuates_low_frequency_more_than_high_frequency() {
        let sample_rate = 8000.0;
        let n = 4096;
        let low_tone = sine_wave(20.0, sample_rate, n);
        let high_tone = sine_wave(2000.0, sample_rate, n);

        let cutoff = 100.0;
        let low_out = high_pass_filter(&low_tone, sample_rate, cutoff);
        let high_out = high_pass_filter(&high_tone, sample_rate, cutoff);

        let low_rms_in = rms(&low_tone[500..]);
        let low_rms_out = rms(&low_out[500..]);
        let high_rms_in = rms(&high_tone[500..]);
        let high_rms_out = rms(&high_out[500..]);

        let low_retention = low_rms_out / low_rms_in;
        let high_retention = high_rms_out / high_rms_in;

        assert!(
            high_retention > 0.9,
            "high-frequency tone should mostly pass through, retained {high_retention}"
        );
        assert!(
            low_retention < 0.2,
            "low-frequency tone should be heavily attenuated, retained {low_retention}"
        );
    }

    #[test]
    fn filters_return_same_length_and_handle_empty() {
        let samples = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(low_pass_filter(&samples, 1000.0, 100.0).len(), 4);
        assert_eq!(high_pass_filter(&samples, 1000.0, 100.0).len(), 4);
        assert!(low_pass_filter(&[], 1000.0, 100.0).is_empty());
        assert!(high_pass_filter(&[], 1000.0, 100.0).is_empty());
    }
}
