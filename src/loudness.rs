//! Perceptual loudness and true-peak estimation for a generic real-valued
//! (possibly multi-channel interleaved) time series.
//!
//! [`integrated_loudness`] implements the ITU-R BS.1770 K-weighting +
//! gated-block-averaging algorithm (the basis of the LUFS unit): each
//! channel is K-weighted, split into overlapping 400ms blocks, and averaged
//! with an absolute gate at -70 LUFS and a relative gate 10 LU below the
//! ungated mean. All channels are weighted equally (`1.0`) — this omits the
//! BS.1770 surround-channel weighting (`1.41` for rear channels), which
//! doesn't apply to the mono/stereo case this crate is built for.
//!
//! [`true_peak_db`] estimates inter-sample peaks by 4x-oversampling with
//! Catmull-Rom cubic interpolation. This is a pragmatic approximation, not
//! the polyphase reconstruction filter specified in BS.1770 Annex 2 — it's
//! good enough to catch peaks that clip on D/A conversion without pulling in
//! a full FIR oversampling filter.

use crate::biquad;

/// K-weighted, gated integrated loudness in LUFS.
///
/// `samples` is interleaved (`channels` values per frame, e.g. `[L, R, L,
/// R, ...]` for stereo). Silence (or a signal too short/quiet to pass the
/// absolute gate) returns `-70.0`, the absolute gate threshold itself.
pub fn integrated_loudness(samples: &[f32], sample_rate_hz: f32, channels: usize) -> f32 {
    const ABSOLUTE_GATE_LUFS: f32 = -70.0;
    let channels = channels.max(1);
    if samples.is_empty() || sample_rate_hz <= 0.0 {
        return ABSOLUTE_GATE_LUFS;
    }

    let frames = samples.len() / channels;
    if frames == 0 {
        return ABSOLUTE_GATE_LUFS;
    }

    let mut per_channel: Vec<Vec<f32>> = vec![Vec::with_capacity(frames); channels];
    for (i, &s) in samples[..frames * channels].iter().enumerate() {
        per_channel[i % channels].push(s);
    }

    // BS.1770 K-weighting: a high shelf ("pre-filter") cascaded with a
    // high-pass ("RLB") filter, applied independently per channel.
    for chan in per_channel.iter_mut() {
        let mut pre_filter =
            biquad::high_shelf(1681.9744509555319, 3.99984385397, 0.7071752369554193, sample_rate_hz);
        let mut rlb_filter = biquad::high_pass(38.13547087602444, 0.5003270373238773, sample_rate_hz);
        let shelved = biquad::apply_biquad(chan, &mut pre_filter);
        *chan = biquad::apply_biquad(&shelved, &mut rlb_filter);
    }

    let block_samples = ((0.4 * sample_rate_hz).round() as usize).max(1);
    let hop_samples = ((0.1 * sample_rate_hz).round() as usize).max(1);

    let mut block_powers: Vec<f32> = Vec::new();
    if frames < block_samples {
        let power: f32 = per_channel.iter().map(|chan| mean_square(chan)).sum();
        block_powers.push(power);
    } else {
        let mut start = 0;
        while start + block_samples <= frames {
            let power: f32 = per_channel
                .iter()
                .map(|chan| mean_square(&chan[start..start + block_samples]))
                .sum();
            block_powers.push(power);
            start += hop_samples;
        }
    }

    let absolute_gated: Vec<f32> = block_powers
        .into_iter()
        .filter(|&p| loudness_of(p) >= ABSOLUTE_GATE_LUFS)
        .collect();
    if absolute_gated.is_empty() {
        return ABSOLUTE_GATE_LUFS;
    }

    let ungated_mean_power = absolute_gated.iter().sum::<f32>() / absolute_gated.len() as f32;
    let relative_threshold = loudness_of(ungated_mean_power) - 10.0;

    let relative_gated: Vec<f32> = absolute_gated
        .into_iter()
        .filter(|&p| loudness_of(p) >= relative_threshold)
        .collect();
    if relative_gated.is_empty() {
        return relative_threshold;
    }

    let gated_mean_power = relative_gated.iter().sum::<f32>() / relative_gated.len() as f32;
    loudness_of(gated_mean_power)
}

fn loudness_of(power: f32) -> f32 {
    -0.691 + 10.0 * power.max(1e-10).log10()
}

fn mean_square(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32
}

/// Estimated true peak in dBFS, via 4x oversampling.
///
/// Unlike a plain sample-peak measurement, this can catch a peak that falls
/// between two samples and would otherwise clip on playback after D/A
/// conversion or resampling. Silence returns a large negative floor rather
/// than `-inf`.
pub fn true_peak_db(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 20.0 * 1e-10f32.log10();
    }

    let n = samples.len();
    let mut peak = samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()));

    if n >= 2 {
        for i in 0..n - 1 {
            let p0 = samples[i.saturating_sub(1)];
            let p1 = samples[i];
            let p2 = samples[i + 1];
            let p3 = samples[(i + 2).min(n - 1)];
            for &t in &[0.25f32, 0.5, 0.75] {
                peak = peak.max(catmull_rom(p0, p1, p2, p3, t).abs());
            }
        }
    }

    20.0 * peak.max(1e-10).log10()
}

fn catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
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

    #[test]
    fn integrated_loudness_of_silence_is_the_absolute_gate_floor() {
        let silence = vec![0.0f32; 48_000 * 2];
        assert_eq!(integrated_loudness(&silence, 48_000.0, 1), -70.0);
        assert!(integrated_loudness(&[], 48_000.0, 1) == -70.0);
    }

    #[test]
    fn integrated_loudness_increases_with_signal_amplitude() {
        let sample_rate = 48_000.0;
        let n = sample_rate as usize * 2; // 2 seconds, several gating blocks
        let quiet = sine_wave(1000.0, sample_rate, 0.05, n);
        let loud = sine_wave(1000.0, sample_rate, 0.5, n);

        let quiet_lufs = integrated_loudness(&quiet, sample_rate, 1);
        let loud_lufs = integrated_loudness(&loud, sample_rate, 1);

        assert!(
            loud_lufs > quiet_lufs,
            "expected louder signal to have higher LUFS: {loud_lufs} vs {quiet_lufs}"
        );
        assert!(loud_lufs < 0.0, "full-scale-ish tone should still be < 0 LUFS, got {loud_lufs}");
    }

    #[test]
    fn integrated_loudness_handles_stereo_interleaved_input() {
        let sample_rate = 48_000.0;
        let n = sample_rate as usize; // 1 second per channel
        let mono = sine_wave(1000.0, sample_rate, 0.3, n);
        let mut stereo = Vec::with_capacity(n * 2);
        for &s in &mono {
            stereo.push(s);
            stereo.push(s);
        }
        let mono_lufs = integrated_loudness(&mono, sample_rate, 1);
        let stereo_lufs = integrated_loudness(&stereo, sample_rate, 2);
        // Per BS.1770, channel powers sum before the log, so identical
        // content duplicated onto a second channel reads ~3.01 LU louder,
        // not the same as mono.
        assert!(
            (stereo_lufs - mono_lufs - 3.0102).abs() < 0.3,
            "mono {mono_lufs} vs stereo {stereo_lufs}"
        );
    }

    #[test]
    fn integrated_loudness_short_signal_does_not_panic() {
        let tiny = vec![0.1, -0.1, 0.2, -0.2];
        let lufs = integrated_loudness(&tiny, 48_000.0, 1);
        assert!(lufs.is_finite());
    }

    #[test]
    fn true_peak_of_silence_is_very_low() {
        let silence = vec![0.0f32; 100];
        assert!(true_peak_db(&silence) < -150.0);
        assert!(true_peak_db(&[]) < -150.0);
    }

    #[test]
    fn true_peak_is_never_below_the_plain_sample_peak() {
        let sample_rate = 8000.0;
        let tone = sine_wave(3000.0, sample_rate, 0.9, 200);
        let sample_peak_db = 20.0 * tone.iter().fold(0.0f32, |m, &s| m.max(s.abs())).log10();
        assert!(true_peak_db(&tone) >= sample_peak_db - 1e-4);
    }

    #[test]
    fn true_peak_catches_an_intersample_overshoot() {
        // A repeated "flat-topped pulse" shape (0, 1, 1, 0, ...) has every
        // sample at or below 1.0, but Catmull-Rom's tangent-driven overshoot
        // pushes the interpolated midpoint between the two 1.0 samples
        // above 1.0 — a plain sample-peak measurement misses this.
        let samples: Vec<f32> = (0..40)
            .map(|i| match i % 4 {
                1 | 2 => 1.0,
                _ => 0.0,
            })
            .collect();
        let sample_peak_db = 20.0 * 1.0f32.log10(); // == 0.0
        assert!(
            true_peak_db(&samples) > sample_peak_db + 0.5,
            "expected true peak to exceed sample peak of 0 dB, got {}",
            true_peak_db(&samples)
        );
    }

    #[test]
    fn true_peak_of_full_scale_dc_is_about_zero_db() {
        let samples = vec![1.0f32; 50];
        let db = true_peak_db(&samples);
        assert!((db - 0.0).abs() < 0.1, "expected ~0 dB, got {db}");
    }
}
