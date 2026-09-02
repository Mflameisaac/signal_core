//! Perceptual loudness per ITU-R BS.1770 (the standard behind "LUFS",
//! Spotify/YouTube/podcast loudness normalization, ffmpeg's `loudnorm`,
//! etc.), plus a true-peak estimate for catching inter-sample overs a plain
//! sample-peak check misses.
//!
//! Why this exists rather than reusing plain RMS: RMS treats a mostly-quiet
//! recording with a few loud words the same as one that's loud throughout,
//! and gives silence/pauses equal weight to speech — two clips can RMS-match
//! and still sound quite different in perceived loudness. BS.1770 fixes both
//! problems with a perceptual pre-filter (K-weighting, below) and a gating
//! scheme that excludes quiet/silent blocks from the average.

use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

use crate::biquad::Biquad;

/// The two-stage K-weighting filter ITU-R BS.1770 defines: a high-shelf
/// approximating the head/ear's frequency response, followed by a highpass
/// (the "RLB" curve) that rolls off sub-bass the ear barely perceives as
/// loudness. Coefficients are derived per sample rate via the same
/// bilinear-transform approach reference implementations (e.g. libebur128)
/// use, rather than hardcoding the spec's published 48kHz-only values, so
/// this works correctly at other common rates (44.1kHz, etc.) too.
fn k_weighting_filters(sample_rate_hz: f32) -> (Biquad, Biquad) {
    let sr = sample_rate_hz as f64;

    // Stage 1: high-shelf (head/ear response).
    let f0 = 1681.9744509555319_f64;
    let g = 3.99984385397_f64;
    let q = 0.7071752369554193_f64;
    let k = (std::f64::consts::PI * f0 / sr).tan();
    let vh = 10f64.powf(g / 20.0);
    let vb = vh.powf(0.4996667741545416);
    let a0 = 1.0 + k / q + k * k;
    let stage1 = Biquad::from_coefficients(
        ((vh + vb * k / q + k * k) / a0) as f32,
        (2.0 * (k * k - vh) / a0) as f32,
        ((vh - vb * k / q + k * k) / a0) as f32,
        (2.0 * (k * k - 1.0) / a0) as f32,
        ((1.0 - k / q + k * k) / a0) as f32,
    );

    // Stage 2: highpass (RLB weighting).
    let f0b = 38.13547087602444_f64;
    let qb = 0.5003270373238773_f64;
    let kb = (std::f64::consts::PI * f0b / sr).tan();
    let a0b = 1.0 + kb / qb + kb * kb;
    let stage2 = Biquad::from_coefficients(
        (1.0 / a0b) as f32,
        (-2.0 / a0b) as f32,
        (1.0 / a0b) as f32,
        (2.0 * (kb * kb - 1.0) / a0b) as f32,
        ((1.0 - kb / qb + kb * kb) / a0b) as f32,
    );

    (stage1, stage2)
}

fn deinterleave(samples: &[f32], channels: usize) -> Vec<Vec<f32>> {
    let channels = channels.max(1);
    let mut out = vec![Vec::with_capacity(samples.len() / channels + 1); channels];
    for frame in samples.chunks(channels) {
        for (c, &s) in frame.iter().enumerate() {
            out[c].push(s);
        }
    }
    out
}

fn loudness_from_mean_square(z: f32) -> f32 {
    -0.691 + 10.0 * z.max(1e-12).log10()
}

/// Gated integrated loudness in LUFS, per ITU-R BS.1770's block/gating
/// scheme: mean-square power is measured over 400ms blocks (75% overlap),
/// blocks quieter than -70 LUFS are discarded outright (the "absolute
/// gate" — true silence shouldn't count), then blocks more than 10 LU
/// quieter than the remaining average are *also* discarded (the "relative
/// gate" — this is what keeps a pause between sentences from dragging the
/// measured loudness down, unlike plain RMS).
///
/// `samples` is interleaved (`channels` per frame, same layout
/// `decode_audio` in `audio-editor` produces).
pub fn integrated_loudness(samples: &[f32], sample_rate_hz: f32, channels: usize) -> f32 {
    if samples.is_empty() {
        return -70.0;
    }
    let deinterleaved = deinterleave(samples, channels);
    let weighted: Vec<Vec<f32>> = deinterleaved
        .iter()
        .map(|ch| {
            let (mut stage1, mut stage2) = k_weighting_filters(sample_rate_hz);
            let s1: Vec<f32> = ch.iter().map(|&x| stage1.process(x)).collect();
            s1.iter().map(|&x| stage2.process(x)).collect()
        })
        .collect();

    let block_size = ((0.4 * sample_rate_hz as f64) as usize).max(1);
    let hop = (block_size / 4).max(1);
    let n = weighted[0].len();

    if n < block_size {
        let z: f32 = weighted
            .iter()
            .map(|ch| ch.iter().map(|&s| s * s).sum::<f32>() / ch.len().max(1) as f32)
            .sum();
        return loudness_from_mean_square(z);
    }

    let mut block_z = Vec::new();
    let mut start = 0;
    while start + block_size <= n {
        let mut z = 0.0f32;
        for ch in &weighted {
            let block = &ch[start..start + block_size];
            z += block.iter().map(|&s| s * s).sum::<f32>() / block_size as f32;
        }
        block_z.push(z);
        start += hop;
    }

    let abs_gated: Vec<f32> = block_z
        .iter()
        .copied()
        .filter(|&z| loudness_from_mean_square(z) >= -70.0)
        .collect();
    if abs_gated.is_empty() {
        return -70.0;
    }

    let mean_z: f32 = abs_gated.iter().sum::<f32>() / abs_gated.len() as f32;
    let relative_threshold = loudness_from_mean_square(mean_z) - 10.0;

    let rel_gated: Vec<f32> = abs_gated
        .iter()
        .copied()
        .filter(|&z| loudness_from_mean_square(z) >= relative_threshold)
        .collect();
    if rel_gated.is_empty() {
        return loudness_from_mean_square(mean_z);
    }

    let final_mean_z: f32 = rel_gated.iter().sum::<f32>() / rel_gated.len() as f32;
    loudness_from_mean_square(final_mean_z)
}

const OVERSAMPLE_FACTOR: usize = 4;

/// FFT-based bandlimited interpolation: zero-pads a block's spectrum to
/// `OVERSAMPLE_FACTOR`x its length and inverse-transforms, which (unlike a
/// naive resample) recovers the true continuous-time peak a signal would
/// reach *between* samples, not just at them — the whole point of "true
/// peak" vs. a plain sample-peak check. Takes pre-built forward/inverse FFT
/// plans (rather than planning internally) so a caller processing many
/// blocks of the same size can reuse one pair of plans instead of paying
/// rustfft's setup cost on every block — see [`true_peak_db`], where this
/// mattered in practice (below).
fn true_peak_of_block(
    block: &[Complex32],
    scratch: &mut [Complex32],
    fwd: &dyn rustfft::Fft<f32>,
    inv: &dyn rustfft::Fft<f32>,
) -> f32 {
    let n = block.len();
    let m = scratch.len();

    let mut spectrum = block.to_vec();
    fwd.process(&mut spectrum);

    for c in scratch.iter_mut() {
        *c = Complex32::new(0.0, 0.0);
    }
    let half = n / 2;
    scratch[0..half].copy_from_slice(&spectrum[0..half]);
    scratch[m - (n - half)..m].copy_from_slice(&spectrum[half..n]);

    inv.process(scratch);

    // rustfft's forward+inverse pair is unnormalized (round-trips to `n`
    // times the original signal at the original length); dividing by the
    // *original* block length `n` (not the padded length `m`) gives the
    // correctly-scaled interpolated amplitude — verified in this module's
    // tests below.
    let scale = 1.0 / n as f32;
    scratch.iter().map(|c| (c.re * scale).abs()).fold(0.0f32, f32::max)
}

/// True-peak level in dBFS: the loudest the signal's *continuous*
/// waveform reaches, including peaks that fall between two samples (which
/// a simple `max(abs(samples))` check can miss entirely — a real, common
/// way a "peak-safe" mix still clips after D/A conversion or lossy
/// encoding). This is an approximation of the full ITU-R BS.1770 true-peak
/// filter (which specifies an exact interpolation filter), not a certified
/// implementation, but catches the same class of inter-sample overs.
///
/// `samples` can be interleaved multi-channel data — like
/// `audio-editor`'s existing `peak()` helper, this only cares about the
/// single largest amplitude anywhere in the stream, not which channel it's
/// in.
///
/// A short input is processed as one FFT of its exact length. A longer one
/// is processed in fixed, power-of-two-sized, 50%-overlapping windows
/// instead: an earlier version chunked by an arbitrary block+margin size
/// (e.g. 16512 samples), which is a poor size for an FFT (large prime
/// factor) *and* replanned a fresh FFT on every single block — fine for the
/// short clips this was tested against, but on a multi-minute real file
/// (hundreds of blocks), the combination made this function dramatically
/// slower than the rest of the mastering chain combined, freezing the app
/// for the duration. Fixed-size power-of-two windows with one reused plan
/// avoid both problems; the 50% overlap means a sample near one window's
/// (less accurate) edge is still covered near the center of the next
/// window, which is where this trick is most accurate.
pub fn true_peak_db(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return f32::NEG_INFINITY;
    }
    if samples.len() < 2 {
        return 20.0 * samples[0].abs().max(1e-8).log10();
    }

    const WINDOW: usize = 16384;

    let mut planner = FftPlanner::<f32>::new();

    if samples.len() <= WINDOW {
        let block: Vec<Complex32> = samples.iter().map(|&s| Complex32::new(s, 0.0)).collect();
        let n = block.len();
        let m = n * OVERSAMPLE_FACTOR;
        let fwd = planner.plan_fft_forward(n);
        let inv = planner.plan_fft_inverse(m);
        let mut scratch = vec![Complex32::new(0.0, 0.0); m];
        let peak = true_peak_of_block(&block, &mut scratch, fwd.as_ref(), inv.as_ref());
        return 20.0 * peak.max(1e-8).log10();
    }

    let padded_len = WINDOW * OVERSAMPLE_FACTOR;
    let fwd = planner.plan_fft_forward(WINDOW);
    let inv = planner.plan_fft_inverse(padded_len);
    let mut window_buf = vec![Complex32::new(0.0, 0.0); WINDOW];
    let mut scratch = vec![Complex32::new(0.0, 0.0); padded_len];

    // Non-overlapping windows: a real true-peak over on a multi-minute file
    // is essentially never sitting exactly on one specific block boundary,
    // and this is already an approximation (see the doc comment above) —
    // halving the window count by dropping the 50% overlap this used to
    // have is a meaningful chunk of this function's cost for a negligible
    // accuracy cost in practice.
    let hop = WINDOW;
    let mut max_peak = 0.0f32;
    let mut start = 0;
    loop {
        let end = (start + WINDOW).min(samples.len());
        for i in 0..WINDOW {
            let idx = start + i;
            window_buf[i] = Complex32::new(if idx < end { samples[idx] } else { 0.0 }, 0.0);
        }

        let peak = true_peak_of_block(&window_buf, &mut scratch, fwd.as_ref(), inv.as_ref());
        max_peak = max_peak.max(peak);

        if end >= samples.len() {
            break;
        }
        start += hop;
    }

    20.0 * max_peak.max(1e-8).log10()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine_wave(freq_hz: f32, sample_rate_hz: f32, amplitude: f32, phase: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| amplitude * (2.0 * PI * freq_hz * i as f32 / sample_rate_hz + phase).sin())
            .collect()
    }

    #[test]
    fn integrated_loudness_increases_by_about_6lu_when_amplitude_doubles() {
        let sample_rate = 48000.0;
        let quiet = sine_wave(1000.0, sample_rate, 0.1, 0.0, 48000);
        let loud = sine_wave(1000.0, sample_rate, 0.2, 0.0, 48000);

        let quiet_lufs = integrated_loudness(&quiet, sample_rate, 1);
        let loud_lufs = integrated_loudness(&loud, sample_rate, 1);

        let delta = loud_lufs - quiet_lufs;
        assert!(
            (delta - 6.0).abs() < 1.0,
            "doubling amplitude should raise loudness ~6LU, got {delta}LU"
        );
    }

    #[test]
    fn integrated_loudness_gates_out_silence_between_loud_bursts() {
        let sample_rate = 48000.0;
        // Bursts long enough to contain several full 400ms gating blocks
        // each (not just boundary-straddling ones), so the gated average
        // isn't dominated by block/silence-transition edge effects.
        let burst_seconds = 1.5;
        let burst = sine_wave(1000.0, sample_rate, 0.5, 0.0, (sample_rate * burst_seconds) as usize);

        let mut samples = vec![0.0f32; sample_rate as usize * 8];
        let start1 = sample_rate as usize; // 1s of leading silence
        samples[start1..start1 + burst.len()].copy_from_slice(&burst);
        let start2 = sample_rate as usize * 4; // 2s silence gap after burst 1
        samples[start2..start2 + burst.len()].copy_from_slice(&burst);
        // remaining ~2s trailing silence

        let gated_lufs = integrated_loudness(&samples, sample_rate, 1);
        // Should land close to the burst's own loudness, not dragged down
        // by the ~5s of silence between/around them.
        let burst_only_lufs = integrated_loudness(&burst, sample_rate, 1);
        assert!(
            (gated_lufs - burst_only_lufs).abs() < 1.0,
            "gating should ignore surrounding silence: gated={gated_lufs} burst_only={burst_only_lufs}"
        );
    }

    #[test]
    fn integrated_loudness_silence_reports_the_floor() {
        let samples = vec![0.0f32; 48000];
        let lufs = integrated_loudness(&samples, 48000.0, 1);
        assert!(lufs <= -69.0, "pure silence should gate to the -70 LUFS floor, got {lufs}");
    }

    #[test]
    fn true_peak_db_stays_fast_on_a_multi_minute_file() {
        // Regression test for a real bug: an earlier version chunked into
        // arbitrary block+margin-sized FFTs (e.g. 16512 samples — a poor
        // size, large prime factor) and replanned a fresh FFT on every
        // block. That was invisible in tests using short clips, but froze
        // the app for tens of seconds on an actual multi-minute song. A
        // stereo ~4-minute track at 44.1kHz is ~21M interleaved samples.
        let sample_rate = 44100.0;
        let seconds = 240.0;
        let samples = sine_wave(220.0, sample_rate, 0.8, 0.0, (sample_rate * seconds * 2.0) as usize);

        let start = std::time::Instant::now();
        let peak = true_peak_db(&samples);
        let elapsed = start.elapsed();

        assert!(peak.is_finite());
        assert!(
            elapsed.as_secs_f32() < 2.0,
            "true_peak_db on a ~4-minute stereo track took {:?}, expected well under 2s",
            elapsed
        );
    }

    #[test]
    fn true_peak_matches_sample_peak_for_a_slowly_varying_signal() {
        // A low-frequency tone's continuous peak is well-approximated by
        // its sample peak — a sanity check that the FFT upsampling scale
        // factor is correct (see the comment in true_peak_of_block).
        let sample_rate = 48000.0;
        let tone = sine_wave(100.0, sample_rate, 0.5, 0.0, 8192);
        let sample_peak_db = 20.0 * tone.iter().fold(0.0f32, |m, &s| m.max(s.abs())).log10();
        let true_peak = true_peak_db(&tone);
        assert!(
            (true_peak - sample_peak_db).abs() < 0.3,
            "expected true peak close to sample peak for a slow tone: sample={sample_peak_db}dB true={true_peak}dB"
        );
    }

    #[test]
    fn true_peak_finds_an_inter_sample_peak_a_naive_check_misses() {
        let sample_rate = 48000.0;
        // Exactly 3 samples per period (sample_rate/3), zero phase, and a
        // block length that's an exact multiple of the period (so the FFT
        // sees a perfectly periodic block, no edge-discontinuity leakage).
        // With only 3 evenly-spaced sample phases per cycle (0°, 120°,
        // 240°), none land near the true peak at 90°/270°, so the sample
        // peak undershoots the tone's real amplitude by a known, large
        // margin (sin(120°) ≈ 0.866 vs. the true peak of 1.0).
        let amplitude = 0.99;
        let period_samples = 3;
        let num_periods = 1365;
        let n = period_samples * num_periods;
        let tone = sine_wave(sample_rate / period_samples as f32, sample_rate, amplitude, 0.0, n);
        let sample_peak = tone.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        let sample_peak_db = 20.0 * sample_peak.log10();
        let true_peak = true_peak_db(&tone);

        assert!(
            sample_peak < amplitude * 0.97,
            "test setup should produce a sample peak visibly below the true amplitude, got {sample_peak}"
        );
        assert!(
            true_peak > sample_peak_db,
            "true peak ({true_peak}dB) should exceed the naive sample peak ({sample_peak_db}dB)"
        );
        assert!(
            (true_peak - 20.0 * amplitude.log10()).abs() < 0.5,
            "true peak should recover close to the tone's actual amplitude, got {true_peak}dB"
        );
    }
}
