//! Fundamental-frequency (pitch) estimation over a generic real-valued
//! series, via the classic normalized-autocorrelation method.
//!
//! Like every other module here, this has no notion of "voice" or "singing"
//! specifically — it finds the lag that maximizes a windowed series'
//! self-similarity, within a caller-specified frequency range, and reports
//! that lag as a frequency. "Pitch" is audio vocabulary (this module exists
//! to serve audio callers, same as [`crate::loudness`]), but the underlying
//! primitive is just periodicity detection and would work equally well on
//! any other quasi-periodic real-valued series.
//!
//! This is deliberately the simplest correct version of the method — no
//! parabolic inter-lag interpolation, no octave-error correction, no
//! FFT-based speedup (see the note on [`estimate_f0`] for why that's a
//! reasonable next step, not a missing requirement). It exists to make the
//! math legible first.

/// Tuning knobs for [`estimate_f0`] and [`estimate_f0_contour`].
#[derive(Debug, Clone, Copy)]
pub struct PitchConfig {
    /// Lowest fundamental frequency to search for, in Hz. Determines the
    /// *longest* lag considered (`sample_rate_hz / min_freq_hz`) — a lower
    /// frequency means a longer period means a longer lag.
    pub min_freq_hz: f32,
    /// Highest fundamental frequency to search for, in Hz. Determines the
    /// *shortest* lag considered (`sample_rate_hz / max_freq_hz`).
    pub max_freq_hz: f32,
    /// Minimum normalized autocorrelation (`R(lag) / R(0)`, bounded in
    /// `[-1, 1]` by Cauchy-Schwarz) required to call a frame "voiced"
    /// (periodic) rather than noise/silence. `0.3` is the commonly-used
    /// default for this method — raise it for stricter voicing decisions,
    /// lower it to tolerate breathier or noisier input.
    pub voicing_threshold: f32,
}

impl Default for PitchConfig {
    fn default() -> Self {
        // 50-1000Hz covers a bass voice's fundamental through a high
        // soprano/falsetto note — wide enough for singing, not just
        // conversational speech's narrower range (which is closer to
        // 80-400Hz).
        PitchConfig {
            min_freq_hz: 50.0,
            max_freq_hz: 1000.0,
            voicing_threshold: 0.3,
        }
    }
}

/// Estimate the fundamental frequency of one frame via normalized
/// autocorrelation.
///
/// For each candidate lag `L` (in samples) within the range implied by
/// `config.min_freq_hz..=config.max_freq_hz`, computes
/// `R(L) = sum_{i=0}^{N-L-1} samples[i] * samples[i+L]`, normalized by the
/// frame's total energy `R(0) = sum samples[i]^2` so the voicing threshold
/// doesn't depend on loudness. Returns `sample_rate_hz / L` for whichever
/// lag maximizes this normalized value.
///
/// Returns `None` when:
/// - `config`'s frequency range is invalid (either bound non-positive, or
///   `min_freq_hz > max_freq_hz`),
/// - the frame is silent (`R(0)` is ~0 — there is no periodicity to find),
/// - the frame is too short to contain even one period of `min_freq_hz`
///   (the implied max lag no longer fits inside the frame), or
/// - the best normalized correlation found is below
///   `config.voicing_threshold` (the frame isn't clearly periodic —
///   unvoiced/noisy/silent-but-not-exactly-zero).
///
/// **Cost is `O(N * lag_range)`** — fine for offline analysis (this is what
/// [`estimate_f0_contour`] calls per frame), but a real-time caller
/// processing large frames at a high lag-range should consider computing
/// autocorrelation via FFT instead (the Wiener-Khinchin theorem: the
/// autocorrelation of a series is the inverse FFT of its power spectrum,
/// which [`crate::fft::power_spectrum`] already computes) — `O(N log N)`
/// instead of `O(N * lag_range)`, same result.
pub fn estimate_f0(samples: &[f32], sample_rate_hz: f32, config: &PitchConfig) -> Option<f32> {
    if config.min_freq_hz <= 0.0 || config.max_freq_hz <= 0.0 || config.min_freq_hz > config.max_freq_hz {
        return None;
    }
    if samples.len() < 2 || sample_rate_hz <= 0.0 {
        return None;
    }

    let r0: f32 = samples.iter().map(|&s| s * s).sum();
    if r0 <= 1e-12 {
        return None; // silence: no periodicity to find, and dividing by
                      // ~0 below would be meaningless anyway.
    }

    let min_lag = ((sample_rate_hz / config.max_freq_hz).round() as usize).max(1);
    let max_lag = ((sample_rate_hz / config.min_freq_hz).round() as usize).min(samples.len() - 1);
    if min_lag > max_lag {
        return None; // frame too short to contain even one period in range
    }

    let mut best_lag = min_lag;
    let mut best_r = f32::NEG_INFINITY;
    for lag in min_lag..=max_lag {
        let r: f32 = (0..samples.len() - lag)
            .map(|i| samples[i] * samples[i + lag])
            .sum();
        let normalized = r / r0;
        if normalized > best_r {
            best_r = normalized;
            best_lag = lag;
        }
    }

    if best_r < config.voicing_threshold {
        return None;
    }
    Some(sample_rate_hz / best_lag as f32)
}

/// Estimate a fundamental-frequency contour across a longer series, by
/// running [`estimate_f0`] over successive `frame_size`-length, `hop_size`-
/// spaced frames (no overlap-padding, unlike [`crate::fft::stft_magnitude`]
/// — every frame is a plain in-bounds slice).
///
/// Returns one entry per frame, in order; each is `None` exactly where
/// [`estimate_f0`] would return `None` for that frame (unvoiced/silent, or
/// unable to find a periodic lag above the voicing threshold). Returns an
/// empty result for empty input, a zero `frame_size`/`hop_size`, or a
/// `frame_size` longer than the whole series (there's no complete frame to
/// analyze).
pub fn estimate_f0_contour(
    samples: &[f32],
    sample_rate_hz: f32,
    frame_size: usize,
    hop_size: usize,
    config: &PitchConfig,
) -> Vec<Option<f32>> {
    if samples.is_empty() || frame_size == 0 || hop_size == 0 {
        return Vec::new();
    }

    let mut contour = Vec::new();
    let mut start = 0;
    while start + frame_size <= samples.len() {
        contour.push(estimate_f0(&samples[start..start + frame_size], sample_rate_hz, config));
        start += hop_size;
    }
    contour
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

    /// A cheap deterministic pseudo-random generator (xorshift32) — good
    /// enough to produce a non-periodic-looking test signal without pulling
    /// in a `rand` dependency for one test.
    fn xorshift_noise(seed: u32, n: usize) -> Vec<f32> {
        let mut state = seed.max(1);
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                // Map to [-1, 1].
                (state as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// The lag-quantization step near `freq_hz`: adjacent integer lags near
    /// `sample_rate_hz / freq_hz` correspond to frequencies roughly
    /// `freq_hz^2 / sample_rate_hz` apart, since this method (deliberately,
    /// see the module doc comment) picks the best *integer* lag with no
    /// sub-sample interpolation. Tests assert against this bound rather
    /// than a fixed magic number so they stay meaningful across frequencies.
    fn quantization_tolerance(freq_hz: f32, sample_rate_hz: f32) -> f32 {
        (freq_hz * freq_hz / sample_rate_hz) + 0.5
    }

    #[test]
    fn detects_pure_tone_across_the_singing_range() {
        let sample_rate = 22050.0;
        let config = PitchConfig::default();
        for &freq in &[110.0f32, 220.0, 440.0, 660.0] {
            // Several dozen periods, comfortably long enough for the
            // longest lag in range (sample_rate / min_freq_hz).
            let n = 4096;
            let samples = sine_wave(freq, sample_rate, n);
            let result = estimate_f0(&samples, sample_rate, &config)
                .unwrap_or_else(|| panic!("expected a voiced result for {freq}Hz"));
            let tolerance = quantization_tolerance(freq, sample_rate);
            assert!(
                (result - freq).abs() < tolerance,
                "expected ~{freq}Hz (+/- {tolerance}), got {result}Hz"
            );
        }
    }

    #[test]
    fn returns_none_for_pure_silence() {
        let samples = vec![0.0f32; 2048];
        assert!(estimate_f0(&samples, 22050.0, &PitchConfig::default()).is_none());
    }

    #[test]
    fn returns_none_for_noise_like_signal_below_voicing_threshold() {
        let samples = xorshift_noise(12345, 4096);
        // Not a formal proof of no coincidental periodicity, but xorshift32
        // output has no structure that would correlate at any specific lag
        // range -- this is the same "noise-like" testing convention
        // fft.rs's dominant_frequency tests already use.
        assert!(estimate_f0(&samples, 22050.0, &PitchConfig::default()).is_none());
    }

    #[test]
    fn returns_none_when_frame_too_short_for_the_requested_range() {
        // Default range's longest lag (sample_rate / 50Hz = 441 @ 22050Hz)
        // doesn't fit in a 10-sample frame.
        let samples = sine_wave(220.0, 22050.0, 10);
        assert!(estimate_f0(&samples, 22050.0, &PitchConfig::default()).is_none());
    }

    #[test]
    fn returns_none_for_degenerate_input() {
        let config = PitchConfig::default();
        assert!(estimate_f0(&[], 22050.0, &config).is_none());
        assert!(estimate_f0(&[1.0], 22050.0, &config).is_none());
        assert!(estimate_f0(&[1.0, 2.0, 3.0], 0.0, &config).is_none());
    }

    #[test]
    fn returns_none_for_invalid_config() {
        let samples = sine_wave(220.0, 22050.0, 4096);
        assert!(estimate_f0(
            &samples,
            22050.0,
            &PitchConfig { min_freq_hz: 0.0, max_freq_hz: 1000.0, voicing_threshold: 0.3 }
        )
        .is_none());
        assert!(estimate_f0(
            &samples,
            22050.0,
            &PitchConfig { min_freq_hz: 1000.0, max_freq_hz: 50.0, voicing_threshold: 0.3 }
        )
        .is_none());
    }

    #[test]
    fn contour_tracks_a_steady_tone_across_multiple_frames() {
        let sample_rate = 22050.0;
        let freq = 220.0;
        let frame_size = 2048;
        let hop_size = 512;
        let n = hop_size * 12 + frame_size;
        let samples = sine_wave(freq, sample_rate, n);

        let contour = estimate_f0_contour(&samples, sample_rate, frame_size, hop_size, &PitchConfig::default());
        let expected_frames = 1 + (n - frame_size) / hop_size;
        assert_eq!(contour.len(), expected_frames);

        let tolerance = quantization_tolerance(freq, sample_rate);
        for (i, entry) in contour.iter().enumerate() {
            let f0 = entry.unwrap_or_else(|| panic!("frame {i} should be voiced"));
            assert!(
                (f0 - freq).abs() < tolerance,
                "frame {i}: expected ~{freq}Hz (+/- {tolerance}), got {f0}Hz"
            );
        }
    }

    #[test]
    fn contour_of_empty_or_degenerate_input_is_empty() {
        let config = PitchConfig::default();
        assert!(estimate_f0_contour(&[], 22050.0, 2048, 512, &config).is_empty());
        assert!(estimate_f0_contour(&[1.0; 4096], 22050.0, 0, 512, &config).is_empty());
        assert!(estimate_f0_contour(&[1.0; 4096], 22050.0, 2048, 0, &config).is_empty());
        // frame_size longer than the whole series: no complete frame exists.
        assert!(estimate_f0_contour(&[1.0; 100], 22050.0, 2048, 512, &config).is_empty());
    }
}
