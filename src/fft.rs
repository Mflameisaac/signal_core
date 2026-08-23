//! Forward FFT and spectral analysis over a generic real-valued time series.
//!
//! None of these functions know or care whether `samples` came from an audio
//! buffer or a price series — callers supply `sample_rate_hz` (samples per
//! second for audio, or samples per unit time for any other series) and get
//! back frequency-domain results in the same units.

use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

/// Compute the forward FFT of a real-valued series.
///
/// The input is zero-padded (or truncated) to `samples.len()` — callers that
/// want a specific FFT size (e.g. a power of two for speed) should pad before
/// calling. Returns the full complex spectrum (length == `samples.len()`).
pub fn forward_fft(samples: &[f32]) -> Vec<Complex32> {
    let mut buffer: Vec<Complex32> = samples.iter().map(|&s| Complex32::new(s, 0.0)).collect();
    if buffer.is_empty() {
        return buffer;
    }
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(buffer.len());
    fft.process(&mut buffer);
    buffer
}

/// Compute the one-sided power spectrum of a real-valued series.
///
/// Returns `floor(n/2) + 1` bins (DC through Nyquist), each the squared
/// magnitude of the corresponding FFT bin. This is the standard "real FFT"
/// convention: bins beyond Nyquist are redundant conjugates for a real input
/// and are omitted.
pub fn power_spectrum(samples: &[f32]) -> Vec<f32> {
    let spectrum = forward_fft(samples);
    if spectrum.is_empty() {
        return Vec::new();
    }
    let bins = spectrum.len() / 2 + 1;
    spectrum[..bins].iter().map(|c| c.norm_sqr()).collect()
}

/// A single frequency-domain peak: its frequency in Hz and its power.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DominantFrequency {
    pub frequency_hz: f32,
    pub power: f32,
}

/// Find the dominant (highest-power) non-DC frequency component in `samples`.
///
/// `sample_rate_hz` is the sampling rate of `samples` in Hz (or more generally,
/// samples per unit time — the returned "frequency" is in cycles per that same
/// unit of time). Returns `None` for inputs shorter than 2 samples, since no
/// non-DC frequency bin exists below that length.
///
/// The DC bin (index 0, representing the series' mean/offset) is always
/// excluded, since it does not represent periodicity.
pub fn dominant_frequency(samples: &[f32], sample_rate_hz: f32) -> Option<DominantFrequency> {
    if samples.len() < 2 {
        return None;
    }
    let power = power_spectrum(samples);
    let n = samples.len();

    power
        .iter()
        .enumerate()
        .skip(1) // exclude DC
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(bin, &p)| DominantFrequency {
            frequency_hz: bin as f32 * sample_rate_hz / n as f32,
            power: p,
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Generate `n` samples of a sine wave at `freq_hz`, sampled at `sample_rate_hz`.
    fn sine_wave(freq_hz: f32, sample_rate_hz: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sample_rate_hz).sin())
            .collect()
    }

    #[test]
    fn fft_of_empty_is_empty() {
        assert!(forward_fft(&[]).is_empty());
    }

    #[test]
    fn fft_of_dc_signal_has_energy_only_at_bin_zero() {
        let samples = vec![1.0f32; 64];
        let spectrum = forward_fft(&samples);
        assert!(spectrum[0].norm() > 60.0); // sum of 64 ones
        for bin in &spectrum[1..] {
            assert!(bin.norm() < 1e-3, "expected ~0 energy, got {}", bin.norm());
        }
    }

    #[test]
    fn power_spectrum_has_expected_bin_count() {
        // n=64 -> 33 one-sided bins (DC..Nyquist inclusive)
        let samples = vec![0.0f32; 64];
        assert_eq!(power_spectrum(&samples).len(), 33);
        // odd length: n=65 -> 33 bins
        let samples = vec![0.0f32; 65];
        assert_eq!(power_spectrum(&samples).len(), 33);
    }

    #[test]
    fn dominant_frequency_detects_pure_sine_tone() {
        let sample_rate = 1024.0;
        let true_freq = 64.0; // exact bin for n=1024 -> bin 64
        let n = 1024;
        let samples = sine_wave(true_freq, sample_rate, n);
        let result = dominant_frequency(&samples, sample_rate).expect("should find a peak");
        assert!(
            (result.frequency_hz - true_freq).abs() < 1e-3,
            "expected ~{true_freq} Hz, got {}",
            result.frequency_hz
        );
    }

    #[test]
    fn dominant_frequency_detects_tone_amid_lower_amplitude_noise_like_signal() {
        let sample_rate = 2048.0;
        let true_freq = 200.0;
        let n = 2048;
        let mut samples = sine_wave(true_freq, sample_rate, n);
        // add a lower-amplitude second tone; the higher-amplitude tone should still win
        for (i, s) in samples.iter_mut().enumerate() {
            *s += 0.1 * (2.0 * PI * 500.0 * i as f32 / sample_rate).sin();
        }
        let result = dominant_frequency(&samples, sample_rate).expect("should find a peak");
        assert!(
            (result.frequency_hz - true_freq).abs() < 2.0,
            "expected ~{true_freq} Hz, got {}",
            result.frequency_hz
        );
    }

    #[test]
    fn dominant_frequency_none_for_short_input() {
        assert!(dominant_frequency(&[], 1000.0).is_none());
        assert!(dominant_frequency(&[1.0], 1000.0).is_none());
    }
}
