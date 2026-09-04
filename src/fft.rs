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

/// Periodic-form window coefficients (denominator `len`, not `len - 1`) —
/// the convention `torch.hann_window`/`torch.stft` and most spectrogram
/// pipelines default to, distinct from [`crate::window::window_coefficients`]'s
/// symmetric form (denominator `len - 1`, matching `numpy.hanning`). The two
/// forms differ subtly and a model trained on one expects the other at
/// inference time — this stays local to [`stft_magnitude`] rather than
/// changing `window`'s existing (symmetric) behavior for its current callers.
fn periodic_window_coefficients(window: crate::window::WindowType, len: usize) -> Vec<f32> {
    use crate::window::WindowType;
    use std::f32::consts::PI;
    if len == 0 {
        return Vec::new();
    }
    let n = len as f32;
    (0..len)
        .map(|i| {
            let x = i as f32 / n;
            match window {
                WindowType::Rectangular => 1.0,
                WindowType::Hann => 0.5 - 0.5 * (2.0 * PI * x).cos(),
                WindowType::Hamming => 0.54 - 0.46 * (2.0 * PI * x).cos(),
                WindowType::Blackman => {
                    0.42 - 0.5 * (2.0 * PI * x).cos() + 0.08 * (4.0 * PI * x).cos()
                }
            }
        })
        .collect()
}

/// Mirror-pad `samples` by `pad` samples on each side without repeating the
/// edge sample (`[a, b, c]` padded by 2 -> `[c, b, a, b, c, b, a]`) — the
/// same "reflect" convention `torch.nn.functional.pad(..., mode="reflect")`
/// uses, which spectrogram pipelines built on `torch.stft(..., center=False)`
/// apply before framing so edge frames aren't zero-padded into silence.
fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    let n = samples.len();
    if n == 0 {
        return vec![0.0; pad * 2];
    }
    let mut out = Vec::with_capacity(n + 2 * pad);
    for i in (1..=pad).rev() {
        out.push(samples[i.min(n - 1)]);
    }
    out.extend_from_slice(samples);
    for i in 1..=pad {
        let idx = (n as isize - 1 - i as isize).max(0) as usize;
        out.push(samples[idx]);
    }
    out
}

/// One-sided magnitude spectrogram: reflect-pads `samples` by
/// `(fft_size - hop_size) / 2` on each side (matching a `center=False`
/// `torch.stft` preceded by that padding, the convention voice/speech models
/// like OpenVoice's tone-color converter expect), frames it into overlapping
/// windows (`fft_size`, hop `hop_size`), applies a periodic-form `window`,
/// and computes each frame's one-sided FFT magnitude (`fft_size / 2 + 1`
/// bins, with the same small epsilon under the square root that PyTorch's
/// own `spectrogram_torch` uses, for numerical parity with models trained
/// against it). Returns one `Vec<f32>` of bins per frame — outer index is
/// the frame/time axis, inner index is the frequency bin.
///
/// Returns an empty result for empty input, an empty or zero `fft_size`, or
/// a `hop_size` of zero.
pub fn stft_magnitude(
    samples: &[f32],
    fft_size: usize,
    hop_size: usize,
    window: crate::window::WindowType,
) -> Vec<Vec<f32>> {
    if samples.is_empty() || fft_size == 0 || hop_size == 0 || hop_size > fft_size {
        return Vec::new();
    }

    let pad = (fft_size - hop_size) / 2;
    let padded = reflect_pad(samples, pad);
    let win = periodic_window_coefficients(window, fft_size);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);
    let n_bins = fft_size / 2 + 1;

    let mut frames = Vec::new();
    let mut start = 0;
    while start + fft_size <= padded.len() {
        let mut buffer: Vec<Complex32> = padded[start..start + fft_size]
            .iter()
            .zip(win.iter())
            .map(|(&s, &w)| Complex32::new(s * w, 0.0))
            .collect();
        fft.process(&mut buffer);
        let magnitudes: Vec<f32> = buffer[..n_bins]
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im + 1e-6).sqrt())
            .collect();
        frames.push(magnitudes);
        start += hop_size;
    }
    frames
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

    #[test]
    fn stft_magnitude_of_empty_or_degenerate_input_is_empty() {
        assert!(stft_magnitude(&[], 1024, 256, crate::window::WindowType::Hann).is_empty());
        assert!(stft_magnitude(&[1.0; 4096], 0, 256, crate::window::WindowType::Hann).is_empty());
        assert!(stft_magnitude(&[1.0; 4096], 1024, 0, crate::window::WindowType::Hann).is_empty());
    }

    #[test]
    fn stft_magnitude_has_expected_frame_and_bin_counts() {
        let sample_rate = 22050.0;
        let fft_size = 1024;
        let hop_size = 256;
        // A handful of hops' worth of samples, long enough for several frames.
        let n = hop_size * 20;
        let samples = sine_wave(440.0, sample_rate, n);

        let frames = stft_magnitude(&samples, fft_size, hop_size, crate::window::WindowType::Hann);

        let pad = (fft_size - hop_size) / 2;
        let expected_frames = 1 + (n + 2 * pad - fft_size) / hop_size;
        assert_eq!(frames.len(), expected_frames);
        for frame in &frames {
            assert_eq!(frame.len(), fft_size / 2 + 1);
        }
    }

    #[test]
    fn stft_magnitude_detects_pure_tone_in_the_expected_bin_every_frame() {
        let sample_rate = 22050.0;
        let fft_size = 1024;
        let hop_size = 256;
        // Bin index for a tone that lands exactly on an FFT bin.
        let bin = 64;
        let freq = bin as f32 * sample_rate / fft_size as f32;
        let n = hop_size * 30;
        let samples = sine_wave(freq, sample_rate, n);

        let frames = stft_magnitude(&samples, fft_size, hop_size, crate::window::WindowType::Hann);
        assert!(!frames.is_empty());

        // Ignore the first/last couple of frames, where reflect-padding can
        // distort which bin dominates.
        for frame in frames.iter().skip(2).take(frames.len().saturating_sub(4)) {
            let (max_bin, _) = frame
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            assert_eq!(max_bin, bin, "expected the tone's own bin to dominate every interior frame");
        }
    }
}
