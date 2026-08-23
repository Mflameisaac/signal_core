//! Windowing functions for pre-conditioning a series before spectral analysis.

use std::f32::consts::PI;

/// A named window function. `Rectangular` applies no shaping (all coefficients are 1.0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    Rectangular,
    Hann,
    Hamming,
    Blackman,
}

/// Compute the `len` window coefficients for the given window type.
///
/// For `len <= 1` every coefficient is `1.0` (a window is not meaningful on 0 or 1 samples).
pub fn window_coefficients(window: WindowType, len: usize) -> Vec<f32> {
    if len <= 1 {
        return vec![1.0; len];
    }
    let n = (len - 1) as f32;
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

/// Apply a window function to `samples`, returning a new windowed series of the same length.
pub fn apply_window(samples: &[f32], window: WindowType) -> Vec<f32> {
    let coeffs = window_coefficients(window, samples.len());
    samples
        .iter()
        .zip(coeffs.iter())
        .map(|(s, c)| s * c)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangular_is_all_ones() {
        let w = window_coefficients(WindowType::Rectangular, 8);
        assert!(w.iter().all(|&c| (c - 1.0).abs() < 1e-6));
    }

    #[test]
    fn hann_endpoints_are_zero_and_peak_is_one() {
        let w = window_coefficients(WindowType::Hann, 9);
        assert!(w[0].abs() < 1e-6, "first sample should be ~0, got {}", w[0]);
        assert!(
            (w[w.len() - 1]).abs() < 1e-6,
            "last sample should be ~0, got {}",
            w[w.len() - 1]
        );
        let mid = w[4];
        assert!((mid - 1.0).abs() < 1e-6, "midpoint should be ~1, got {mid}");
    }

    #[test]
    fn hamming_endpoints_are_nonzero() {
        let w = window_coefficients(WindowType::Hamming, 9);
        // Hamming's defining trait vs Hann: endpoints don't touch zero.
        assert!((w[0] - 0.08).abs() < 1e-3, "expected ~0.08, got {}", w[0]);
    }

    #[test]
    fn blackman_endpoints_near_zero() {
        let w = window_coefficients(WindowType::Blackman, 9);
        assert!(w[0].abs() < 1e-3);
    }

    #[test]
    fn apply_window_scales_samples() {
        let samples = vec![1.0; 8];
        let windowed = apply_window(&samples, WindowType::Hann);
        let coeffs = window_coefficients(WindowType::Hann, 8);
        for (w, c) in windowed.iter().zip(coeffs.iter()) {
            assert!((w - c).abs() < 1e-6);
        }
    }

    #[test]
    fn degenerate_lengths_return_ones() {
        assert_eq!(window_coefficients(WindowType::Hann, 0), Vec::<f32>::new());
        assert_eq!(window_coefficients(WindowType::Hann, 1), vec![1.0]);
    }
}
