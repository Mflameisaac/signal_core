//! Peak, transient, and flat-region detection over a generic real-valued series.
//!
//! These are pure shape-detection primitives: they know nothing about
//! "silence" or "price anomalies". A caller renders "silence detection" by
//! calling [`find_flat_regions`] with a threshold near the series' noise
//! floor, and "anomaly detection" by calling the same function (or
//! [`find_transients`]) with thresholds tuned to price volatility. The
//! threshold tuning is entirely the caller's responsibility.

/// Configuration for [`find_local_maxima`].
#[derive(Debug, Clone, Copy)]
pub struct PeakConfig {
    /// Minimum sample value to be considered a peak. `None` means no minimum.
    pub min_height: Option<f32>,
    /// Minimum number of samples between two reported peaks. Peaks closer
    /// together than this are suppressed, keeping only the taller one.
    pub min_distance: usize,
}

impl Default for PeakConfig {
    fn default() -> Self {
        PeakConfig {
            min_height: None,
            min_distance: 1,
        }
    }
}

/// Find indices of local maxima in `samples`.
///
/// A sample is a local maximum if it is strictly greater than its
/// left neighbor and greater than or equal to its right neighbor (this
/// picks a single representative index for a flat-topped plateau peak).
/// Endpoints (index 0 and the last index) are never reported, since they
/// have no two-sided neighborhood.
pub fn find_local_maxima(samples: &[f32], config: &PeakConfig) -> Vec<usize> {
    if samples.len() < 3 {
        return Vec::new();
    }

    let mut candidates: Vec<usize> = (1..samples.len() - 1)
        .filter(|&i| samples[i] > samples[i - 1] && samples[i] >= samples[i + 1])
        .filter(|&i| config.min_height.map_or(true, |h| samples[i] >= h))
        .collect();

    if config.min_distance <= 1 || candidates.len() < 2 {
        return candidates;
    }

    // Greedily keep the tallest candidates, suppressing any other candidate
    // within `min_distance` samples of one already accepted.
    candidates.sort_by(|&a, &b| {
        samples[b]
            .partial_cmp(&samples[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut accepted: Vec<usize> = Vec::new();
    for c in candidates {
        let too_close = accepted
            .iter()
            .any(|&a| (a as isize - c as isize).unsigned_abs() < config.min_distance);
        if !too_close {
            accepted.push(c);
        }
    }
    accepted.sort_unstable();
    accepted
}

/// Find indices where the sample-to-sample change exceeds `threshold`.
///
/// For each `i > 0`, reports `i` if `|samples[i] - samples[i-1]| > threshold`.
/// This is the same primitive whether it's called to flag a transient in an
/// audio waveform or a sudden jump in a price series — only `threshold` differs.
pub fn find_transients(samples: &[f32], threshold: f32) -> Vec<usize> {
    if samples.len() < 2 {
        return Vec::new();
    }
    (1..samples.len())
        .filter(|&i| (samples[i] - samples[i - 1]).abs() > threshold)
        .collect()
}

/// Find contiguous regions where every sample's absolute value is at or
/// below `amplitude_threshold`, and the region is at least `min_length`
/// samples long.
///
/// Returns `(start, end)` index pairs, `end` exclusive. Tune
/// `amplitude_threshold` near the noise floor for silence detection, or
/// near an expected-volatility band for flat/no-activity detection in a
/// price series.
pub fn find_flat_regions(
    samples: &[f32],
    amplitude_threshold: f32,
    min_length: usize,
) -> Vec<(usize, usize)> {
    let mut regions = Vec::new();
    let mut start: Option<usize> = None;

    for (i, &s) in samples.iter().enumerate() {
        let is_flat = s.abs() <= amplitude_threshold;
        match (is_flat, start) {
            (true, None) => start = Some(i),
            (false, Some(s0)) => {
                if i - s0 >= min_length {
                    regions.push((s0, i));
                }
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s0) = start {
        if samples.len() - s0 >= min_length {
            regions.push((s0, samples.len()));
        }
    }
    regions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_local_maxima_in_simple_signal() {
        let samples = vec![0.0, 1.0, 0.0, 2.0, 0.5, 3.0, 0.0];
        let peaks = find_local_maxima(&samples, &PeakConfig::default());
        assert_eq!(peaks, vec![1, 3, 5]);
    }

    #[test]
    fn short_signals_have_no_peaks() {
        assert!(find_local_maxima(&[], &PeakConfig::default()).is_empty());
        assert!(find_local_maxima(&[1.0], &PeakConfig::default()).is_empty());
        assert!(find_local_maxima(&[1.0, 2.0], &PeakConfig::default()).is_empty());
    }

    #[test]
    fn min_height_filters_out_small_peaks() {
        let samples = vec![0.0, 1.0, 0.0, 5.0, 0.0];
        let config = PeakConfig {
            min_height: Some(2.0),
            min_distance: 1,
        };
        let peaks = find_local_maxima(&samples, &config);
        assert_eq!(peaks, vec![3]);
    }

    #[test]
    fn min_distance_keeps_only_the_tallest_nearby_peak() {
        // two close peaks (indices 1 and 3), one far peak (index 8)
        let samples = vec![0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0];
        let config = PeakConfig {
            min_height: None,
            min_distance: 4,
        };
        let peaks = find_local_maxima(&samples, &config);
        // index 3 (height 3) should suppress index 1 (height 2); index 8 stands alone
        assert_eq!(peaks, vec![3, 8]);
    }

    #[test]
    fn transients_flag_sudden_jumps() {
        let mut samples = vec![0.0f32; 10];
        samples[5] = 10.0; // sudden jump up, then a jump back down at index 6
        let transients = find_transients(&samples, 1.0);
        assert_eq!(transients, vec![5, 6]);
    }

    #[test]
    fn no_transients_below_threshold() {
        let samples = vec![0.0, 0.1, 0.2, 0.1, 0.0];
        assert!(find_transients(&samples, 1.0).is_empty());
    }

    #[test]
    fn finds_flat_region_in_the_middle_of_a_signal() {
        let mut samples = vec![1.0f32; 20];
        for s in samples.iter_mut().take(15).skip(5) {
            *s = 0.0; // "silent"/flat region from index 5 to 14 inclusive
        }
        let regions = find_flat_regions(&samples, 0.01, 5);
        assert_eq!(regions, vec![(5, 15)]);
    }

    #[test]
    fn ignores_flat_regions_shorter_than_min_length() {
        let samples = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        assert!(find_flat_regions(&samples, 0.01, 3).is_empty());
    }

    #[test]
    fn flat_region_extending_to_end_of_series_is_reported() {
        let samples = vec![1.0, 1.0, 0.0, 0.0, 0.0];
        let regions = find_flat_regions(&samples, 0.01, 2);
        assert_eq!(regions, vec![(2, 5)]);
    }
}
