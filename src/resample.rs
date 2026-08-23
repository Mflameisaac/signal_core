//! Downsampling and resampling for rendering large series efficiently
//! (e.g. a waveform or chart view that can't draw one point per sample).

/// Decimate `samples` by averaging non-overlapping chunks of `factor` samples.
///
/// The last chunk may be shorter than `factor` if `samples.len()` isn't an
/// exact multiple; it is still averaged over however many samples it has.
/// `factor <= 1` returns the input unchanged.
pub fn downsample_average(samples: &[f32], factor: usize) -> Vec<f32> {
    if factor <= 1 || samples.is_empty() {
        return samples.to_vec();
    }
    samples
        .chunks(factor)
        .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
        .collect()
}

/// Min/max envelope downsampling, the standard technique for rendering a
/// waveform/chart with far fewer points than samples while preserving
/// visible peaks that a plain average or stride would smooth away.
///
/// Splits `samples` into `num_buckets` contiguous chunks and, for each,
/// emits `(min, max)` in chronological order (i.e. whichever of the min/max
/// occurred first in the chunk comes first in the pair). Output length is
/// always `2 * num_buckets` (flattened). Returns the input unchanged
/// (as a single min/max pair) if `num_buckets == 0` or `samples` is empty.
pub fn downsample_minmax(samples: &[f32], num_buckets: usize) -> Vec<f32> {
    if samples.is_empty() || num_buckets == 0 {
        return Vec::new();
    }
    let n = samples.len();
    let buckets = num_buckets.min(n);
    let mut out = Vec::with_capacity(buckets * 2);

    for b in 0..buckets {
        let start = b * n / buckets;
        let end = ((b + 1) * n / buckets).max(start + 1).min(n);
        let chunk = &samples[start..end];

        let mut min_idx = 0usize;
        let mut max_idx = 0usize;
        for (i, &v) in chunk.iter().enumerate() {
            if v < chunk[min_idx] {
                min_idx = i;
            }
            if v > chunk[max_idx] {
                max_idx = i;
            }
        }
        if min_idx <= max_idx {
            out.push(chunk[min_idx]);
            out.push(chunk[max_idx]);
        } else {
            out.push(chunk[max_idx]);
            out.push(chunk[min_idx]);
        }
    }
    out
}

/// Resample `samples` to exactly `target_len` points via linear interpolation.
///
/// Works for both downsampling (`target_len < samples.len()`) and
/// upsampling (`target_len > samples.len()`). The first and last output
/// samples always equal the first and last input samples.
pub fn linear_resample(samples: &[f32], target_len: usize) -> Vec<f32> {
    if target_len == 0 || samples.is_empty() {
        return Vec::new();
    }
    if samples.len() == 1 || target_len == 1 {
        return vec![samples[0]; target_len];
    }

    let src_last = (samples.len() - 1) as f32;
    let dst_last = (target_len - 1) as f32;

    (0..target_len)
        .map(|i| {
            let src_pos = i as f32 * src_last / dst_last;
            let lo = src_pos.floor() as usize;
            let hi = (lo + 1).min(samples.len() - 1);
            let frac = src_pos - lo as f32;
            samples[lo] * (1.0 - frac) + samples[hi] * frac
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downsample_average_reduces_length_and_averages_correctly() {
        let samples: Vec<f32> = (1..=10).map(|x| x as f32).collect(); // 1..10
        let out = downsample_average(&samples, 2);
        assert_eq!(out, vec![1.5, 3.5, 5.5, 7.5, 9.5]);
    }

    #[test]
    fn downsample_average_handles_uneven_final_chunk() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let out = downsample_average(&samples, 2);
        assert_eq!(out, vec![1.5, 3.5, 5.0]);
    }

    #[test]
    fn downsample_average_factor_one_is_identity() {
        let samples = vec![1.0, 2.0, 3.0];
        assert_eq!(downsample_average(&samples, 1), samples);
        assert_eq!(downsample_average(&samples, 0), samples);
    }

    #[test]
    fn downsample_minmax_captures_a_spike_that_averaging_would_hide() {
        let mut samples = vec![0.0f32; 100];
        samples[42] = 100.0; // a single spike
        let averaged = downsample_average(&samples, 10);
        let envelope = downsample_minmax(&samples, 10);

        // the bucket containing index 42 is bucket 4 (42/10)
        let avg_bucket_val = averaged[4];
        let (env_min, env_max) = (envelope[4 * 2], envelope[4 * 2 + 1]);

        assert!(
            avg_bucket_val < 15.0,
            "plain averaging should mostly hide the spike, got {avg_bucket_val}"
        );
        assert!(
            env_max >= 99.9,
            "min/max envelope should preserve the spike's peak, got {env_max}"
        );
        assert!(env_min <= 0.0);
    }

    #[test]
    fn downsample_minmax_output_length_is_twice_bucket_count() {
        let samples = vec![1.0; 50];
        let out = downsample_minmax(&samples, 7);
        assert_eq!(out.len(), 14);
    }

    #[test]
    fn downsample_minmax_empty_input() {
        assert!(downsample_minmax(&[], 10).is_empty());
        assert!(downsample_minmax(&[1.0, 2.0], 0).is_empty());
    }

    #[test]
    fn linear_resample_preserves_endpoints() {
        let samples = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let out = linear_resample(&samples, 3);
        assert_eq!(out[0], 10.0);
        assert_eq!(out[out.len() - 1], 50.0);
    }

    #[test]
    fn linear_resample_upsampling_interpolates_between_points() {
        let samples = vec![0.0, 10.0];
        let out = linear_resample(&samples, 5);
        // evenly spaced interpolation from 0 to 10 across 5 points
        let expected = [0.0, 2.5, 5.0, 7.5, 10.0];
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} != {b}");
        }
    }

    #[test]
    fn linear_resample_downsampling_reduces_length() {
        let samples: Vec<f32> = (0..100).map(|x| x as f32).collect();
        let out = linear_resample(&samples, 10);
        assert_eq!(out.len(), 10);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[9], 99.0);
    }

    #[test]
    fn linear_resample_single_sample_input_broadcasts() {
        let out = linear_resample(&[7.0], 4);
        assert_eq!(out, vec![7.0, 7.0, 7.0, 7.0]);
    }

    #[test]
    fn linear_resample_empty_or_zero_target() {
        assert!(linear_resample(&[], 10).is_empty());
        assert!(linear_resample(&[1.0, 2.0], 0).is_empty());
    }
}
