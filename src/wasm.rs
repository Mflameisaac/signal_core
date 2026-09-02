//! wasm-bindgen surface: thin typed-array wrappers around the pure-Rust
//! primitives in [`crate::fft`], [`crate::window`], [`crate::filter`],
//! [`crate::peaks`], and [`crate::resample`].
//!
//! No signal-processing logic lives in this module — it only converts
//! between `Float32Array`/`Uint32Array` and the native `Vec<f32>`/`&[f32]`
//! types the core functions use, so both TypeScript consumers can call
//! straight through without writing their own glue.

use js_sys::{Float32Array, Uint32Array};
use wasm_bindgen::prelude::*;

use crate::{biquad, dynamics, fft, filter, loudness, peaks, resample, window};

#[cfg(feature = "console_error_panic_hook")]
#[wasm_bindgen(js_name = initPanicHook)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

fn to_f32_vec(samples: &Float32Array) -> Vec<f32> {
    samples.to_vec()
}

fn to_f32_array(v: Vec<f32>) -> Float32Array {
    Float32Array::from(v.as_slice())
}

fn to_u32_array(v: &[usize]) -> Uint32Array {
    let converted: Vec<u32> = v.iter().map(|&i| i as u32).collect();
    Uint32Array::from(converted.as_slice())
}

// ---------------------------------------------------------------------
// Windowing
// ---------------------------------------------------------------------

/// Window function kind, mirroring [`window::WindowType`] for JS callers.
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum WindowType {
    Rectangular,
    Hann,
    Hamming,
    Blackman,
}

impl From<WindowType> for window::WindowType {
    fn from(w: WindowType) -> Self {
        match w {
            WindowType::Rectangular => window::WindowType::Rectangular,
            WindowType::Hann => window::WindowType::Hann,
            WindowType::Hamming => window::WindowType::Hamming,
            WindowType::Blackman => window::WindowType::Blackman,
        }
    }
}

#[wasm_bindgen(js_name = applyWindow)]
pub fn apply_window(samples: &Float32Array, window_type: WindowType) -> Float32Array {
    let input = to_f32_vec(samples);
    to_f32_array(window::apply_window(&input, window_type.into()))
}

// ---------------------------------------------------------------------
// FFT / spectral analysis
// ---------------------------------------------------------------------

/// Complex FFT result, exposed as separate real/imaginary Float32Arrays.
#[wasm_bindgen]
pub struct FftResult {
    real: Vec<f32>,
    imag: Vec<f32>,
}

#[wasm_bindgen]
impl FftResult {
    #[wasm_bindgen(getter)]
    pub fn real(&self) -> Float32Array {
        Float32Array::from(self.real.as_slice())
    }

    #[wasm_bindgen(getter)]
    pub fn imag(&self) -> Float32Array {
        Float32Array::from(self.imag.as_slice())
    }
}

#[wasm_bindgen(js_name = forwardFft)]
pub fn forward_fft(samples: &Float32Array) -> FftResult {
    let input = to_f32_vec(samples);
    let spectrum = fft::forward_fft(&input);
    FftResult {
        real: spectrum.iter().map(|c| c.re).collect(),
        imag: spectrum.iter().map(|c| c.im).collect(),
    }
}

#[wasm_bindgen(js_name = powerSpectrum)]
pub fn power_spectrum(samples: &Float32Array) -> Float32Array {
    let input = to_f32_vec(samples);
    to_f32_array(fft::power_spectrum(&input))
}

/// Dominant frequency-domain peak, exposed as scalar getters.
#[wasm_bindgen]
pub struct DominantFrequencyResult {
    frequency_hz: f32,
    power: f32,
}

#[wasm_bindgen]
impl DominantFrequencyResult {
    #[wasm_bindgen(getter, js_name = frequencyHz)]
    pub fn frequency_hz(&self) -> f32 {
        self.frequency_hz
    }

    #[wasm_bindgen(getter)]
    pub fn power(&self) -> f32 {
        self.power
    }
}

#[wasm_bindgen(js_name = dominantFrequency)]
pub fn dominant_frequency(
    samples: &Float32Array,
    sample_rate_hz: f32,
) -> Option<DominantFrequencyResult> {
    let input = to_f32_vec(samples);
    fft::dominant_frequency(&input, sample_rate_hz).map(|d| DominantFrequencyResult {
        frequency_hz: d.frequency_hz,
        power: d.power,
    })
}

// ---------------------------------------------------------------------
// Filtering
// ---------------------------------------------------------------------

#[wasm_bindgen(js_name = movingAverage)]
pub fn moving_average(samples: &Float32Array, window_size: usize) -> Float32Array {
    let input = to_f32_vec(samples);
    to_f32_array(filter::moving_average(&input, window_size))
}

#[wasm_bindgen(js_name = lowPassFilter)]
pub fn low_pass_filter(samples: &Float32Array, sample_rate_hz: f32, cutoff_hz: f32) -> Float32Array {
    let input = to_f32_vec(samples);
    to_f32_array(filter::low_pass_filter(&input, sample_rate_hz, cutoff_hz))
}

#[wasm_bindgen(js_name = highPassFilter)]
pub fn high_pass_filter(samples: &Float32Array, sample_rate_hz: f32, cutoff_hz: f32) -> Float32Array {
    let input = to_f32_vec(samples);
    to_f32_array(filter::high_pass_filter(&input, sample_rate_hz, cutoff_hz))
}

// ---------------------------------------------------------------------
// Biquad EQ (parametric/shelf, 2-pole low/high-pass)
// ---------------------------------------------------------------------

#[wasm_bindgen(js_name = peakingEq)]
pub fn peaking_eq(samples: &Float32Array, freq_hz: f32, gain_db: f32, q: f32, sample_rate_hz: f32) -> Float32Array {
    let input = to_f32_vec(samples);
    let mut b = biquad::peaking_eq(freq_hz, gain_db, q, sample_rate_hz);
    to_f32_array(biquad::apply_biquad(&input, &mut b))
}

#[wasm_bindgen(js_name = lowShelf)]
pub fn low_shelf(samples: &Float32Array, freq_hz: f32, gain_db: f32, q: f32, sample_rate_hz: f32) -> Float32Array {
    let input = to_f32_vec(samples);
    let mut b = biquad::low_shelf(freq_hz, gain_db, q, sample_rate_hz);
    to_f32_array(biquad::apply_biquad(&input, &mut b))
}

#[wasm_bindgen(js_name = highShelf)]
pub fn high_shelf(samples: &Float32Array, freq_hz: f32, gain_db: f32, q: f32, sample_rate_hz: f32) -> Float32Array {
    let input = to_f32_vec(samples);
    let mut b = biquad::high_shelf(freq_hz, gain_db, q, sample_rate_hz);
    to_f32_array(biquad::apply_biquad(&input, &mut b))
}

#[wasm_bindgen(js_name = biquadLowPass)]
pub fn biquad_low_pass(samples: &Float32Array, freq_hz: f32, q: f32, sample_rate_hz: f32) -> Float32Array {
    let input = to_f32_vec(samples);
    let mut b = biquad::low_pass(freq_hz, q, sample_rate_hz);
    to_f32_array(biquad::apply_biquad(&input, &mut b))
}

#[wasm_bindgen(js_name = biquadHighPass)]
pub fn biquad_high_pass(samples: &Float32Array, freq_hz: f32, q: f32, sample_rate_hz: f32) -> Float32Array {
    let input = to_f32_vec(samples);
    let mut b = biquad::high_pass(freq_hz, q, sample_rate_hz);
    to_f32_array(biquad::apply_biquad(&input, &mut b))
}

// ---------------------------------------------------------------------
// Dynamics (compressor / limiter)
// ---------------------------------------------------------------------

#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn compress(
    samples: &Float32Array,
    sample_rate_hz: f32,
    threshold_db: f32,
    ratio: f32,
    attack_ms: f32,
    release_ms: f32,
    makeup_db: f32,
) -> Float32Array {
    let input = to_f32_vec(samples);
    to_f32_array(dynamics::compress(
        &input,
        sample_rate_hz,
        threshold_db,
        ratio,
        attack_ms,
        release_ms,
        makeup_db,
    ))
}

#[wasm_bindgen]
pub fn limit(samples: &Float32Array, sample_rate_hz: f32, ceiling_db: f32, release_ms: f32) -> Float32Array {
    let input = to_f32_vec(samples);
    to_f32_array(dynamics::limit(&input, sample_rate_hz, ceiling_db, release_ms))
}

// ---------------------------------------------------------------------
// Loudness (ITU-R BS.1770)
// ---------------------------------------------------------------------

#[wasm_bindgen(js_name = integratedLoudness)]
pub fn integrated_loudness(samples: &Float32Array, sample_rate_hz: f32, channels: usize) -> f32 {
    let input = to_f32_vec(samples);
    loudness::integrated_loudness(&input, sample_rate_hz, channels)
}

#[wasm_bindgen(js_name = truePeakDb)]
pub fn true_peak_db(samples: &Float32Array) -> f32 {
    let input = to_f32_vec(samples);
    loudness::true_peak_db(&input)
}

// ---------------------------------------------------------------------
// Peak / transient / flat-region detection
// ---------------------------------------------------------------------

/// Local maxima result: indices and the sample value at each index.
#[wasm_bindgen]
pub struct PeakResult {
    indices: Vec<usize>,
    values: Vec<f32>,
}

#[wasm_bindgen]
impl PeakResult {
    #[wasm_bindgen(getter)]
    pub fn indices(&self) -> Uint32Array {
        to_u32_array(&self.indices)
    }

    #[wasm_bindgen(getter)]
    pub fn values(&self) -> Float32Array {
        Float32Array::from(self.values.as_slice())
    }
}

#[wasm_bindgen(js_name = findLocalMaxima)]
pub fn find_local_maxima(
    samples: &Float32Array,
    min_height: Option<f32>,
    min_distance: usize,
) -> PeakResult {
    let input = to_f32_vec(samples);
    let config = peaks::PeakConfig {
        min_height,
        min_distance: min_distance.max(1),
    };
    let indices = peaks::find_local_maxima(&input, &config);
    let values = indices.iter().map(|&i| input[i]).collect();
    PeakResult { indices, values }
}

#[wasm_bindgen(js_name = findTransients)]
pub fn find_transients(samples: &Float32Array, threshold: f32) -> Uint32Array {
    let input = to_f32_vec(samples);
    to_u32_array(&peaks::find_transients(&input, threshold))
}

/// Flat-region result: parallel `starts`/`ends` index arrays (end exclusive).
#[wasm_bindgen]
pub struct FlatRegionsResult {
    starts: Vec<usize>,
    ends: Vec<usize>,
}

#[wasm_bindgen]
impl FlatRegionsResult {
    #[wasm_bindgen(getter)]
    pub fn starts(&self) -> Uint32Array {
        to_u32_array(&self.starts)
    }

    #[wasm_bindgen(getter)]
    pub fn ends(&self) -> Uint32Array {
        to_u32_array(&self.ends)
    }
}

#[wasm_bindgen(js_name = findFlatRegions)]
pub fn find_flat_regions(
    samples: &Float32Array,
    amplitude_threshold: f32,
    min_length: usize,
) -> FlatRegionsResult {
    let input = to_f32_vec(samples);
    let regions = peaks::find_flat_regions(&input, amplitude_threshold, min_length);
    FlatRegionsResult {
        starts: regions.iter().map(|&(s, _)| s).collect(),
        ends: regions.iter().map(|&(_, e)| e).collect(),
    }
}

// ---------------------------------------------------------------------
// Downsampling / resampling
// ---------------------------------------------------------------------

#[wasm_bindgen(js_name = downsampleAverage)]
pub fn downsample_average(samples: &Float32Array, factor: usize) -> Float32Array {
    let input = to_f32_vec(samples);
    to_f32_array(resample::downsample_average(&input, factor))
}

/// Min/max envelope result: parallel `mins`/`maxes` arrays, one pair per bucket.
#[wasm_bindgen]
pub struct MinMaxResult {
    mins: Vec<f32>,
    maxes: Vec<f32>,
}

#[wasm_bindgen]
impl MinMaxResult {
    #[wasm_bindgen(getter)]
    pub fn mins(&self) -> Float32Array {
        Float32Array::from(self.mins.as_slice())
    }

    #[wasm_bindgen(getter)]
    pub fn maxes(&self) -> Float32Array {
        Float32Array::from(self.maxes.as_slice())
    }
}

#[wasm_bindgen(js_name = downsampleMinMax)]
pub fn downsample_minmax(samples: &Float32Array, num_buckets: usize) -> MinMaxResult {
    let input = to_f32_vec(samples);
    let interleaved = resample::downsample_minmax(&input, num_buckets);
    MinMaxResult {
        mins: interleaved.iter().step_by(2).copied().collect(),
        maxes: interleaved.iter().skip(1).step_by(2).copied().collect(),
    }
}

#[wasm_bindgen(js_name = linearResample)]
pub fn linear_resample(samples: &Float32Array, target_len: usize) -> Float32Array {
    let input = to_f32_vec(samples);
    to_f32_array(resample::linear_resample(&input, target_len))
}
