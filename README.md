# signal-core

Domain-agnostic signal-processing primitives — FFT/spectral analysis,
windowing, filtering, peak/transient detection, and downsampling — for any
real-valued time series. Compiled once, in Rust, and consumed either as a
native Cargo dependency or as a WASM+JS package.

This library is designed for reuse across unrelated numeric domains —
audio processing and financial time-series analysis are both examples of
where the same primitives apply unchanged. Everything here operates on a
plain `&[f32]` series plus explicit parameters like sample rate — nothing
in this crate knows or assumes it's looking at audio or price data.
Domain-specific concepts ("silence detection", "anomaly detection") are
just this crate's generic primitives (peak detection, flat-region
detection) called with app-specific threshold tuning, and that tuning
happens in the consuming app, not here.

This repo is **not** a workspace member of any downstream app. It's an
independent, independently-versioned crate that consuming apps pull in as
a dependency.

## What's in it

| Module | Purpose |
|---|---|
| `window` | `Hann`, `Hamming`, `Blackman`, `Rectangular` window coefficients + application |
| `fft` | Forward FFT (via `rustfft`), one-sided power spectrum, dominant-frequency detection |
| `filter` | Centered moving average, single-pole low-pass / high-pass filters |
| `biquad` | Second-order IIR filters (RBJ cookbook) — low/high-pass, low/high shelf, peaking EQ |
| `dynamics` | Feed-forward compressor and brickwall peak limiter, log-domain envelope following |
| `loudness` | ITU-R BS.1770 K-weighted gated integrated loudness (LUFS) and an oversampled true-peak estimate |
| `peaks` | Local maxima, sample-to-sample transient detection, flat/silent region detection |
| `resample` | Decimation-by-average, min/max envelope downsampling, arbitrary-length linear resampling |
| `wasm` | Thin `wasm-bindgen` wrappers over all of the above (only compiled for `wasm32` targets) |

Every function is unit-tested against synthetic reference signals (e.g. a
known-frequency sine wave for FFT peak detection, a known spike for
envelope downsampling) — see each module's `#[cfg(test)]` block.

## Building

Native (runs the full test suite, no WASM toolchain needed):

```sh
cargo test
```

WASM package (requires [`wasm-pack`](https://rustwasm.github.io/wasm-pack/installer/)):

```sh
# ES module target (bundler-free, works with a plain <script type="module">
# or any modern bundler's `target: "web"` handling)
wasm-pack build --target web

# or, if the consuming app uses webpack/Vite's older CJS-friendly resolution
wasm-pack build --target bundler
```

Either command produces a `pkg/` directory containing the compiled
`.wasm` binary, the generated JS glue, and a `.d.ts` file with full
TypeScript types for every exported function and result struct. `pkg/` is
gitignored — it's a build artifact, not source.

CI (`.github/workflows/ci.yml`) runs `cargo test` and `wasm-pack build` on
every push/PR to `main`, so a broken change here is caught before it
propagates to either downstream app.

## Consuming from a Rust app (Cargo git dependency)

If a downstream app is itself Rust (e.g. a Tauri shell), depend on this
repo directly via Cargo's git dependency support — no publishing required:

```toml
# In the consuming app's Cargo.toml
[dependencies]
signal-core = { git = "ssh://git@github.com/Mflameisaac/signal_core.git", tag = "v0.1.0" }
```

Pin to a `tag` (or `rev` for a specific commit) rather than tracking a
branch, so an update to this library only reaches the consuming app when
you deliberately bump the pin. Tag releases here with semver (`v0.1.0`,
`v0.2.0`, ...) and bump the consuming apps' `Cargo.toml` pins deliberately
when you want to pick up changes.

## Consuming from a frontend (WASM + JS/TS)

1. Build the package for your target bundler (see above), producing `pkg/`.
2. Either:
   - Copy/symlink `pkg/` into the consuming app's dependency tree and
     reference it as a local package:
     ```json
     // consuming app's package.json
     "dependencies": {
       "signal-core": "file:../path/to/signal-core/pkg"
     }
     ```
   - Or, in CI, build `pkg/` as a step and publish it to a private npm
     registry / GitHub Packages, then depend on it like any other npm
     package (`"signal-core": "^0.1.0"`). This is the better long-term
     setup once both apps are consuming it regularly, since it decouples
     "rebuild the WASM package" from "consume a new version of it".

3. From TypeScript:

   ```ts
   import init, {
     forwardFft,
     powerSpectrum,
     dominantFrequency,
     applyWindow,
     WindowType,
     movingAverage,
     lowPassFilter,
     highPassFilter,
     findLocalMaxima,
     findTransients,
     findFlatRegions,
     downsampleAverage,
     downsampleMinMax,
     linearResample,
   } from "signal-core";

   await init(); // instantiates the wasm module once

   const samples = new Float32Array(/* your series */);

   // Spectral analysis
   const dom = dominantFrequency(samples, /* sampleRateHz */ 44100);
   if (dom) console.log(dom.frequencyHz, dom.power);

   // Windowing before FFT
   const windowed = applyWindow(samples, WindowType.Hann);

   // Peak detection — same function powers "silence detection" (small
   // amplitude threshold) and "price anomaly detection" (large amplitude
   // threshold); the threshold is entirely the caller's choice.
   const peaks = findLocalMaxima(samples, /* minHeight */ 0.5, /* minDistance */ 10);
   console.log(peaks.indices, peaks.values);

   const flat = findFlatRegions(samples, /* amplitudeThreshold */ 0.01, /* minLength */ 100);
   console.log(flat.starts, flat.ends);

   // Downsampling for chart/waveform rendering
   const envelope = downsampleMinMax(samples, /* numBuckets */ 800);
   ```

   Every exported function takes/returns `Float32Array`/`Uint32Array`
   directly, or a small result struct (`FftResult`, `PeakResult`,
   `FlatRegionsResult`, `MinMaxResult`, `DominantFrequencyResult`) exposing
   its fields as JS getters — see `pkg/signal_core.d.ts` after building for
   the exact, generated signatures.

## API reference

Full signatures always live in the source (`cargo doc --open` generates
browsable docs from the doc comments in each module) — this section is a
quicker, example-driven tour of every public function, including a plain-
language explanation of the DSP concept for anyone newer to signal
processing.

### `window` — shaping a series before FFT

FFT math implicitly assumes the series repeats forever. A raw slice of
samples almost never actually loops smoothly, so the FFT sees a sharp
"seam" at the edges and reports fake extra frequencies (called
*spectral leakage*) that aren't really in the signal. A window function
tapers the edges of the series down toward zero before the FFT so that
seam disappears.

- `WindowType` — enum: `Rectangular` (no tapering — the default/no-op),
  `Hann`, `Hamming`, `Blackman` (increasingly aggressive tapering; Hann is
  the most common general-purpose choice).
- `window_coefficients(window: WindowType, len: usize) -> Vec<f32>` —
  the raw multiplier for each of `len` positions, if you want to inspect
  or reuse them directly.
- `apply_window(samples: &[f32], window: WindowType) -> Vec<f32>` —
  multiplies `samples` by the window's coefficients, returning a new
  vector the same length as the input. Use this right before
  `fft::forward_fft` / `fft::power_spectrum`.

### `fft` — frequency-domain analysis

- `forward_fft(samples: &[f32]) -> Vec<Complex32>` — the raw FFT. Converts
  a series from "value over time" into "how much of each frequency is
  present," as complex numbers (real + imaginary parts encode both
  amplitude and phase per frequency bin). Most callers want
  `power_spectrum` or `dominant_frequency` instead of this directly.
- `power_spectrum(samples: &[f32]) -> Vec<f32>` — the FFT's magnitude
  squared, one-sided (only DC through Nyquist — the redundant mirrored
  half of a real-valued signal's spectrum is dropped). This is "how much
  energy is at each frequency," which is what you plot as a spectrogram
  bar or feed into peak-picking.
- `dominant_frequency(samples: &[f32], sample_rate_hz: f32) -> Option<DominantFrequency>` —
  convenience wrapper that runs `power_spectrum` and returns the single
  strongest non-DC frequency bin, converted from a bin index into Hz
  using `sample_rate_hz`. Returns `None` for inputs under 2 samples.
  `DominantFrequency` has two fields: `frequency_hz` and `power`.

### `pitch` — fundamental-frequency (F0) estimation

- `PitchConfig { min_freq_hz, max_freq_hz, voicing_threshold }` — the
  search range (in Hz) and the minimum normalized autocorrelation required
  to call a frame periodic ("voiced") at all. `Default` is `50-1000Hz` at
  a `0.3` threshold — wide enough to cover singing, not just
  conversational speech's narrower range.
- `estimate_f0(samples: &[f32], sample_rate_hz: f32, config: &PitchConfig) -> Option<f32>` —
  the classic normalized-autocorrelation method: for each candidate lag in
  the range `config` implies, computes `R(lag) / R(0)` and returns
  `sample_rate_hz / lag` for whichever lag maximizes it. `None` for
  silence, a frame too short for the requested range, an invalid `config`,
  or a best correlation under `voicing_threshold` (not clearly periodic).
- `estimate_f0_contour(samples: &[f32], sample_rate_hz: f32, frame_size: usize, hop_size: usize, config: &PitchConfig) -> Vec<Option<f32>>` —
  runs `estimate_f0` over successive frames, one entry per frame (`None`
  wherever `estimate_f0` would be).

Deliberately the simplest correct version of this method: no sub-sample
(parabolic) interpolation between lags, no octave-error correction, no
FFT-based speedup. See `estimate_f0`'s own doc comment for the
Wiener-Khinchin route (autocorrelation via inverse-FFT-of-power-spectrum)
if `O(N * lag_range)` per frame ever needs to become `O(N log N)`.

### `filter` — smoothing and frequency-selective filtering

- `moving_average(samples: &[f32], window_size: usize) -> Vec<f32>` — for
  each sample, averages it together with its neighbors within
  `window_size / 2` on each side (the window shrinks near the array edges
  rather than reading out of bounds). Flattens noise/jitter; the bigger
  `window_size`, the smoother — and blurrier — the result.
- `low_pass_filter(samples: &[f32], sample_rate_hz: f32, cutoff_hz: f32) -> Vec<f32>` —
  lets slow changes through, damps fast ones. A single-pole (first-order
  RC-equivalent) filter — simple and cheap, not a "brick wall": frequencies
  right at `cutoff_hz` are attenuated by about half power (-3dB), and it
  keeps attenuating more gradually above that, rather than cutting off
  sharply.
- `high_pass_filter(samples: &[f32], sample_rate_hz: f32, cutoff_hz: f32) -> Vec<f32>` —
  the mirror image of `low_pass_filter`: damps slow changes, lets fast ones
  through. Useful for removing a slowly drifting baseline/DC offset.

### `biquad` — second-order IIR filters

A biquad is a two-pole, two-zero IIR filter — one stage buys a much
sharper frequency response than `filter`'s single-pole filters, at the
cost of being stateful (it remembers the last two inputs and outputs).
Coefficients here follow the RBJ Audio EQ Cookbook formulas.

- `Biquad` — a stateful filter instance returned by the constructors
  below, with a `process(&mut self, x: f32) -> f32` method for one
  sample at a time. Carries its own internal delay line, so feeding it a
  signal in chunks across multiple `apply_biquad` calls (or `process`
  calls) gives the same result as one call with the whole signal.
- `high_pass(freq_hz: f32, q: f32, sample_rate_hz: f32) -> Biquad` /
  `low_pass(freq_hz: f32, q: f32, sample_rate_hz: f32) -> Biquad` — damp
  everything below/above `freq_hz`, with a much steeper rolloff than
  `filter`'s single-pole versions. `q = 0.7071` gives a maximally flat
  (Butterworth) response; higher `q` adds resonance/ringing right at the
  corner frequency.
- `peaking_eq(freq_hz: f32, gain_db: f32, q: f32, sample_rate_hz: f32) -> Biquad` —
  boosts (`gain_db > 0`) or cuts (`gain_db < 0`) a band centered at
  `freq_hz`; `q` controls how narrow that band is. This is a standard
  parametric EQ band.
- `low_shelf(freq_hz: f32, gain_db: f32, q: f32, sample_rate_hz: f32) -> Biquad` /
  `high_shelf(freq_hz: f32, gain_db: f32, q: f32, sample_rate_hz: f32) -> Biquad` —
  boost or cut everything below/above `freq_hz` by a flat `gain_db`,
  rather than a narrow band. `q = 0.7071` gives a classic shelf slope.
- `apply_biquad(samples: &[f32], filter: &mut Biquad) -> Vec<f32>` — runs
  `samples` through `filter` in order, updating its internal state as it
  goes.

### `dynamics` — compression and limiting

Both functions track a signal's level with a log-domain (dB) envelope
follower — separate attack/release time constants control how fast the
tracked level reacts to the signal getting louder vs. quieter — then
apply a gain curve to that envelope.

- `compress(samples: &[f32], sample_rate_hz: f32, threshold_db: f32, ratio: f32, attack_ms: f32, release_ms: f32, makeup_gain_db: f32) -> Vec<f32>` —
  reduces the level of anything louder than `threshold_db`, by `ratio`
  (e.g. `4.0` means every 4dB over the threshold becomes 1dB of output).
  `attack_ms`/`release_ms` control how quickly gain reduction engages and
  recovers; `makeup_gain_db` is a flat gain applied afterward to
  compensate for the average level lost to compression.
- `limit(samples: &[f32], sample_rate_hz: f32, ceiling_db: f32, release_ms: f32) -> Vec<f32>` —
  a lookahead peak limiter: guarantees no output sample exceeds
  `ceiling_db` in magnitude, by scanning a few milliseconds ahead of each
  sample for the loudest upcoming peak and reducing gain in advance —
  since this runs offline over a whole in-memory buffer, lookahead is
  "free" (no real-time output delay to manage). Gain reduction is instant;
  it recovers back toward unity over `release_ms` once a peak has passed,
  to avoid audible pumping.

### `loudness` — perceptual loudness and true-peak estimation

- `integrated_loudness(samples: &[f32], sample_rate_hz: f32, channels: usize) -> f32` —
  K-weighted, gated integrated loudness in LUFS, per ITU-R BS.1770 (the
  standard behind streaming platforms' loudness normalization).
  `samples` is interleaved (`channels` values per frame, e.g. `[L, R, L,
  R, ...]` for stereo). Each channel is K-weighted (a shelf + high-pass
  filter pair approximating human loudness perception), split into
  overlapping 400ms blocks, and averaged with an absolute gate at -70
  LUFS and a relative gate 10 LU below the ungated mean — this gating is
  what keeps quiet pauses from dragging down the measured loudness of an
  otherwise-consistent recording. All channels are weighted equally
  (`1.0`); this omits BS.1770's surround-channel weighting (`1.41` for
  rear channels), since this crate targets mono/stereo material.
- `true_peak_db(samples: &[f32]) -> f32` — an inter-sample peak estimate
  in dBFS, via FFT-based bandlimited interpolation (zero-padding each
  block's spectrum 4x and inverse-transforming). A plain sample-peak
  measurement can miss a peak that falls *between* two samples and would
  clip on D/A conversion or resampling; this recovers the true
  continuous-time peak instead.

### `peaks` — finding interesting points or regions

These are the primitives domain-specific "detectors" get built from —
see [Design notes](#design-notes) below for why nothing here is named
`detect_silence` or `detect_anomaly`.

- `PeakConfig { min_height: Option<f32>, min_distance: usize }` — tuning
  knobs for `find_local_maxima`. `min_height` discards peaks below a
  value; `min_distance` discards peaks too close to a taller
  already-accepted peak (so one wide bump doesn't get reported as ten
  separate near-duplicate peaks).
- `find_local_maxima(samples: &[f32], config: &PeakConfig) -> Vec<usize>` —
  indices where the series turns from rising to falling (a local hump).
- `find_transients(samples: &[f32], threshold: f32) -> Vec<usize>` —
  indices where the value jumps by more than `threshold` from the
  previous sample — i.e. a sudden, sharp change rather than a gradual
  hump. This is the primitive that powers "transient detection" in audio
  or "sudden price move" detection in a price series.
- `find_flat_regions(samples: &[f32], amplitude_threshold: f32, min_length: usize) -> Vec<(usize, usize)>` —
  contiguous `(start, end)` index ranges (end exclusive) where every
  sample's absolute value stays at or under `amplitude_threshold`, and the
  range is at least `min_length` samples long. Tune the threshold near
  the noise floor for silence detection, or near an expected-volatility
  band for "nothing interesting is happening here" detection in a price
  series.

### `resample` — rendering large series efficiently

- `downsample_average(samples: &[f32], factor: usize) -> Vec<f32>` —
  splits the series into chunks of `factor` samples and averages each
  chunk down to one point. Simple and cheap, but a brief spike gets
  smeared out and can disappear entirely.
- `downsample_minmax(samples: &[f32], num_buckets: usize) -> Vec<f32>` —
  the standard "waveform view" technique: splits into `num_buckets`
  chunks and keeps the *min and max* of each chunk (returned flattened,
  in chronological order per pair, so the output length is always
  `2 * num_buckets`). Unlike `downsample_average`, a single loud spike
  still shows up as a tall bar, because the max of its chunk is preserved
  exactly.
- `linear_resample(samples: &[f32], target_len: usize) -> Vec<f32>` —
  resamples to *any* target length (not just an integer downsampling
  factor) by linearly interpolating between the nearest input samples.
  Works for both shrinking and growing the series; the first and last
  output samples always exactly match the first and last input samples.

## Design notes

- **f32 throughout.** Internal computation and the WASM boundary both use
  `f32`/`Float32Array`, avoiding any conversion at the JS interop boundary.
  If a consuming app needs more precision than `f32` offers for very large
  magnitudes (e.g. raw price data in the billions), normalize/scale the
  series before calling in — the shape-detection functions here care about
  relative structure, not absolute magnitude.
- **No domain concepts.** There is no `detect_silence` or
  `detect_price_anomaly` function here, on purpose — both are the same
  underlying primitive (`find_flat_regions`, `find_local_maxima`,
  `find_transients`) with different threshold tuning, and that tuning is
  domain knowledge that belongs in the consuming app.
- **Sample rate / interval is always an explicit parameter**, never
  assumed — pass your audio sample rate in Hz, or your bar/tick interval
  as a rate, and frequency-domain results come back in the same units.
- **`true_peak_db` reuses one FFT plan pair across blocks.** An earlier
  version replanned the FFT per block with an arbitrary block+margin
  size; on a real multi-minute file in an unoptimized build that measured
  60+ seconds. Processing in fixed power-of-two windows with one reused
  forward/inverse `rustfft` plan pair fixed that — see
  `true_peak_of_block`'s doc comment in `loudness.rs`.

## Further reading

If you're new to digital signal processing and want the theory behind
what this crate implements (FFT, windowing, filtering), roughly in order
of increasing depth:

1. [*The Scientist and Engineer's Guide to Digital Signal Processing*](https://www.dspguide.com/)
   by Steven W. Smith — free online, and the most approachable starting
   point: practical and light on formal math.
2. *Understanding Digital Signal Processing* by Richard G. Lyons — the
   standard "practical engineer's" DSP book; covers FFT, spectral
   leakage/windowing, and filter design in more depth than the Smith
   book, still without requiring a heavy signals-and-systems background.
3. *Discrete-Time Signal Processing* by Alan V. Oppenheim and Ronald W.
   Schafer — the rigorous academic reference most university DSP courses
   are built around. Worth it once the practical books' explanations
   raise questions the practical books don't answer.
4. L.R. Rabiner, "On the Use of Autocorrelation Analysis for Pitch
   Detection" (IEEE Transactions on Acoustics, Speech, and Signal
   Processing, 1977) — the specific method `pitch::estimate_f0`
   implements, if you want the original paper rather than a textbook
   summary of it.

## License

Dual-licensed under either of:

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option — this is the standard convention for Rust crates, and lets
consumers pick whichever license fits their own project's requirements.
Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this crate is licensed as above, without any
additional terms or conditions.
