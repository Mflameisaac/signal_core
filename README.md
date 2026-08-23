# signal-core

Domain-agnostic signal-processing primitives — FFT/spectral analysis,
windowing, filtering, peak/transient detection, and downsampling — for any
real-valued time series. Compiled once, in Rust, and consumed either as a
native Cargo dependency or as a WASM+JS package.

This library exists to be shared between two otherwise-unrelated
applications: an audio/podcast editor and a stock-market analysis tool.
Everything here operates on a plain `&[f32]` series plus explicit parameters
like sample rate — nothing in this crate knows or assumes it's looking at
audio or price data. Domain-specific concepts ("silence detection",
"anomaly detection") are just this crate's generic primitives (peak
detection, flat-region detection) called with app-specific threshold
tuning, and that tuning happens in the consuming app, not here.

This repo is **not** a workspace member of either downstream app. It's an
independent, independently-versioned crate that both apps pull in as a
dependency.

## What's in it

| Module | Purpose |
|---|---|
| `window` | `Hann`, `Hamming`, `Blackman`, `Rectangular` window coefficients + application |
| `fft` | Forward FFT (via `rustfft`), one-sided power spectrum, dominant-frequency detection |
| `filter` | Centered moving average, single-pole low-pass / high-pass filters |
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
signal-core = { git = "ssh://git@github.com/YOUR_ORG/signal-core.git", tag = "v0.1.0" }
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

## License

MIT — see `LICENSE`.
