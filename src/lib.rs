//! signal-core: domain-agnostic signal-processing primitives.
//!
//! Every function here operates on a generic real-valued time series
//! (`&[f32]`) plus explicit parameters like sample rate — nothing in this
//! crate assumes the series is audio or price data. Domain-specific naming
//! ("silence detection", "anomaly detection") belongs in the consuming app,
//! not here; see [`peaks`] for the underlying shared primitive.
//!
//! See the crate README for build/consumption instructions (native Rust via
//! Cargo git dependency, or WASM via `wasm-pack build`).

pub mod fft;
pub mod filter;
pub mod peaks;
pub mod resample;
pub mod window;

#[cfg(target_arch = "wasm32")]
pub mod wasm;
