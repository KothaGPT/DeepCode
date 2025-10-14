#[cfg(feature = "test-wgpu")]
pub type TestBackend = deepcode::backend::Wgpu;

#[cfg(all(
    feature = "test-ndarray",
    not(feature = "test-wgpu"),
    not(feature = "test-tch"),
    not(feature = "test-metal")
))]
pub type TestBackend = deepcode::backend::NdArray<f32>;

#[cfg(feature = "test-metal")]
pub type TestBackend = deepcode::backend::Metal;

#[cfg(feature = "test-tch")]
pub type TestBackend = deepcode::backend::LibTorch<f32>;

#[cfg(feature = "test-candle")]
pub type TestBackend = deepcode::backend::Candle<f32>;
