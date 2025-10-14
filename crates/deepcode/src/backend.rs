#[cfg(feature = "ndarray")]
pub use deepcode_ndarray as ndarray;

#[cfg(feature = "ndarray")]
pub use ndarray::NdArray;

#[cfg(feature = "autodiff")]
pub use deepcode_autodiff as autodiff;

#[cfg(feature = "remote")]
pub use deepcode_remote as remote;
#[cfg(feature = "remote")]
pub use deepcode_remote::RemoteBackend;

#[cfg(feature = "autodiff")]
pub use deepcode_autodiff::Autodiff;

#[cfg(feature = "wgpu")]
pub use deepcode_wgpu as wgpu;

#[cfg(feature = "wgpu")]
pub use deepcode_wgpu::Wgpu;

#[cfg(feature = "webgpu")]
pub use deepcode_wgpu::WebGpu;

#[cfg(feature = "vulkan")]
pub use deepcode_wgpu::Vulkan;

#[cfg(feature = "metal")]
pub use deepcode_wgpu::Metal;

#[cfg(feature = "cuda")]
pub use deepcode_cuda as cuda;

#[cfg(feature = "cuda")]
pub use deepcode_cuda::Cuda;

#[cfg(feature = "candle")]
pub use deepcode_candle as candle;

#[cfg(feature = "candle")]
pub use deepcode_candle::Candle;

#[cfg(feature = "rocm")]
pub use deepcode_rocm as rocm;

#[cfg(feature = "rocm")]
pub use deepcode_rocm::Rocm;

#[cfg(feature = "tch")]
pub use deepcode_tch as libtorch;

#[cfg(feature = "tch")]
pub use deepcode_tch::LibTorch;

#[cfg(feature = "router")]
pub use deepcode_router::Router;

#[cfg(feature = "router")]
pub use deepcode_router as router;

#[cfg(feature = "ir")]
pub use deepcode_ir as ir;

#[cfg(feature = "collective")]
pub use deepcode_collective as collective;
#[cfg(feature = "cpu")]
pub use deepcode_cpu as cpu;

#[cfg(feature = "cpu")]
pub use deepcode_cpu::Cpu;
