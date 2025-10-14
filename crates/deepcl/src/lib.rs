pub use deepcl_core::*;

pub use deepcl_runtime::config;
pub use deepcl_runtime::features;
pub use deepcl_runtime::memory_management::MemoryAllocationMode;

#[cfg(feature = "wgpu")]
pub use deepcl_wgpu as wgpu;

#[cfg(feature = "cuda")]
pub use deepcl_cuda as cuda;

#[cfg(feature = "hip")]
pub use deepcl_hip as hip;

#[cfg(feature = "matmul")]
pub use deepcl_matmul as matmul;

#[cfg(feature = "convolution")]
pub use deepcl_convolution as convolution;

#[cfg(feature = "stdlib")]
pub use deepcl_std as std;

#[cfg(feature = "reduce")]
pub use deepcl_reduce as reduce;

#[cfg(feature = "random")]
pub use deepcl_random as random;

#[cfg(feature = "cpu")]
pub use deepcl_cpu as cpu;
