#![cfg_attr(docsrs, feature(doc_auto_cfg))]

extern crate alloc;

use deepcode_deepcl::CubeBackend;
pub use deepcl::cuda::CudaDevice;
use deepcl::cuda::CudaRuntime;

#[cfg(not(feature = "fusion"))]
pub type Cuda<F = f32, I = i32> = CubeBackend<CudaRuntime, F, I, u8>;

#[cfg(feature = "fusion")]
pub type Cuda<F = f32, I = i32> = deepcode_fusion::Fusion<CubeBackend<CudaRuntime, F, I, u8>>;

#[cfg(test)]
mod tests {
    use deepcode_deepcl::CubeBackend;
    //use half::{bf16, f16};

    pub type TestRuntime = deepcl::cuda::CudaRuntime;

    // TODO: Add tests for bf16
    //deepcode_deepcl::testgen_all!([bf16, f16, f32], [i8, i16, i32, i64], [u8, u32]);
    deepcode_deepcl::testgen_all!([f32], [i32], [u32]);
}
