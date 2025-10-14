#![cfg_attr(docsrs, feature(doc_auto_cfg))]
extern crate alloc;

use deepcode_deepcl::CubeBackend;

pub use deepcl::hip::AmdDevice as RocmDevice;

use deepcl::hip::HipRuntime;

#[cfg(not(feature = "fusion"))]
pub type Rocm<F = f32, I = i32, B = u8> = CubeBackend<HipRuntime, F, I, B>;

#[cfg(feature = "fusion")]
pub type Rocm<F = f32, I = i32, B = u8> = deepcode_fusion::Fusion<CubeBackend<HipRuntime, F, I, B>>;

#[cfg(test)]
mod tests {
    use deepcode_deepcl::CubeBackend;

    pub type TestRuntime = deepcl::hip::HipRuntime;
    use half::f16;

    // TODO: Add tests for bf16
    // deepcode_deepcl::testgen_all!([f16, f32], [i8, i16, i32, i64], [u8, u32]);
    deepcode_deepcl::testgen_all!([f16, f32], [i32], [u32]);
}
