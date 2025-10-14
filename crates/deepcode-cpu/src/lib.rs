#![cfg(target_os = "linux")]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

extern crate alloc;

use deepcode_deepcl::CubeBackend;
pub use deepcl::cpu::CpuDevice;
use deepcl::cpu::CpuRuntime;

#[cfg(not(feature = "fusion"))]
pub type Cpu<F = f32, I = i32> = CubeBackend<CpuRuntime, F, I, u8>;

#[cfg(feature = "fusion")]
pub type Cpu<F = f32, I = i32> = deepcode_fusion::Fusion<CubeBackend<CpuRuntime, F, I, u8>>;

#[cfg(test)]
mod tests {
    use deepcode_deepcl::CubeBackend;

    pub type TestRuntime = deepcl::cpu::CpuRuntime;

    deepcode_deepcl::testgen_all!([f32], [i8, i16, i32, i64], [u32]);
}
