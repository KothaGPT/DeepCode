#[macro_use]
extern crate derive_new;

#[cfg(test)]
#[allow(unexpected_cfgs)]
mod tests {
    pub type TestRuntime = crate::CpuRuntime;

    pub use half::f16;

    deepcl_core::testgen_all!(f32: [f16, f32, f64], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);
    deepcl_std::testgen!();
    deepcl_std::testgen_tensor_identity!([f16, f32, u32]);
    deepcl_random::testgen_random!();
    deepcl_matmul::testgen_matmul_simple!([f16, f32]);
    deepcl_matmul::testgen_matmul_unit!();
    deepcl_convolution::testgen_conv2d_accelerated!([f16: f16, f32: f32]);
    deepcl_reduce::testgen_shared_sum!([f16, f32, f64]);

    deepcl_reduce::testgen_reduce!([f16, f32, f64]);
}

pub mod compiler;
pub mod compute;
pub mod device;
pub mod runtime;

pub use device::CpuDevice;
pub use runtime::*;
