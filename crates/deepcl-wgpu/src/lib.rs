#[macro_use]
extern crate derive_new;

extern crate alloc;

mod backend;
mod compiler;
mod compute;
mod device;
mod element;
mod graphics;
mod runtime;

pub use compiler::base::*;
pub use compiler::wgsl::WgslCompiler;
pub use compute::*;
pub use device::*;
pub use element::*;
pub use graphics::*;
pub use runtime::*;

#[cfg(feature = "spirv")]
pub use backend::vulkan;

#[cfg(all(feature = "msl", target_os = "macos"))]
pub use backend::metal;

#[cfg(all(test, not(feature = "spirv"), not(feature = "msl")))]
#[allow(unexpected_cfgs)]
mod tests {
    pub type TestRuntime = crate::WgpuRuntime;

    deepcl_core::testgen_all!();
    deepcl_std::testgen!();
    deepcl_std::testgen_tensor_identity!([flex32, f32, u32]);
    deepcl_matmul::testgen_matmul_simple!([flex32, f32]);
    deepcl_matmul::testgen_matmul_plane_vecmat!();
    deepcl_matmul::testgen_matmul_unit!();
    deepcl_reduce::testgen_reduce!();
    deepcl_random::testgen_random!();
    deepcl_attention::testgen_attention!();
    deepcl_reduce::testgen_shared_sum!([f32]);
    deepcl_quant::testgen_quant!();
}

#[cfg(all(test, feature = "spirv"))]
#[allow(unexpected_cfgs)]
mod tests_spirv {
    pub type TestRuntime = crate::WgpuRuntime;
    use deepcl_core::flex32;
    use half::f16;

    deepcl_core::testgen_all!(f32: [f16, flex32, f32], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);
    deepcl_std::testgen!();
    deepcl_std::testgen_tensor_identity!([f16, flex32, f32, u32]);
    deepcl_convolution::testgen_conv2d_accelerated!([f16: f16]);
    deepcl_matmul::testgen_matmul_simple!([f32]);
    deepcl_matmul::testgen_matmul_plane_accelerated!();
    deepcl_matmul::testgen_matmul_plane_vecmat!();
    deepcl_matmul::testgen_matmul_unit!();
    deepcl_reduce::testgen_reduce!();
    deepcl_random::testgen_random!();
    deepcl_reduce::testgen_shared_sum!([f32]);
    deepcl_quant::testgen_quant!();
}

#[cfg(all(test, feature = "msl"))]
#[allow(unexpected_cfgs)]
mod tests_msl {
    pub type TestRuntime = crate::WgpuRuntime;
    use half::f16;

    deepcl_core::testgen_all!(f32: [f16, f32], i32: [i16, i32], u32: [u16, u32]);
    deepcl_std::testgen!();
    deepcl_std::testgen_tensor_identity!([f16, flex32, f32, u32]);
    deepcl_convolution::testgen_conv2d_accelerated!([f16: f16]);
    deepcl_matmul::testgen_matmul_simple!([f16, f32]);
    deepcl_matmul::testgen_matmul_plane_accelerated!();
    deepcl_matmul::testgen_matmul_plane_vecmat!();
    deepcl_matmul::testgen_matmul_unit!();
    deepcl_attention::testgen_attention!();
    deepcl_reduce::testgen_reduce!();
    deepcl_random::testgen_random!();
    deepcl_reduce::testgen_shared_sum!([f32]);
}
