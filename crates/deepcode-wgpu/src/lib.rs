#![cfg_attr(docsrs, feature(doc_auto_cfg))]

extern crate alloc;

#[cfg(feature = "template")]
pub use deepcode_deepcl::{
    kernel::{KernelMetadata, into_contiguous},
    kernel_source,
    template::{KernelSource, SourceKernel, SourceTemplate, build_info},
};

pub use deepcode_deepcl::{BoolElement, FloatElement, IntElement};
pub use deepcode_deepcl::{CubeBackend, tensor::CubeTensor};
pub use deepcl::CubeDim;
pub use deepcl::flex32;

pub use deepcl::wgpu::{
    AutoCompiler, MemoryConfiguration, RuntimeOptions, WgpuDevice, WgpuResource, WgpuRuntime,
    WgpuSetup, WgpuStorage, init_device, init_setup, init_setup_async,
};
// Vulkan and WebGpu would have conflicting type names
pub mod graphics {
    pub use deepcl::wgpu::{AutoGraphicsApi, Dx12, GraphicsApi, Metal, OpenGl, Vulkan, WebGpu};
}

#[cfg(feature = "deepcl-wgsl")]
pub use deepcl::wgpu::WgslCompiler;
#[cfg(feature = "deepcl-spirv")]
pub use deepcl::wgpu::vulkan::VkSpirvCompiler;

#[cfg(feature = "fusion")]
/// Tensor backend that uses the wgpu crate for executing GPU compute shaders.
///
/// This backend can target multiple graphics APIs, including:
///   - [Vulkan][crate::graphics::Vulkan] on Linux, Windows, and Android.
///   - [OpenGL](crate::graphics::OpenGl) on Linux, Windows, and Android.
///   - [DirectX 12](crate::graphics::Dx12) on Windows.
///   - [Metal][crate::graphics::Metal] on Apple hardware.
///   - [WebGPU](crate::graphics::WebGpu) on supported browsers and `wasm` runtimes.
///
/// To configure the wgpu backend, eg. to select what graphics API to use or what memory strategy to use,
/// you have to manually initialize the runtime. For example:
///
/// ```rust, ignore
/// fn custom_init() {
///     let device = Default::default();
///     deepcode::backend::wgpu::init_setup::<deepcode::backend::wgpu::graphics::Vulkan>(
///         &device,
///         Default::default(),
///     );
/// }
/// ```
/// will mean the given device (in this case the default) will be initialized to use Vulkan as the graphics API.
/// It's also possible to use an existing wgpu device, by using `init_device`.
///
/// # Notes
///
/// This version of the wgpu backend uses [deepcode_fusion] to compile and optimize streams of tensor
/// operations for improved performance.
///
/// You can disable the `fusion` feature flag to remove that functionality, which might be
/// necessary on `wasm` for now.
pub type Wgpu<F = f32, I = i32, B = u32> =
    deepcode_fusion::Fusion<CubeBackend<deepcl::wgpu::WgpuRuntime, F, I, B>>;

#[cfg(not(feature = "fusion"))]
/// Tensor backend that uses the wgpu crate for executing GPU compute shaders.
///
/// This backend can target multiple graphics APIs, including:
///   - [Vulkan] on Linux, Windows, and Android.
///   - [OpenGL](crate::OpenGl) on Linux, Windows, and Android.
///   - [DirectX 12](crate::Dx12) on Windows.
///   - [Metal] on Apple hardware.
///   - [WebGPU](crate::WebGpu) on supported browsers and `wasm` runtimes.
///
/// To configure the wgpu backend, eg. to select what graphics API to use or what memory strategy to use,
/// you have to manually initialize the runtime. For example:
///
/// ```rust, ignore
/// fn custom_init() {
///     let device = Default::default();
///     deepcode::backend::wgpu::init_setup::<deepcode::backend::wgpu::graphics::Vulkan>(
///         &device,
///         Default::default(),
///     );
/// }
/// ```
/// will mean the given device (in this case the default) will be initialized to use Vulkan as the graphics API.
/// It's also possible to use an existing wgpu device, by using `init_device`.
///
/// # Notes
///
/// This version of the wgpu backend doesn't use [deepcode_fusion] to compile and optimize streams of tensor
/// operations.
///
/// You can enable the `fusion` feature flag to add that functionality, which might improve
/// performance.
pub type Wgpu<F = f32, I = i32, B = u32> = CubeBackend<deepcl::wgpu::WgpuRuntime, F, I, B>;

#[cfg(feature = "vulkan")]
/// Tensor backend that leverages the Vulkan graphics API to execute GPU compute shaders compiled to SPIR-V.
pub type Vulkan<F = f32, I = i32, B = u8> = Wgpu<F, I, B>;

#[cfg(feature = "webgpu")]
/// Tensor backend that uses the wgpu crate to execute GPU compute shaders written in WGSL.
pub type WebGpu<F = f32, I = i32, B = u32> = Wgpu<F, I, B>;

#[cfg(feature = "metal")]
/// Tensor backend that leverages the Metal graphics API to execute GPU compute shaders compiled to MSL.
pub type Metal<F = f32, I = i32, B = u8> = Wgpu<F, I, B>;

#[cfg(test)]
mod tests {
    use deepcode_deepcl::CubeBackend;
    #[cfg(feature = "vulkan")]
    pub use half::f16;
    #[cfg(feature = "metal")]
    pub use half::f16;

    pub type TestRuntime = deepcl::wgpu::WgpuRuntime;

    // Don't test `flex32` for now, deepcode sees it as `f32` but is actually `f16` precision, so it
    // breaks a lot of tests from precision issues
    #[cfg(feature = "vulkan")]
    deepcode_deepcl::testgen_all!([f16, f32], [i8, i16, i32, i64], [u8, u32]);
    #[cfg(feature = "metal")]
    deepcode_deepcl::testgen_all!([f16, f32], [i16, i32], [u32]);
    #[cfg(all(not(feature = "vulkan"), not(feature = "metal")))]
    deepcode_deepcl::testgen_all!([f32], [i32], [u32]);
}
