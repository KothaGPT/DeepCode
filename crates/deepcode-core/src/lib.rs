#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![recursion_limit = "135"]

//! The core crate of Deepcode.

#[macro_use]
extern crate derive_new;

/// Re-export serde for proc macros.
pub use serde;

/// The configuration module.
pub mod config;

/// Data module.
#[cfg(feature = "std")]
pub mod data;

/// Module for the neural network module.
pub mod module;

/// Module for the recorder.
pub mod record;

/// Module for the tensor.
pub mod tensor;
// Tensor at root: `deepcode::Tensor`
pub use tensor::Tensor;

/// Module for visual operations
#[cfg(feature = "vision")]
pub mod vision;

extern crate alloc;

/// Backend for test cases
#[cfg(all(
    test,
    not(feature = "test-tch"),
    not(feature = "test-wgpu"),
    not(feature = "test-cuda"),
    not(feature = "test-rocm")
))]
pub type TestBackend = deepcode_ndarray::NdArray<f32>;

#[cfg(all(test, feature = "test-tch"))]
/// Backend for test cases
pub type TestBackend = deepcode_tch::LibTorch<f32>;

#[cfg(all(test, feature = "test-wgpu"))]
/// Backend for test cases
pub type TestBackend = deepcode_wgpu::Wgpu;

#[cfg(all(test, feature = "test-cuda"))]
/// Backend for test cases
pub type TestBackend = deepcode_cuda::Cuda;

#[cfg(all(test, feature = "test-rocm"))]
/// Backend for test cases
pub type TestBackend = deepcode_rocm::Rocm;

/// Backend for autodiff test cases
#[cfg(test)]
pub type TestAutodiffBackend = deepcode_autodiff::Autodiff<TestBackend>;

#[cfg(all(test, feature = "test-memory-checks"))]
mod tests {
    deepcode_fusion::memory_checks!();
}

#[cfg(test)]
mod test_utils {
    use crate as deepcode;
    use crate::module::Module;
    use crate::module::Param;
    use deepcode_tensor::Tensor;
    use deepcode_tensor::backend::Backend;

    /// Simple linear module.
    #[derive(Module, Debug)]
    pub struct SimpleLinear<B: Backend> {
        pub weight: Param<Tensor<B, 2>>,
        pub bias: Option<Param<Tensor<B, 1>>>,
    }

    impl<B: Backend> SimpleLinear<B> {
        pub fn new(in_features: usize, out_features: usize, device: &B::Device) -> Self {
            let weight = Tensor::random(
                [out_features, in_features],
                deepcode_tensor::Distribution::Default,
                device,
            );
            let bias = Tensor::random([out_features], deepcode_tensor::Distribution::Default, device);

            Self {
                weight: Param::from_tensor(weight),
                bias: Some(Param::from_tensor(bias)),
            }
        }
    }
}

pub mod prelude {
    //! Structs and macros used by most projects. Add `use
    //! deepcode::prelude::*` to your code to quickly get started with
    //! Deepcode.
    pub use crate::{
        config::Config,
        module::Module,
        tensor::{
            Bool, Device, ElementConversion, Float, Int, Shape, SliceArg, Tensor, TensorData,
            backend::Backend, cast::ToElement, s,
        },
    };
    pub use deepcode_common::device::Device as DeviceOps;
}
