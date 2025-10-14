#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! Deepcode ndarray backend.

#[cfg(any(
    feature = "blas-netlib",
    feature = "blas-openblas",
    feature = "blas-openblas-system",
))]
extern crate blas_src;

mod backend;
mod element;
mod ops;
mod parallel;
mod rand;
mod sharing;
mod tensor;

pub use backend::*;
pub use element::*;
pub(crate) use sharing::*;
pub use tensor::*;

extern crate alloc;

#[cfg(test)]
mod tests {
    type TestBackend = crate::NdArray<f32>;
    type TestTensor<const D: usize> = deepcode_tensor::Tensor<TestBackend, D>;
    type TestTensorInt<const D: usize> = deepcode_tensor::Tensor<TestBackend, D, deepcode_tensor::Int>;
    type TestTensorBool<const D: usize> = deepcode_tensor::Tensor<TestBackend, D, deepcode_tensor::Bool>;

    use alloc::format;
    use alloc::vec;
    use alloc::vec::Vec;

    deepcode_tensor::testgen_all!();

    #[cfg(feature = "std")]
    deepcode_autodiff::testgen_all!();

    // Quantization
    deepcode_tensor::testgen_calibration!();
    deepcode_tensor::testgen_scheme!();
    deepcode_tensor::testgen_quantize!();
    deepcode_tensor::testgen_q_data!();
}
