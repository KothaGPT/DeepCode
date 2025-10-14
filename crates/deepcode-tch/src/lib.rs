#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![allow(clippy::single_range_in_vec_init)]

//! Deepcode Tch Backend

mod backend;
mod element;
mod ops;
mod tensor;

pub use backend::*;
pub use element::*;
pub use tensor::*;

#[cfg(test)]
mod tests {
    extern crate alloc;

    type TestBackend = crate::LibTorch<f32>;
    type TestTensor<const D: usize> = deepcode_tensor::Tensor<TestBackend, D>;
    type TestTensorInt<const D: usize> = deepcode_tensor::Tensor<TestBackend, D, deepcode_tensor::Int>;
    type TestTensorBool<const D: usize> = deepcode_tensor::Tensor<TestBackend, D, deepcode_tensor::Bool>;

    deepcode_tensor::testgen_all!();
    deepcode_autodiff::testgen_all!();
}
