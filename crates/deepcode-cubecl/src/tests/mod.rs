#![allow(missing_docs)]

mod avg_pool2d;
mod bernoulli;
mod cast;
mod cat;
mod clamp;
mod conv2d;
mod conv3d;
mod conv_transpose2d;
mod conv_transpose3d;
mod cross;
mod gather;
mod mask_fill;
mod mask_where;
mod matmul;
mod max_pool2d;
mod max_pool2d_backward;
mod normal;
mod quantization;
mod reduce;
mod repeat_dim;
mod scatter;
mod select;
mod select_assign;
mod slice;
mod slice_assign;
mod unary;
mod uniform;

// Re-export dependencies for tests
pub use crate::ops::base::into_data_sync;
pub use deepcode_autodiff;
pub use deepcode_fusion;
pub use deepcode_ndarray;
pub use deepcode_tensor;
pub use serial_test;

#[macro_export]
macro_rules! testgen_all {
    () => {
        use deepcode_tensor::{Float, Int, Bool};
        $crate::testgen_all!([Float], [Int], [Bool]);

    };
    ([$($float:ident),*], [$($int:ident),*], [$($bool:ident),*]) => {
        mod cube {
            deepcode_deepcl::testgen_jit!([$($float),*], [$($int),*], [$($bool),*]);

            mod kernel {
                use super::*;

                deepcode_deepcl::testgen_conv2d!();
                deepcode_deepcl::testgen_conv3d!();
                deepcode_deepcl::testgen_conv_transpose2d!();
                deepcode_deepcl::testgen_conv_transpose3d!();

                deepcode_deepcl::testgen_repeat_dim!();
                deepcode_deepcl::testgen_gather!();
                deepcode_deepcl::testgen_scatter!();

                deepcode_deepcl::testgen_select!();
                deepcode_deepcl::testgen_select_assign!();

                deepcode_deepcl::testgen_slice!();
                deepcode_deepcl::testgen_slice_assign!();

                deepcode_deepcl::testgen_mask_where!();
                deepcode_deepcl::testgen_mask_fill!();

                deepcode_deepcl::testgen_avg_pool2d!();
                deepcode_deepcl::testgen_max_pool2d!();
                deepcode_deepcl::testgen_max_pool2d_backward!();

                deepcode_deepcl::testgen_bernoulli!();
                deepcode_deepcl::testgen_normal!();
                deepcode_deepcl::testgen_uniform!();

                deepcode_deepcl::testgen_cast!();
                deepcode_deepcl::testgen_cat!();
                deepcode_deepcl::testgen_clamp!();
                deepcode_deepcl::testgen_unary!();

                deepcode_deepcl::testgen_reduce!();

                deepcode_deepcl::testgen_cross!();

                deepcode_deepcl::testgen_quantization!();
            }
        }
        mod cube_fusion {
            deepcode_deepcl::testgen_jit_fusion!([$($float),*], [$($int),*], [$($bool),*]);
            deepcode_fusion::memory_checks!();
        }
    };
}

#[macro_export]
macro_rules! testgen_jit {
    () => {
        use deepcode_tensor::{Float, Int, Bool};
        $crate::testgen_jit!([Float], [Int], [Bool]);
    };
    ([$($float:ident),*], [$($int:ident),*], [$($bool:ident),*]) => {
        pub use super::*;
        use deepcode_deepcl::tests::{deepcode_autodiff, deepcode_ndarray, deepcode_tensor, serial_test};

        pub type TestBackend = CubeBackend<TestRuntime, f32, i32, u32>;
        pub type TestBackend2<F, I, B> = CubeBackend<TestRuntime, F, I, B>;
        pub type ReferenceBackend = deepcode_ndarray::NdArray<f32>;

        pub type TestTensor<const D: usize> = deepcode_tensor::Tensor<TestBackend, D>;
        pub type TestTensor2<F, I, B, const D: usize> = deepcode_tensor::Tensor<TestBackend2<F, I, B>, D>;
        pub type TestTensorInt<const D: usize> =
            deepcode_tensor::Tensor<TestBackend, D, deepcode_tensor::Int>;
        pub type TestTensorInt2<F, I, B, const D: usize> =
            deepcode_tensor::Tensor<TestBackend2<F, I, B>, D, deepcode_tensor::Int>;
        pub type TestTensorBool<const D: usize> =
            deepcode_tensor::Tensor<TestBackend, D, deepcode_tensor::Bool>;
        pub type TestTensorBool2<F, I, B, const D: usize> =
            deepcode_tensor::Tensor<TestBackend2<F, I, B>, D, deepcode_tensor::Bool>;

        pub type ReferenceTensor<const D: usize> = deepcode_tensor::Tensor<ReferenceBackend, D>;

        deepcode_tensor::testgen_all!([$($float),*], [$($int),*], [$($bool),*]);
        deepcode_autodiff::testgen_all!([$($float),*]);

        use deepcode_tensor::tests::qtensor::*;

        deepcode_tensor::testgen_q_matmul!();
        deepcode_tensor::testgen_calibration!();
        deepcode_tensor::testgen_scheme!();
        deepcode_tensor::testgen_quantize!();
        deepcode_tensor::testgen_q_data!();
    }
}

#[macro_export]
macro_rules! testgen_jit_fusion {
    () => {
        use deepcode_tensor::{Float, Int};
        $crate::testgen_jit_fusion!([Float], [Int]);
    };
    ([$($float:ident),*], [$($int:ident),*], [$($bool:ident),*]) => {
        use super::*;
        use deepcode_deepcl::tests::{deepcode_autodiff, deepcode_fusion, deepcode_ndarray, deepcode_tensor};

        pub type TestBackend = deepcode_fusion::Fusion<CubeBackend<TestRuntime, f32, i32, u32>>;
        pub type TestBackend2<F, I, B> = deepcode_fusion::Fusion<CubeBackend<TestRuntime, F, I, B>>;
        pub type ReferenceBackend = deepcode_ndarray::NdArray<f32>;

        pub type TestTensor<const D: usize> = deepcode_tensor::Tensor<TestBackend, D>;
        pub type TestTensor2<F, I, B, const D: usize> = deepcode_tensor::Tensor<TestBackend2<F, I, B>, D>;
        pub type TestTensorInt<const D: usize> =
            deepcode_tensor::Tensor<TestBackend, D, deepcode_tensor::Int>;
        pub type TestTensorInt2<F, I, B, const D: usize> =
            deepcode_tensor::Tensor<TestBackend2<F, I, B>, D, deepcode_tensor::Int>;
        pub type TestTensorBool<const D: usize> =
            deepcode_tensor::Tensor<TestBackend, D, deepcode_tensor::Bool>;
        pub type TestTensorBool2<F, I, B, const D: usize> =
            deepcode_tensor::Tensor<TestBackend2<F, I, B>, D, deepcode_tensor::Bool>;

        pub type ReferenceTensor<const D: usize> = deepcode_tensor::Tensor<ReferenceBackend, D>;

        deepcode_tensor::testgen_all!([$($float),*], [$($int),*], [$($bool),*]);
        deepcode_autodiff::testgen_all!([$($float),*]);

        use deepcode_tensor::tests::qtensor::*;

        deepcode_tensor::testgen_scheme!();
        deepcode_tensor::testgen_quantize!();
    };
}
