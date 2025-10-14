#![allow(missing_docs)]

mod abs;
mod adaptive_avgpool1d;
mod adaptive_avgpool2d;
mod add;
mod aggregation;
mod avgpool1d;
mod avgpool2d;
mod backward;
mod bridge;
mod broadcast;
mod cat;
mod ceil;
mod checkpoint;
mod complex;
mod conv1d;
mod conv2d;
mod conv3d;
mod conv_transpose1d;
mod conv_transpose2d;
mod conv_transpose3d;
mod cos;
mod cross;
mod cross_entropy;
mod cumsum;
mod deform_conv2d;
mod div;
mod erf;
mod exp;
mod expand;
mod flip;
mod floor;
mod gather_scatter;
mod gelu;
mod gradients;
mod log;
mod log1p;
mod log_sigmoid;
mod mask;
mod matmul;
mod maxmin;
mod maxpool1d;
mod maxpool2d;
mod memory_management;
mod mul;
mod multithread;
mod nearest_interpolate;
mod neg;
mod nonzero;
mod permute;
mod pow;
mod recip;
mod relu;
mod remainder;
mod repeat_dim;
mod reshape;
mod round;
mod select;
mod sigmoid;
mod sign;
mod sin;
mod slice;
mod slice_assign;
mod softmax;
mod sort;
mod sqrt;
mod sub;
mod tanh;
mod transpose;

#[macro_export]
macro_rules! testgen_all {
    // Avoid using paste dependency with no parameters
    () => {
        mod autodiff {
            pub use super::*;
            type TestAutodiffBackend = deepcode_autodiff::Autodiff<TestBackend>;
            type TestAutodiffTensor<const D: usize> = deepcode_tensor::Tensor<TestAutodiffBackend, D>;

            pub type FloatType = <TestBackend as deepcode_tensor::backend::Backend>::FloatElem;
            pub type IntType = <TestBackend as deepcode_tensor::backend::Backend>::IntElem;
            pub type BoolType = <TestBackend as deepcode_tensor::backend::Backend>::BoolTensorPrimitive;

            $crate::testgen_with_float_param!();
        }
        mod autodiff_checkpointing {
            pub use super::*;
            type TestAutodiffBackend = deepcode_autodiff::Autodiff<TestBackend, deepcode_autodiff::checkpoint::strategy::BalancedCheckpointing>;
            type TestAutodiffTensor<const D: usize> = deepcode_tensor::Tensor<TestAutodiffBackend, D>;

            pub type FloatType = <TestBackend as deepcode_tensor::backend::Backend>::FloatElem;
            pub type IntType = <TestBackend as deepcode_tensor::backend::Backend>::IntElem;
            pub type BoolType = <TestBackend as deepcode_tensor::backend::Backend>::BoolTensorPrimitive;

            $crate::testgen_with_float_param!();
        }
    };
    ([$($float:ident),*]) => {
        mod autodiff_checkpointing {
            pub use super::*;
            type TestAutodiffBackend = deepcode_autodiff::Autodiff<TestBackend, deepcode_autodiff::checkpoint::strategy::BalancedCheckpointing>;
            type TestAutodiffTensor<const D: usize> = deepcode_tensor::Tensor<TestAutodiffBackend, D>;

            pub type FloatType = <TestBackend as deepcode_tensor::backend::Backend>::FloatElem;
            pub type IntType = <TestBackend as deepcode_tensor::backend::Backend>::IntElem;
            pub type BoolType = <TestBackend as deepcode_tensor::backend::Backend>::BoolElem;

            ::paste::paste! {
                $(mod [<$float _ty>] {
                    pub use super::*;

                    pub type TestBackend = TestBackend2<$float, IntType, BoolType>;
                    pub type TestAutodiffBackend = deepcode_autodiff::Autodiff<TestBackend, deepcode_autodiff::checkpoint::strategy::BalancedCheckpointing>;
                    pub type TestAutodiffTensor<const D: usize> = deepcode_tensor::Tensor<TestAutodiffBackend, D>;
                    pub type TestTensor<const D: usize> = TestTensor2<$float, IntType, BoolType, D>;
                    pub type TestTensorInt<const D: usize> = TestTensorInt2<$float, IntType, BoolType, D>;
                    pub type TestTensorBool<const D: usize> = TestTensorBool2<$float, IntType, BoolType, D>;

                    type FloatType = $float;

                    $crate::testgen_with_float_param!();
                })*
            }
        }

        mod autodiff {
            pub use super::*;
            type TestAutodiffBackend = deepcode_autodiff::Autodiff<TestBackend>;
            type TestAutodiffTensor<const D: usize> = deepcode_tensor::Tensor<TestAutodiffBackend, D>;

            pub type FloatType = <TestBackend as deepcode_tensor::backend::Backend>::FloatElem;
            pub type IntType = <TestBackend as deepcode_tensor::backend::Backend>::IntElem;
            pub type BoolType = <TestBackend as deepcode_tensor::backend::Backend>::BoolElem;

            ::paste::paste! {
                $(mod [<$float _ty>] {
                    pub use super::*;

                    pub type TestBackend = TestBackend2<$float, IntType, BoolType>;
                    pub type TestAutodiffBackend = deepcode_autodiff::Autodiff<TestBackend>;
                    pub type TestAutodiffTensor<const D: usize> = deepcode_tensor::Tensor<TestAutodiffBackend, D>;
                    pub type TestTensor<const D: usize> = TestTensor2<$float, IntType, BoolType, D>;
                    pub type TestTensorInt<const D: usize> = TestTensorInt2<$float, IntType, BoolType, D>;
                    pub type TestTensorBool<const D: usize> = TestTensorBool2<$float, IntType, BoolType, D>;

                    type FloatType = $float;

                    $crate::testgen_with_float_param!();
                })*
            }
        }
    };
}

#[macro_export]
macro_rules! testgen_with_float_param {
    () => {
        // Behaviour
        deepcode_autodiff::testgen_ad_broadcast!();
        deepcode_autodiff::testgen_gradients!();
        deepcode_autodiff::testgen_bridge!();
        deepcode_autodiff::testgen_checkpoint!();
        deepcode_autodiff::testgen_memory_management!();

        // Activation
        deepcode_autodiff::testgen_ad_relu!();
        deepcode_autodiff::testgen_ad_gelu!();

        // Modules
        deepcode_autodiff::testgen_ad_conv1d!();
        deepcode_autodiff::testgen_ad_conv2d!();
        deepcode_autodiff::testgen_ad_conv3d!();
        // #[cfg(not(target_os = "macos"))] // Wgpu on MacOS currently doesn't support atomic compare exchange
        // deepcode_autodiff::testgen_ad_deform_conv2d!(); // This kernel in deepcl isn't implemented without atomics
        deepcode_autodiff::testgen_ad_conv_transpose1d!();
        deepcode_autodiff::testgen_ad_conv_transpose2d!();
        deepcode_autodiff::testgen_ad_conv_transpose3d!();
        deepcode_autodiff::testgen_ad_max_pool1d!();
        deepcode_autodiff::testgen_ad_max_pool2d!();
        deepcode_autodiff::testgen_ad_avg_pool1d!();
        deepcode_autodiff::testgen_ad_avg_pool2d!();
        deepcode_autodiff::testgen_ad_adaptive_avg_pool1d!();
        deepcode_autodiff::testgen_ad_adaptive_avg_pool2d!();
        deepcode_autodiff::testgen_module_backward!();
        deepcode_autodiff::testgen_ad_nearest_interpolate!();

        // Tensor
        deepcode_autodiff::testgen_ad_complex!();
        deepcode_autodiff::testgen_ad_multithread!();
        deepcode_autodiff::testgen_ad_add!();
        deepcode_autodiff::testgen_ad_aggregation!();
        deepcode_autodiff::testgen_ad_maxmin!();
        deepcode_autodiff::testgen_ad_cat!();
        deepcode_autodiff::testgen_ad_cos!();
        deepcode_autodiff::testgen_ad_cross!();
        deepcode_autodiff::testgen_ad_cross_entropy_loss!();
        deepcode_autodiff::testgen_ad_cumsum!();
        deepcode_autodiff::testgen_ad_div!();
        deepcode_autodiff::testgen_ad_remainder!();
        deepcode_autodiff::testgen_ad_erf!();
        deepcode_autodiff::testgen_ad_exp!();
        deepcode_autodiff::testgen_ad_slice!();
        deepcode_autodiff::testgen_ad_slice_assign!();
        deepcode_autodiff::testgen_ad_gather_scatter!();
        deepcode_autodiff::testgen_ad_select!();
        deepcode_autodiff::testgen_ad_log!();
        deepcode_autodiff::testgen_ad_log1p!();
        deepcode_autodiff::testgen_ad_mask!();
        deepcode_autodiff::testgen_ad_matmul!();
        deepcode_autodiff::testgen_ad_mul!();
        deepcode_autodiff::testgen_ad_neg!();
        deepcode_autodiff::testgen_ad_powf!();
        deepcode_autodiff::testgen_ad_recip!();
        deepcode_autodiff::testgen_ad_reshape!();
        deepcode_autodiff::testgen_ad_sin!();
        deepcode_autodiff::testgen_ad_softmax!();
        deepcode_autodiff::testgen_ad_sqrt!();
        deepcode_autodiff::testgen_ad_abs!();
        deepcode_autodiff::testgen_ad_sub!();
        deepcode_autodiff::testgen_ad_tanh!();
        deepcode_autodiff::testgen_ad_round!();
        deepcode_autodiff::testgen_ad_floor!();
        deepcode_autodiff::testgen_ad_ceil!();
        deepcode_autodiff::testgen_ad_sigmoid!();
        deepcode_autodiff::testgen_ad_log_sigmoid!();
        deepcode_autodiff::testgen_ad_transpose!();
        deepcode_autodiff::testgen_ad_permute!();
        deepcode_autodiff::testgen_ad_flip!();
        deepcode_autodiff::testgen_ad_nonzero!();
        deepcode_autodiff::testgen_ad_sign!();
        deepcode_autodiff::testgen_ad_expand!();
        deepcode_autodiff::testgen_ad_sort!();
        deepcode_autodiff::testgen_ad_repeat_dim!();
    };
}
