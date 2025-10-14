mod activation;
mod clone_invariance;
mod grid;
mod linalg;
mod module;
#[cfg(feature = "std")]
mod multi_threads;
mod ops;
mod primitive;
mod quantization;
mod stats;

pub use deepcl::prelude::{Float, Int, Numeric};
pub use num_traits::Float as NumFloat;

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_all {
    () => {
        pub mod tensor {
            pub use super::*;

            pub type FloatType = <TestBackend as $crate::backend::Backend>::FloatElem;
            pub type IntType = <TestBackend as $crate::backend::Backend>::IntElem;
            pub type BoolType = <TestBackend as $crate::backend::Backend>::BoolElem;

            $crate::testgen_with_float_param!();
            $crate::testgen_no_param!();
        }
    };
    ([$($float:ident),*], [$($int:ident),*], [$($bool:ident),*]) => {
        pub mod tensor {
            pub use super::*;

            pub type FloatType = <TestBackend as $crate::backend::Backend>::FloatElem;
            pub type IntType = <TestBackend as $crate::backend::Backend>::IntElem;
            pub type BoolType = <TestBackend as $crate::backend::Backend>::BoolElem;

            ::paste::paste! {
                $(mod [<$float _ty>] {
                    pub use super::*;

                    pub type TestBackend = TestBackend2<$float, IntType, BoolType>;
                    pub type TestTensor<const D: usize> = TestTensor2<$float, IntType, BoolType, D>;
                    pub type TestTensorInt<const D: usize> = TestTensorInt2<$float, IntType, BoolType, D>;
                    pub type TestTensorBool<const D: usize> = TestTensorBool2<$float, IntType, BoolType, D>;

                    pub type FloatType = $float;

                    $crate::testgen_with_float_param!();
                })*
                $(mod [<$int _ty>] {
                    pub use super::*;

                    pub type TestBackend = TestBackend2<FloatType, $int, BoolType>;
                    pub type TestTensor<const D: usize> = TestTensor2<FloatType, $int, BoolType, D>;
                    pub type TestTensorInt<const D: usize> = TestTensorInt2<FloatType, $int, BoolType, D>;
                    pub type TestTensorBool<const D: usize> = TestTensorBool2<FloatType, $int, BoolType, D>;

                    pub type IntType = $int;

                    $crate::testgen_with_int_param!();
                })*
                $(mod [<$bool _bool_ty>] {
                    pub use super::*;

                    pub type TestBackend = TestBackend2<FloatType, IntType, $bool>;
                    pub type TestTensor<const D: usize> = TestTensor2<FloatType, IntType, $bool, D>;
                    pub type TestTensorInt<const D: usize> = TestTensorInt2<FloatType, IntType, $bool, D>;
                    pub type TestTensorBool<const D: usize> = TestTensorBool2<FloatType, IntType, $bool, D>;

                    pub type BoolType = $bool;

                    $crate::testgen_with_bool_param!();
                })*
            }
            $crate::testgen_no_param!();
        }
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_quantization {
    () => {
        pub use deepcode_tensor::tests::qtensor::*;

        // test quantization
        deepcode_tensor::testgen_calibration!();
        deepcode_tensor::testgen_scheme!();
        deepcode_tensor::testgen_quantize!();
        deepcode_tensor::testgen_q_data!();

        // test ops
        deepcode_tensor::testgen_q_abs!();
        deepcode_tensor::testgen_q_add!();
        deepcode_tensor::testgen_q_aggregation!();
        deepcode_tensor::testgen_q_all!();
        deepcode_tensor::testgen_q_any!();
        deepcode_tensor::testgen_q_arg!();
        deepcode_tensor::testgen_q_cat!();
        deepcode_tensor::testgen_q_chunk!();
        deepcode_tensor::testgen_q_clamp!();
        deepcode_tensor::testgen_q_cos!();
        deepcode_tensor::testgen_q_cosh!();
        deepcode_tensor::testgen_q_div!();
        deepcode_tensor::testgen_q_erf!();
        deepcode_tensor::testgen_q_exp!();
        deepcode_tensor::testgen_q_expand!();
        deepcode_tensor::testgen_q_flip!();
        deepcode_tensor::testgen_q_gather_scatter!();
        deepcode_tensor::testgen_q_log!();
        deepcode_tensor::testgen_q_log1p!();
        deepcode_tensor::testgen_q_map_comparison!();
        deepcode_tensor::testgen_q_mask!();
        deepcode_tensor::testgen_q_matmul!();
        deepcode_tensor::testgen_q_maxmin!();
        deepcode_tensor::testgen_q_mul!();
        deepcode_tensor::testgen_q_narrow!();
        deepcode_tensor::testgen_q_neg!();
        deepcode_tensor::testgen_q_permute!();
        deepcode_tensor::testgen_q_powf_scalar!();
        deepcode_tensor::testgen_q_powf!();
        deepcode_tensor::testgen_q_recip!();
        deepcode_tensor::testgen_q_remainder!();
        deepcode_tensor::testgen_q_repeat_dim!();
        deepcode_tensor::testgen_q_reshape!();
        deepcode_tensor::testgen_q_round!();
        deepcode_tensor::testgen_q_select!();
        deepcode_tensor::testgen_q_sin!();
        deepcode_tensor::testgen_q_sinh!();
        deepcode_tensor::testgen_q_slice!();
        deepcode_tensor::testgen_q_sort_argsort!();
        deepcode_tensor::testgen_q_split!();
        deepcode_tensor::testgen_q_sqrt!();
        deepcode_tensor::testgen_q_stack!();
        deepcode_tensor::testgen_q_sub!();
        deepcode_tensor::testgen_q_tan!();
        deepcode_tensor::testgen_q_tanh!();
        deepcode_tensor::testgen_q_topk!();
        deepcode_tensor::testgen_q_transpose!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_with_float_param {
    () => {
        // test activation
        deepcode_tensor::testgen_gelu!();
        deepcode_tensor::testgen_mish!();
        deepcode_tensor::testgen_relu!();
        deepcode_tensor::testgen_leaky_relu!();
        deepcode_tensor::testgen_softmax!();
        deepcode_tensor::testgen_softmin!();
        deepcode_tensor::testgen_softplus!();
        deepcode_tensor::testgen_sigmoid!();
        deepcode_tensor::testgen_log_sigmoid!();
        deepcode_tensor::testgen_silu!();
        deepcode_tensor::testgen_tanh_activation!();

        // test grid
        deepcode_tensor::testgen_meshgrid!();
        deepcode_tensor::testgen_affine_grid!();

        // test linalg
        deepcode_tensor::testgen_vector_norm!();
        deepcode_tensor::testgen_diag!();
        deepcode_tensor::testgen_cosine_similarity!();
        deepcode_tensor::testgen_trace!();
        deepcode_tensor::testgen_outer!();
        deepcode_tensor::testgen_lu_decomposition!();

        // test module
        deepcode_tensor::testgen_module_conv1d!();
        deepcode_tensor::testgen_module_conv2d!();
        deepcode_tensor::testgen_module_conv3d!();
        deepcode_tensor::testgen_module_forward!();
        deepcode_tensor::testgen_module_deform_conv2d!();
        deepcode_tensor::testgen_module_conv_transpose1d!();
        deepcode_tensor::testgen_module_conv_transpose2d!();
        deepcode_tensor::testgen_module_conv_transpose3d!();
        deepcode_tensor::testgen_module_unfold4d!();
        deepcode_tensor::testgen_module_max_pool1d!();
        deepcode_tensor::testgen_module_max_pool2d!();
        deepcode_tensor::testgen_module_avg_pool1d!();
        deepcode_tensor::testgen_module_avg_pool2d!();
        deepcode_tensor::testgen_module_adaptive_avg_pool1d!();
        deepcode_tensor::testgen_module_adaptive_avg_pool2d!();
        deepcode_tensor::testgen_module_nearest_interpolate!();
        deepcode_tensor::testgen_module_bilinear_interpolate!();
        deepcode_tensor::testgen_module_bicubic_interpolate!();
        deepcode_tensor::testgen_module_linear!();

        // test ops
        deepcode_tensor::testgen_gather_scatter!();
        deepcode_tensor::testgen_narrow!();
        deepcode_tensor::testgen_add!();
        deepcode_tensor::testgen_aggregation!();
        deepcode_tensor::testgen_arange!();
        deepcode_tensor::testgen_arange_step!();
        deepcode_tensor::testgen_arg!();
        deepcode_tensor::testgen_cast!();
        deepcode_tensor::testgen_cat!();
        deepcode_tensor::testgen_chunk!();
        deepcode_tensor::testgen_clamp!();
        deepcode_tensor::testgen_close!();
        deepcode_tensor::testgen_cos!();
        deepcode_tensor::testgen_cosh!();
        deepcode_tensor::testgen_create_like!();
        deepcode_tensor::testgen_cross!();
        deepcode_tensor::testgen_cumsum!();
        deepcode_tensor::testgen_div!();
        deepcode_tensor::testgen_dot!();
        deepcode_tensor::testgen_erf!();
        deepcode_tensor::testgen_exp!();
        deepcode_tensor::testgen_flatten!();
        deepcode_tensor::testgen_full!();
        deepcode_tensor::testgen_init!();
        deepcode_tensor::testgen_iter_dim!();
        deepcode_tensor::testgen_log!();
        deepcode_tensor::testgen_log1p!();
        deepcode_tensor::testgen_map_comparison!();
        deepcode_tensor::testgen_mask!();
        deepcode_tensor::testgen_matmul!();
        deepcode_tensor::testgen_maxmin!();
        deepcode_tensor::testgen_mul!();
        deepcode_tensor::testgen_neg!();
        deepcode_tensor::testgen_one_hot!();
        deepcode_tensor::testgen_powf_scalar!();
        deepcode_tensor::testgen_random!();
        deepcode_tensor::testgen_recip!();
        deepcode_tensor::testgen_repeat_dim!();
        deepcode_tensor::testgen_repeat!();
        deepcode_tensor::testgen_reshape!();
        deepcode_tensor::testgen_roll!();
        deepcode_tensor::testgen_sin!();
        deepcode_tensor::testgen_sinh!();
        deepcode_tensor::testgen_slice!();
        deepcode_tensor::testgen_slice_assign!();
        deepcode_tensor::testgen_stack!();
        deepcode_tensor::testgen_sqrt!();
        deepcode_tensor::testgen_abs!();
        deepcode_tensor::testgen_squeeze!();
        deepcode_tensor::testgen_sub!();
        deepcode_tensor::testgen_tan!();
        deepcode_tensor::testgen_tanh!();
        deepcode_tensor::testgen_transpose!();
        deepcode_tensor::testgen_tri!();
        deepcode_tensor::testgen_powf!();
        deepcode_tensor::testgen_any!();
        deepcode_tensor::testgen_all_op!();
        deepcode_tensor::testgen_permute!();
        deepcode_tensor::testgen_movedim!();
        deepcode_tensor::testgen_flip!();
        deepcode_tensor::testgen_bool!();
        deepcode_tensor::testgen_argwhere_nonzero!();
        deepcode_tensor::testgen_sign!();
        deepcode_tensor::testgen_expand!();
        deepcode_tensor::testgen_tri_mask!();
        deepcode_tensor::testgen_sort_argsort!();
        deepcode_tensor::testgen_topk!();
        deepcode_tensor::testgen_remainder!();
        deepcode_tensor::testgen_cartesian_grid!();
        deepcode_tensor::testgen_nan!();
        deepcode_tensor::testgen_inf!();
        deepcode_tensor::testgen_finite!();
        deepcode_tensor::testgen_round!();
        deepcode_tensor::testgen_floor!();
        deepcode_tensor::testgen_ceil!();
        deepcode_tensor::testgen_trunc!();
        deepcode_tensor::testgen_fmod!();
        deepcode_tensor::testgen_select!();
        deepcode_tensor::testgen_take!();
        deepcode_tensor::testgen_split!();
        deepcode_tensor::testgen_prod!();
        deepcode_tensor::testgen_grid_sample!();
        deepcode_tensor::testgen_unfold!();

        // test stats
        deepcode_tensor::testgen_var!();
        deepcode_tensor::testgen_cov!();
        deepcode_tensor::testgen_eye!();

        // test padding
        deepcode_tensor::testgen_padding!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_with_int_param {
    () => {
        // test ops
        deepcode_tensor::testgen_add!();
        deepcode_tensor::testgen_aggregation!();
        deepcode_tensor::testgen_arg!();
        deepcode_tensor::testgen_cast!();
        deepcode_tensor::testgen_bool!();
        deepcode_tensor::testgen_cat!();
        deepcode_tensor::testgen_cumsum!();
        deepcode_tensor::testgen_div!();
        deepcode_tensor::testgen_expand!();
        deepcode_tensor::testgen_flip!();
        deepcode_tensor::testgen_mask!();
        deepcode_tensor::testgen_movedim!();
        deepcode_tensor::testgen_mul!();
        deepcode_tensor::testgen_permute!();
        deepcode_tensor::testgen_reshape!();
        deepcode_tensor::testgen_select!();
        deepcode_tensor::testgen_take!();
        deepcode_tensor::testgen_sign!();
        deepcode_tensor::testgen_sort_argsort!();
        deepcode_tensor::testgen_stack!();
        deepcode_tensor::testgen_sub!();
        deepcode_tensor::testgen_transpose!();
        deepcode_tensor::testgen_gather_scatter!();
        deepcode_tensor::testgen_bitwise!();
        deepcode_tensor::testgen_matmul!();
        deepcode_tensor::testgen_unfold!();

        // test stats
        deepcode_tensor::testgen_eye!();

        // test padding
        deepcode_tensor::testgen_padding!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_with_bool_param {
    () => {
        deepcode_tensor::testgen_all_op!();
        deepcode_tensor::testgen_any!();
        deepcode_tensor::testgen_argwhere_nonzero!();
        deepcode_tensor::testgen_cast!();
        deepcode_tensor::testgen_cat!();
        deepcode_tensor::testgen_expand!();
        deepcode_tensor::testgen_full!();
        deepcode_tensor::testgen_map_comparison!();
        deepcode_tensor::testgen_mask!();
        deepcode_tensor::testgen_repeat_dim!();
        deepcode_tensor::testgen_repeat!();
        deepcode_tensor::testgen_reshape!();
        deepcode_tensor::testgen_stack!();
        deepcode_tensor::testgen_transpose!();
        deepcode_tensor::testgen_tri_mask!();
        deepcode_tensor::testgen_unfold!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_no_param {
    () => {
        // test stats
        deepcode_tensor::testgen_display!();

        // test clone invariance
        deepcode_tensor::testgen_clone_invariance!();

        // test primitive
        deepcode_tensor::testgen_primitive!();

        // test multi threads
        #[cfg(feature = "std")]
        deepcode_tensor::testgen_multi_threads!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! as_bytes {
    ($ty:ident: $($elem:expr),*) => {
        F::as_bytes(&[$($ty::new($elem),)*])
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! as_type {
    ($ty:ident: [$($elem:tt),*]) => {
        [$($crate::as_type![$ty: $elem]),*]
    };
    ($ty:ident: [$($elem:tt,)*]) => {
        [$($crate::as_type![$ty: $elem]),*]
    };
    ($ty:ident: $elem:expr) => {
        {
            use $crate::tests::{Float, Int};

            $ty::new($elem)
        }
    };
}

// Quantized tensor utilities
pub mod qtensor {
    use core::marker::PhantomData;

    use crate::{
        Tensor, TensorData,
        backend::Backend,
        quantization::{QTensorPrimitive, QuantValue},
    };

    pub struct QTensor<B: Backend, const D: usize> {
        b: PhantomData<B>,
    }

    impl<B: Backend, const D: usize> QTensor<B, D> {
        /// Creates a quantized int8 tensor from the floating point data using the default quantization scheme
        /// (i.e., per-tensor symmetric quantization).
        pub fn int8<F: Into<TensorData>>(floats: F) -> Tensor<B, D> {
            Self::int8_symmetric(floats)
        }

        /// Creates a quantized int8 tensor from the floating point data using per-tensor symmetric quantization.
        pub fn int8_symmetric<F: Into<TensorData>>(floats: F) -> Tensor<B, D> {
            Tensor::from_floats(floats, &Default::default()).quantize_dynamic(
                &<B::QuantizedTensorPrimitive as QTensorPrimitive>::default_scheme()
                    .with_value(QuantValue::Q8S),
            )
        }
    }
}
