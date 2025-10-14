#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![allow(unused)] // TODO remove when backend filled

//! Deepcode Candle Backend

#[macro_use]
extern crate derive_new;

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
    use super::*;

    pub type TestBackend = Candle<f32, i64>;
    pub type ReferenceBackend = deepcode_tch::LibTorch<f32>;

    pub type TestTensor<const D: usize> = deepcode_tensor::Tensor<TestBackend, D>;
    pub type ReferenceTensor<const D: usize> = deepcode_tensor::Tensor<ReferenceBackend, D>;
    pub type TestTensorInt<const D: usize> = deepcode_tensor::Tensor<TestBackend, D, deepcode_tensor::Int>;
    pub type TestTensorBool<const D: usize> =
        deepcode_tensor::Tensor<TestBackend, D, deepcode_tensor::Bool>;

    type TestAutodiffBackend = deepcode_autodiff::Autodiff<TestBackend>;
    type TestAutodiffTensor<const D: usize> = deepcode_tensor::Tensor<TestAutodiffBackend, D>;

    pub type FloatType = f32;
    pub type IntType = i64;

    // test activation
    deepcode_tensor::testgen_gelu!();
    deepcode_tensor::testgen_prelu!();
    deepcode_tensor::testgen_relu!();
    deepcode_tensor::testgen_softmax!();
    deepcode_tensor::testgen_sigmoid!();
    deepcode_tensor::testgen_hard_sigmoid!();
    deepcode_tensor::testgen_silu!();

    // test module
    deepcode_tensor::testgen_module_forward!();
    deepcode_tensor::testgen_module_conv1d!();
    deepcode_tensor::testgen_module_nearest_interpolate!();
    // deepcode_tensor::testgen_module_conv2d!();
    // deepcode_tensor::testgen_module_conv_transpose1d!();
    // deepcode_tensor::testgen_module_conv_transpose2d!();
    // deepcode_tensor::testgen_module_max_pool1d!();
    // deepcode_tensor::testgen_module_max_pool2d!();
    // deepcode_tensor::testgen_module_avg_pool1d!();
    // deepcode_tensor::testgen_module_avg_pool2d!();
    // deepcode_tensor::testgen_module_adaptive_avg_pool1d!();
    // deepcode_tensor::testgen_module_adaptive_avg_pool2d!();

    // test ops
    deepcode_tensor::testgen_add!();
    // deepcode_tensor::testgen_aggregation!();
    deepcode_tensor::testgen_arange!();
    deepcode_tensor::testgen_arange_step!();
    deepcode_tensor::testgen_arg!();
    deepcode_tensor::testgen_bool!();
    deepcode_tensor::testgen_cast!();
    deepcode_tensor::testgen_cat!();
    deepcode_tensor::testgen_recip!();
    deepcode_tensor::testgen_clamp!();
    deepcode_tensor::testgen_cos!();
    deepcode_tensor::testgen_close!();
    // deepcode_tensor::testgen_div!();
    deepcode_tensor::testgen_erf!();
    deepcode_tensor::testgen_exp!();
    deepcode_tensor::testgen_flatten!();
    deepcode_tensor::testgen_full!();
    deepcode_tensor::testgen_gather_scatter!();
    deepcode_tensor::testgen_init!();
    deepcode_tensor::testgen_log!();
    deepcode_tensor::testgen_log1p!();
    deepcode_tensor::testgen_map_comparison!();
    deepcode_tensor::testgen_mask!();
    deepcode_tensor::testgen_matmul!();
    deepcode_tensor::testgen_maxmin!();
    deepcode_tensor::testgen_mul!();
    deepcode_tensor::testgen_neg!();
    deepcode_tensor::testgen_permute!();
    // commented out due to macos CI failure, see #2427
    // deepcode_tensor::testgen_remainder!();
    deepcode_tensor::testgen_flip!();
    deepcode_tensor::testgen_argwhere_nonzero!();
    deepcode_tensor::testgen_sign!();
    deepcode_tensor::testgen_nan!();
    deepcode_tensor::testgen_inf!();
    deepcode_tensor::testgen_finite!();
    deepcode_tensor::testgen_round!();
    deepcode_tensor::testgen_floor!();
    deepcode_tensor::testgen_ceil!();

    // TODO: https://github.com/kothagpt/deepcode/issues/1237
    //
    // deepcode_tensor::testgen_powf_scalar!();
    // deepcode_tensor::testgen_powf!();

    deepcode_tensor::testgen_random!();
    deepcode_tensor::testgen_repeat_dim!();
    deepcode_tensor::testgen_reshape!();
    deepcode_tensor::testgen_select!();
    deepcode_tensor::testgen_sin!();
    deepcode_tensor::testgen_slice!();
    deepcode_tensor::testgen_slice_assign!();
    deepcode_tensor::testgen_sqrt!();
    deepcode_tensor::testgen_abs!();
    deepcode_tensor::testgen_squeeze!();
    deepcode_tensor::testgen_sub!();
    deepcode_tensor::testgen_tanh!();
    deepcode_tensor::testgen_transpose!();
    deepcode_tensor::testgen_expand!();
    deepcode_tensor::testgen_cumsum!();

    // test stats
    deepcode_tensor::testgen_var!();
    deepcode_tensor::testgen_display!();

    // Behavior
    // deepcode_autodiff::testgen_ad_broadcast!();

    // Activation
    deepcode_autodiff::testgen_ad_relu!();
    deepcode_autodiff::testgen_ad_gelu!();

    // Modules
    // deepcode_autodiff::testgen_ad_conv1d!();
    // deepcode_autodiff::testgen_ad_conv2d!();
    // deepcode_autodiff::testgen_ad_conv_transpose1d!();
    // deepcode_autodiff::testgen_ad_conv_transpose2d!();
    // deepcode_autodiff::testgen_ad_max_pool1d!();
    // deepcode_autodiff::testgen_ad_max_pool2d!();
    // deepcode_autodiff::testgen_ad_avg_pool1d!();
    // deepcode_autodiff::testgen_ad_avg_pool2d!();
    // deepcode_autodiff::testgen_ad_adaptive_avg_pool1d!();
    // deepcode_autodiff::testgen_ad_adaptive_avg_pool2d!();
    deepcode_autodiff::testgen_module_backward!();

    // Tensor
    deepcode_autodiff::testgen_ad_complex!();
    deepcode_autodiff::testgen_ad_multithread!();
    deepcode_autodiff::testgen_ad_add!();
    deepcode_autodiff::testgen_ad_aggregation!();
    deepcode_autodiff::testgen_ad_maxmin!();
    // deepcode_autodiff::testgen_ad_cat!();
    deepcode_autodiff::testgen_ad_cos!();
    deepcode_autodiff::testgen_ad_cross_entropy_loss!();
    deepcode_autodiff::testgen_ad_div!();
    deepcode_autodiff::testgen_ad_erf!();
    deepcode_autodiff::testgen_ad_exp!();
    deepcode_autodiff::testgen_ad_slice!();
    deepcode_autodiff::testgen_ad_gather_scatter!();
    deepcode_autodiff::testgen_ad_select!();
    deepcode_autodiff::testgen_ad_log!();
    deepcode_autodiff::testgen_ad_log1p!();
    deepcode_autodiff::testgen_ad_mask!();
    deepcode_autodiff::testgen_ad_matmul!();
    deepcode_autodiff::testgen_ad_mul!();
    deepcode_autodiff::testgen_ad_neg!();
    deepcode_autodiff::testgen_ad_recip!();
    // commented out due to macos CI failure, see #2427
    // deepcode_autodiff::testgen_ad_remainder!();
    deepcode_autodiff::testgen_ad_reshape!();
    deepcode_autodiff::testgen_ad_sin!();
    deepcode_autodiff::testgen_ad_softmax!();
    deepcode_autodiff::testgen_ad_sqrt!();
    deepcode_autodiff::testgen_ad_abs!();
    deepcode_autodiff::testgen_ad_sub!();
    deepcode_autodiff::testgen_ad_tanh!();
    deepcode_autodiff::testgen_ad_transpose!();
    deepcode_autodiff::testgen_ad_expand!();
    deepcode_autodiff::testgen_ad_round!();
    deepcode_autodiff::testgen_ad_floor!();
    deepcode_autodiff::testgen_ad_ceil!();
    deepcode_autodiff::testgen_ad_slice_assign!();
    deepcode_autodiff::testgen_ad_cumsum!();
}
