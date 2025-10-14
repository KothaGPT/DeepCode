pub mod reinterpret_slice;
pub mod tensor;

#[macro_export]
macro_rules! testgen {
    () => {
        mod test_deepcl_std {
            use super::*;
            use half::{bf16, f16};

            deepcl_std::testgen_reinterpret_slice!();
        }
    };
}
