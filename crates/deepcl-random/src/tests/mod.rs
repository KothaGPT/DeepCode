pub mod bernoulli;
pub mod interval;
pub mod normal;
pub mod uniform;

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_random {
    () => {
        use deepcl::prelude::*;
        use deepcl_core as deepcl;

        use deepcl::{client::ComputeClient, prelude::TensorHandleRef};
        use deepcl_random::*;
        use deepcl_std::tensor::TensorHandle;

        use core::f32;

        deepcl_random::testgen_random_bernoulli!();
        deepcl_random::testgen_random_normal!();
        deepcl_random::testgen_random_uniform!();
        deepcl_random::testgen_random_interval!();
    };
}
