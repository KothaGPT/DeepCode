use deepcode_core as deepcode;
use deepcode_core::record::Record;

use deepcode_tensor::Tensor;
use deepcode_tensor::backend::Backend;

// It compiles
#[derive(Record)]
pub struct TestWithBackendRecord<B: Backend> {
    tensor: Tensor<B, 2>,
}

// It compiles
#[derive(Record)]
pub struct TestWithoutBackendRecord {
    _tensor: usize,
}
