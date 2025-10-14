use deepcode_tensor::Shape;
use deepcl::prelude::*;

use crate::{CubeRuntime, element::CubeElement, ops::numeric::empty_device, tensor::CubeTensor};

/// Pseudo-random generator with uniform distribution
pub fn random_normal<R: CubeRuntime, E: CubeElement + Numeric>(
    shape: Shape,
    device: &R::Device,
    mean: E,
    std: E,
) -> CubeTensor<R> {
    let client = R::client(device);
    let output = empty_device::<R, E>(client.clone(), device.clone(), shape);
    let output_handle = output.as_handle_ref();

    deepcl::random::random_normal(&client, mean, std, output_handle);

    output
}
