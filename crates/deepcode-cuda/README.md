# Deepcode CUDA Backend

[Deepcode](https://github.com/kothagpt/deepcode) CUDA backend

[![Current Crates.io Version](https://img.shields.io/crates/v/deepcode-cuda.svg)](https://crates.io/crates/deepcode-cuda)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/kothagpt/deepcode-cuda/blob/master/README.md)

This crate provides a CUDA backend for [Deepcode](https://github.com/kothagpt/deepcode) using the
[deepcl](https://github.com/tracel-ai/deepcl.git) and [cudarc](https://github.com/coreylowman/cudarc.git)
crates.

## Usage Example

```rust
#[cfg(feature = "cuda")]
mod cuda {
    use deepcode_autodiff::Autodiff;
    use deepcode_cuda::{Cuda, CudaDevice};
    use mnist::training;

    pub fn run() {
        let device = CudaDevice::default();
        training::run::<Autodiff<Cuda<f32, i32>>>(device);
    }
}
```

## Dependencies

Requires CUDA 12.x to be installed and on the `PATH`.