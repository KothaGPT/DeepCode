#[cfg(any(
    feature = "ndarray",
    feature = "ndarray-blas-netlib",
    feature = "ndarray-blas-openblas",
    feature = "ndarray-blas-accelerate",
))]
mod ndarray {
    use deepcode::backend::{
        Autodiff,
        ndarray::{NdArray, NdArrayDevice},
    };
    use mnist::training;

    pub fn run() {
        let device = NdArrayDevice::Cpu;
        training::run::<Autodiff<NdArray>>(device);
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use deepcode::backend::{
        Autodiff,
        libtorch::{LibTorch, LibTorchDevice},
    };
    use mnist::training;

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        training::run::<Autodiff<LibTorch>>(device);
    }
}

#[cfg(any(feature = "wgpu", feature = "metal", feature = "vulkan"))]
mod wgpu {
    use deepcode::backend::{
        Autodiff,
        wgpu::{Wgpu, WgpuDevice},
    };
    use mnist::training;

    pub fn run() {
        let device = WgpuDevice::default();
        training::run::<Autodiff<Wgpu>>(device);
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use deepcode::backend::{Autodiff, Cuda};
    use mnist::training;

    pub fn run() {
        let device = Default::default();
        training::run::<Autodiff<Cuda>>(device);
    }
}

#[cfg(feature = "rocm")]
mod rocm {
    use deepcode::backend::{Autodiff, Rocm};
    use mnist::training;

    pub fn run() {
        let device = Default::default();
        training::run::<Autodiff<Rocm>>(device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use deepcode::backend::{
        Autodiff,
        libtorch::{LibTorch, LibTorchDevice},
    };
    use mnist::training;

    pub fn run() {
        let device = LibTorchDevice::Cpu;
        training::run::<Autodiff<LibTorch>>(device);
    }
}

#[cfg(feature = "remote")]
mod remote {
    use deepcode::backend::{Autodiff, RemoteBackend};
    use mnist::training;

    pub fn run() {
        training::run::<Autodiff<RemoteBackend>>(Default::default());
    }
}

fn main() {
    #[cfg(any(
        feature = "ndarray",
        feature = "ndarray-blas-netlib",
        feature = "ndarray-blas-openblas",
        feature = "ndarray-blas-accelerate",
    ))]
    ndarray::run();
    #[cfg(feature = "tch-gpu")]
    tch_gpu::run();
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run();
    #[cfg(any(feature = "wgpu", feature = "metal", feature = "vulkan"))]
    wgpu::run();
    #[cfg(feature = "cuda")]
    cuda::run();
    #[cfg(feature = "rocm")]
    rocm::run();
    #[cfg(feature = "remote")]
    remote::run();
}
