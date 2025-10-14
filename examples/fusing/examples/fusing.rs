fn main() {
    #[cfg(feature = "cuda")]
    fusing::launch::<deepcl::cuda::CudaRuntime>(&Default::default());
    #[cfg(feature = "wgpu")]
    fusing::launch::<deepcl::wgpu::WgpuRuntime>(&Default::default());
}
