fn main() {
    #[cfg(feature = "cuda")]
    normalization::launch::<deepcl::cuda::CudaRuntime>(&Default::default());
    #[cfg(feature = "wgpu")]
    normalization::launch::<deepcl::wgpu::WgpuRuntime>(&Default::default());
}
