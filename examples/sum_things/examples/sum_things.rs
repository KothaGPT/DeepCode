fn main() {
    #[cfg(feature = "cuda")]
    sum_things::launch::<deepcl::cuda::CudaRuntime>(&Default::default());
    #[cfg(feature = "wgpu")]
    sum_things::launch::<deepcl::wgpu::WgpuRuntime>(&Default::default());
}
