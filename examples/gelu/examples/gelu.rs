fn main() {
    #[cfg(feature = "cuda")]
    gelu::launch::<deepcl::cuda::CudaRuntime>(&Default::default());
    #[cfg(feature = "wgpu")]
    gelu::launch::<deepcl::wgpu::WgpuRuntime>(&Default::default());
    #[cfg(feature = "cpu")]
    gelu::launch::<deepcl::cpu::CpuRuntime>(&Default::default());
}
