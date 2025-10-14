fn main() {
    #[cfg(feature = "wgpu")]
    {
        let setup_shared = device_sharing::create_wgpu_setup_from_raw();
        let device_deepcl = deepcl::wgpu::init_device(setup_shared.clone(), Default::default());
        device_sharing::assert_wgpu_device_existing(&device_deepcl);
        sum_things::launch::<deepcl::wgpu::WgpuRuntime>(&device_deepcl);
    }
}
