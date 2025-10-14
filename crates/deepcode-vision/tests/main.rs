#[cfg(all(test, feature = "test-cpu"))]
mod tests_cpu {
    pub type TestBackend = deepcode_ndarray::NdArray<f32, i32>;

    deepcode_vision::testgen_all!();
}

#[cfg(all(test, feature = "test-wgpu"))]
mod tests_wgpu {
    pub type TestBackend = deepcode_wgpu::Wgpu;

    deepcode_vision::testgen_all!();
}

#[cfg(all(test, feature = "test-cuda"))]
mod tests_cuda {
    pub type TestBackend = deepcode_cuda::Cuda;

    deepcode_vision::testgen_all!();
}
