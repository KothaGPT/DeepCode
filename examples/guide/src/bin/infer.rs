#![recursion_limit = "131"]
use deepcode::{backend::WebGpu, data::dataset::Dataset};
use guide::inference;

fn main() {
    type MyBackend = WebGpu<f32, i32>;

    let device = deepcode::backend::wgpu::WgpuDevice::default();

    // All the training artifacts are saved in this directory
    let artifact_dir = "/tmp/guide";

    // Infer the model
    inference::infer::<MyBackend>(
        artifact_dir,
        device,
        deepcode::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
}
