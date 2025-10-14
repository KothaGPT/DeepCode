use deepcode::optim::decay::WeightDecayConfig;
use text_generation::{DbPediaDataset, training::ExperimentConfig};

#[cfg(feature = "f16")]
type Elem = deepcode::tensor::f16;
#[cfg(not(feature = "f16"))]
type Elem = f32;

type Backend = deepcode::backend::Autodiff<deepcode::backend::LibTorch<Elem>>;

fn main() {
    let config = ExperimentConfig::new(
        deepcode::nn::transformer::TransformerEncoderConfig::new(384, 1536, 12, 6)
            .with_norm_first(true),
        deepcode::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-6))),
    );

    text_generation::training::train::<Backend, DbPediaDataset>(
        if cfg!(target_os = "macos") {
            deepcode::tensor::Device::<Backend>::Mps
        } else {
            deepcode::tensor::Device::<Backend>::Cuda(0)
        },
        DbPediaDataset::train(),
        DbPediaDataset::test(),
        config,
        "/tmp/text-generation",
    );
}
