fn main() {
    #[cfg(feature = "cuda")]
    multi_gpus::run::<deepcode::backend::Cuda>();
    #[cfg(feature = "rocm")]
    multi_gpus::run::<deepcode::backend::Rocm>();
    #[cfg(feature = "tch-gpu")]
    multi_gpus::run::<deepcode::backend::LibTorch>();
}
