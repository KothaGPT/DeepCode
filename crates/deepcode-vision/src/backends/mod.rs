pub(crate) mod cpu;
#[cfg(feature = "deepcl-backend")]
mod cube;

pub use cpu::{KernelShape, create_structuring_element};
