/// Dataloader module.
#[cfg(feature = "dataset")]
pub mod dataloader;

/// Dataset module.
#[cfg(feature = "dataset")]
pub mod dataset {
    pub use deepcode_dataset::*;
}

/// Network module.
#[cfg(feature = "network")]
pub mod network {
    pub use deepcode_common::network::*;
}
