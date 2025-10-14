use crate::{Fusion, FusionBackend};
use deepcode_tensor::ops::ActivationOps;

impl<B: FusionBackend> ActivationOps<Self> for Fusion<B> {}
