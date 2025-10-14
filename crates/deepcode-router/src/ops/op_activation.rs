use crate::{BackendRouter, RunnerChannel};
use deepcode_tensor::ops::ActivationOps;

impl<R: RunnerChannel> ActivationOps<Self> for BackendRouter<R> {}
