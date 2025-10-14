use std::marker::PhantomData;

use deepcl_core::client::ComputeClient;

use crate::components::{
    Args, AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    attention_types::*,
    batch::{
        BatchAttentionFamily,
        dummy::{DummyBatchAttention, config::DummyBatchConfig},
        entry_point::attention,
    },
    global::GlobalAttentionFamily,
};

pub struct DummyBatchAttentionFamily<GA: GlobalAttentionFamily> {
    _phantom: PhantomData<GA>,
}

impl<GA: GlobalAttentionFamily> BatchAttentionFamily for DummyBatchAttentionFamily<GA> {
    type Attention<AP: AttentionPrecision> = DummyBatchAttention<AP, GA::Attention<AP>>;
    type Config = DummyBatchConfig<GA::Config>;

    fn setup<AP: crate::components::AttentionPrecision, R: deepcl_core::Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
    ) -> Result<Self::Config, crate::components::AttentionSetupError> {
        let global_config = GA::setup::<AP, R>(client, problem, selection, line_sizes)?;

        DummyBatchConfig::new(
            global_config,
            selection
                .hypercube_selection
                .to_hypercube_config(problem, client.properties().hardware.max_cube_count.clone()),
            problem.seq_kv as u32,
        )
        .validate(problem)
    }

    unsafe fn launch_unchecked<
        'a,
        AS: crate::components::AttentionSpec,
        R: deepcl_core::Runtime,
    >(
        client: &deepcl_core::prelude::ComputeClient<
            <R as deepcl_core::Runtime>::Server,
            <R as deepcl_core::Runtime>::Channel,
        >,
        cube_dim: deepcl_core::CubeDim,
        cube_count: deepcl_core::CubeCount,
        input: crate::components::InputRuntimeArg<'a, AS, R>,
        output: crate::components::OutputRuntimeArg<'a, AS, R>,
        cube_count_input: crate::components::batch::CubeCountInputArgs<'a, R>,
        config: Self::Config,
    ) {
        unsafe {
            attention::launch_unchecked::<
                Args<AS>,
                QG<AS>,
                QT<AS>,
                KG<AS>,
                KS<AS>,
                VG<AS>,
                VS<AS>,
                KVT<AS>,
                SM<AS>,
                ACC<AS>,
                MSK<AS>,
                OG<AS>,
                OS<AS>,
                Self,
                R,
            >(
                client,
                cube_count,
                cube_dim,
                input,
                output,
                cube_count_input,
                config,
            );
        }
    }
}
