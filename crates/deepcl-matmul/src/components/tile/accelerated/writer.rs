use deepcl::prelude::*;
use deepcl_core as deepcl;

use crate::components::{as_cmma_layout, tile::StridedTile};

/// Writer using the cmma store function.
#[derive(CubeType)]
pub struct CmmaStageWriter {}

#[cube]
impl CmmaStageWriter {
    pub fn store_fragment<E: Numeric, V: Numeric>(
        tile: &mut StridedTile<V, ReadWrite>,
        fragment: &cmma::Matrix<E>,
    ) {
        let layout = as_cmma_layout(tile.layout);
        let (mut slice, stride) = tile.as_unlined();
        cmma::store(&mut slice, fragment, stride, layout);
    }
}
