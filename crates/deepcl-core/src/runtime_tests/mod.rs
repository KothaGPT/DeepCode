pub mod assign;
pub mod atomic;
pub mod barrier;
pub mod binary;
pub mod branch;
pub mod cluster;
pub mod cmma;
pub mod comparison;
pub mod const_match;
pub mod constants;
pub mod debug;
pub mod different_rank;
pub mod enums;
pub mod index;
pub mod launch;
pub mod line;
pub mod metadata;
pub mod minifloat;
pub mod plane;
pub mod saturating;
pub mod sequence;
pub mod slice;
pub mod stream;
pub mod synchronization;
pub mod tensor;
pub mod tensormap;
pub mod to_client;
pub mod topology;
pub mod traits;
pub mod unary;
pub mod unroll;

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_all {
    () => {
        use $crate::Runtime;

        type FloatType = f32;
        type IntType = i32;
        type UintType = u32;

        $crate::testgen_float!();
        $crate::testgen_int!();
        $crate::testgen_uint!();
        $crate::testgen_untyped!();
    };
    ($f_def:ident: [$($float:ident),*], $i_def:ident: [$($int:ident),*], $u_def:ident: [$($uint:ident),*]) => {
        use $crate::Runtime;

        ::paste::paste! {
            $(mod [<$float _ty>] {
                use super::*;

                type FloatType = $float;
                type IntType = $i_def;
                type UintType = $u_def;

                $crate::testgen_float!();
            })*
            $(mod [<$int _ty>] {
                use super::*;

                type FloatType = $f_def;
                type IntType = $int;
                type UintType = $u_def;

                $crate::testgen_int!();
            })*
            $(mod [<$uint _ty>] {
                use super::*;

                type FloatType = $f_def;
                type IntType = $i_def;
                type UintType = $uint;

                $crate::testgen_uint!();
            })*
        }
        $crate::testgen_untyped!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_float {
    () => {
        deepcl_core::testgen_assign!();
        deepcl_core::testgen_barrier!();
        deepcl_core::testgen_binary!();
        deepcl_core::testgen_branch!();
        deepcl_core::testgen_different_rank!();
        deepcl_core::testgen_index!();
        deepcl_core::testgen_launch!();
        deepcl_core::testgen_line!();
        deepcl_core::testgen_plane!();
        deepcl_core::testgen_sequence!();
        deepcl_core::testgen_slice!();
        deepcl_core::testgen_stream!();
        deepcl_core::testgen_unary!();
        deepcl_core::testgen_atomic_float!();
        deepcl_core::testgen_tensormap!();
        deepcl_core::testgen_minifloat!();
        deepcl_core::testgen_unroll!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_int {
    () => {
        deepcl_core::testgen_unary_int!();
        deepcl_core::testgen_atomic_int!();
        deepcl_core::testgen_saturating_int!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_uint {
    () => {
        deepcl_core::testgen_const_match!();
        deepcl_core::testgen_saturating_uint!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_untyped {
    () => {
        deepcl_core::testgen_cmma!();
        deepcl_core::testgen_metadata!();
        deepcl_core::testgen_topology!();

        deepcl_core::testgen_constants!();
        deepcl_core::testgen_sync_plane!();
        deepcl_core::testgen_tensor_indexing!();
        deepcl_core::testgen_debug!();
        deepcl_core::testgen_binary_untyped!();
        deepcl_core::testgen_cluster!();

        deepcl_core::testgen_enums!();
        deepcl_core::testgen_comparison!();

        deepcl_core::testgen_to_client!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! as_bytes {
    ($ty:ident: $($elem:expr),*) => {
        $ty::as_bytes(&[$($ty::new($elem),)*])
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! as_type {
    ($ty:ident: $($elem:expr),*) => {
        &[$($ty::new($elem),)*]
    };
}
