#![allow(missing_docs)]

#[macro_export]
macro_rules! testgen_matmul_simple {
    () => {
        mod simple {
            $crate::testgen_matmul_simple!(f32);
        }
    };
    ($float:ident) => {
            use super::*;
            use deepcl_matmul::tests;
            use deepcl_core::flex32;

            pub type FloatT = $float;

            #[test]
            pub fn test_small() {
                deepcl_matmul::tests::naive::tests::test_small::<TestRuntime, FloatT>(
                    &Default::default(),
                )
            }

            #[test]
            pub fn test_simple_matmul_large() {
                deepcl_matmul::tests::naive::tests::test_large::<TestRuntime, FloatT>(
                    &Default::default(),
                )
            }

            #[test]
            pub fn test_with_check_bounds() {
                deepcl_matmul::tests::naive::tests::test_with_check_bounds::<
                    TestRuntime,
                    FloatT,
                >(&Default::default())
            }

            #[test]
            pub fn test_with_batches() {
                deepcl_matmul::tests::naive::tests::test_with_batches::<
                    TestRuntime,
                    FloatT,
                >(&Default::default())
            }
    };
    ([$($float:ident),*]) => {
        mod simple {
            use super::*;
            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;

                    $crate::testgen_matmul_simple!($float);
                })*
            }
        }
    };
}
