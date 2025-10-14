use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::deepcode::{Scope, TensorType, ToTokens, Type};
use deepcode::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct FlattenNode {
    pub input: TensorType,
    pub output: TensorType,
    pub axis: usize,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for FlattenNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        if self.axis == 0 {
            quote! {
                let #output = #input.reshape::<2>([1, -1]);
            }
        } else {
            let axis = self.axis.to_tokens();
            quote! {
                let #output = {
                    let leading_dim = #input.shape().dims[..#axis].iter().product::<usize>() as i32;
                    #input.reshape::<2, _>([leading_dim, -1])
                };
            }
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Flatten(self)
    }
}

impl OnnxIntoNode for FlattenNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let input = match crate::deepcode::Type::from(node.inputs.first().unwrap()) {
            crate::deepcode::Type::Tensor(t) => t,
            _ => panic!("Flatten expects tensor input"),
        };
        let output = match crate::deepcode::Type::from(node.outputs.first().unwrap()) {
            crate::deepcode::Type::Tensor(t) => t,
            _ => panic!("Flatten expects tensor output"),
        };
        let axis = onnx_ir::node::flatten::flatten_config(&node);
        Self::new(input, output, axis)
    }
}

#[cfg(test)]
mod tests {
    use deepcode::record::FullPrecisionSettings;

    use super::*;
    use crate::deepcode::{TensorType, graph::DeepcodeGraph, node::test::assert_tokens};

    #[test]
    fn test_codegen_flatten() {
        let mut graph = DeepcodeGraph::<FullPrecisionSettings>::default();

        graph.register(FlattenNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 2),
            1,
        ));

        graph.register_input_output(vec!["tensor1".to_string()], vec!["tensor2".to_string()]);

        let expected = quote! {
            use deepcode::prelude::*;

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: deepcode::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: deepcode::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 2> {
                    let tensor2 = {
                        let leading_dim = tensor1.shape().dims[..1].iter().product::<usize>() as i32;
                        tensor1.reshape::<2, _>([leading_dim, -1])
                    };
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
