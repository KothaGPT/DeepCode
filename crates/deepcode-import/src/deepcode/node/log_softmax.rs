use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::deepcode::{Scope, TensorType, ToTokens, Type};
use deepcode::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct LogSoftmaxNode {
    pub input: TensorType,
    pub output: TensorType,
    pub dim: usize,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for LogSoftmaxNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let dim = self.dim.to_tokens();

        quote! {
            let #output = deepcode::tensor::activation::log_softmax(#input, #dim);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::LogSoftmax(self)
    }
}

impl OnnxIntoNode for LogSoftmaxNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let input = match Type::from(node.inputs.first().unwrap()) {
            Type::Tensor(t) => t,
            _ => panic!("LogSoftmax expects tensor input"),
        };
        let output = match Type::from(node.outputs.first().unwrap()) {
            Type::Tensor(t) => t,
            _ => panic!("LogSoftmax expects tensor output"),
        };
        let dim = onnx_ir::node::log_softmax::log_softmax_config(&node);
        Self::new(input, output, dim)
    }
}

#[cfg(test)]
mod tests {
    use deepcode::record::FullPrecisionSettings;

    use super::*;
    use crate::deepcode::{TensorType, graph::DeepcodeGraph, node::test::assert_tokens};

    #[test]
    fn test_codegen_log_softmax() {
        let mut graph = DeepcodeGraph::<FullPrecisionSettings>::default();

        graph.register(LogSoftmaxNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
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
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = deepcode::tensor::activation::log_softmax(tensor1, 1);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
