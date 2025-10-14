use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::deepcode::{Scope, TensorType, Type};
use deepcode::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct RandomNormalLikeNode {
    pub mean: f64,
    pub scale: f64,
    pub input: TensorType,
    pub output: TensorType,
}

impl RandomNormalLikeNode {
    // Set distribution parameters based on mean and scale
    fn get_distribution(&self) -> TokenStream {
        let mean = self.mean;
        let std_deviation = self.scale;
        quote! { Distribution::Normal(#mean, #std_deviation) }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for RandomNormalLikeNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let output = &self.output.name;
        let input = scope.tensor_use_owned(&self.input, node_position);
        let dist = self.get_distribution();
        quote! {
            let #output = #input.random_like(#dist);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::RandomNormalLike(self)
    }

    fn register_imports(&self, imports: &mut crate::deepcode::DeepcodeImports) {
        imports.register("deepcode::tensor::Distribution");
    }
}

impl OnnxIntoNode for RandomNormalLikeNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let mean = node
            .attrs
            .get("mean")
            .map(|val| val.clone().into_f32() as f64)
            .unwrap_or(0.0f64);
        let scale = node
            .attrs
            .get("scale")
            .map(|val| val.clone().into_f32() as f64)
            .unwrap_or(1.0f64);
        Self::new(mean, scale, input, output)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::deepcode::{TensorKind, TensorType, graph::DeepcodeGraph, node::test::assert_tokens};
    use deepcode::record::FullPrecisionSettings;

    #[test]
    fn test_random_normal_like_codegen() {
        let mut graph = DeepcodeGraph::<FullPrecisionSettings>::default();

        graph.register(RandomNormalLikeNode::new(
            0.0f64,
            1.0f64,
            TensorType::new("input", 2, TensorKind::Float),
            TensorType::new("output", 2, TensorKind::Float),
        ));

        graph.register_input_output(vec!["input".to_string()], vec!["output".to_string()]);

        let expected = quote! {
            use deepcode::prelude::*;
            use deepcode::tensor::Distribution;

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
                pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
                    let output = input.random_like(Distribution::Normal(0f64, 1f64));
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
