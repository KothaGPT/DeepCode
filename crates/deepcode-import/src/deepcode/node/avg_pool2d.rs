use onnx_ir::node::avg_pool2d::AvgPool2dConfig;
use proc_macro2::TokenStream;
use quote::quote;

use deepcode::record::PrecisionSettings;

use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::deepcode::{DeepcodeImports, OtherType, Scope, TensorType, ToTokens, Type};

#[derive(Debug, Clone)]
pub struct AvgPool2dNode {
    pub field: OtherType,
    pub input: TensorType,
    pub output: TensorType,
    pub config: AvgPool2dConfig,
}

impl AvgPool2dNode {
    pub fn new<S: AsRef<str>>(
        name: S,
        input: TensorType,
        output: TensorType,
        config: AvgPool2dConfig,
    ) -> Self {
        Self {
            field: OtherType::new(
                name,
                quote! {
                    AvgPool2d
                },
            ),
            input,
            output,
            config,
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for AvgPool2dNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }
    fn field_type(&self) -> Option<Type> {
        Some(Type::Other(self.field.clone()))
    }

    fn field_init(&self) -> Option<TokenStream> {
        let name = &self.field.name;
        let kernel_size = self.config.kernel_size.to_tokens();
        let strides = self.config.strides.to_tokens();
        let padding = self.config.padding.to_tokens();
        let count_include_pad = self.config.count_include_pad;

        let tokens = quote! {
            let #name = AvgPool2dConfig::new(#kernel_size)
                .with_strides(#strides)
                .with_padding(#padding)
                .with_count_include_pad(#count_include_pad)
                .init();
        };

        Some(tokens)
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let field = &self.field.name;

        quote! {
            let #output = self.#field.forward(#input);
        }
    }

    fn register_imports(&self, imports: &mut DeepcodeImports) {
        imports.register("deepcode::nn::PaddingConfig2d");
        imports.register("deepcode::nn::pool::AvgPool2d");
        imports.register("deepcode::nn::pool::AvgPool2dConfig");
    }

    fn into_node(self) -> Node<PS> {
        Node::AveragePool2d(self)
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        S::serialize_none(serializer)
    }
}

impl OnnxIntoNode for AvgPool2dNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = onnx_ir::node::avg_pool2d::avg_pool2d_config(&node);
        let name = &node.name;
        Self::new(name, input, output, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deepcode::{
        TensorType,
        graph::DeepcodeGraph,
        node::{avg_pool2d::AvgPool2dNode, test::assert_tokens},
    };
    use deepcode::record::FullPrecisionSettings;
    use onnx_ir::node::padding::PaddingConfig2d;

    #[test]
    fn test_codegen() {
        let mut graph = DeepcodeGraph::<FullPrecisionSettings>::default();

        graph.register(AvgPool2dNode::new(
            "avg_pool2d",
            TensorType::new_float("input", 4),
            TensorType::new_float("output", 4),
            AvgPool2dConfig::new([3, 3], [1, 1], PaddingConfig2d::Valid, true),
        ));

        graph.register_input_output(vec!["input".to_string()], vec!["output".to_string()]);

        let expected = quote! {
            use deepcode::prelude::*;
            use deepcode::nn::PaddingConfig2d;
            use deepcode::nn::pool::AvgPool2d;
            use deepcode::nn::pool::AvgPool2dConfig;

            #[derive(Module, Debug)]
            pub struct Model <B: Backend> {
                avg_pool2d: AvgPool2d,
                phantom: core::marker::PhantomData<B>,
                device: deepcode::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let avg_pool2d = AvgPool2dConfig::new([3, 3])
                        .with_strides([1, 1])
                        .with_padding(PaddingConfig2d::Valid)
                        .with_count_include_pad(true)
                        .init();

                    Self {
                        avg_pool2d,
                        phantom: core::marker::PhantomData,
                        device: deepcode::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
                    let output = self.avg_pool2d.forward(input);

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
