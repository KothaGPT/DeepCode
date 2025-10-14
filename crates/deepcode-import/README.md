# Deepcode Import

The `deepcode-import` crate enables seamless integration of pre-trained models from popular machine
learning frameworks into the Deepcode ecosystem. This functionality allows you to leverage existing
models while benefiting from Deepcode's performance optimizations and native Rust integration.

## Supported Import Formats

Deepcode currently supports three primary model import formats, each serving different use cases:

| Format                                                                              | Description                               | Use Case                                                                                               |
| ----------------------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| [**ONNX** (Guide)](https://deepcode.dev/books/deepcode/import/onnx-model.html)               | Open Neural Network Exchange format       | Direct import of complete model architectures and weights from any framework that supports ONNX export |
| [**PyTorch** (Guide)](https://deepcode.dev/books/deepcode/import/pytorch-model.html)         | PyTorch weights (.pt, .pth)               | Loading weights from PyTorch models into a matching Deepcode architecture                                  |
| [**Safetensors** (Guide)](https://deepcode.dev/books/deepcode/import/safetensors-model.html) | Hugging Face's model serialization format | Loading a model's tensor weights into a matching Deepcode architecture                                     |

## ONNX Contributor Resources

- [ONNX to Deepcode conversion guide](https://deepcode.dev/books/contributor/guides/onnx-to-deepcode-conversion-tool.html) -
  Instructions for adding support for additional ONNX operators
- [ONNX tests README](https://github.com/kothagpt/deepcode/blob/main/crates/deepcode-import/onnx-tests/README.md) -
  Testing procedures for ONNX operators
- [Supported ONNX Operators table](https://github.com/kothagpt/deepcode/blob/main/crates/deepcode-import/SUPPORTED-ONNX-OPS.md) -
  Complete list of currently supported ONNX operators
