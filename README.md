<div align="center">
<img src="https://raw.githubusercontent.com/kothagpt/deepcode/main/assets/logo-deepcode-neutral.webp" width="350px"/>

[![Discord](https://img.shields.io/discord/1038839012602941528.svg?color=7289da&&logo=discord)](https://discord.gg/uPEBbYYDB6)
[![Current Crates.io Version](https://img.shields.io/crates/v/deepcl.svg)](https://crates.io/crates/deepcl)
[![Minimum Supported Rust Version](https://img.shields.io/crates/msrv/deepcl)](https://crates.io/crates/deepcl)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://deepcode.dev/docs/deepcl)
[![Test Status](https://github.com/kothagpt/deepcode/actions/workflows/test.yml/badge.svg)](https://github.com/kothagpt/deepcode/actions/workflows/test.yml)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](#license)

---

**🚀 DeepCL - Multi-Platform High-Performance Compute Language for Rust**

*Blazingly fast tensor operations and deep learning across GPUs, CPUs, and WebAssembly*

</div>

---

## 🌟 Overview

**DeepCL** is a next-generation high-performance compute framework for Rust that brings GPU-accelerated tensor operations and deep learning capabilities to native applications. Built from the ground up for performance, portability, and developer experience, DeepCL enables you to harness the full power of modern hardware while maintaining the safety and expressiveness of Rust.

### ✨ Key Features

- **🚀 Multi-Platform GPU Support**: CUDA, WGPU, HIP, SPIR-V backends
- **⚡ Zero-Cost Abstractions**: High-level APIs with optimal performance
- **🌐 WebAssembly Ready**: Run ML models directly in browsers
- **🔧 Hardware Agnostic**: Write once, run anywhere
- **🧠 Deep Learning Primitives**: Convolution, MatMul, Attention, and more
- **📦 ONNX & PyTorch Integration**: Import existing models seamlessly
- **🛠️ Custom Kernels**: Write GPU shaders in Rust
- **📊 Real-time Training Dashboard**: Monitor progress with TUI

## 🏗️ Architecture

DeepCL is organized into modular crates for maximum flexibility:

| Crate | Description |
|-------|-------------|
| [`deepcl-core`](crates/deepcl-core/) | Core tensor operations and compute graph |
| [`deepcl-runtime`](crates/deepcl-runtime/) | Async runtime for high-performance execution |
| [`deepcl-cuda`](crates/deepcl-cuda/) | NVIDIA CUDA backend |
| [`deepcl-wgpu`](crates/deepcl-wgpu/) | WebGPU and SPIR-V support |
| [`deepcl-cpu`](crates/deepcl-cpu/) | Optimized CPU backend |
| [`deepcl-convolution`](crates/deepcl-convolution/) | Convolutional neural network operations |
| [`deepcl-attention`](crates/deepcl-attention/) | Transformer attention mechanisms |
| [`deepcl-matmul`](crates/deepcl-matmul/) | Matrix multiplication optimizations |
| [`deepcl-opt`](crates/deepcl-opt/) | Kernel optimization and fusion |

## 🚀 Quick Start

### Basic Tensor Operations

```rust
use deepcl::{prelude::*, tensor::{Distribution, Tensor}};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize runtime
    let device = CudaDevice::default();

    // Create tensors
    let x: Tensor<Cuda, 2> = Tensor::random([32, 32], Distribution::Default, &device);
    let y: Tensor<Cuda, 2> = Tensor::random([32, 32], Distribution::Default, &device);

    // Perform operations
    let result = x.matmul(y)?;

    println!("Result shape: {:?}", result.shape());
    Ok(())
}
```

### Custom GPU Kernels

```rust
use deepcl::{prelude::*, ir::{Instruction, Variable}};

#[cube] // DeepCL's GPU kernel language
fn custom_kernel(input: &Tensor<f32>) -> Tensor<f32> {
    let value = input[ABSOLUTE_POS];

    // Custom computation
    let result = value * 2.0 + 1.0;

    Tensor::new(result)
}
```

### Deep Learning Model

```rust
use deepcl::nn::{Linear, Module, Sequential};

#[derive(Module)]
pub struct MLP<B: Backend> {
    layers: Sequential<B, (Linear<B, 784, 256>, Linear<B, 256, 10>)>,
}

impl<B: Backend> MLP<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.layers.forward(input)
    }
}
```

## 🎯 Backends

DeepCL supports multiple execution backends for maximum flexibility:

### GPU Backends

| Backend | Platforms | Use Case |
|---------|-----------|----------|
| **CUDA** | NVIDIA GPUs | High-performance training & inference |
| **WGPU** | All GPUs + WebAssembly | Cross-platform GPU computing |
| **HIP** | AMD GPUs | ROCm ecosystem integration |
| **SPIR-V** | Vulkan-compatible GPUs | Open standard GPU computing |

### CPU Backends

| Backend | Platforms | Use Case |
|---------|-----------|----------|
| **CPU** | x86, ARM | Optimized CPU operations |
| **NdArray** | All platforms | NumPy-style operations |

## 🌐 WebAssembly Support

Run DeepCL models directly in web browsers:

```bash
# Build for web
wasm-pack build --target web --out-dir pkg

# Example: MNIST inference in browser
cd examples/mnist-inference-web
npm install
npm run serve
```

## 📚 Examples

Explore comprehensive examples covering various use cases:

| Example | Description |
|---------|-------------|
| [`mnist`](examples/mnist/) | Complete CNN training on MNIST |
| [`image-classification-web`](examples/image-classification-web/) | Web-based image classification |
| [`onnx-inference`](examples/onnx-inference/) | Import and run ONNX models |
| [`custom-wgpu-kernel`](examples/custom-wgpu-kernel/) | Write custom GPU shaders |
| [`text-classification`](examples/text-classification/) | NLP with transformers |
| [`wgan`](examples/wgan/) | Generative adversarial networks |

## 🛠️ Installation

### Cargo

```bash
# Basic installation
cargo add deepcl

# With CUDA support
cargo add deepcl --features cuda

# With WGPU support
cargo add deepcl --features wgpu

# Full installation
cargo add deepcl --features "cuda,wgpu,cpu,convolution,matmul"
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `cuda` | NVIDIA CUDA backend |
| `wgpu` | WebGPU/SPIR-V backend |
| `cpu` | CPU backend |
| `convolution` | CNN operations |
| `matmul` | Matrix multiplication |
| `attention` | Transformer attention |
| `stdlib` | Standard library functions |

## 🔬 Advanced Usage

### Custom Operations

```rust
use deepcl::ir::{Operation, Operator};

// Define custom tensor operation
#[derive(Debug)]
pub struct CustomOp {
    pub factor: f32,
}

impl Operation for CustomOp {
    fn args(&self) -> Vec<Variable> {
        // Implementation
    }
}
```

### Performance Optimization

```rust
use deepcl::prelude::*;

// Enable kernel fusion for better performance
#[cfg(feature = "fusion")]
use deepcl_fusion::Fusion;

type Backend = Fusion<Cuda>;
```

### Distributed Computing

```rust
use deepcl::backend::{RemoteBackend, Router};

// Multi-GPU setup
type MultiGpuBackend = Router<(Cuda, Cuda)>;

// Remote execution
type RemoteBackend = deepcl_remote::RemoteBackend;
```

## 🧪 Testing

```bash
# Run all tests
cargo test

# Run specific backend tests
cargo test --features cuda

# Run benchmarks
cargo bench

# Test WebAssembly build
wasm-pack test --node
```

## 📖 Documentation

- **[DeepCL Book](https://deepcode.dev/books/deepcl/)** - Comprehensive guide and tutorials
- **[API Documentation](https://docs.rs/deepcl)** - Complete API reference
- **[Examples](./examples/)** - Practical usage examples
- **[Contributing Guide](./CONTRIBUTING.md)** - How to contribute

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

1. **🐛 Bug Reports**: Use [GitHub Issues](https://github.com/kothagpt/deepcode/issues)
2. **💡 Feature Requests**: Open an issue with your ideas
3. **📝 Documentation**: Help improve our guides and examples
4. **🧪 Testing**: Add tests for new functionality
5. **⚡ Performance**: Optimize existing code or add new backends

### Development Setup

```bash
# Clone the repository
git clone https://github.com/kothagpt/deepcode.git
cd deepcode

# Install dependencies
cargo fetch

# Run tests
cargo test

# Build examples
cargo build --examples
```

## 🏢 Enterprise

DeepCL is designed for production use cases:

- **🔒 Memory Safety**: Rust's ownership system prevents memory bugs
- **⚡ Performance**: Competitive with CUDA/C++ implementations
- **🔧 Maintainable**: Strong typing and clear abstractions
- **🌐 Portable**: Deploy anywhere from servers to browsers

## 📄 License

DeepCL is distributed under the terms of both the MIT license and the Apache License (Version 2.0). See [LICENSE-APACHE](./LICENSE-APACHE) and [LICENSE-MIT](./LICENSE-MIT) for details.

## 🙏 Acknowledgments

DeepCL builds upon the excellent work of the Rust community and draws inspiration from frameworks like PyTorch, TensorFlow, and JAX. Special thanks to:

- The Rust community for the amazing ecosystem
- Contributors to WGPU, CUDA, and Vulkan ecosystems
- The original CubeCL project for the foundation

---

<div align="center">

**Made with ❤️ by the DeepCL community**

[⭐ Star us on GitHub](https://github.com/kothagpt/deepcode) • [💬 Join our Discord](https://discord.gg/uPEBbYYDB6) • [📖 Read the Book](https://deepcode.dev/books/deepcl/)

</div>