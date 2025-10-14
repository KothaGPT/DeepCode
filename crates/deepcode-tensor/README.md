# Deepcode Tensor

> [Deepcode](https://github.com/kothagpt/deepcode) Tensor Library

[![Current Crates.io Version](https://img.shields.io/crates/v/deepcode-tensor.svg)](https://crates.io/crates/deepcode-tensor)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/kothagpt/deepcode-tensor/blob/master/README.md)

This library provides the core abstractions required to run tensor operations with Deepcode.

`Tensor`s are generic over the backend to allow users to perform operations using different
`Backend` implementations. Deepcode's tensors also support auto-differentiation thanks to the
`AutodiffBackend` trait.
