# Deepcode NdArray

> [Deepcode](https://github.com/kothagpt/deepcode) ndarray backend

[![Current Crates.io Version](https://img.shields.io/crates/v/deepcode-ndarray.svg)](https://crates.io/crates/deepcode-ndarray)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/kothagpt/deepcode-ndarray/blob/master/README.md)

## Feature Flags

This crate can be used without the standard library (`#![no_std]`) with `alloc` by disabling the
default `std` feature.

The following flags support various BLAS options:

- `blas-accelerate` - Accelerate framework (macOS only)
- `blas-netlib` - Netlib
- `blas-openblas` - OpenBLAS static linked
- `blas-openblas-system` - OpenBLAS from the system

Note: under the `no_std` mode, the seed is fixed if the seed is not
initialized by by `Backend::seed` method.

### Platform Support

| Option     | CPU | GPU | Linux | MacOS | Windows | Android | iOS | WASM |
| :--------- | :-: | :-: | :---: | :---: | :-----: | :-----: | :-: | :--: |
| Pure Rust  | Yes | No  |  Yes  |  Yes  |   Yes   |   Yes   | Yes | Yes  |
| Accelerate | Yes | No  |  No   |  Yes  |   No    |   No    | Yes |  No  |
| Netlib     | Yes | No  |  Yes  |  Yes  |   Yes   |   No    | No  |  No  |
| Openblas   | Yes | No  |  Yes  |  Yes  |   Yes   |   Yes   | Yes |  No  |
