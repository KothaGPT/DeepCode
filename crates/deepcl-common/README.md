# DeepCL Common

The `deepcl-common` package hosts code that _must_ be shared between deepcl packages (with `std` or
`no_std` enabled). No other code should be placed in this package unless unavoidable.

The package must build with `cargo build --no-default-features` as well.
