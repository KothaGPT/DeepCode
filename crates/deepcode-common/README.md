# Deepcode Common

The `deepcode-common` package hosts code that _must_ be shared between deepcode packages (with `std` or
`no_std` enabled). No other code should be placed in this package unless unavoidable.

The package must build with `cargo build --no-default-features` as well.
