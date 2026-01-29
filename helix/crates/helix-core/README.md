# helix-core

Shared foundation types, traits, and utilities used across all HELIX crates.

## Purpose

Zero external dependencies on other helix crates. Provides:

- **BoundedValue<T>** - Values with tracked error bounds
- **ErrorMargin** - Absolute and relative error representations
- **Precision** - Precision levels (F32, F16, INT8, custom)
- **BoundedTensor** - N-dimensional bounded values
- **Traits** - `ApproximateOp`, `Provable`, `Serializable`

## Usage

```rust
use helix_core::types::{BoundedValue, ErrorMargin};

let value = BoundedValue::new(1.0, ErrorMargin::absolute(0.01));
```
