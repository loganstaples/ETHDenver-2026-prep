# helix-avm

The Approximate Virtual Machine (AVM) - a complete approximate execution engine.

## Purpose

Standalone execution engine that can be built, tested, and used independently. Provides:

- **Approximate Arithmetic** - Float, fixed-point, and quantized operations with error tracking
- **ML Operations** - Matrix multiply, convolution, activations, normalization
- **VM Engine** - Opcodes, instruction execution, memory model, gas metering
- **Neural Network Layers** - Linear, attention, MLP, transformer blocks
- **Automatic Differentiation** - Backward pass with error bound tracking
- **Witness Collection** - Capture intermediate values for ZK proof generation

## Key Insight

Instead of computing `y = Wx` exactly, compute `y ≈ Wx` where the approximation is bounded and the bound is tracked through the computation graph.

## Usage

```rust
use helix_avm::ops::matmul::approx_matmul;
use helix_avm::vm::executor::Executor;

let result = approx_matmul(&a, &b, precision);
// result.value() - the computed value
// result.error_bound() - guaranteed error bound
```
