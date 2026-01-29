# helix-circuits

Halo2 ZK circuits that prove bounded computation.

## Purpose

Circuits that prove statements like:
- "The output matrix is within ε of the true matrix multiplication"
- "The computed gradient has L2 distance < δ from the true gradient"
- "Error accumulation across N operations remains within bound B"

## Modules

- **gadgets/** - Range checks, comparisons, field arithmetic, lookups
- **approximate/** - Bounded add, mul, matmul, error accumulation
- **ml/** - Linear layer, attention, gradient verification
- **commitment/** - Model and gradient commitments, state transitions
- **verifier/** - Native Rust and Solidity verifiers

## Why This Works

Proving bounded computation is exponentially cheaper than proving exact equality:
- Range proofs are simpler than equality proofs
- Approximate arithmetic allows cheaper circuit representations
- Error bounds can be composed algebraically
