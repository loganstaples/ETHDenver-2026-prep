# helix-prover

Proof generation orchestration layer.

## Purpose

Coordinates proof generation by taking execution traces from the AVM and producing ZK proofs.

## Modules

- **pipeline/** - Witness preparation, chunking, scheduling, aggregation
- **provers/** - Specialized provers for matmul, layers, gradients, training steps
- **cache/** - Proving key and proof caching
- **output/** - Final proof bundling with public inputs and metadata

## Usage

```rust
use helix_prover::pipeline::ProofPipeline;

let pipeline = ProofPipeline::new(config);
let proof = pipeline.prove(&execution_trace)?;
```
