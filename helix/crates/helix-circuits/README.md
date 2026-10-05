# helix-circuits

Halo2 ZK circuits for verifiable ML training on BN254 (PSE fork, KZG commitment scheme).

## Architecture

```
helix-circuits/
├── approximate/     Error-bounded arithmetic (add, mul, matmul, accumulation)
├── cache/           Circuit structure and witness caching (LRU/LFU)
├── commitment/      Model weight hashing, state transition verification
├── gadgets/         Primitives: arithmetic gates, Poseidon hash, range checks, lookups
├── ivc.rs           Nova-style IVC with folding and Poseidon state commitments
├── lookup/          Activation tables: ReLU, GELU, sigmoid, softmax (+ gradients)
├── ml/              ML circuits: training step, transformer, attention, embedding, aggregation
├── params/          SRS generation, proving/verification key management
├── quantization/    INT4/INT8 quantization circuits with calibration
└── verifier/        EVM proof serialization, Solidity codegen, native verifier
```

## Primary Circuit: MLTrainingStepV2

Proves a single gradient step (forward pass, loss, backward pass, weight update) for a 2-layer MLP. Exposes 8 public inputs verified on-chain:

| PI | Name | Constraint |
|----|------|-----------|
| 0 | old_hash_lo | Direct (constrain_instance) |
| 1 | old_hash_hi | Direct (constrain_instance) |
| 2 | new_hash_lo | Direct (constrain_instance) |
| 3 | new_hash_hi | Direct (constrain_instance) |
| 4 | loss | Direct (in-circuit forward pass) |
| 5 | total_error | Direct (in-circuit error accumulation via s_error_acc gate) |
| 6 | step_number | Indirect (committed via PI[7] Poseidon hash) |
| 7 | error_checksum | In-circuit (3 Poseidon hashes) |

## Key Features

- **Error-bounded arithmetic**: Every operation tracks numerical error with copy-constrained gadgets
- **Transformer verification**: Layer norm, multi-head attention, FFN with GELU -- all constrained (not self-equality)
- **IVC/folding**: Nova-style incremental verification with real KZG proofs
- **Proof aggregation**: SHPLONK aggregation circuit with Fiat-Shamir challenges
- **EVM integration**: Proof serialization matching `Halo2Verifier.sol` format (320-byte proofs)
- **Activation lookups**: ReLU, LeakyReLU, ReLU6, PReLU, GELU, sigmoid, tanh, softmax

## Build & Test

```bash
cargo check -p helix-circuits          # Quick compile check
cargo test -p helix-circuits           # Run all 318 tests
cargo test -p helix-circuits -- ivc    # Run IVC tests only
cargo bench -p helix-circuits          # Run benchmarks
```

## Proof Format

- **Commitment scheme**: KZG on BN254
- **Opening scheme**: SHPLONK (2 opening proof points: W, W')
- **Transcript**: Blake2b (proving), Keccak256 (EVM verification)
- **Proof size**: 320 bytes (3 advice commits + 2 opening proofs, each 64 bytes)

## Known Limitations

1. **Merkle path verification** is native-only (SHA-256 out-of-circuit). The in-circuit `s_hash` gate is documented as a stub.
2. **SolidityGenerator** produces a simplified KZG check. Use `Halo2Verifier.sol` in `contracts/` for production.
3. **ReLU lookup range** covers [-128, 128). Weights must be quantized within this range.
4. **Model architecture** limited to 2-layer MLP in training_step_v2. Transformer circuit handles full blocks.
5. **Poseidon constants** use SHA-256-derived round constants (not standard Grain LFSR).

See [SECURITY.md](SECURITY.md) for the full security model.
