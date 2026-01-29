# helix-node

Complete node software for participating in the HELIX training network.

## Purpose

Full node implementation supporting multiple roles:

- **Compute Node** - Execute training, generate proofs, submit gradients
- **Aggregator Node** - Collect gradients, verify proofs, run aggregation
- **Verifier Node** - Spot-check proofs, maintain network integrity

## Modules

- **roles/** - Role-specific logic (compute, aggregator, verifier)
- **training/** - Round management, data loading, checkpointing
- **network/** - libp2p peer discovery, gradient gossip, model sync
- **storage/** - IPFS, local storage, Ethereum integration
- **api/** - JSON-RPC server, Prometheus metrics

## Running a Node

```bash
helix-node --role compute --config node.toml
```
