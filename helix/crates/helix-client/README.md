# helix-client

CLI and client library for interacting with the HELIX network.

## Purpose

User-facing tools for training management and network interaction.

## Commands

```bash
helix init      # Initialize new training run
helix join      # Join existing training as compute node
helix status    # Check training status
helix query     # Query trained model
helix export    # Export model weights
```

## Library Usage

```rust
use helix_client::Client;

let client = Client::new(config)?;
let status = client.training_status().await?;
```
