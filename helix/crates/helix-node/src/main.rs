mod sc_client;

use crate::sc_client::{SCClient, TrainingProofInputs};
use helix_node::trainer::Trainer;
use helix_prover::halo2curves::bn256::Fr;
use helix_prover::halo2curves::ff::PrimeField;
use tokio::time::{sleep, Duration};
use log::{info, error};

use ethers::types::U256;

/// Converts a Fr field element to U256 for smart contract calls.
fn fr_to_u256(fr: &Fr) -> U256 {
    let repr = fr.to_repr();
    U256::from_little_endian(repr.as_ref())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();
    info!("Starting Helix Training Node...");

    let client = match SCClient::new().await {
        Ok(c) => c,
        Err(e) => {
            error!("Failed to initialize SCClient: {}", e);
            return Ok(());
        }
    };

    // Real 2-layer MLP: 4 inputs, 8 hidden, 2 outputs.
    let mut trainer = Trainer::new(4, 8, 2, 0.01, 42);
    let model_id: u64 = 0;

    let args: Vec<String> = std::env::args().collect();
    let one_shot = args.contains(&"--one-shot".to_string());

    info!(
        "Node initialized (model: {}×{}×{}, {} params). Polling for rounds...",
        trainer.model().d_in,
        trainer.model().d_hid,
        trainer.model().d_out,
        trainer.model().num_params(),
    );

    // Demo input/target (would come from DataLoader in production).
    let x = vec![1.0, 0.5, -0.3, 0.8];
    let target = vec![1.0, 0.0];

    loop {
        match client.get_model_state(model_id).await {
            Ok(state) => {
                info!("Current Round: {}, Active: {}", state.current_round, state.active);

                if !state.active {
                    info!("Model not active, waiting...");
                    sleep(Duration::from_secs(10)).await;
                    continue;
                }

                info!("Training step {}...", trainer.step_count() + 1);
                match trainer.train_step(&x, &target) {
                    Ok(result) => {
                        info!(
                            "Proof generated ({} bytes, loss={:.6})! Submitting...",
                            result.proof.len(),
                            result.loss
                        );

                        // Convert public inputs from Fr to U256
                        let public_inputs: Vec<U256> = result.public_inputs
                            .iter()
                            .map(fr_to_u256)
                            .collect();

                        match client
                            .submit_proof_raw(
                                model_id,
                                state.current_round,
                                result.proof,
                                public_inputs,
                            )
                            .await
                        {
                            Ok(receipt) => {
                                info!(
                                    "Update submitted: {:?}",
                                    receipt.transaction_hash
                                );
                                if one_shot {
                                    info!("One-shot mode: exiting.");
                                    return Ok(());
                                }
                            }
                            Err(e) => error!("Submission failed: {}", e),
                        }
                    }
                    Err(e) => error!("Training failed: {}", e),
                }
            }
            Err(e) => error!("Error fetching model state: {}", e),
        }

        sleep(Duration::from_secs(10)).await;
    }
}
