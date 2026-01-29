mod sc_client;
mod trainer;

use crate::sc_client::SCClient;
use crate::trainer::Trainer;
use tokio::time::{sleep, Duration};
use log::{info, error};

use ethers::types::U256;

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
    
    let trainer = Trainer::new();
    let model_id = U256::from(0); // Default/Mock model ID 0

    let args: Vec<String> = std::env::args().collect();
    let one_shot = args.contains(&"--one-shot".to_string());

    info!("Node initialized. Polling for rounds for Model ID {}... (One-shot: {})", model_id, one_shot);

    loop {
        match client.get_model_state(model_id).await {
            Ok((round, old_commitment_u256)) => {
                info!("Current Round: {}", round);
                
                // Logic to check if we should train would go here
                // For now, we simulate a training step every loop for demo
                
                // Convert old commitment to [u8; 32] for prover
                let mut old_commitment_bytes = [0u8; 32];
                old_commitment_u256.to_big_endian(&mut old_commitment_bytes);

                info!("Training...");
                match trainer.train_and_prove(&[], &[]) {
                    Ok((proof, commitment)) => {
                        info!("Proof generated! Submitting...");
                        // Use the correct update params
                        match client.submit_update(model_id, round, proof, commitment, old_commitment_bytes).await {
                            Ok(receipt) => {
                                info!("Update submitted: {:?}", receipt.transaction_hash);
                                if one_shot {
                                    info!("One-shot mode: Exiting after successful submission.");
                                    return Ok(());
                                }
                            }
                            Err(e) => error!("Submission failed: {}", e),
                        }
                    }
                    Err(e) => error!("Training failed: {}", e),
                }

            }
            Err(e) => error!("Error fetching round: {}", e),
        }

        if one_shot {
             // If we failed to submit in one-shot mode, we should probably retry or exit with error.
             // But for this simple test loop, let's just wait a bit and retry until success or manual timeout.
             // Actually, the loop logic above only returns on success.
             // So if we are here, it means we either errored or didn't attempt submission (e.g. valid checks failed)
             // For simplicity, let's just continue polling.
        }

        sleep(Duration::from_secs(10)).await;
    }
}
