mod sc_client;
mod trainer;

use crate::sc_client::SCClient;
use crate::trainer::Trainer;
use tokio::time::{sleep, Duration};
use log::{info, error};

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
    let model_id = [0u8; 32]; // Default/Mock model ID

    info!("Node initialized. Polling for rounds...");

    loop {
        match client.get_current_round().await {
            Ok(round) => {
                info!("Current Round: {}", round);
                
                // Logic to check if we should train would go here
                // For now, we simulate a training step every loop for demo
                
                info!("Training...");
                match trainer.train_and_prove(&[], &[]) {
                    Ok((proof, commitment)) => {
                        info!("Proof generated! Submitting...");
                        match client.submit_update(model_id, proof, commitment).await {
                            Ok(receipt) => info!("Update submitted: {:?}", receipt.transaction_hash),
                            Err(e) => error!("Submission failed: {}", e),
                        }
                    }
                    Err(e) => error!("Training failed: {}", e),
                }

            }
            Err(e) => error!("Error fetching round: {}", e),
        }

        sleep(Duration::from_secs(10)).await;
    }
}
