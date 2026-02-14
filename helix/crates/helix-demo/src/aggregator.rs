//! Aggregator verification demo.
//!
//! Wires in the `AggregatorNode` from `helix-node` to collect worker proofs,
//! validate them, aggregate commitments, and verify that invalid proofs
//! are correctly rejected (triggering slashing).

use anyhow::Result;
use helix_node::network::messages::PeerId;
use helix_node::roles::aggregator::{AggregatorConfig, AggregatorNode};
use sha2::{Digest, Sha256};

use crate::worker::WorkerResult;

/// Results from aggregator verification.
#[derive(Debug)]
pub struct AggregatorResult {
    /// Number of proofs submitted to aggregator.
    pub proofs_submitted: usize,
    /// Number of proofs accepted by aggregator.
    pub proofs_accepted: usize,
    /// Number of invalid proofs rejected.
    pub invalid_rejected: usize,
}

/// Runs the aggregator verification demo.
///
/// Creates an aggregator node, feeds worker proofs as gradient shares,
/// aggregates them, and then tests rejection of an invalid proof.
pub async fn run_aggregator_demo(
    worker_results: &[WorkerResult],
) -> Result<AggregatorResult> {
    let aggregator_id = PeerId::from_string("aggregator-0");
    let config = AggregatorConfig {
        min_participants: 1,
        max_participants: worker_results.len() + 1,
        collection_timeout_secs: 60,
        generate_proof: false,
        max_error_bound: 1000.0,
        commitment_aggregation: helix_node::roles::aggregator::CommitmentAggregation::HashBased,
        byzantine_strategy: None, // No Byzantine filtering for demo
        min_stake_amount: 0,
    };

    let aggregator = AggregatorNode::new(aggregator_id.clone(), config);

    // Start a training round
    let model_hash = {
        let mut h = Sha256::new();
        h.update(b"helix-demo-model-v1");
        let result = h.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    };

    let params = helix_node::network::messages::TrainingParams {
        learning_rate: 0.01,
        batch_size: 1,
        local_epochs: 1,
        max_error_bound: 1000.0,
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        model_seed: 42,
        num_layers: 2,
        activation_type: 0,
    };

    let _round_msg = aggregator.start_round(model_hash, params).await;
    crate::display::info("Aggregator started training round");

    // Register workers as participants
    let mut worker_peer_ids = Vec::new();
    for (i, _) in worker_results.iter().enumerate() {
        let peer_id = PeerId::from_string(format!("worker-{}", i));
        worker_peer_ids.push(peer_id.clone());

        let current_round = 1u64;
        if let Some(_response) = aggregator
            .handle_participate_request(peer_id, current_round)
            .await
        {
            // Participation registered
        }
    }

    // Transition to collection phase
    aggregator.start_collection().await;
    crate::display::info("Aggregator collecting gradient proofs...");

    // Submit valid proofs from workers
    let mut proofs_submitted = 0;
    let mut proofs_accepted = 0;

    for (i, result) in worker_results.iter().enumerate() {
        for bundle in &result.evm_bundles {
            let commitment = compute_proof_commitment(&bundle.evm_proof);
            let peer_id = worker_peer_ids[i].clone();

            let accepted = aggregator
                .handle_gradient_share(
                    peer_id,
                    1, // round_id
                    commitment,
                    0.001, // error_bound
                    bundle.evm_proof.clone(),
                )
                .await;

            proofs_submitted += 1;
            if accepted {
                proofs_accepted += 1;
            }
        }
    }

    crate::display::success(&format!(
        "Aggregator accepted {}/{} valid proofs",
        proofs_accepted, proofs_submitted,
    ));

    // Attempt aggregation
    let _aggregation_complete = if let Some(agg_result) = aggregator.aggregate().await {
        crate::display::success(&format!(
            "Aggregation complete: {} participants, commitment={:?}",
            agg_result.num_participants,
            &agg_result.commitment[..8],
        ));
        true
    } else {
        crate::display::info("Aggregation pending (waiting for more participants)");
        // Even without full aggregation, the proof collection worked
        true
    };

    // Test invalid proof rejection
    let invalid_rejected = test_invalid_proof_rejection(&aggregator, &worker_peer_ids).await;

    Ok(AggregatorResult {
        proofs_submitted,
        proofs_accepted,
        invalid_rejected,
    })
}

/// Computes a SHA-256 commitment for a proof (for aggregator tracking).
fn compute_proof_commitment(proof_bytes: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(proof_bytes);
    let result = h.finalize();
    let mut commitment = [0u8; 32];
    commitment.copy_from_slice(&result);
    commitment
}

/// Tests that the aggregator rejects proofs from unregistered peers.
async fn test_invalid_proof_rejection(
    aggregator: &AggregatorNode,
    _worker_peer_ids: &[PeerId],
) -> usize {
    let mut rejected = 0;

    // Test 1: Proof from unregistered peer should be rejected
    let unknown_peer = PeerId::from_string("adversary-0");
    let fake_commitment = [0xDE; 32];
    let fake_proof = vec![0xAB; 320];

    let accepted = aggregator
        .handle_gradient_share(unknown_peer, 1, fake_commitment, 0.001, fake_proof)
        .await;

    if !accepted {
        crate::display::success("Aggregator rejected proof from unregistered peer");
        rejected += 1;
    }

    // Test 2: Proof for wrong round should be rejected
    if let Some(peer_id) = _worker_peer_ids.first() {
        let accepted = aggregator
            .handle_gradient_share(
                peer_id.clone(),
                999, // wrong round
                [0x00; 32],
                0.001,
                vec![0x00; 320],
            )
            .await;

        if !accepted {
            crate::display::success("Aggregator rejected proof for wrong round");
            rejected += 1;
        }
    }

    rejected
}
