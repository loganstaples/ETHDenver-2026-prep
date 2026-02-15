# HELIX Implementation Stages

These are standalone prompts to give to AI agents in order. Each agent gets full context and implements one stage. Execute them sequentially — each builds on the previous.

Read SYSTEM_DESIGN.md in this same directory first for the full architecture.

IMPORTANT: ZK proofs are an OPTIONAL feature in HELIX. The core system runs entirely on MPC + SPDZ MACs + multi-party attestations. ZK is a user-configurable add-on for extra external verifiability. Stages 1-8 build the core system WITHOUT any ZK dependency. Stage 9 adds ZK as an optional feature. Stage 10 is the demo and polish.

---

## Stage 1: SPDZ MAC Integration into MPC Training Loop

You are working on HELIX's MPC layer (helix-mpc crate). Read helix/docs/SYSTEM_DESIGN.md for the full architecture.

HELIX uses SPDZ-style MPC with information-theoretic MACs as the PRIMARY mechanism for detecting cheaters during training. The SPDZ MAC infrastructure exists in the codebase (implemented in a previous round of work — search for "spdz", "mac", "Mac" in helix-mpc/src/). But it is NOT wired into the actual training computation loop as a per-step verification mechanism.

What to build:
1. Find the existing SPDZ MAC code in helix-mpc. Understand how MAC tags are generated and verified. The MAC system uses BN254 Fr field elements with a global MAC key alpha that is secret-shared among parties.

2. Wire MAC verification into the MPC training loop. In helix-mpc/src/training/ (the MPC trainer), after each training step (or configurable batch of K steps), run the SPDZ MAC batch verification protocol: each party computes sigma_i = gamma_i - alpha_i * x for a random linear combination of recently computed values. Parties exchange commitments to sigma values, then reveal and verify sum(sigma_i) = 0.

3. Implement pairwise cheater identification. When sum(sigma_i) != 0 (MAC failure detected), run the identification protocol: for each pair of parties (i, j), do a 2-party MAC cross-check. The cheater will fail cross-checks with all honest parties. Honest parties will pass their mutual cross-checks. Build a function identify_cheater(parties, sigma_values) -> Result<PartyId, IdentificationError> that returns the identified cheater's ID.

4. Build a MACFailureReport struct containing: session_id, step_number, identified_cheater (party ID), sigma_values from all parties, commitments, and the identifying evidence. This report will be signed and submitted on-chain in a later stage.

5. Implement training halt and rollback. When MAC failure is detected: immediately stop the training loop, run identification, produce the failure report, and return the last known-good state (from the most recent checkpoint).

6. Write tests:
   - test_honest_training_passes_mac: 3 parties do 10 honest training steps, MAC verification passes every step.
   - test_cheater_detected: 3 parties, party 2 corrupts their gradient share at step 5, MAC verification fails at step 5.
   - test_cheater_identified: Same as above but verify that identify_cheater() correctly returns party 2.
   - test_training_continues_after_removal: 3 parties, party 2 cheats, is removed, training continues with parties 1 and 3.
   - test_mac_check_frequency: Verify that setting K=5 (check every 5 steps) allows a cheater to corrupt up to 5 steps but is caught at the check.

Read helix-mpc/src/training/trainer.rs for the current MPC training loop. Read helix-mpc/src/spdz/ or search for MAC-related code. Read helix-mpc/src/session/ for session management. All existing 540 helix-mpc tests must continue to pass.

---

## Stage 2: Distributed Beaver Triple Generation (Remove Trusted Dealer)

You are working on HELIX's MPC layer (helix-mpc crate). Read helix/docs/SYSTEM_DESIGN.md for the full architecture.

HELIX currently uses a TrustedDealer for Beaver triple generation. The dealer is a single entity that generates all correlated randomness — this is a trusted party that we want to eliminate. The system should have NO trusted party.

What to build:
1. Find the current TrustedDealer implementation in helix-mpc (search for "TrustedDealer", "trusted_dealer", "dealer"). Understand how it generates Beaver triples (a, b, c) where c = a*b, and how it distributes shares.

2. Implement OT-based distributed triple generation. The MASCOT protocol works as follows: for each pair of parties (i, j), they run oblivious transfer to generate correlated random values. Party i picks random a_i, party j picks random b_j. Via OT, they jointly compute shares of a_i * b_j without either seeing the other's value. Combining all pairwise products gives shares of c = a*b. HELIX already has x25519-based OT in helix-mpc/src/comparison/ or helix-mpc/src/ot/ — use it as the OT primitive.

3. Build a DistributedTripleGenerator that takes a set of connected MPC parties (via the existing transport layer) and generates Beaver triples collaboratively. It should produce the same triple format that the TrustedDealer produces so the rest of the system works unchanged.

4. Add a BeaverTriplePool that pre-generates a configurable number of triples before training starts. The pool should support: generate(num_triples) -> fills the pool, take() -> returns the next triple (or errors if pool is empty), remaining() -> count of unused triples. Pre-generation can run in parallel across party pairs.

5. Replace TrustedDealer usage in the MPC training pipeline with DistributedTripleGenerator. The TrustedDealer code should still exist (useful for testing and as a fast fallback) but the production path should use distributed generation.

6. Write tests:
   - test_distributed_triple_correctness: Generate 1000 triples distributedly with 3 parties, verify a*b = c for each.
   - test_no_party_sees_full_triple: Verify that each party only has shares, not the full (a, b, c).
   - test_distributed_vs_dealer_equivalence: Training with distributed triples produces the same results as training with dealer triples (same initial weights, same data, same randomness seed for shares).
   - test_pool_exhaustion: Verify that running out of triples produces a clear error, not silent corruption.

Read helix-mpc/src/beaver/ for existing Beaver triple infrastructure. Read helix-mpc/src/ot/ or helix-mpc/src/comparison/ for the OT implementation. All existing tests must continue to pass.

---

## Stage 3: Share Distribution, Reconstruction, and Pedersen Commitments

You are working on HELIX's client and MPC layers. Read helix/docs/SYSTEM_DESIGN.md for the full architecture.

The model owner needs to split initial weights into secret shares, distribute them to workers, and reconstruct the trained weights at the end. At checkpoints, workers need to jointly compute Pedersen commitments to the weights WITHOUT reconstructing them.

What to build:
1. In helix-client, build a ShareDistributor module. Given a model's weight tensor (Vec<f64> or similar) and N worker public keys (x25519): (a) Convert weights to BN254 Fr field elements (use the existing Fr encoding in helix-core or helix-mpc). (b) Generate N-1 random Fr shares, compute the Nth share as the remainder. (c) Encrypt each share to the corresponding worker's x25519 public key using DH key agreement + AES-GCM (the OT module in helix-mpc already does this exact pattern — reuse it). (d) Return the encrypted shares ready for network transmission. Also compute and return the initial Pedersen commitment: C0 = g^W * h^r where g and h are fixed public generators on BN254 G1.

2. In helix-mpc, build a ShareReceiver that accepts an encrypted share, decrypts with the worker's x25519 private key, and initializes the worker's MPC state with their share.

3. Build a PedersenCommitment module (in helix-mpc or helix-core). Implement: (a) commit(value_share, blinding_share) -> G1 point. (b) combine_commitments(commitments: Vec<G1>) -> G1 (multiply points). (c) verify_commitment(full_value, full_blinding, commitment) -> bool. Workers compute their commitment share independently. The combined commitment is computed by anyone (just multiply the published G1 points). Use the BN254 G1 curve from halo2curves (already in the dependency tree).

4. Build the checkpoint commitment flow: each worker publishes their Pedersen commitment share. Any party (or the aggregator) combines them to get the full commitment. Nobody reconstructed the weights. This commitment goes on-chain as the checkpoint.

5. Build the reconstruction flow for training completion: each worker encrypts their final weight share to the owner's public key and sends it. The owner's client collects all shares, decrypts, sums them to reconstruct the full weights, and converts back to f64. Verify the reconstruction matches the last checkpoint commitment.

6. Write tests:
   - test_share_roundtrip: Owner splits weights into 3 shares, workers receive shares, workers send shares back, owner reconstructs, verify reconstruction matches original.
   - test_pedersen_commitment_verification: Split weights, compute per-party commitment shares, combine, verify against full commitment.
   - test_commitment_without_reconstruction: 3 workers each compute their commitment share, combine them, verify the combined commitment matches — without any party seeing the full weights.
   - test_encrypted_share_transfer: Encrypt a share to a worker's key, have the worker decrypt, verify correctness.
   - test_large_model: Run the full flow with a 25K-param model (784x32x10 dimensions).

Read helix-mpc/src/ot/ for x25519 + AES-GCM patterns. Read helix-mpc/src/spdz/ for Fr field element handling. Read helix-core/src/types/ for tensor types.

---

## Stage 4: Node P2P Networking and MPC Session Coordination

You are working on HELIX's node layer (helix-node crate). Read helix/docs/SYSTEM_DESIGN.md for the full architecture.

Nodes need to discover each other, establish connections, and coordinate MPC training sessions. The wire protocol (bincode framing, CRC32 checksums, HELX magic bytes) and message types (ed25519 signed, replay-protected) already exist. The MPC layer has a NodeTransport bridge (helix-mpc/src/session/node_transport.rs) designed to run MPC over node P2P connections. But nothing is actually wired together.

What to build:
1. A connection manager in helix-node that establishes and maintains TCP connections using the existing wire protocol. An aggregator/coordinator node listens on a port. Worker nodes connect to the aggregator by address (seed node model — no complex discovery needed for the hackathon). Implement handshake: worker sends its role, ed25519 public key, x25519 public key, and Ethereum address. Aggregator maintains a peer registry of connected workers.

2. Heartbeat and connection health: workers send periodic pings (every 5 seconds). Aggregator removes workers that miss 3 consecutive heartbeats. Automatic reconnection on the worker side.

3. MPC session setup: when a training job starts, the aggregator creates an MPC session for the registered workers. It assigns party IDs, distributes session parameters (model architecture, hyperparameters, checkpoint frequency), and signals workers to begin. The MPC message transport is bridged through the node P2P connections using the existing NodeTransport in helix-mpc/src/session/node_transport.rs — wire this bridge into the node's connection manager.

4. Message routing: MPC messages between workers are routed through the P2P layer. Each MPC message is tagged with (session_id, sender_party_id, recipient_party_id). The aggregator can relay messages or workers can communicate directly (prefer direct for efficiency). Use the existing TaggedMessage and MPCMessageRouter from helix-mpc/src/session/node_transport.rs.

5. Session lifecycle management: the aggregator tracks session state (setup, training, checkpointing, completed, failed). It coordinates phase transitions: "all workers ready" -> start training, "checkpoint interval reached" -> coordinate checkpoint, "MAC failure" -> coordinate identification and recovery.

6. Write tests:
   - test_peer_connection: Boot aggregator + 2 workers in separate tokio tasks with different ports, verify they connect and handshake.
   - test_mpc_message_routing: Send an MPC message from worker 1 to worker 2 via the P2P layer, verify it arrives correctly.
   - test_heartbeat_timeout: Connect a worker, stop sending heartbeats, verify the aggregator removes it after timeout.
   - test_session_setup: Aggregator creates a session, assigns party IDs to 3 connected workers, verify all workers receive session parameters.

Read helix-node/src/network/wire.rs (wire protocol), helix-node/src/network/messages.rs (message types), helix-node/src/roles/ (role definitions), helix-mpc/src/session/node_transport.rs (MPC-node bridge). All existing 840 helix-node tests must continue to pass.

---

## Stage 5: MPC Training at MNIST Scale

You are working on HELIX's MPC training pipeline. Read helix/docs/SYSTEM_DESIGN.md for the full architecture.

Previous stages built: SPDZ MAC verification in the training loop (Stage 1), distributed Beaver triple generation (Stage 2), share distribution (Stage 3), and node networking (Stage 4). Now make the MPC training pipeline work with a real MNIST model.

What to build:
1. An end-to-end MPC training test with a 784x32x10 MLP on MNIST data. Load MNIST images (28x28 = 784 inputs, 10 output classes). Split model weights into 3 shares. Run 100-500 training steps via MPC with SPDZ MAC verification every step. Verify: loss decreases monotonically (or near-monotonically), final accuracy > 90% after 500 steps.

2. Performance optimization for MNIST scale. The key bottleneck is Beaver triple consumption: each matrix multiply W*x where W is (m x n) requires m*n Beaver triples. Layer 1 (784x32) = 25,088 triples per forward pass just for that multiply. Optimize: (a) Batch all multiplications in a layer into a single communication round. (b) Pre-generate all needed triples before training starts. (c) Profile the per-step time and identify bottlenecks.

3. Garbled circuit ReLU at scale: 32 neurons means 32 garbled circuit evaluations per step. Profile this and optimize. The garbled circuits should be batched — generate all 32 ReLU circuits at once, transfer labels via batch OT.

4. Loss revelation: after each step, workers jointly reveal the loss value (all parties open their loss shares). The loss is a single scalar — revealing it doesn't leak model information. Use the revealed loss for progress monitoring and convergence detection.

5. Accuracy evaluation: periodically (every 50-100 steps), run inference on a test batch via MPC and jointly reveal the accuracy (number of correct predictions). This provides meaningful progress metrics without leaking the model.

6. Write tests:
   - test_mnist_mpc_training_converges: 3-party MPC training on MNIST for 200 steps, verify loss < 0.5 and accuracy > 85%.
   - test_mnist_mpc_matches_native: Compare MPC training result to native (non-MPC) training with same initial weights and data. The results should match within the error tolerance of the fixed-point field arithmetic.
   - test_mnist_mpc_with_cheater: 3-party training, party 2 sends random gradient shares at step 50, verify MAC detection catches it immediately.
   - test_mnist_training_time: Verify 100 steps complete in under 60 seconds (targeting ~500ms per step).

Read helix-mpc/src/training/ for the MPC trainer. Read helix-avm/src/ for the native training engine (for comparison). Read helix-core/src/data/ for data loading utilities. MNIST data can be downloaded or generated synthetically for testing.

---

## Stage 6: Smart Contract Updates for MPC-Primary Architecture

You are working on HELIX's smart contracts. Read helix/docs/SYSTEM_DESIGN.md for the full architecture.

The contracts need to support the new MPC-primary architecture: multi-party attestation checkpoints (instead of ZK proof checkpoints), MAC failure reporting with individual slashing, and simplified payment flow.

What to build:
1. Create HelixCoordinatorV4.sol (or modify V3) with the following functions:
   - registerTrainingJob(bytes32 architectureHash, uint256 checkpointFreq, uint256 numRounds, uint256 paymentAmount) payable: Owner registers a job, deposits payment. Returns jobId.
   - stakeAndJoin(uint256 jobId): Worker stakes tokens and joins a job. Records worker address and stake.
   - submitCheckpoint(uint256 jobId, uint256 stepNumber, bytes32 weightCommitment, uint256 loss, bytes[] signatures): Accepts a checkpoint attestation signed by all active workers. Verifies all signatures are from registered workers for this job. Stores the commitment and step number. Emits CheckpointSubmitted event.
   - reportMACFailure(uint256 jobId, uint256 stepNumber, address cheater, bytes evidence, bytes[] reporterSignatures): Accepts a MAC failure report signed by a majority of workers. Verifies: (a) reporters are registered workers, (b) reporters constitute a majority, (c) cheater is a registered worker. Slashes the cheater's full stake. Distributes 10% as bounty to reporters, rest to treasury. Removes cheater from the active worker set. Emits WorkerSlashed event.
   - completeTraining(uint256 jobId, bytes32 finalCommitment, bytes[] signatures): Final checkpoint. Releases payment to all active (non-slashed) workers proportionally.
   - submitCheckpointWithProof(uint256 jobId, uint256 stepNumber, bytes32 weightCommitment, uint256 loss, bytes proof, uint256[] publicInputs): Optional. Verifies a ZK proof of the state transition via the existing Halo2Verifier. This is for users who enable ZK verification (--zk-proofs flag). Not required for the default flow.

2. Update the Staking.sol contract (or integrate staking directly into V4) so that: workers can stake per-job (not just globally), slashing is per-job, and stake is returned after job completion + cooldown.

3. Update Deploy.s.sol to deploy the new V4 coordinator. Keep V2 and V3 deployments available for backwards compatibility.

4. Write Solidity tests:
   - test_full_job_lifecycle: Register job, 3 workers stake and join, submit 5 checkpoints with valid signatures, complete training, verify payment distributed.
   - test_mac_failure_slashing: 3 workers joined, submit MAC failure report against worker 2 signed by workers 1 and 3 (majority), verify worker 2's stake is slashed.
   - test_insufficient_signatures_rejected: Submit checkpoint with only 1 of 3 signatures, verify rejection.
   - test_fake_reporter_rejected: Submit MAC failure report with a signature from a non-registered worker, verify rejection.
   - test_payment_proportional: 3 workers, one slashed at step 50, remaining two complete 100 steps, verify payment is proportional to participation.
   - test_double_slash_prevention: Submit same MAC failure report twice, verify second is rejected.

Read helix/contracts/src/core/HelixCoordinatorV2.sol and V3 for existing patterns. Read helix/contracts/src/token/Staking.sol, Rewards.sol. Read helix/contracts/test/ for test patterns. All existing contract tests (639 passing) must continue to pass — add the new contract alongside existing ones, don't break them.

---

## Stage 7: End-to-End Integration (Wiring Everything Together)

You are working on HELIX's integration layer. Read helix/docs/SYSTEM_DESIGN.md for the full architecture.

Previous stages built all the individual pieces: SPDZ MAC verification (Stage 1), distributed triple generation (Stage 2), share distribution (Stage 3), node P2P (Stage 4), MNIST-scale MPC training (Stage 5), and updated contracts (Stage 6). Now wire everything together into a working end-to-end system.

What to build:
1. The owner flow in helix-client: a single function (or CLI command) that takes model weights, worker endpoints, and training config, then: (a) connects to the smart contract, registers the training job, (b) waits for workers to stake and join, (c) splits weights into shares, encrypts and distributes to workers via P2P, (d) signals the aggregator to begin training, (e) monitors progress by polling checkpoint attestations from the contract, (f) at completion, collects encrypted shares from workers, reconstructs final weights.

2. The worker flow in helix-node: when a worker node starts and is assigned to a job, it: (a) receives encrypted weight share, decrypts, (b) participates in distributed Beaver triple pre-generation, (c) runs MPC training steps with SPDZ MAC verification, (d) at checkpoint intervals, computes Pedersen commitment share, signs attestation, (e) if MAC failure: participates in pairwise identification, signs blame report, (f) at completion, encrypts final share to owner and sends it.

3. The aggregator flow in helix-node: the aggregator orchestrates but does NOT participate in MPC computation. It: (a) manages worker connections and session setup, (b) relays MPC messages between workers (or facilitates direct connections), (c) coordinates phase transitions (setup -> training -> checkpoint -> ...), (d) collects attestation signatures and submits checkpoints on-chain, (e) handles MAC failure reports: collects blame signatures, submits on-chain, coordinates worker removal and training resumption.

4. Wire the flows together into a single integration test that does the full E2E: boot 1 aggregator + 3 workers (as separate tokio tasks with real TCP connections on localhost), register a job on Anvil, distribute shares for a small model (e.g., 4x4x2 for speed), run 20 MPC training steps with per-step MAC verification, submit 2 checkpoints on-chain, verify checkpoint attestations are accepted, complete training, reconstruct weights, verify they match expected training result.

5. A second integration test with cheater detection: same setup but worker 2 corrupts gradient shares at step 10. Verify: MAC check fails at step 10, worker 2 is identified, blame report submitted on-chain, worker 2 is slashed, training continues with workers 1 and 3 from last checkpoint.

Read helix-client/src/orchestration.rs for existing orchestration patterns. Read helix-node/src/roles/ for role implementations. Read helix/tests/integration/ for existing test patterns and helpers. Read helix/tests/common/ for shared test infrastructure (Anvil helpers, contract deployment).

---

## Stage 8: Error Containment, Recovery, and Final Polish

You are working on HELIX's reliability and resilience. Read helix/docs/SYSTEM_DESIGN.md for the full architecture.

The system needs to handle real-world failure modes without letting errors propagate or data get lost.

What to build:
1. Error containment: When a MAC verification fails, training MUST halt immediately before the next step executes. No corrupted values should flow into subsequent computation. Verify this with a test: inject corruption at step N, verify that step N+1 never runs, and rollback begins immediately.

2. Checkpoint rollback: After removing a cheater, training must resume from the last verified checkpoint, not from the corrupted state. The remaining honest workers reload their shares from the checkpoint state. Build the share redistribution protocol: when a party is removed, the remaining parties need to compensate for the missing share. Since the checkpoint commitment is known, and each honest party knows their own share, they can compute what the removed party's share was (via the commitment) and redistribute it. Alternatively, the owner can re-split and redistribute fresh shares from the checkpoint weights.

3. Worker disconnection handling: if a worker's heartbeat stops (network failure, crash), the aggregator waits a grace period (30 seconds), then removes the worker. If honest majority still holds, training continues. If not, training pauses and the owner is notified. The disconnected worker's stake enters a cooldown — if they reconnect within the cooldown, they can rejoin without re-staking.

4. Beaver triple exhaustion: if the pre-generated triple pool runs out mid-training, pause training, generate more triples, then resume. This should be transparent — just a brief pause, not a failure.

5. Chain transaction failures: if a checkpoint submission fails (gas too low, RPC down), retry with exponential backoff up to 3 times. If all retries fail, log the failed checkpoint locally and continue training. Submit the checkpoint retroactively when the chain becomes available.

6. Graceful shutdown: on SIGTERM/SIGINT, the demo should: complete the current MPC round (don't leave in a half-computed state), trigger an emergency checkpoint, save all worker states locally, and exit cleanly.

7. Write comprehensive integration tests:
   - test_error_containment: Inject corruption at step N, verify step N+1 never executes and rollback begins immediately.
   - test_checkpoint_rollback_integrity: Cheater corrupts 10 steps before detection, verify rollback to last checkpoint and no corrupted values persist.
   - test_worker_disconnect_recovery: Worker drops at step 75, training continues with remaining workers, completes successfully.
   - test_triple_exhaustion_recovery: Start with only enough triples for 50 steps, verify pause-and-regenerate at step 50, training completes.
   - test_graceful_shutdown: Send SIGTERM during training, verify clean state save and resumability.

Read helix-mpc/src/session/health.rs for existing health monitoring. Read helix-mpc/src/session/checkpoint.rs for existing checkpoint infrastructure. Read helix-node/src/roles/ for node role management.

---

## Stage 9: Optional ZK Proof Feature

You are working on HELIX's optional ZK proof layer. Read helix/docs/SYSTEM_DESIGN.md for the full architecture.

IMPORTANT: ZK proofs are NOT required for HELIX's core operation. The MPC + SPDZ MAC system (Stages 1-8) provides complete security. ZK is an OPTIONAL add-on that provides external verifiability — a third party who wasn't part of the MPC can verify a state transition without trusting any of the workers. This is enabled by the user with `--zk-proofs` and `--zk-checkpoint-freq N` flags. Default is OFF.

What to build:
1. A lightweight StateTransitionCircuit in helix-circuits. This circuit proves: "given old weight commitment C_old and new weight commitment C_new, the transition is valid." It does NOT prove the full forward/backward pass (that would be 1000x overhead). Instead it proves: (a) C_old is a valid Pedersen commitment to old weights. (b) C_new = C_old + delta where delta is the weight update. (c) The weight update delta matches a claimed gradient (Poseidon hash of the gradient is committed). (d) The error bound of the transition is within tolerance. This is a simple circuit — matrix subtraction and commitment verification, not full ML.

2. A CheckpointProver in helix-prover that wraps the StateTransitionCircuit. It takes: old commitment, new commitment, weight delta (available to the model owner who reconstructed weights), and produces a KZG proof compatible with the existing Halo2Verifier contract. Use the existing PSE halo2 pipeline (ParamsKZG, Keccak256Transcript from halo2_solidity_verifier, SHPLONK prover). The proof format should be 1856 bytes matching the existing EVM verifier.

3. Integration with the training pipeline: when the user enables `--zk-proofs`, after each N-th checkpoint (where N is set by `--zk-checkpoint-freq`, default every checkpoint), the model owner generates a ZK proof of the state transition for that checkpoint. The proof is submitted on-chain via submitCheckpointWithProof() (from Stage 6). If the ZK proof fails verification on-chain, the checkpoint is still valid via attestation — the ZK failure just triggers a warning event.

4. CLI integration: add `--zk-proofs` and `--zk-checkpoint-freq N` flags to the demo binary and client. When `--zk-proofs` is enabled, show proof generation progress and verification results. When disabled (default), ZK is completely absent from the pipeline — no proof generation, no circuit setup, no SRS loading.

5. Write tests:
   - test_state_transition_circuit: Create a valid state transition, prove it, verify it passes.
   - test_invalid_transition_rejected: Create an invalid transition (wrong delta), prove it, verify MockProver rejects it.
   - test_evm_proof_format: Generate a proof, verify it's 1856 bytes and matches the EVM verifier format.
   - test_zk_disabled_no_overhead: Run training with ZK disabled, verify no circuit/prover code is invoked.
   - test_zk_checkpoint_frequency: Enable ZK with freq=3, run 10 checkpoints, verify exactly 3 ZK proofs generated (at checkpoints 3, 6, 9).

Read helix-circuits/src/ for existing circuit patterns. Read helix-prover/src/pipeline.rs for the existing KZG proof pipeline. Read helix/contracts/src/verification/Halo2Verifier.sol for the EVM verifier. All existing circuit and prover tests must continue to pass.

---

## Stage 10: Demo Binary and MNIST Showcase

You are working on HELIX's demo binary. Read helix/docs/SYSTEM_DESIGN.md for the full architecture.

Build the demo binary that will be run live at ETHDenver. It should demonstrate the full system in 3-5 minutes with a real MNIST model achieving real accuracy.

What to build:
1. Rebuild helix-demo (at helix/crates/helix-demo/) as a single binary that orchestrates the full demo flow. CLI: `helix-demo [--steps N] [--workers N] [--checkpoint-freq N] [--simulate-cheater] [--zk-proofs] [--zk-checkpoint-freq N] [--verbose]`

2. Default flow (no flags, ZK OFF): (a) Start Anvil in the background. (b) Deploy HelixCoordinatorV4 + HelixToken + Staking contracts. (c) Boot 3 worker nodes as background tokio tasks. (d) Register MNIST training job on-chain. Workers auto-stake and join. (e) Split initial MNIST model (784x32x10) into 3 shares, distribute encrypted. (f) Pre-generate Beaver triples (show progress bar). (g) Run 500 MPC training steps with per-step MAC verification. (h) Show live progress: step number, loss, accuracy, MACs verified, checkpoints submitted. (i) After every 100 steps: compute checkpoint commitment, collect attestation signatures, submit on-chain. (j) At completion: reconstruct trained model, evaluate on test set, print final accuracy. Target: >95% accuracy.

3. Cheater simulation (--simulate-cheater flag): At step 250 (halfway through training), one worker starts sending corrupted gradient shares. The demo shows: MAC failure detected at step 251, pairwise identification runs, worker identified, slashing transaction submitted on-chain, training resumes with 2 workers, accuracy continues improving. This is the dramatic moment in the demo.

4. ZK proof demo (--zk-proofs flag): When enabled, also generate ZK proofs at the configured frequency. Show proof generation time, proof size, gas cost for on-chain verification. This demonstrates the optional ZK layer as extra verifiability. The demo should clearly label this as "Optional ZK Verification" to emphasize it's an add-on.

5. Terminal UI: use colored output and clear formatting. Show a dashboard with: current step, loss, accuracy, workers active, checkpoints submitted, gas used, time elapsed. When a cheater is caught, highlight it prominently. When training completes, show a summary.

6. Two demo modes for the presentation:
   - Quick demo (default, ~3 min): 200 steps, 3 workers, cheater simulation, no ZK. Shows core MPC + MAC system.
   - Full demo (--zk-proofs, ~5 min): 200 steps, 3 workers, cheater simulation, plus ZK proofs at every other checkpoint. Shows the complete system with optional ZK.

7. Performance targets: Total demo time should be under 5 minutes. Pre-generation of Beaver triples should take under 30 seconds. Each MPC training step should take under 200ms. Each checkpoint submission should take under 5 seconds.

Read helix/crates/helix-demo/src/ for the existing demo structure (main.rs, worker.rs, chain.rs, bench.rs, display.rs). Read helix/scripts/run_demo.sh for the existing demo runner. The existing demo generates real ZK proofs — the new demo should be similarly impressive but with the MPC-primary architecture.
