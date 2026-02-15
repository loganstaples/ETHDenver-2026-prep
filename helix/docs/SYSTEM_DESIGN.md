# HELIX System Design — Final Architecture

## One-Sentence Summary

HELIX is a decentralized ML training protocol where workers train a model on secret-shared weights using MPC, with SPDZ information-theoretic MACs providing mathematically certain cheater detection, individual bad actor identification, and on-chain settlement — all without anyone ever seeing the model weights.

## What Makes This Novel

No existing project combines all of these:

1. INFORMATION-THEORETIC SECURITY: SPDZ MACs cannot be broken even with unlimited computing power. This is strictly stronger than computational assumptions used by ZK proofs, TEEs, or any other approach. The guarantee is mathematical impossibility, not "this would take a long time to break."

2. INDIVIDUAL ACCOUNTABILITY: When a worker cheats, the protocol doesn't just detect "someone cheated" — it identifies the exact bad actor through pairwise MAC verification. That specific worker is slashed on-chain. No group punishment, no ambiguity.

3. ZERO WEIGHT LEAKAGE: Model weights exist only as secret shares during training. No single worker, no coordinator, no on-chain contract ever sees the full model. The model owner is the only entity that can reconstruct — and only at the very end.

4. SELF-HEALING: When a bad actor is detected and removed, training continues with the remaining honest workers. No restart, no data loss. The system degrades gracefully.

5. NEAR-NATIVE EFFICIENCY: 5-15x overhead vs 100-1000x for ZK-every-step approaches. MPC computation on secret shares is expensive but not catastrophically so. The overhead comes from communication between parties, not from proof generation.

Existing projects for comparison:
- EZKL, Giza, Modulus: ZK proofs for inference only (not training). 100-1000x overhead.
- Bittensor: Decentralized inference with validator re-computation. No privacy (validators see everything). No training verification.
- Gensyn: Decentralized training with probabilistic proof-of-learning. No weight privacy. Random spot checks, not information-theoretic guarantees.
- Ritual: Decentralized inference with TEEs. Trusts hardware manufacturers (Intel/AMD). SGX has been broken multiple times via side channels.
- io.net: GPU marketplace. No verification at all.

HELIX is the only system that provides private, verified, decentralized TRAINING with information-theoretic security.


## Participants

MODEL OWNER (the person submitting a model for training):
- Has initial model weights and training configuration
- Submits a training job to the network via CLI: `helix train --model ./weights.bin --architecture "784,32,10" --data mnist --workers 3 --rounds 1000`
- The CLI automatically handles weight splitting and share distribution
- Owner does NOT participate in training computation
- Owner does NOT need to be online during training (can disconnect after setup)
- Owner reconnects at the end to collect the trained model
- Computation required from owner: effectively zero (weight splitting is milliseconds, fully automated by the CLI)

WORKERS (3-7 nodes providing compute):
- Each holds one additive share of the model weights
- Perform MPC training computation collaboratively
- Stake tokens on-chain before participating (economic security)
- Earn payment proportional to their honest participation
- Never see the full model weights — only their share
- Run the SPDZ MAC verification protocol to catch cheaters among themselves
- Can be slashed if they cheat (identified by MAC verification)

SMART CONTRACT (on-chain coordinator):
- Manages worker registration and staking
- Stores model weight commitments at checkpoints
- Accepts multi-party attestations (signed by worker majority)
- Processes MAC failure reports (signed by honest majority) and slashes cheaters
- Releases payment to workers on training completion
- Does NOT perform heavy computation — just signature verification and bookkeeping


## Trust Model

The system has NO trusted party. Specifically:

- The model owner is NOT trusted by workers (workers verify the initial share distribution via commitment checks)
- Workers are NOT trusted by anyone (SPDZ MACs catch any cheating)
- No single coordinator is trusted (aggregator orchestrates but cannot corrupt computation)
- The smart contract is trusted only to execute its code (standard Ethereum trust model)
- Beaver triples are generated distributedly via OT between workers — no trusted dealer

Assumption: HONEST MAJORITY among workers. If more than half of workers collude, they can reconstruct the model weights. This is the fundamental MPC assumption and applies to all secret-sharing-based systems. For 3 workers, we need at least 2 honest. For 5 workers, at least 3.

If a worker is caught cheating (via MAC failure), they are removed and training continues if honest majority still holds. If too many workers are removed and honest majority cannot be maintained, training pauses and new workers can be recruited.


## Protocol Flow

### Phase 1: Job Setup

Step 1 — Model Registration:
Owner runs the CLI. The CLI connects to the Helix smart contract (via any Ethereum RPC endpoint). The CLI registers a training job on-chain with: model architecture (public — e.g., "784,32,10"), training hyperparameters (public — learning rate, batch size, number of rounds), checkpoint frequency (e.g., every 100 steps), payment amount (deposited into contract), minimum number of workers required, and optionally whether ZK proofs are enabled and at what frequency (off by default).

Step 2 — Worker Registration:
Workers discover the training job (by watching contract events or querying the aggregator node). Each worker stakes tokens on-chain to join the job. The contract records each worker's address and stake amount. Workers must meet the minimum stake requirement. Once enough workers are registered, the job can begin.

Step 3 — Weight Distribution:
The owner's CLI takes the initial model weights W0 and generates N additive shares: W0 = s1 + s2 + ... + sN. Shares s1 through s(N-1) are random field elements. Share sN = W0 - s1 - s2 - ... - s(N-1) (the remainder). Each share is encrypted to the corresponding worker's x25519 public key using Diffie-Hellman key agreement + AES-GCM. Encrypted shares are sent to workers via direct TCP connections. The owner also publishes a Pedersen commitment to the initial weights: C0 = g^W0 * h^r0, where g and h are public generators and r0 is a random blinding factor. This commitment goes on-chain. Workers can verify the commitment against their shares: each worker commits to their share Ci = g^si * h^ri, and the product of all Ci should equal C0.

Step 4 — Beaver Triple Pre-Generation:
Workers collaboratively generate Beaver triples using OT-based distributed generation (MASCOT-style protocol). Each pair of workers runs oblivious transfer to generate correlated random triples (a, b, c) where c = a*b, without anyone seeing the full triple. Triples are generated in bulk before training begins — enough for the entire training run. For a 784x32x10 model with 1000 training steps: approximately 75K multiplications per step * 1000 steps = 75M triples needed. At ~10K triples/second per pair (realistic OT throughput), pre-generation takes ~2 hours for 3 parties. For the hackathon demo (100 steps, smaller model), pre-generation takes seconds.

Step 5 — SPDZ MAC Key Setup:
Workers jointly generate the global MAC key alpha. Each worker i generates a random alpha_i. The global key alpha = alpha_1 + alpha_2 + ... + alpha_N (but nobody knows the full alpha). For each secret-shared value x = x_1 + x_2 + ... + x_N, the MAC tag gamma(x) = alpha * x is also secret-shared: gamma_i(x) = alpha_i * x + noise_i (where noise terms cancel in aggregate). MAC tags are generated during the Beaver triple pre-generation phase as part of the SPDZ offline protocol.


### Phase 2: MPC Training (the main loop, runs entirely off-chain)

For each training step:

FORWARD PASS:
- Input data is public (for the hackathon — MNIST images are not secret). Each worker has the same training batch.
- Layer 1 matmul: Workers compute h_pre = W1 * x + b1 on secret-shared W1 and b1, using Beaver triples for multiplication. Each multiplication a * b works as follows: each worker opens (a - alpha) and (b - beta) where (alpha, beta, gamma) is a pre-computed Beaver triple with gamma = alpha * beta. From the opened values, each worker locally computes their share of a*b. This requires 1 round of communication per multiplication (all multiplications in a layer can be batched into 1 round).
- ReLU activation: Workers compute h = ReLU(h_pre) using garbled circuits via oblivious transfer. One worker acts as the garbler (constructs the circuit), others act as evaluators. The comparison "is h_pre_i > 0?" is evaluated without anyone learning h_pre_i. Each worker gets their share of h. This is the most expensive operation per element.
- Layer 2 matmul: Same Beaver triple protocol for y = W2 * h + b2.

LOSS COMPUTATION:
- Workers jointly compute loss = sum((y_i - target_i)^2).
- The loss value is PUBLICLY REVEALED (all workers open their shares). This is safe — the loss is a single scalar that doesn't reveal model weights. It provides training progress monitoring.

BACKWARD PASS:
- Same MPC protocols applied in reverse.
- dy = 2 * (y - target): computed on shares, no communication needed (linear operation).
- dW2 = dy * h^T: Beaver triple multiplication on shares.
- dh = W2^T * dy: Beaver triple multiplication.
- dh_pre = dh * relu_mask: element-wise multiply, requires Beaver triples.
- dW1 = dh_pre * x^T: Beaver triple multiplication.
- Bias gradients: sums of shares, no communication.

WEIGHT UPDATE:
- Each worker locally updates their share: si_new = si_old - lr * grad_i
- NO COMMUNICATION NEEDED. Additive secret sharing is linear: subtracting a share of the gradient from a share of the weights gives a share of the updated weights. This is the key efficiency advantage of additive sharing.

SPDZ MAC VERIFICATION (every K steps, configurable, default K=1 for maximum security):
- After a batch of operations, workers verify the MACs on a random linear combination of recent values.
- Each worker i computes: sigma_i = gamma_i(x) - alpha_i * x (for each value x that was opened/used).
- Workers commit to their sigma values: each publishes hash(sigma_i || nonce_i).
- After all commitments, workers reveal sigma_i values.
- If sum(sigma_i) = 0: all MACs are valid, training continues.
- If sum(sigma_i) != 0: CHEATER DETECTED. Proceed to identification (see Bad Actor Detection below).


### Phase 3: Checkpoints (periodic on-chain anchoring)

At every checkpoint interval (e.g., every 100 steps):

Step 1 — Joint Commitment:
Workers jointly compute a Pedersen commitment to the current weights WITHOUT reconstructing them. Each worker i computes their commitment share: Ci = g^(si) * h^(ri) where si is their weight share and ri is a random blinding factor. Workers publish all Ci. The combined commitment C = product(Ci) = g^(sum(si)) * h^(sum(ri)) = g^W * h^r. Nobody learned W.

Step 2 — Multi-Party Attestation:
All workers sign an attestation message: "Training job {job_id}: Steps {start} to {end} completed. New weight commitment: {C}. Current loss: {L}. Participants: {worker_addresses}."
Each worker signs with their Ethereum private key (ECDSA).

Step 3 — On-Chain Submission:
One worker (or the aggregator) submits the attestation + all signatures to the smart contract. The contract verifies: (a) all registered workers signed, (b) the attestation is for the correct job and step range, (c) the loss is not NaN or infinite. The contract stores the new commitment and step number.

Cost: ~50-80K gas per checkpoint (signature verification + storage). Compare to 7.5M gas for a ZK proof.

Step 4 — Owner Notification (optional):
If the owner is online, they receive the checkpoint attestation. They can optionally request weight shares from workers at any checkpoint to verify progress. If offline, they verify everything at the end.


### Phase 4: Bad Actor Detection and Individual Punishment

When a SPDZ MAC check fails (sum(sigma_i) != 0):

Step 1 — Halt Training:
Training immediately stops. No more steps are executed until the cheater is identified and removed. This prevents error propagation — a cheating worker cannot corrupt more than K steps (where K is the MAC check frequency).

Step 2 — Pairwise Identification:
Workers run pairwise MAC verification. For each pair (i, j):
- Party i and party j share their sigma values for the failed batch.
- They verify each other's contributions against their shared Beaver triple data.
- If party j cheated, the pairwise check between j and every honest party will fail, while pairwise checks between honest parties will succeed.

With 3 workers and 1 cheater:
- Pair (1,2): if 2 is the cheater, this FAILS
- Pair (1,3): PASSES (both honest)
- Pair (2,3): FAILS (2 is dishonest)
Party 2 is in both failing pairs. Parties 1 and 3 are in the passing pair. Party 2 is unambiguously identified as the cheater.

With N workers and 1 cheater: the cheater appears in all failing pairs. Honest workers only appear in passing pairs (with each other). Identification is unambiguous with honest majority.

Step 3 — Signed Blame Report:
The honest majority of workers sign a blame report: "Worker {cheater_address} identified as cheating at step {K}. MAC verification evidence: {sigma values and commitments}. Signed by workers {honest_addresses}."

Step 4 — On-Chain Slashing:
The blame report is submitted to the smart contract. The contract verifies: (a) the report is signed by a majority of registered workers for this job, (b) the reported worker is a registered participant. The contract slashes the cheater's full stake. A portion (e.g., 10%) goes to the reporters as a bounty. The rest goes to the protocol treasury.

Step 5 — Resume Training:
If enough honest workers remain (honest majority maintained), training resumes from the last verified checkpoint. The cheater's share is redistributed: remaining workers collectively adjust their shares so that the sum still equals the correct weights. This is possible because the last checkpoint commitment is known and honest workers' shares are known to themselves.

If honest majority is lost (e.g., 2 of 3 workers were cheating), training cannot continue. The job is paused, remaining honest workers are paid for completed work, and the owner is notified that new workers are needed.


### Phase 5: Training Completion

Step 1 — Final Checkpoint:
After all rounds complete, a final checkpoint attestation is submitted on-chain. All workers sign.

Step 2 — Weight Reconstruction:
Each worker encrypts their final weight share to the model owner's public key and sends it to the owner. The owner's CLI collects all shares and reconstructs: W_final = s1 + s2 + ... + sN. This is the ONLY time the full model exists outside of shares. Only the owner sees it.

Step 3 — Owner Verification (optional):
Owner can locally verify: run a few inference samples through the model to check accuracy. If the accuracy matches expectations (loss was decreasing in attestations), training was successful.

Step 4 — Payment Release:
Owner signals acceptance on-chain (or acceptance is automatic after a timeout period). The contract releases payment to workers proportional to their participation. Workers who were slashed receive nothing. Workers who joined late receive payment proportional to the steps they participated in.

Step 5 — Optional ZK Proofs (user-configurable, OFF by default):
ZK proofs are an optional feature that the model owner can enable at job registration time. They are NEVER required — the MPC attestation chain provides complete security on its own. ZK proofs exist as an extra layer for users who want external third-party verifiability (anyone in the world can verify training correctness without trusting the worker set).

If enabled, the owner configures:
- --zk-proofs: Enables ZK proof generation (default: off)
- --zk-checkpoint-freq N: Generate a ZK proof every N checkpoints (default: only at the end)

When ZK is enabled, the owner generates lightweight state-transition proofs at the configured frequency. Each proof demonstrates: "The model weights with commitment C_old transitioned to commitment C_new via N steps of SGD with total error within bound E." The circuit is just matrix subtraction + Poseidon commitment verification (NOT the full forward/backward pass), so proofs take 1-3 seconds regardless of model size and cost 7.5M gas to verify on-chain.

Example configurations:
- `helix train --model ./weights.bin --workers 3 --rounds 1000` — No ZK. Pure MPC + attestations. Fastest and cheapest.
- `helix train --model ./weights.bin --workers 3 --rounds 1000 --zk-proofs` — ZK proof generated only at the very end. One extra proof, one extra on-chain verification.
- `helix train --model ./weights.bin --workers 3 --rounds 1000 --zk-proofs --zk-checkpoint-freq 5` — ZK proof at every 5th checkpoint. More external verifiability, more gas cost.
- `helix train --model ./weights.bin --workers 3 --rounds 1000 --zk-proofs --zk-checkpoint-freq 1` — ZK proof at every checkpoint. Maximum external verifiability. Most expensive but still far cheaper than ZK-every-step (proves checkpoint transitions, not individual steps).

Why ZK is not necessary:
- SPDZ MACs already guarantee correctness with information-theoretic security (stronger than ZK's computational assumptions)
- Multi-party attestations already provide on-chain anchoring
- The model owner can independently verify training by reconstructing weights and checking accuracy
- ZK adds external verifiability for untrusted third parties — a nice-to-have, not a security requirement


## Error Containment

Errors and bad behavior do NOT propagate through the system. Here is how:

MAC CHECK FREQUENCY: By default, SPDZ MAC verification runs after every single training step. This means a cheater can corrupt AT MOST 1 step before being caught. The cost of per-step MAC verification is minimal — it is a linear operation on secret-shared values that adds <5% to the step time.

IMMEDIATE HALT ON DETECTION: When a MAC check fails, training stops immediately. No more computation happens until the cheater is identified and removed. This is critical — continuing to compute on corrupted values would propagate errors.

ROLLBACK TO LAST CHECKPOINT: After removing the cheater, training rolls back to the last verified checkpoint (the last on-chain attestation). Any work done between the last checkpoint and the detection is discarded. This bounds the maximum wasted work to (checkpoint_interval) steps.

SHARE REDISTRIBUTION: When a party is removed, their share must be accounted for. The remaining honest parties adjust their shares using the checkpoint commitment. Specifically: the honest parties know their own shares and the total commitment from the checkpoint. They can compute what the removed party's share SHOULD have been (via the commitment) and redistribute it among themselves. Training resumes from the checkpoint with N-1 parties.

LOSS MONITORING: The publicly revealed loss value after each step provides an independent sanity check. If loss suddenly increases dramatically or becomes NaN, something is wrong even if MACs haven't caught it yet (this could indicate a bug, not just cheating). The aggregator can pause training for investigation.


## Performance Characteristics

For a 784x32x10 MNIST model (25K parameters, 3 workers):

OVERHEAD BREAKDOWN:
- Native training step: ~5-10ms
- MPC overhead (Beaver triple consumption + communication): ~30-80ms per step
- Garbled circuit ReLU (32 neurons): ~10-30ms per step
- SPDZ MAC verification: ~2-5ms per step
- Total per step: ~50-120ms
- Overhead ratio: ~10-15x native

ON-CHAIN COSTS (per checkpoint):
- Multi-party attestation: ~50-80K gas (~$0.10-0.50 on L2)
- No ZK proof verification needed at checkpoints

TRAINING TIME (1000 steps, 3 workers):
- Native: ~5-10 seconds
- MPC: ~50-120 seconds (~1-2 minutes)
- Plus checkpoint overhead: ~10 seconds total for 10 checkpoints
- Total: ~2 minutes for the full training run

COMMUNICATION:
- Per step: ~2-5MB total across all worker pairs (Beaver triple openings + share exchanges)
- Per checkpoint: ~100KB (attestation + signatures)
- Total for 1000 steps: ~2-5GB network traffic across all parties

PRE-GENERATION (Beaver triples, one-time before training):
- For 1000 steps of 25K-param model: ~75M triples
- Generation time with OT: ~30-120 seconds (parallelized across worker pairs)
- This happens ONCE before training begins


## On-Chain Components

SMART CONTRACT (HelixCoordinator):
- registerTrainingJob(architecture, hyperparams, checkpointFreq, payment, zkEnabled) — owner registers job. zkEnabled is a boolean (default false).
- stakeAndJoin(jobId) — worker stakes tokens and joins
- submitCheckpoint(jobId, stepNumber, commitment, attestation, signatures[]) — stores checkpoint (always used)
- submitCheckpointWithProof(jobId, stepNumber, commitment, proof, publicInputs, signatures[]) — stores checkpoint AND verifies a ZK proof via Halo2Verifier (only called when ZK is enabled for the job)
- reportMACFailure(jobId, stepNumber, cheater, evidence, signatures[]) — slashes cheater
- completeTraining(jobId) — releases payment after final checkpoint
- withdrawModel(jobId) — owner confirms receipt of trained model

The contract works identically whether ZK is enabled or not. ZK proofs are verified via the existing Halo2Verifier when present but are never required for the protocol to function.

TOKEN (HelixToken):
- Standard ERC20 with staking hooks
- Workers must hold and stake tokens to participate
- Slashed tokens go to reporters (bounty) and treasury

STAKING:
- Minimum stake configurable per job (set by owner)
- Full stake slashed on MAC failure (severe — cheating in training is unacceptable)
- Stake returned after training completion + cooldown period


## Component Mapping to Codebase

helix-core: Types, tensors, bounded values, error tracking, serialization. KEEP AS-IS. This is the foundation and works well.

helix-avm: Training engine (forward, backward, autodiff, quantization). KEEP AS-IS for native training reference. Workers use this for computing on their shares — same arithmetic, just on share values instead of full values.

helix-mpc: MPC protocols. MAJOR EXTENSION. This becomes the primary computation layer. Extend: wire SPDZ MACs into the training loop, implement pairwise cheater identification, add distributed Beaver triple generation via OT, integrate with node P2P transport.

helix-circuits: ZK circuits. SIMPLIFY. Add a lightweight StateTransitionCircuit for optional ZK proofs. Keep the Halo2/KZG infrastructure. The full MLTrainingStepV2Circuit can remain in the codebase but is not used in the primary flow. ZK is an optional feature — the system works completely without it.

helix-prover: Proof pipeline. SIMPLIFY. Keep for optional ZK proof generation (when user enables --zk-proofs). The prover is only invoked if the user explicitly opts in. Remove per-step proving logic.

helix-node: P2P networking and worker roles. MAJOR REWORK. Workers become MPC participants. Aggregator becomes session coordinator. Wire MPC transport through the P2P connection layer. Add checkpoint attestation flow. Add MAC failure reporting.

helix-client: SDK and CLI. MODERATE REWORK. Becomes the model owner's interface. Handles: weight splitting + distribution, job registration on-chain (including optional --zk-proofs and --zk-checkpoint-freq flags), progress monitoring via attestation polling, final weight reconstruction. If ZK is enabled, the client invokes the CheckpointProver at the configured frequency — but this code path is entirely optional and never runs unless the user explicitly enables it.

contracts: Smart contracts. MODERATE CHANGES. Replace proof verification checkpoints with attestation verification. Add MAC failure report handling. Simplify to focus on staking, attestation, and slashing.

helix-demo: Demo binary. REBUILD for new architecture. Single command that orchestrates the full flow for live demo.


## Demo Script (ETHDenver)

What the audience sees (3-5 minute live demo):

1. "I have an MNIST model — 25K parameters, 784 input neurons, 32 hidden, 10 output classes. I want to train it without revealing the model weights to anyone."

2. Run: `helix train --model ./mnist_init.bin --arch 784,32,10 --data mnist --workers 3 --rounds 500 --checkpoints 50 --zk-proofs --zk-checkpoint-freq 5`

3. Terminal shows: "Deploying contracts to chain... Done. Registering job... Done. 3 workers staked and joined. Distributing weight shares via encrypted channels... Done. No worker has seen the full model. Starting MPC training..."

4. Training progress: "Step 50/500 — Loss: 2.31 — Accuracy: 15% — All MACs verified — Checkpoint 1 submitted on-chain"

5. Progress continues: "Step 200/500 — Loss: 0.45 — Accuracy: 87% — 4 checkpoints on-chain"

6. Simulated attack: "Worker 2 is now submitting corrupt gradients..."
   "Step 251: MAC VERIFICATION FAILED. Pairwise check: Worker 2 identified as cheater. Slashing Worker 2 (100% of stake). Resuming with 2 workers from checkpoint 5."

7. Training completes: "Step 500 — Loss: 0.12 — Accuracy: 97.3% — Model weights reconstructed."

8. "Let me verify: running inference on test set... 97.3% accuracy. The model was trained by 3 independent workers. Nobody ever saw the weights. One worker tried to cheat and was caught in 1 step, identified, and slashed. All verified on-chain."

9. "And because we enabled ZK proofs for this run, every 5th checkpoint also has a zero-knowledge proof verified on-chain. Anyone in the world can independently verify this training was correct — no need to trust us or the workers. But even without ZK, the SPDZ MACs provide information-theoretic security that's strictly stronger than any ZK proof system."

10. "Let me show you it works without ZK too:" Run: `helix train --model ./mnist_init2.bin --arch 784,32,10 --data mnist --workers 3 --rounds 100` (no --zk-proofs flag). Show training completes with just MPC + attestations. "Same security guarantees, lower gas cost, faster. ZK is a nice extra, not a requirement."
