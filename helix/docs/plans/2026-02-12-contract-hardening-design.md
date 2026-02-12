# Smart Contract Hardening: Multi-Model, Multi-Round, Real Economics

## Problem

The HELIX contracts work (378+ tests, real proof verification, slashing) but lack production-ready economics:
- Rounds complete after a single proof with no thresholds or timeouts
- Rewards are flat per-round, not tied to compute contribution
- Model owners don't pay workers; no training fee mechanism
- Model version history exists as checkpoints but lacks chain traversal
- DAO encoding bug in `createParameterProposal()`
- Batch proof submission exists in V2 but not V3

## Design

All changes target **HelixCoordinatorV3** and its associated contracts. V2 remains unchanged.

### 1. Round Lifecycle with Thresholds, Timeouts, and Dispute Period

**Current**: `startRound()` creates round, first `submitProof()` completes it.

**New lifecycle**:
```
Created → Active → Submission → Dispute → Finalized
                               └→ Expired (if < minParticipants)
```

**Round struct additions**:
- `minParticipants`: minimum valid proofs required (set at round creation, default 3)
- `validProofs`: counter incremented on each valid proof
- `disputeDeadline`: `submissionDeadline + DISPUTE_PERIOD` (1 hour)
- `participants`: mapping of prover → proof data (loss, commitment, timestamp)
- `bestLoss` / `bestCommitment` / `bestProver`: tracked as proofs arrive
- `finalized`: bool

**Functions**:
- `startRound(modelId, duration, minParticipants)`: Creates round with thresholds
- `submitProof()`: Records proof but does NOT complete round. Updates best loss tracker.
- `finalizeRound(modelId, roundId)`: Callable by anyone after dispute deadline. Requires `validProofs >= minParticipants`. Updates model commitment to best result. Triggers reward distribution.
- `expireRound(modelId, roundId)`: Callable after submission deadline if threshold not met. Refunds training fees.

### 2. Compute-First Reward Model

**Principle**: Workers are paid for compute provided. Stake is a security bond, not a yield source.

**Staking requirement**:
- Workers declare compute slots when joining a round (how many proofs they commit to)
- Required stake = `slotsCommitted * stakePerSlot` (e.g., 100 HELIX per slot)
- Stake scales with compute commitment, not as an investment

**Reward pools per round** (from training fees + protocol emissions):

| Pool | Share | Distribution |
|------|-------|-------------|
| Compute | 70% | Per valid proof: `computePool / totalValidProofs` |
| Quality | 20% | Among proofs beating median loss, weighted by improvement |
| Timeliness | 10% | Full for first 75% of deadline, linear decay to 0 at deadline |

A worker submitting 10 valid proofs earns ~10x someone submitting 1. Directly proportional to useful compute.

**Quality bonus calculation**:
```
medianLoss = median of all valid proof losses in the round
For each proof with loss < medianLoss:
  improvement = (medianLoss - proofLoss) / medianLoss
qualityReward[i] = (improvement[i] / sum(improvements)) * qualityPool
```

**Timeliness calculation**:
```
elapsed = proofTimestamp - roundStartTime
roundDuration = submissionDeadline - roundStartTime
progress = elapsed / roundDuration
if progress <= 0.75: multiplier = 1.0 (full bonus)
else: multiplier = 1.0 - ((progress - 0.75) / 0.25)  // linear decay
timelinessReward[i] = multiplier * (timelinessPool / totalValidProofs)
```

### 3. Training Fee Mechanism

**New `TrainingJob` concept in V3**:

```solidity
struct TrainingJob {
    uint256 modelId;
    address owner;
    uint256 totalDeposit;      // HELIX deposited by model owner
    uint256 totalRounds;       // How many rounds funded
    uint256 completedRounds;
    uint256 feePerRound;       // totalDeposit / totalRounds
    uint256 remainingBalance;
    bool active;
}
```

**Flow**:
1. `createTrainingJob(modelId, rounds)` + token transfer: Owner deposits payment
2. `feePerRound = deposit / rounds`, held in contract
3. `finalizeRound()` distributes `feePerRound` to workers via reward pools
4. `expireRound()` refunds `feePerRound` to owner
5. `cancelTrainingJob()`: Owner reclaims funds for uncompleted rounds

### 4. Model Version History (ModelRegistry.sol)

**Existing**: Checkpoints with `roundId, commitment, ipfsHash, timestamp, errorBound, proofHash`. Version field increments.

**Additions**:
- `parentVersion` field in Checkpoint struct (v2 points to v1)
- `getVersionChain(modelId, fromVersion, count)` for traversal
- `commitmentToVersion` mapping for reverse lookup (given commitment, find version)
- `ModelVersionCreated` event with parentVersion

### 5. DAO Encoding Bug Fix

**Bug**: `createParameterProposal()` at line 198 pre-computes `proposalId = proposalCount + 1` then passes it to `createProposal()`. The encoded calldata bakes in a potentially wrong ID.

**Fix**: `createProposal()` returns the actual `proposalId`. Encode calldata after creation using the returned ID. Or encode with a placeholder and patch after.

Since the calldata is stored in the proposal itself and `applyParameters` uses `proposalId` to look up params, the cleanest fix is to have `createParameterProposal` call `createProposal` first, get the real ID, then store the params keyed by that ID.

### 6. Batch Proof Submission in V3

**Current V3**: No batch submission. V2 has `submitProofBatch()`.

**Add to V3**:
- `submitProofBatch(BatchSubmission[])`: Process multiple proofs in one tx
- Integrates with new round lifecycle (each proof recorded, not completing round)
- Gas optimization: amortize base tx cost (21K gas) across proofs
- `batchFinalizeRounds(modelId, roundIds[])`: Finalize multiple rounds in one tx

## Files Modified

| File | Changes |
|------|---------|
| `HelixCoordinatorV3.sol` | Round lifecycle, training jobs, batch submission, compute slot staking |
| `Rewards.sol` | Compute/quality/timeliness pools, per-proof distribution, fee pool integration |
| `ModelRegistry.sol` | parentVersion, version chain traversal, reverse lookup |
| `TrainingDAO.sol` | Fix encoding bug in createParameterProposal |
| `Deploy.s.sol` | Updated deployment for new params |
| `test/ContractHardening.t.sol` | New comprehensive test suite |

## Files NOT Modified

- `HelixCoordinatorV2.sol` (backward compat)
- `Halo2Verifier.sol`, `PoseidonHasher.sol` (verification layer unchanged)
- `Staking.sol` (already production-grade with severity system)
- `HelixToken.sol` (no changes needed)

## Testing Strategy

New test file `ContractHardening.t.sol` covering:
- Round lifecycle: create → submit multiple proofs → finalize after dispute
- Round expiry: insufficient participants → refund
- Compute rewards: 10-proof worker gets ~10x reward of 1-proof worker
- Quality bonus: lower-loss proofs get disproportionate reward
- Timeliness: early vs late submission reward difference
- Training jobs: deposit → rounds → distribute → refund unused
- Model versioning: chain traversal, reverse lookup
- DAO bug fix: proposal ID correctness
- Batch submission: multiple proofs in one tx
- Edge cases: zero participants, single participant at threshold, dispute period challenges
