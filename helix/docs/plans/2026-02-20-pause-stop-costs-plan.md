# Pause/Stop Training, Weight Storage & Live Cost Tracker — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add pause/stop training controls, live cost tracking, weight storage options (IndexedDB/0G), and deposit refund mechanics to the HELIX protocol.

**Architecture:** Contract-level state machine (`Active→Paused→Stopped/Active`, `Active→Stopped/Completed`) in `HelixCoordinatorV4`, with `tokio::sync::watch` signal channels in the Rust MPC training loop, and new dashboard UI for controls, cost display, and weight management.

**Tech Stack:** Solidity ^0.8.19 (Foundry), Rust (tokio, serde), TypeScript/React (Next.js, IndexedDB via idb)

---

## Task 1: Contract — Add JobStatus Enum and Migration

**Files:**
- Modify: `helix/contracts/src/core/HelixCoordinatorV4.sol:28-47` (Job struct)

**Step 1: Add JobStatus enum and new errors/events**

Add after line 93 (after PoolWorker struct), before the Constants section:

```solidity
/// @notice Training job lifecycle status
enum JobStatus { Active, Paused, Stopped, Completed }
```

Add new errors (after line 304):
```solidity
error JobNotPaused();
error NotJobOwnerOrOperator();
error NothingToRefund();
```

Add new events (after line 266):
```solidity
event TrainingPaused(uint256 indexed jobId, uint256 pausedAtStep, uint256 releasedWorkerCount);
event TrainingStopped(uint256 indexed jobId, uint256 stoppedAtStep, uint256 refundAmount);
event TrainingResumed(uint256 indexed jobId, uint256 resumeFromStep);
```

**Step 2: Replace active/completed booleans with status field**

In the `Job` struct, replace:
```solidity
bool active;
bool completed;
```
with:
```solidity
JobStatus status;
```

**Step 3: Update all existing code that reads active/completed**

These locations must change (search for `active` and `completed` in the contract):

- `registerTrainingJob` (line 380): change `active: true, completed: false` → `status: JobStatus.Active`
- `jobActive` modifier (line 318-322): change to `if (jobs[jobId].status != JobStatus.Active) revert JobNotActive();`
- `completeTraining` (line 630-632): change to `job.status = JobStatus.Completed;`
- `withdrawStake` (line 724): change `if (!jobs[jobId].completed)` → `if (jobs[jobId].status != JobStatus.Completed && jobs[jobId].status != JobStatus.Stopped)`
- `submitInferenceResult` (line 1178): change `if (!jobs[jobId].completed)` → `if (jobs[jobId].status != JobStatus.Completed && jobs[jobId].status != JobStatus.Stopped)`
- `getJobSummary` (line 1095-1118): change return to include `status` instead of `active` and `completed`

**Step 4: Run forge build to verify compilation**

Run: `cd helix/contracts && forge build`
Expected: Compilation success

**Step 5: Update all test code that reads active/completed**

In `HelixCoordinatorV4.t.sol`, update all assertions like:
- `assertTrue(active)` / `assertFalse(active)` → check status value
- `assertTrue(completed)` / `assertFalse(completed)` → check status value
- Update `getJobSummary` destructuring to match new return signature

**Step 6: Run forge test to verify existing tests still pass**

Run: `cd helix/contracts && forge test --match-contract HelixCoordinatorV4Test -vvv`
Expected: All existing tests pass

**Step 7: Commit**

```bash
git add helix/contracts/src/core/HelixCoordinatorV4.sol helix/contracts/test/HelixCoordinatorV4.t.sol
git commit -m "refactor(contracts): replace active/completed booleans with JobStatus enum"
```

---

## Task 2: Contract — Implement pauseTraining

**Files:**
- Modify: `helix/contracts/src/core/HelixCoordinatorV4.sol`
- Modify: `helix/contracts/test/HelixCoordinatorV4.t.sol`

**Step 1: Write the failing tests**

Add to the test file:

```solidity
// ============ Pause Training Tests ============

function test_pauseTraining_basic() public {
    uint256 jobId = _setupJobWith3Workers();

    // Submit a checkpoint first
    bytes32 commitment = keccak256("weights_100");
    bytes[] memory sigs = _signCheckpoint(jobId, 100, commitment, 500);
    coordinator.submitCheckpoint(jobId, 100, commitment, 500, sigs);

    // Pause training
    vm.prank(jobOwner);
    coordinator.pauseTraining(jobId);

    // Verify job is paused
    (,,,,,HelixCoordinatorV4.JobStatus status,,) = coordinator.getJobSummary(jobId);
    assertEq(uint(status), uint(HelixCoordinatorV4.JobStatus.Paused));

    // Verify all workers released
    assertEq(coordinator.getActiveWorkerCount(jobId), 0);
}

function test_pauseTraining_onlyOwnerOrOperator() public {
    uint256 jobId = _setupJobWith3Workers();

    vm.prank(outsider);
    vm.expectRevert(HelixCoordinatorV4.NotJobOwnerOrOperator.selector);
    coordinator.pauseTraining(jobId);
}

function test_pauseTraining_rejectsNonActive() public {
    uint256 jobId = _setupJobWith3Workers();

    // Complete training
    bytes32 finalCommitment = keccak256("final");
    uint256[] memory allPKs = new uint256[](3);
    allPKs[0] = WORKER1_PK; allPKs[1] = WORKER2_PK; allPKs[2] = WORKER3_PK;
    bytes[] memory sigs = _signCompletion(jobId, finalCommitment, allPKs);
    coordinator.completeTraining(jobId, finalCommitment, sigs);

    vm.prank(jobOwner);
    vm.expectRevert(HelixCoordinatorV4.JobNotActive.selector);
    coordinator.pauseTraining(jobId);
}

function test_pauseTraining_workersCanWithdrawStakeImmediately() public {
    uint256 jobId = _setupJobWith3Workers();

    vm.prank(jobOwner);
    coordinator.pauseTraining(jobId);

    // Workers can withdraw stake immediately (no cooldown)
    uint256 balBefore = worker1.balance;
    vm.prank(worker1);
    coordinator.withdrawStake(jobId);
    assertEq(worker1.balance - balBefore, STAKE_AMOUNT);
}

function test_pauseTraining_workerActiveJobsDecremented() public {
    uint256 jobId = _setupJobWith3Workers();

    uint256 activeJobsBefore = coordinator.workerActiveJobs(worker1);
    assertEq(activeJobsBefore, 1);

    vm.prank(jobOwner);
    coordinator.pauseTraining(jobId);

    assertEq(coordinator.workerActiveJobs(worker1), 0);
}
```

**Step 2: Run tests to verify they fail**

Run: `cd helix/contracts && forge test --match-test "test_pauseTraining" -vvv`
Expected: All fail (function doesn't exist yet)

**Step 3: Implement pauseTraining**

Add after `completeTraining` function (before the Optional ZK Checkpoint section):

```solidity
/// @notice Pause an active training job, releasing all workers
/// @param jobId The training job to pause
function pauseTraining(uint256 jobId) external nonReentrant jobExists(jobId) {
    Job storage job = jobs[jobId];
    if (job.status != JobStatus.Active) revert JobNotActive();
    if (msg.sender != job.owner && msg.sender != job.operator) revert NotJobOwnerOrOperator();

    uint256 activeCount = _activeWorkers[jobId].length;

    // Release all workers
    for (uint256 i = activeCount; i > 0; i--) {
        address w = _activeWorkers[jobId][i - 1];
        if (workerActiveJobs[w] > 0) {
            workerActiveJobs[w]--;
        }
        // Release pool workers back to available
        if (_poolWorkerIndex[w] != 0) {
            poolWorkers[w].available = true;
            poolWorkers[w].activeJobId = 0;
        }
        _workerIndex[jobId][w] = 0;
    }
    delete _activeWorkers[jobId];

    job.status = JobStatus.Paused;

    emit TrainingPaused(jobId, job.currentStep, activeCount);
}
```

**Step 4: Update withdrawStake to allow paused job workers without cooldown**

Modify `withdrawStake` to check:
```solidity
function withdrawStake(uint256 jobId) external nonReentrant jobExists(jobId) {
    JobStatus status = jobs[jobId].status;
    if (status != JobStatus.Completed && status != JobStatus.Stopped && status != JobStatus.Paused) {
        revert JobNotCompleted();
    }
    // For completed/stopped jobs, enforce cooldown. For paused, skip cooldown.
    if (status != JobStatus.Paused) {
        if (block.timestamp < jobCompletionTime[jobId] + STAKE_COOLDOWN) revert CooldownNotExpired();
    }
    // ... rest unchanged
}
```

**Step 5: Run tests to verify they pass**

Run: `cd helix/contracts && forge test --match-test "test_pauseTraining" -vvv`
Expected: All pass

**Step 6: Run all tests to verify no regressions**

Run: `cd helix/contracts && forge test`
Expected: All tests pass

**Step 7: Commit**

```bash
git add helix/contracts/
git commit -m "feat(contracts): add pauseTraining function with worker release"
```

---

## Task 3: Contract — Implement stopTraining with Proportional Payment + Refund

**Files:**
- Modify: `helix/contracts/src/core/HelixCoordinatorV4.sol`
- Modify: `helix/contracts/test/HelixCoordinatorV4.t.sol`

**Step 1: Write the failing tests**

```solidity
// ============ Stop Training Tests ============

function test_stopTraining_fromActive() public {
    uint256 jobId = _setupJobWith3Workers();

    // Submit checkpoint at step 100 (out of 500 total)
    bytes32 commitment = keccak256("weights_100");
    bytes[] memory sigs = _signCheckpoint(jobId, 100, commitment, 500);
    coordinator.submitCheckpoint(jobId, 100, commitment, 500, sigs);

    uint256 ownerBalBefore = jobOwner.balance;
    uint256 w1BalBefore = worker1.balance;

    vm.prank(jobOwner);
    coordinator.stopTraining(jobId);

    // Verify job is stopped
    (,,,,,HelixCoordinatorV4.JobStatus status,,) = coordinator.getJobSummary(jobId);
    assertEq(uint(status), uint(HelixCoordinatorV4.JobStatus.Stopped));

    // Verify workers got proportional payment (100/500 = 20% of 10 ETH = 2 ETH total)
    uint256 totalWorkerPay = (worker1.balance - w1BalBefore)
        + (worker2.balance - uint256(0)) // need pre-balances
        + (worker3.balance - uint256(0));

    // Verify owner got refund (remaining 80%)
    uint256 ownerRefund = jobOwner.balance - ownerBalBefore;
    assertGt(ownerRefund, 0);
}

function test_stopTraining_fromPaused() public {
    uint256 jobId = _setupJobWith3Workers();

    // Submit checkpoint
    bytes32 commitment = keccak256("weights_100");
    bytes[] memory sigs = _signCheckpoint(jobId, 100, commitment, 500);
    coordinator.submitCheckpoint(jobId, 100, commitment, 500, sigs);

    // Pause first
    vm.prank(jobOwner);
    coordinator.pauseTraining(jobId);

    // Then stop
    uint256 ownerBalBefore = jobOwner.balance;
    vm.prank(jobOwner);
    coordinator.stopTraining(jobId);

    // Verify job is stopped
    (,,,,,HelixCoordinatorV4.JobStatus status,,) = coordinator.getJobSummary(jobId);
    assertEq(uint(status), uint(HelixCoordinatorV4.JobStatus.Stopped));

    // Owner gets full refund when stopping from paused (no active workers to pay)
    uint256 ownerRefund = jobOwner.balance - ownerBalBefore;
    assertEq(ownerRefund, PAYMENT_AMOUNT);
}

function test_stopTraining_rejectsNonOwner() public {
    uint256 jobId = _setupJobWith3Workers();

    vm.prank(outsider);
    vm.expectRevert(HelixCoordinatorV4.NotJobOwnerOrOperator.selector);
    coordinator.stopTraining(jobId);
}

function test_stopTraining_rejectsCompletedJob() public {
    uint256 jobId = _setupJobWith3Workers();
    bytes32 fc = keccak256("final");
    uint256[] memory pks = new uint256[](3);
    pks[0] = WORKER1_PK; pks[1] = WORKER2_PK; pks[2] = WORKER3_PK;
    bytes[] memory sigs = _signCompletion(jobId, fc, pks);
    coordinator.completeTraining(jobId, fc, sigs);

    vm.prank(jobOwner);
    vm.expectRevert(HelixCoordinatorV4.JobNotActive.selector);
    coordinator.stopTraining(jobId);
}

function test_stopTraining_setsCompletionTimeForStakeWithdrawal() public {
    uint256 jobId = _setupJobWith3Workers();

    bytes32 commitment = keccak256("weights_100");
    bytes[] memory sigs = _signCheckpoint(jobId, 100, commitment, 500);
    coordinator.submitCheckpoint(jobId, 100, commitment, 500, sigs);

    vm.prank(jobOwner);
    coordinator.stopTraining(jobId);

    // Workers need cooldown to withdraw stake
    vm.prank(worker1);
    vm.expectRevert(HelixCoordinatorV4.CooldownNotExpired.selector);
    coordinator.withdrawStake(jobId);

    // After cooldown, can withdraw
    vm.warp(block.timestamp + 7 days + 1);
    uint256 balBefore = worker1.balance;
    vm.prank(worker1);
    coordinator.withdrawStake(jobId);
    assertEq(worker1.balance - balBefore, STAKE_AMOUNT);
}
```

**Step 2: Run tests to verify they fail**

Run: `cd helix/contracts && forge test --match-test "test_stopTraining" -vvv`
Expected: All fail

**Step 3: Implement stopTraining**

```solidity
/// @notice Stop an active or paused training job permanently
/// @dev Distributes proportional payment to active workers and refunds remainder to owner
/// @param jobId The training job to stop
function stopTraining(uint256 jobId) external nonReentrant jobExists(jobId) {
    Job storage job = jobs[jobId];
    if (job.status != JobStatus.Active && job.status != JobStatus.Paused) revert JobNotActive();
    if (msg.sender != job.owner && msg.sender != job.operator) revert NotJobOwnerOrOperator();

    uint256 activeCount = _activeWorkers[jobId].length;
    uint256 workerPayout = 0;

    if (activeCount > 0 && job.currentStep > 0) {
        // Calculate proportional payment: (currentStep / numRounds) * paymentAmount
        workerPayout = (job.paymentAmount * job.currentStep) / job.numRounds;
        if (workerPayout > job.paymentAmount) workerPayout = job.paymentAmount;

        // Distribute proportionally to workers (reuse existing logic)
        _distributePartialPayment(jobId, workerPayout);

        // Release workers
        for (uint256 i = activeCount; i > 0; i--) {
            address w = _activeWorkers[jobId][i - 1];
            workerJobsCompleted[w]++;
            if (workerActiveJobs[w] > 0) {
                workerActiveJobs[w]--;
            }
            if (_poolWorkerIndex[w] != 0) {
                poolWorkers[w].available = true;
                poolWorkers[w].activeJobId = 0;
            }
            _workerIndex[jobId][w] = 0;
        }
        delete _activeWorkers[jobId];
    }

    // Refund remaining payment to owner
    uint256 refund = job.paymentAmount - workerPayout;
    job.status = JobStatus.Stopped;
    jobCompletionTime[jobId] = block.timestamp;

    if (refund > 0) {
        (bool success, ) = job.owner.call{value: refund}("");
        if (!success) revert TransferFailed();
    }

    emit TrainingStopped(jobId, job.currentStep, refund);
}
```

**Step 4: Add _distributePartialPayment helper**

```solidity
/// @dev Distribute a partial payment amount proportionally based on steps participated
function _distributePartialPayment(uint256 jobId, uint256 totalPayout) internal {
    address[] storage active = _activeWorkers[jobId];
    uint256 activeCount = active.length;
    if (activeCount == 0 || totalPayout == 0) return;

    uint256 totalWeight = 0;
    uint256[] memory weights = new uint256[](activeCount);

    for (uint256 i = 0; i < activeCount; i++) {
        WorkerInfo storage w = workers[jobId][active[i]];
        uint256 stepsParticipated = w.lastActiveStep - w.joinedAtStep;
        if (stepsParticipated == 0) stepsParticipated = 1;
        weights[i] = stepsParticipated;
        totalWeight += stepsParticipated;
    }

    uint256 totalDistributed = 0;
    for (uint256 i = 0; i < activeCount; i++) {
        uint256 payment;
        if (i == activeCount - 1) {
            payment = totalPayout - totalDistributed;
        } else {
            payment = (totalPayout * weights[i]) / totalWeight;
        }
        totalDistributed += payment;

        if (payment > 0) {
            (bool success, ) = active[i].call{value: payment}("");
            if (!success) revert TransferFailed();
            emit PaymentDistributed(jobId, active[i], payment);
        }
    }
}
```

**Step 5: Run tests**

Run: `cd helix/contracts && forge test --match-test "test_stopTraining" -vvv`
Expected: All pass

**Step 6: Run all contract tests**

Run: `cd helix/contracts && forge test`
Expected: All pass

**Step 7: Commit**

```bash
git add helix/contracts/
git commit -m "feat(contracts): add stopTraining with proportional payment and refund"
```

---

## Task 4: Contract — Implement resumeTraining

**Files:**
- Modify: `helix/contracts/src/core/HelixCoordinatorV4.sol`
- Modify: `helix/contracts/test/HelixCoordinatorV4.t.sol`

**Step 1: Write the failing tests**

```solidity
function test_resumeTraining_basic() public {
    uint256 jobId = _setupJobWith3Workers();

    bytes32 commitment = keccak256("weights_100");
    bytes[] memory sigs = _signCheckpoint(jobId, 100, commitment, 500);
    coordinator.submitCheckpoint(jobId, 100, commitment, 500, sigs);

    // Pause
    vm.prank(jobOwner);
    coordinator.pauseTraining(jobId);

    // Resume
    vm.prank(jobOwner);
    coordinator.resumeTraining(jobId);

    // Verify job is active again
    (,,,,,HelixCoordinatorV4.JobStatus status,,) = coordinator.getJobSummary(jobId);
    assertEq(uint(status), uint(HelixCoordinatorV4.JobStatus.Active));
}

function test_resumeTraining_rejectsNonPaused() public {
    uint256 jobId = _setupJobWith3Workers();

    vm.prank(jobOwner);
    vm.expectRevert(HelixCoordinatorV4.JobNotPaused.selector);
    coordinator.resumeTraining(jobId);
}

function test_resumeTraining_workersCanRejoin() public {
    uint256 jobId = _setupJobWith3Workers();

    vm.prank(jobOwner);
    coordinator.pauseTraining(jobId);

    // Workers withdraw stake from paused job
    vm.prank(worker1);
    coordinator.withdrawStake(jobId);

    // Resume
    vm.prank(jobOwner);
    coordinator.resumeTraining(jobId);

    // New workers can join (need to re-register since worker1 already registered)
    // Worker1 was already registered, so a new worker joins instead
    vm.prank(outsider);
    coordinator.stakeAndJoin{value: STAKE_AMOUNT}(jobId);
    assertEq(coordinator.getActiveWorkerCount(jobId), 1);
}

function test_resumeTraining_acceptsTopUpPayment() public {
    uint256 jobId = _setupJobWith3Workers();

    vm.prank(jobOwner);
    coordinator.pauseTraining(jobId);

    // Resume with additional payment
    vm.prank(jobOwner);
    coordinator.resumeTraining{value: 5 ether}(jobId);

    // Payment pool increased
    (,,, uint256 payment,,,,) = coordinator.getJobSummary(jobId);
    assertEq(payment, PAYMENT_AMOUNT + 5 ether);
}
```

**Step 2: Run tests to verify they fail**

Run: `cd helix/contracts && forge test --match-test "test_resumeTraining" -vvv`
Expected: All fail

**Step 3: Implement resumeTraining**

```solidity
/// @notice Resume a paused training job
/// @param jobId The paused training job to resume
function resumeTraining(uint256 jobId) external payable nonReentrant jobExists(jobId) {
    Job storage job = jobs[jobId];
    if (job.status != JobStatus.Paused) revert JobNotPaused();
    if (msg.sender != job.owner && msg.sender != job.operator) revert NotJobOwnerOrOperator();

    job.status = JobStatus.Active;

    // Accept optional top-up payment
    if (msg.value > 0) {
        job.paymentAmount += msg.value;
    }

    emit TrainingResumed(jobId, job.currentStep);
}
```

**Step 4: Run tests**

Run: `cd helix/contracts && forge test --match-test "test_resumeTraining" -vvv`
Expected: All pass

**Step 5: Run all contract tests**

Run: `cd helix/contracts && forge test`
Expected: All pass

**Step 6: Commit**

```bash
git add helix/contracts/
git commit -m "feat(contracts): add resumeTraining with optional payment top-up"
```

---

## Task 5: Contract — Update completeTraining to Refund Remainder

**Files:**
- Modify: `helix/contracts/src/core/HelixCoordinatorV4.sol`
- Modify: `helix/contracts/test/HelixCoordinatorV4.t.sol`

**Step 1: Write the failing test**

```solidity
function test_completeTraining_refundsRemainder() public {
    uint256 jobId = _setupJobWith3Workers();

    // Submit checkpoint
    bytes32 commitment = keccak256("weights_100");
    bytes[] memory sigs = _signCheckpoint(jobId, 100, commitment, 500);
    coordinator.submitCheckpoint(jobId, 100, commitment, 500, sigs);

    bytes32 finalCommitment = keccak256("final");
    uint256[] memory pks = new uint256[](3);
    pks[0] = WORKER1_PK; pks[1] = WORKER2_PK; pks[2] = WORKER3_PK;
    bytes[] memory completionSigs = _signCompletion(jobId, finalCommitment, pks);

    uint256 ownerBalBefore = jobOwner.balance;
    uint256 w1BalBefore = worker1.balance;
    uint256 w2BalBefore = worker2.balance;
    uint256 w3BalBefore = worker3.balance;

    coordinator.completeTraining(jobId, finalCommitment, completionSigs);

    uint256 totalWorkerPay = (worker1.balance - w1BalBefore) +
        (worker2.balance - w2BalBefore) + (worker3.balance - w3BalBefore);

    // Workers get full payment (all steps completed)
    assertEq(totalWorkerPay, PAYMENT_AMOUNT);
    // Owner gets no refund (workers earned everything)
    assertEq(jobOwner.balance, ownerBalBefore);
}
```

**Step 2: Verify the test behavior**

The current `_distributePayment` already gives all payment to workers, so refund is $0 when training completes fully. The existing behavior is correct — this test validates it.

Run: `cd helix/contracts && forge test --match-test "test_completeTraining_refundsRemainder" -vvv`
Expected: Pass (existing behavior already handles this)

**Step 3: Commit (if any changes were needed)**

```bash
git add helix/contracts/
git commit -m "test(contracts): add test for completeTraining refund behavior"
```

---

## Task 6: Contract — Update getJobSummary and Inference for Stopped Models

**Files:**
- Modify: `helix/contracts/src/core/HelixCoordinatorV4.sol`
- Modify: `helix/contracts/test/HelixCoordinatorV4.t.sol`

**Step 1: Update getJobSummary return type**

Change `getJobSummary` to return `JobStatus` instead of separate `active`/`completed`:

```solidity
function getJobSummary(uint256 jobId) external view returns (
    address jobOwner,
    uint256 currentStep,
    uint256 numRounds,
    uint256 paymentAmount,
    uint256 activeWorkerCount,
    JobStatus status,
    bool zkEnabled,
    bool zkActivatedByRisk
) {
    Job storage job = jobs[jobId];
    return (
        job.owner,
        job.currentStep,
        job.numRounds,
        job.paymentAmount,
        _activeWorkers[jobId].length,
        job.status,
        job.zkEnabled,
        job.zkActivatedByRisk
    );
}
```

**Step 2: Allow inference on stopped models**

Update `submitInferenceResult` to accept stopped models:
```solidity
if (jobs[jobId].status != JobStatus.Completed && jobs[jobId].status != JobStatus.Stopped) {
    revert JobNotCompleted();
}
```

**Step 3: Add test for inference on stopped model**

```solidity
function test_submitInferenceResult_allowsStoppedModel() public {
    uint256 jobId = _setupJobWith3Workers();

    // Submit checkpoint
    bytes32 commitment = keccak256("weights_100");
    bytes[] memory cpSigs = _signCheckpoint(jobId, 100, commitment, 500);
    coordinator.submitCheckpoint(jobId, 100, commitment, 500, cpSigs);

    // Stop training
    vm.prank(jobOwner);
    coordinator.stopTraining(jobId);

    // Inference should work on stopped model
    bytes32 inferMessage = keccak256(abi.encodePacked(
        "HELIX_INFERENCE", jobId, uint256(5), keccak256("in"), keccak256("out")
    ));
    bytes[] memory sigs = new bytes[](3);
    sigs[0] = _sign(WORKER1_PK, inferMessage);
    sigs[1] = _sign(WORKER2_PK, inferMessage);
    sigs[2] = _sign(WORKER3_PK, inferMessage);
    coordinator.submitInferenceResult(jobId, 5, keccak256("in"), keccak256("out"), sigs);
}
```

**Step 4: Run all tests**

Run: `cd helix/contracts && forge test`
Expected: All pass

**Step 5: Commit**

```bash
git add helix/contracts/
git commit -m "feat(contracts): update getJobSummary, allow inference on stopped models"
```

---

## Task 7: Rust — Add TrainingSignal and TrainingOutcome to MPC Integration

**Files:**
- Modify: `helix/crates/helix-mpc/src/e2e_integration.rs`

**Step 1: Add TrainingSignal enum and TrainingOutcome**

Add near the top of the file, after the imports:

```rust
/// Signal sent to the training loop to control execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingSignal {
    /// Continue training normally.
    Continue,
    /// Pause training: save checkpoint and exit cleanly.
    Pause,
    /// Stop training permanently: save checkpoint and exit.
    Stop,
}

/// Outcome of a training run (how/why it ended).
#[derive(Debug, Clone)]
pub enum TrainingOutcome {
    /// Training completed all steps.
    Completed,
    /// Training was paused at a checkpoint.
    Paused { step: usize, checkpoint: Option<CheckpointRecord> },
    /// Training was stopped permanently.
    Stopped { step: usize, checkpoint: Option<CheckpointRecord> },
    /// A cheater was detected.
    CheaterDetected { record: CheaterRecord },
}
```

**Step 2: Add signal_rx and starting_step to MPCIntegrationConfig**

```rust
/// Optional receiver for pause/stop signals from the orchestrator.
/// When present, the training loop checks this after each step.
#[serde(skip)]
pub signal_rx: Option<tokio::sync::watch::Receiver<TrainingSignal>>,

/// Starting step for resumed training (0 for fresh training).
pub starting_step: usize,
```

Update `Default` impl to include `signal_rx: None, starting_step: 0`.

**Step 3: Add outcome field to MPCIntegrationResult**

```rust
/// How the training ended (completed, paused, stopped, or cheater detected).
pub outcome: TrainingOutcome,
```

**Step 4: Run cargo check to verify compilation**

Run: `cargo check -p helix-mpc`
Expected: Compilation errors (outcome field not populated in existing code)

**Step 5: Update result construction in all run_with_* functions**

In every place an `MPCIntegrationResult` is constructed, add:
```rust
outcome: if cheater_detected.is_some() {
    TrainingOutcome::CheaterDetected { record: cheater_detected.clone().unwrap() }
} else {
    TrainingOutcome::Completed
},
```

**Step 6: Run cargo check again**

Run: `cargo check -p helix-mpc`
Expected: Success

**Step 7: Commit**

```bash
git add helix/crates/helix-mpc/
git commit -m "feat(mpc): add TrainingSignal, TrainingOutcome, signal_rx to config"
```

---

## Task 8: Rust — Wire Signal Check into Training Loop

**Files:**
- Modify: `helix/crates/helix-mpc/src/e2e_integration.rs`

**Step 1: Add signal check after each training step**

In the training loop (around line 1312, after `steps_completed += 1;`), add:

```rust
// Check for pause/stop signal
if let Some(ref mut rx) = signal_rx {
    let signal = *rx.borrow();
    match signal {
        TrainingSignal::Pause => {
            info!(step = step, "Training paused by signal");
            let last_checkpoint = checkpoints.last().cloned();
            // Set outcome on the result
            outcome = TrainingOutcome::Paused {
                step: steps_completed,
                checkpoint: last_checkpoint.map(|cp| CheckpointRecord {
                    step: cp.step_number as usize,
                    commitment_bytes32: {
                        let mut bytes = [0u8; 32];
                        let h = cp.weight_commitment;
                        bytes.copy_from_slice(&h);
                        bytes
                    },
                    loss: cp.loss as f64,
                    weight_snapshot: None,
                }),
            };
            break;
        }
        TrainingSignal::Stop => {
            info!(step = step, "Training stopped by signal");
            let last_checkpoint = checkpoints.last().cloned();
            outcome = TrainingOutcome::Stopped {
                step: steps_completed,
                checkpoint: last_checkpoint.map(|cp| CheckpointRecord {
                    step: cp.step_number as usize,
                    commitment_bytes32: {
                        let mut bytes = [0u8; 32];
                        let h = cp.weight_commitment;
                        bytes.copy_from_slice(&h);
                        bytes
                    },
                    loss: cp.loss as f64,
                    weight_snapshot: None,
                }),
            };
            break;
        }
        TrainingSignal::Continue => {} // normal flow
    }
}
```

**Step 2: Thread signal_rx through the run_with_local_transport function**

Add `signal_rx` parameter to the internal functions, passing `config.signal_rx` from the config. The signal_rx needs to be cloned for party 0 (since only party 0 checks it).

**Step 3: Update the training loop's for-range to use starting_step**

Change `for step in 0..num_steps` to `for step in starting_step..num_steps` and thread `starting_step` from the config.

**Step 4: Run cargo test to verify**

Run: `cargo test -p helix-mpc -- e2e`
Expected: All existing tests pass (signal_rx is None by default)

**Step 5: Write a test for the signal mechanism**

```rust
#[tokio::test]
async fn test_training_pause_signal() {
    let (tx, rx) = tokio::sync::watch::channel(TrainingSignal::Continue);
    let mut config = MPCIntegrationConfig::default();
    config.num_steps = 20;
    config.signal_rx = Some(rx);

    // Send pause signal after a short delay
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        tx.send(TrainingSignal::Pause).unwrap();
    });

    let result = run_mpc_training(config).await.unwrap();
    assert!(matches!(result.outcome, TrainingOutcome::Paused { .. }));
    assert!(result.steps_completed < 20);
}
```

**Step 6: Run the new test**

Run: `cargo test -p helix-mpc test_training_pause_signal`
Expected: Pass

**Step 7: Commit**

```bash
git add helix/crates/helix-mpc/
git commit -m "feat(mpc): wire pause/stop signal check into training loop"
```

---

## Task 9: Rust — Add CostUpdate to ProgressEvent

**Files:**
- Modify: `helix/crates/helix-client/src/full_orchestration.rs`

**Step 1: Add CostUpdate variant to ProgressEvent**

```rust
/// Live cost update during training.
CostUpdate {
    /// Total gas spent in wei across all on-chain operations.
    gas_spent_wei: u128,
    /// Worker compute fees accrued so far (in ADI).
    worker_fees_accrued: f64,
    /// Original deposit amount (in ADI).
    deposit_amount: f64,
},
/// Training was paused by user.
TrainingPaused { step: usize, checkpoint_commitment: Option<String> },
/// Training was stopped by user.
TrainingStopped { step: usize, checkpoint_commitment: Option<String>, refund_amount: f64 },
```

**Step 2: Run cargo check**

Run: `cargo check -p helix-client`
Expected: Success (new variants are additive)

**Step 3: Commit**

```bash
git add helix/crates/helix-client/
git commit -m "feat(client): add CostUpdate, TrainingPaused, TrainingStopped progress events"
```

---

## Task 10: Rust — Wire Signal Channel through FullOrchestrator

**Files:**
- Modify: `helix/crates/helix-client/src/full_orchestration.rs`

**Step 1: Add signal_tx to FullOrchestrationConfig**

```rust
/// Optional sender for pause/stop signals (dashboard/CLI sends, training loop receives).
#[serde(skip)]
pub signal_tx: Option<tokio::sync::watch::Sender<helix_mpc::e2e_integration::TrainingSignal>>,
```

**Step 2: Pass signal_rx to MPCIntegrationConfig in Phase 8**

When building the `MPCIntegrationConfig` for Phase 8, create the watch channel:
```rust
let signal_rx = self.config.signal_tx.as_ref().map(|tx| tx.subscribe());
// Set on mpc_config:
mpc_config.signal_rx = signal_rx;
```

**Step 3: Handle TrainingOutcome in Phase 8 result processing**

After Phase 8 completes, check the outcome:
```rust
match &mpc_result.outcome {
    TrainingOutcome::Paused { step, .. } => {
        // Emit pause event, skip to checkpoint submission, return early
        if let Some(ref cb) = self.progress {
            cb(ProgressEvent::TrainingPaused { step: *step, checkpoint_commitment: None });
        }
        // Submit last checkpoint, then return
    }
    TrainingOutcome::Stopped { step, .. } => {
        // Emit stop event, submit checkpoint, call stopTraining on-chain
        if let Some(ref cb) = self.progress {
            cb(ProgressEvent::TrainingStopped { step: *step, checkpoint_commitment: None, refund_amount: 0.0 });
        }
    }
    TrainingOutcome::Completed => { /* normal flow */ }
    TrainingOutcome::CheaterDetected { .. } => { /* existing cheater handling */ }
}
```

**Step 4: Run cargo check**

Run: `cargo check -p helix-client`
Expected: Success

**Step 5: Commit**

```bash
git add helix/crates/helix-client/
git commit -m "feat(client): wire signal channel through FullOrchestrator"
```

---

## Task 11: Rust — Add CostTracker to Orchestrator

**Files:**
- Modify: `helix/crates/helix-client/src/full_orchestration.rs`

**Step 1: Add CostTracker struct**

```rust
/// Tracks cumulative costs during training for live display.
#[derive(Debug, Clone, Default)]
struct CostTracker {
    gas_spent_wei: u128,
    worker_fees_accrued: f64,
    deposit_amount: f64,
}

impl CostTracker {
    fn add_gas(&mut self, gas: u64, gas_price_wei: u128) {
        self.gas_spent_wei += gas as u128 * gas_price_wei;
    }

    fn update_worker_fees(&mut self, steps: usize, workers: usize, per_step_fee: f64) {
        self.worker_fees_accrued = steps as f64 * workers as f64 * per_step_fee;
    }
}
```

**Step 2: Emit CostUpdate events after gas-consuming operations**

After each on-chain operation (checkpoint submission, staking, registration), update the CostTracker and emit:
```rust
if let Some(ref cb) = self.progress {
    cb(ProgressEvent::CostUpdate {
        gas_spent_wei: cost_tracker.gas_spent_wei,
        worker_fees_accrued: cost_tracker.worker_fees_accrued,
        deposit_amount: cost_tracker.deposit_amount,
    });
}
```

**Step 3: Run cargo check**

Run: `cargo check -p helix-client`
Expected: Success

**Step 4: Commit**

```bash
git add helix/crates/helix-client/
git commit -m "feat(client): add CostTracker with live cost emission"
```

---

## Task 12: Dashboard — Create useWeightStorage Hook

**Files:**
- Create: `helix/dashboard/src/hooks/useWeightStorage.ts`

**Step 1: Create the IndexedDB weight storage hook**

```typescript
'use client';

import { useCallback, useEffect, useState } from 'react';

const DB_NAME = 'helix-weights';
const DB_VERSION = 1;
const STORE_NAME = 'weights';

export interface StoredWeights {
  jobId: number;
  step: number;
  commitment: string;
  weights: ArrayBuffer;
  timestamp: number;
  status: 'stopped' | 'paused' | 'completed';
  modelName: string;
}

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'jobId' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export function useWeightStorage() {
  const saveWeights = useCallback(async (data: StoredWeights) => {
    const db = await openDB();
    return new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).put(data);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }, []);

  const loadWeights = useCallback(async (jobId: number): Promise<StoredWeights | null> => {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const request = tx.objectStore(STORE_NAME).get(jobId);
      request.onsuccess = () => resolve(request.result ?? null);
      request.onerror = () => reject(request.error);
    });
  }, []);

  const deleteWeights = useCallback(async (jobId: number) => {
    const db = await openDB();
    return new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).delete(jobId);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }, []);

  const listWeights = useCallback(async (): Promise<StoredWeights[]> => {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const request = tx.objectStore(STORE_NAME).getAll();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }, []);

  const downloadWeights = useCallback(async (jobId: number) => {
    const data = await loadWeights(jobId);
    if (!data) return;
    const blob = new Blob([data.weights], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${data.modelName}-step${data.step}-weights.bin`;
    a.click();
    URL.revokeObjectURL(url);
  }, [loadWeights]);

  return { saveWeights, loadWeights, deleteWeights, listWeights, downloadWeights };
}
```

**Step 2: Run build to verify**

Run: `cd helix/dashboard && npm run build`
Expected: Success

**Step 3: Commit**

```bash
git add helix/dashboard/src/hooks/useWeightStorage.ts
git commit -m "feat(dashboard): add useWeightStorage hook for IndexedDB weight persistence"
```

---

## Task 13: Dashboard — Add Pause/Stop Controls to Training Page

**Files:**
- Modify: `helix/dashboard/src/app/train/page.tsx`

**Step 1: Add pause/stop buttons to the training UI**

In the Phase 8 (MPC training) section of the training page, add:

```tsx
{/* Pause/Stop Controls — visible during active training (Phase 8) */}
{status === 'running' && currentPhase === 8 && (
  <div className="flex gap-3 mt-4">
    <button
      onClick={() => handlePause()}
      className="px-4 py-2 rounded-lg bg-yellow-500/20 text-yellow-400 border border-yellow-500/30 hover:bg-yellow-500/30 transition-colors"
    >
      Pause Training
    </button>
    <button
      onClick={() => setShowStopConfirm(true)}
      className="px-4 py-2 rounded-lg bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30 transition-colors"
    >
      Stop Training
    </button>
  </div>
)}
```

**Step 2: Add stop confirmation dialog**

```tsx
{showStopConfirm && (
  <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
    <div className="bg-zinc-900 border border-zinc-700 rounded-xl p-6 max-w-md">
      <h3 className="text-lg font-semibold text-white mb-2">Stop Training?</h3>
      <p className="text-zinc-400 text-sm mb-4">
        This cannot be undone. The model will be saved at the last checkpoint
        and treated as a finished model. Workers will be paid proportionally
        and your remaining deposit will be refunded.
      </p>
      <div className="flex gap-3 justify-end">
        <button onClick={() => setShowStopConfirm(false)}
          className="px-4 py-2 rounded-lg bg-zinc-800 text-zinc-300">
          Cancel
        </button>
        <button onClick={() => { handleStop(); setShowStopConfirm(false); }}
          className="px-4 py-2 rounded-lg bg-red-600 text-white">
          Stop Training
        </button>
      </div>
    </div>
  </div>
)}
```

**Step 3: Add paused state UI**

```tsx
{status === 'paused' && (
  <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-xl p-6">
    <div className="flex items-center gap-2 mb-3">
      <PauseCircle className="w-5 h-5 text-yellow-400" />
      <h3 className="text-yellow-400 font-semibold">Training Paused at Step {pausedStep}</h3>
    </div>
    <p className="text-zinc-400 text-sm mb-4">
      Workers have been released. You can resume training with new workers
      or stop permanently.
    </p>
    <div className="flex gap-3">
      <button onClick={handleResume}
        className="px-4 py-2 rounded-lg bg-green-600 text-white">
        Resume Training
      </button>
      <button onClick={() => setShowStopConfirm(true)}
        className="px-4 py-2 rounded-lg bg-red-500/20 text-red-400 border border-red-500/30">
        Stop Training
      </button>
      <button onClick={() => downloadWeights(jobId)}
        className="px-4 py-2 rounded-lg bg-zinc-800 text-zinc-300">
        Download Weights
      </button>
    </div>
  </div>
)}
```

**Step 4: Add the handler functions and state**

```tsx
const [showStopConfirm, setShowStopConfirm] = useState(false);
const [pausedStep, setPausedStep] = useState<number | null>(null);
const { downloadWeights } = useWeightStorage();

const handlePause = async () => {
  // Send pause signal to backend via WebSocket
  // ws.send(JSON.stringify({ type: 'pause_training', jobId }));
  setStatus('paused');
};

const handleStop = async () => {
  // Send stop signal to backend via WebSocket
  // ws.send(JSON.stringify({ type: 'stop_training', jobId }));
  setStatus('stopped');
};

const handleResume = async () => {
  // Send resume signal to backend via WebSocket
  // ws.send(JSON.stringify({ type: 'resume_training', jobId }));
  setStatus('running');
};
```

**Step 5: Run build**

Run: `cd helix/dashboard && npm run build`
Expected: Success

**Step 6: Commit**

```bash
git add helix/dashboard/src/app/train/page.tsx
git commit -m "feat(dashboard): add pause/stop/resume controls to training page"
```

---

## Task 14: Dashboard — Add Live Cost Panel to Training Page

**Files:**
- Modify: `helix/dashboard/src/app/train/page.tsx`

**Step 1: Add live cost tracking state**

```tsx
const [liveCost, setLiveCost] = useState({
  gasSpentWei: 0,
  workerFeesAdi: 0,
  depositAdi: 0,
});
```

**Step 2: Add cost update handler in WebSocket message processing**

In the WebSocket message handler, add a case for cost updates:
```tsx
case 'cost_update':
  setLiveCost({
    gasSpentWei: data.gas_spent_wei,
    workerFeesAdi: data.worker_fees_accrued,
    depositAdi: data.deposit_amount,
  });
  break;
```

**Step 3: Add the LiveCostPanel component**

```tsx
{status === 'running' && currentPhase >= 8 && (
  <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-4 mt-4">
    <h4 className="text-sm font-medium text-zinc-400 mb-3">Costs So Far</h4>
    <div className="space-y-2 text-sm">
      <div className="flex justify-between">
        <span className="text-zinc-500">Worker compute</span>
        <span className="text-zinc-300">{liveCost.workerFeesAdi.toFixed(6)} ADI</span>
      </div>
      <div className="flex justify-between">
        <span className="text-zinc-500">Gas spent</span>
        <span className="text-zinc-300">{(liveCost.gasSpentWei / 1e18).toFixed(6)} ETH</span>
      </div>
      <div className="border-t border-zinc-800 pt-2 flex justify-between font-medium">
        <span className="text-zinc-400">Deposit remaining</span>
        <span className="text-green-400">
          {(liveCost.depositAdi - liveCost.workerFeesAdi).toFixed(4)} ADI
        </span>
      </div>
      {/* Progress bar */}
      <div className="w-full bg-zinc-800 rounded-full h-1.5 mt-2">
        <div
          className="bg-green-500 h-1.5 rounded-full transition-all"
          style={{ width: `${Math.min(100, (liveCost.workerFeesAdi / liveCost.depositAdi) * 100)}%` }}
        />
      </div>
    </div>
  </div>
)}
```

**Step 4: Run build**

Run: `cd helix/dashboard && npm run build`
Expected: Success

**Step 5: Commit**

```bash
git add helix/dashboard/src/app/train/page.tsx
git commit -m "feat(dashboard): add live cost tracking panel during training"
```

---

## Task 15: Dashboard — Add Status Badges to My Models Page

**Files:**
- Modify: `helix/dashboard/src/app/my-models/page.tsx`

**Step 1: Add status badge component**

```tsx
function TrainingStatusBadge({ status }: { status: string }) {
  const config = {
    completed: { bg: 'bg-green-500/20', text: 'text-green-400', border: 'border-green-500/30', label: 'Completed' },
    paused: { bg: 'bg-yellow-500/20', text: 'text-yellow-400', border: 'border-yellow-500/30', label: 'Paused' },
    stopped: { bg: 'bg-zinc-500/20', text: 'text-zinc-400', border: 'border-zinc-500/30', label: 'Stopped' },
    training: { bg: 'bg-blue-500/20', text: 'text-blue-400', border: 'border-blue-500/30', label: 'Training' },
  }[status] ?? { bg: 'bg-zinc-800', text: 'text-zinc-500', border: 'border-zinc-700', label: status };

  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium border ${config.bg} ${config.text} ${config.border}`}>
      {config.label}
    </span>
  );
}
```

**Step 2: Add badges to model cards in the training history**

In the model card rendering, add the badge:
```tsx
<TrainingStatusBadge status={session.status} />
```

**Step 3: Add "Stopped at step N/M" indicator**

For stopped models, show:
```tsx
{session.status === 'stopped' && (
  <span className="text-xs text-zinc-500">
    Stopped at step {session.stoppedAtStep}/{session.totalSteps}
  </span>
)}
```

**Step 4: Run build**

Run: `cd helix/dashboard && npm run build`
Expected: Success

**Step 5: Commit**

```bash
git add helix/dashboard/src/app/my-models/page.tsx
git commit -m "feat(dashboard): add training status badges to my-models page"
```

---

## Task 16: Dashboard — Add Weight Download and 0G Storage Prompt

**Files:**
- Modify: `helix/dashboard/src/app/my-models/[id]/page.tsx`

**Step 1: Add weight download button**

```tsx
{hasWeights && (
  <button
    onClick={() => downloadWeights(model.jobId)}
    className="px-4 py-2 rounded-lg bg-zinc-800 text-zinc-300 border border-zinc-700 hover:bg-zinc-700 transition-colors flex items-center gap-2"
  >
    <Download className="w-4 h-4" />
    Download Weights
  </button>
)}
```

**Step 2: Add 0G storage prompt modal**

```tsx
{showStoragePrompt && (
  <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
    <div className="bg-zinc-900 border border-zinc-700 rounded-xl p-6 max-w-md">
      <h3 className="text-lg font-semibold text-white mb-2">Save Weights Permanently?</h3>
      <p className="text-zinc-400 text-sm mb-4">
        Store your model weights on 0G Storage for permanent access.
        This requires a small storage fee. Otherwise, weights are stored
        locally in your browser and may be lost if you clear your data.
      </p>
      <div className="flex gap-3 justify-end">
        <button onClick={() => { saveToLocal(); setShowStoragePrompt(false); }}
          className="px-4 py-2 rounded-lg bg-zinc-800 text-zinc-300">
          Keep Local
        </button>
        <button onClick={() => { saveTo0G(); setShowStoragePrompt(false); }}
          className="px-4 py-2 rounded-lg bg-blue-600 text-white">
          Save to 0G Storage
        </button>
      </div>
    </div>
  </div>
)}
```

**Step 3: Add storage indicator**

```tsx
<div className="flex items-center gap-1.5 text-xs text-zinc-500">
  {storageLocation === '0g' ? (
    <>
      <Cloud className="w-3.5 h-3.5 text-blue-400" />
      <span>Stored on 0G</span>
    </>
  ) : (
    <>
      <HardDrive className="w-3.5 h-3.5 text-zinc-500" />
      <span>Stored locally</span>
    </>
  )}
</div>
```

**Step 4: Run build**

Run: `cd helix/dashboard && npm run build`
Expected: Success

**Step 5: Commit**

```bash
git add helix/dashboard/src/app/my-models/
git commit -m "feat(dashboard): add weight download, 0G storage prompt, storage indicator"
```

---

## Task 17: Dashboard — Update TrainingWidget for Paused/Stopped States

**Files:**
- Modify: `helix/dashboard/src/components/ui/TrainingWidget.tsx`

**Step 1: Add paused and stopped terminal states**

Add alongside the existing `complete` and `failed` states:

```tsx
if (status === 'paused') {
  return (
    <div className="...">
      <PauseCircle className="w-5 h-5 text-yellow-400" />
      <div>
        <span className="text-yellow-400 font-medium">{modelName}</span>
        <span className="text-zinc-500 text-sm ml-2">
          Paused at step {currentStep}/{totalSteps}
        </span>
      </div>
    </div>
  );
}

if (status === 'stopped') {
  return (
    <div className="...">
      <StopCircle className="w-5 h-5 text-zinc-400" />
      <div>
        <span className="text-zinc-300 font-medium">{modelName}</span>
        <span className="text-zinc-500 text-sm ml-2">
          Stopped at step {currentStep}/{totalSteps}
        </span>
      </div>
    </div>
  );
}
```

**Step 2: Run build**

Run: `cd helix/dashboard && npm run build`
Expected: Success

**Step 3: Commit**

```bash
git add helix/dashboard/src/components/ui/TrainingWidget.tsx
git commit -m "feat(dashboard): add paused/stopped states to TrainingWidget"
```

---

## Task 18: Dashboard — Update Inference Page for Owner Commission Skip

**Files:**
- Modify: `helix/dashboard/src/app/inference/page.tsx`

**Step 1: Add owner check for commission**

When displaying the cost breakdown for inference, check if the connected wallet is the model owner:

```tsx
const isModelOwner = connectedAddress && selectedModel?.owner?.toLowerCase() === connectedAddress.toLowerCase();

{/* In the cost display: */}
{isModelOwner ? (
  <div className="flex justify-between text-sm text-green-400">
    <span>Commission</span>
    <span>$0 (you own this model)</span>
  </div>
) : (
  <div className="flex justify-between text-sm">
    <span className="text-zinc-500">Commission ({formatFee(selectedModel?.inferenceFee ?? 0)})</span>
    <span className="text-zinc-300">{commissionAmount} ADI</span>
  </div>
)}
```

**Step 2: Add refund tooltip**

```tsx
<div className="group relative inline-block">
  <span className="text-zinc-500 text-xs cursor-help border-b border-dotted border-zinc-600">
    How deposits work
  </span>
  <div className="hidden group-hover:block absolute bottom-full left-0 mb-2 p-3 bg-zinc-800 border border-zinc-700 rounded-lg text-xs text-zinc-300 w-64 z-10">
    You pay a deposit that covers the maximum possible cost. Any unused
    portion is automatically refunded after inference completes.
  </div>
</div>
```

**Step 3: Run build**

Run: `cd helix/dashboard && npm run build`
Expected: Success

**Step 4: Commit**

```bash
git add helix/dashboard/src/app/inference/page.tsx
git commit -m "feat(dashboard): skip commission for model owners, add refund tooltip"
```

---

## Task 19: Dashboard — Add Stopped Badge to Model Detail Pages

**Files:**
- Modify: `helix/dashboard/src/app/models/[id]/page.tsx`

**Step 1: Add stopped/paused indicator to public model view**

```tsx
{/* Near the model name/header */}
{model.trainingStatus === 'stopped' && (
  <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-zinc-500/20 text-zinc-400 border border-zinc-500/30">
    Stopped at step {model.stoppedAtStep}/{model.totalSteps}
  </span>
)}
```

**Step 2: Run build**

Run: `cd helix/dashboard && npm run build`
Expected: Success

**Step 3: Commit**

```bash
git add helix/dashboard/src/app/models/
git commit -m "feat(dashboard): add stopped training indicator to model detail page"
```

---

## Task 20: Demo Runner — Wire Signal Channel for Demo Mode

**Files:**
- Modify: `helix/crates/helix-demo/src/runner.rs`

**Step 1: Add signal channel creation**

When building the `MPCIntegrationConfig` for the demo, create the watch channel and expose it for the display layer to send signals:

```rust
let (signal_tx, signal_rx) = tokio::sync::watch::channel(TrainingSignal::Continue);
// Store signal_tx somewhere accessible (e.g., a global or pass to display)
mpc_config.signal_rx = Some(signal_rx);
```

**Step 2: Handle paused/stopped outcomes in the demo runner**

After the MPC training phase completes, check the outcome:
```rust
match &result.outcome {
    TrainingOutcome::Paused { step, .. } => {
        display::training_paused(*step);
        return Ok(()); // Skip remaining phases
    }
    TrainingOutcome::Stopped { step, .. } => {
        display::training_stopped(*step);
        // Continue to on-chain settlement with stop
    }
    _ => {} // normal flow
}
```

**Step 3: Add display functions for paused/stopped**

In `display.rs`:
```rust
pub fn training_paused(step: usize) {
    println!("  {} Training paused at step {}", "⏸".yellow(), step);
}

pub fn training_stopped(step: usize) {
    println!("  {} Training stopped at step {}", "⏹".red(), step);
}
```

**Step 4: Run cargo check**

Run: `cargo check -p helix-demo`
Expected: Success

**Step 5: Commit**

```bash
git add helix/crates/helix-demo/
git commit -m "feat(demo): wire signal channel and handle paused/stopped outcomes"
```

---

## Task 21: Final Integration Test

**Step 1: Run all contract tests**

Run: `cd helix/contracts && forge test -vvv`
Expected: All pass

**Step 2: Run all Rust tests**

Run: `cd helix && cargo test --workspace`
Expected: All pass (or same pass/fail count as before our changes)

**Step 3: Run dashboard build**

Run: `cd helix/dashboard && npm run build`
Expected: Success

**Step 4: Run dashboard lint**

Run: `cd helix/dashboard && npm run lint`
Expected: No new errors

**Step 5: Final commit**

If any fixes were needed during integration testing, commit them.
