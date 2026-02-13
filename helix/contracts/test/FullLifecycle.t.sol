// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../src/core/HelixCoordinatorV3.sol";
import "../src/core/ModelRegistry.sol";
import "../src/token/HelixToken.sol";
import "../src/token/Staking.sol";
import "../src/token/Rewards.sol";
import "./ProofFixtures.t.sol";

/// @title FullLifecycleTest
/// @notice End-to-end integration test: 3-round training job with 3 workers,
///         reward distribution, and claim verification.
contract FullLifecycleTest is Test {
    HelixCoordinatorV3 public coordinator;
    HelixToken public token;
    Staking public staking;
    Rewards public rewards;
    ModelRegistry public registry;
    MockVerifierForLifecycle public mockVerifier;

    address public modelOwner;
    address public treasuryAddr;
    address public worker1;
    address public worker2;
    address public worker3;

    uint256 constant STAKE_AMOUNT = 200e18;
    uint256 constant MIN_STAKE = 100e18;
    uint256 constant ROUND_DURATION = 1 hours;
    uint256 constant JOB_DEPOSIT = 3000e18;
    uint256 constant TOTAL_ROUNDS = 3;
    uint256 constant REWARD_PER_ROUND = 300e18;
    uint256 constant REWARD_POOL = 10000e18;

    function setUp() public {
        modelOwner = address(this);
        treasuryAddr = makeAddr("treasury");
        worker1 = makeAddr("worker1");
        worker2 = makeAddr("worker2");
        worker3 = makeAddr("worker3");

        // Deploy full V3 stack
        token = new HelixToken(treasuryAddr);
        mockVerifier = new MockVerifierForLifecycle();
        staking = new Staking(address(token), MIN_STAKE, 7 days, 5000);
        rewards = new Rewards(address(token));
        registry = new ModelRegistry();

        coordinator = new HelixCoordinatorV3(
            address(mockVerifier),
            address(staking),
            address(rewards),
            address(registry),
            treasuryAddr
        );

        // Wire contracts
        staking.setOperator(address(coordinator));
        rewards.setCoordinator(address(coordinator));
        rewards.setStakingContract(address(staking));
        registry.setCoordinator(address(coordinator));

        // Fund workers with HELIX tokens
        token.mint(worker1, 5000e18);
        token.mint(worker2, 5000e18);
        token.mint(worker3, 5000e18);

        // Workers approve staking and stake
        _stakeWorker(worker1);
        _stakeWorker(worker2);
        _stakeWorker(worker3);

        // Fund model owner for training job
        token.mint(modelOwner, 50000e18);
        token.approve(address(coordinator), type(uint256).max);
        token.approve(address(rewards), type(uint256).max);

        // Fund reward pool
        rewards.fundRewardPool(REWARD_POOL, REWARD_PER_ROUND, 365 days);
    }

    // ============ Full 3-Round Lifecycle ============

    function test_FullLifecycle_3Rounds_3Workers() public {
        // Step 1: Register model
        uint256 oldHashLo = 12345;
        uint256 oldHashHi = 67890;
        uint256 initialCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        uint256 modelId = coordinator.registerModel("TestModel", "Full lifecycle test", "QmTestHash", initialCommitment, 4, 8, 2, 2, 0);

        // Verify model registered
        (uint256 round, uint256 commitment, bool active) = coordinator.getModelState(modelId);
        assertEq(round, 0);
        assertEq(commitment, initialCommitment);
        assertTrue(active);

        // Step 2: Create training job (3000 tokens for 3 rounds = 1000/round)
        uint256 jobId = coordinator.createTrainingJob(modelId, TOTAL_ROUNDS, JOB_DEPOSIT);
        assertGt(jobId, 0);

        // Step 3: Run 3 multi-participant rounds
        uint256 currentOldLo = oldHashLo;
        uint256 currentOldHi = oldHashHi;

        // Track new commitment lo/hi for the best prover each round
        uint256[3] memory bestNewLos;
        uint256[3] memory bestNewHis;

        for (uint256 r = 1; r <= TOTAL_ROUNDS; r++) {
            // Start multi-participant round requiring 3 workers
            coordinator.startRoundWithThreshold(modelId, ROUND_DURATION, 3);

            (uint256 currentRound,,) = coordinator.getModelState(modelId);
            assertEq(currentRound, r);

            // Each worker submits a proof with different loss values
            // Worker1: loss=100, Worker2: loss=50 (best), Worker3: loss=150
            uint256[3] memory losses = [uint256(100), uint256(50), uint256(150)];
            address[3] memory workers = [worker1, worker2, worker3];
            uint256[3] memory newLos;
            uint256[3] memory newHis;

            for (uint256 w = 0; w < 3; w++) {
                // Each worker produces different new weights
                newLos[w] = uint256(keccak256(abi.encodePacked("newLo", r, w)));
                newHis[w] = uint256(keccak256(abi.encodePacked("newHi", r, w)));

                uint256[] memory publicInputs = _buildPublicInputs(
                    currentOldLo, currentOldHi,
                    newLos[w], newHis[w],
                    losses[w], 10, r, modelId
                );

                // Unique proof per worker per round
                bytes memory proof = abi.encodePacked(
                    ProofFixtureHardcoded.createValidProof(),
                    bytes32(uint256(uint160(workers[w]))),
                    bytes32(r)
                );

                vm.prank(workers[w]);
                coordinator.submitProof(modelId, r, proof, publicInputs);
            }

            // Multi-participant round should NOT auto-finalize
            (,,, bool isCompleted,) = coordinator.rounds(modelId, r);
            assertFalse(isCompleted, "Multi-participant round should not auto-finalize");

            // Wait for dispute period to end
            vm.warp(block.timestamp + ROUND_DURATION + 1 hours + 1);

            // Finalize round
            coordinator.finalizeRound(modelId, r);

            (,,, isCompleted,) = coordinator.rounds(modelId, r);
            assertTrue(isCompleted, "Round should be completed after finalization");

            // Best prover is worker2 (loss=50, index=1)
            bestNewLos[r - 1] = newLos[1];
            bestNewHis[r - 1] = newHis[1];

            // Update commitment to best prover's new commitment
            (, uint256 newModelCommitment,) = coordinator.getModelState(modelId);
            assertGt(newModelCommitment, 0);
            uint256 expectedCommitment = uint256(keccak256(abi.encodePacked(newLos[1], newHis[1])));
            assertEq(newModelCommitment, expectedCommitment, "Commitment should match best prover");

            currentOldLo = newLos[1];
            currentOldHi = newHis[1];
        }

        // Step 4: Verify accumulated error (10 per round × 3 rounds)
        uint256 totalError = coordinator.getAccumulatedErrorBound(modelId);
        assertEq(totalError, 10 * TOTAL_ROUNDS, "Error should accumulate across rounds");

        // Step 5: Verify registry checkpoints (initial + 3 rounds)
        uint256 checkpoints = registry.getCheckpointCount(modelId);
        assertEq(checkpoints, 1 + TOTAL_ROUNDS, "Should have initial + round checkpoints");

        // Step 6: Claim rewards for all workers
        uint256[] memory modelIds = new uint256[](TOTAL_ROUNDS);
        uint256[] memory roundIds = new uint256[](TOTAL_ROUNDS);
        for (uint256 i = 0; i < TOTAL_ROUNDS; i++) {
            modelIds[i] = modelId;
            roundIds[i] = i + 1;
        }

        uint256 w1BalBefore = token.balanceOf(worker1);
        uint256 w2BalBefore = token.balanceOf(worker2);
        uint256 w3BalBefore = token.balanceOf(worker3);

        vm.prank(worker1);
        rewards.claimRoundRewards(modelIds, roundIds);
        vm.prank(worker2);
        rewards.claimRoundRewards(modelIds, roundIds);
        vm.prank(worker3);
        rewards.claimRoundRewards(modelIds, roundIds);

        uint256 w1Earned = token.balanceOf(worker1) - w1BalBefore;
        uint256 w2Earned = token.balanceOf(worker2) - w2BalBefore;
        uint256 w3Earned = token.balanceOf(worker3) - w3BalBefore;

        // All workers should have earned something
        assertGt(w1Earned, 0, "Worker1 should earn rewards");
        assertGt(w2Earned, 0, "Worker2 should earn rewards");
        assertGt(w3Earned, 0, "Worker3 should earn rewards");

        // Worker2 (best loss) should earn >= other workers (quality bonus)
        assertGe(w2Earned, w1Earned, "Best worker should earn >= average worker");
        assertGe(w2Earned, w3Earned, "Best worker should earn >= worst worker");

        // Total earned should match REWARD_PER_ROUND * TOTAL_ROUNDS (approximately)
        uint256 totalEarned = w1Earned + w2Earned + w3Earned;
        assertGt(totalEarned, 0, "Total earned should be > 0");

        // Job fees should also have been distributed (1000 per round split 3 ways)
        // Each worker gets ~333 per round × 3 rounds = ~999 in job fees

        // Verify no double-claim
        vm.prank(worker1);
        vm.expectRevert("No rewards to claim");
        rewards.claimRoundRewards(modelIds, roundIds);
    }

    // ============ Multi-Participant Round with Finalization ============

    function test_MultiParticipantRound_FinalizeAndRewards() public {
        uint256 oldHashLo = 111;
        uint256 oldHashHi = 222;
        uint256 initialCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        uint256 modelId = coordinator.registerModel("MultiModel", "Multi-participant", "QmMulti", initialCommitment, 4, 8, 2, 2, 0);

        // Start a multi-participant round (requires 3 workers)
        coordinator.startRoundWithThreshold(modelId, ROUND_DURATION, 3);

        // All 3 workers submit
        address[3] memory workers = [worker1, worker2, worker3];
        uint256[3] memory losses = [uint256(200), uint256(100), uint256(300)];

        for (uint256 w = 0; w < 3; w++) {
            uint256 newLo = uint256(keccak256(abi.encodePacked("multi_lo", w)));
            uint256 newHi = uint256(keccak256(abi.encodePacked("multi_hi", w)));

            uint256[] memory publicInputs = _buildPublicInputs(
                oldHashLo, oldHashHi,
                newLo, newHi,
                losses[w], 5, 1, modelId
            );

            bytes memory proof = abi.encodePacked(
                ProofFixtureHardcoded.createValidProof(),
                bytes32(uint256(uint160(workers[w]))),
                bytes32(uint256(999))
            );

            vm.prank(workers[w]);
            coordinator.submitProof(modelId, 1, proof, publicInputs);
        }

        // Round should NOT be auto-finalized (minParticipants > 1)
        (,,, bool isCompleted,) = coordinator.rounds(modelId, 1);
        assertFalse(isCompleted, "Multi-participant round should not auto-finalize");

        // Verify round ext data
        (uint32 minP, uint32 validP,,,, address bestProver, uint256 bestLoss) = coordinator.getRoundExt(modelId, 1);
        assertEq(minP, 3);
        assertEq(validP, 3);
        assertEq(bestProver, worker2); // worker2 had lowest loss=100
        assertEq(bestLoss, 100);

        // Wait for dispute period to end
        vm.warp(block.timestamp + ROUND_DURATION + 1 hours + 1);

        // Finalize round
        coordinator.finalizeRound(modelId, 1);

        // Now round should be completed
        (,,, isCompleted,) = coordinator.rounds(modelId, 1);
        assertTrue(isCompleted, "Round should be completed after finalization");

        // Best prover (worker2) should be the round prover
        (,,,, address roundProver) = coordinator.rounds(modelId, 1);
        assertEq(roundProver, worker2, "Best prover should be selected");

        // Error should be accumulated
        assertEq(coordinator.getAccumulatedErrorBound(modelId), 5);

        // Claim rewards
        uint256[] memory mIds = new uint256[](1);
        uint256[] memory rIds = new uint256[](1);
        mIds[0] = modelId;
        rIds[0] = 1;

        for (uint256 i = 0; i < 3; i++) {
            uint256 balBefore = token.balanceOf(workers[i]);
            vm.prank(workers[i]);
            rewards.claimRoundRewards(mIds, rIds);
            uint256 earned = token.balanceOf(workers[i]) - balBefore;
            assertGt(earned, 0, "Each worker should earn rewards");
        }
    }

    // ============ Training Job Fee Distribution ============

    function test_TrainingJob_FeeDistribution() public {
        uint256 oldHashLo = 333;
        uint256 oldHashHi = 444;
        uint256 initialCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        uint256 modelId = coordinator.registerModel("FeeModel", "Fee test", "QmFee", initialCommitment, 4, 8, 2, 2, 0);

        // Create training job: 3000 tokens / 3 rounds = 1000 tokens per round
        uint256 jobId = coordinator.createTrainingJob(modelId, 3, JOB_DEPOSIT);

        uint256 currentOldLo = oldHashLo;
        uint256 currentOldHi = oldHashHi;

        for (uint256 r = 1; r <= 3; r++) {
            coordinator.startRound(modelId, ROUND_DURATION);

            uint256 newLo = uint256(keccak256(abi.encodePacked("fee_lo", r)));
            uint256 newHi = uint256(keccak256(abi.encodePacked("fee_hi", r)));

            // Only worker1 submits (single-participant auto-finalize)
            uint256[] memory publicInputs = _buildPublicInputs(
                currentOldLo, currentOldHi,
                newLo, newHi,
                100, 10, r, modelId
            );

            bytes memory proof = abi.encodePacked(
                ProofFixtureHardcoded.createValidProof(),
                bytes32(uint256(r)),
                bytes32(uint256(0xFEE))
            );

            uint256 w1BalBefore = token.balanceOf(worker1);

            vm.prank(worker1);
            coordinator.submitProof(modelId, r, proof, publicInputs);

            // Worker1 should receive job fees (1000 tokens per round / 1 participant)
            uint256 w1BalAfter = token.balanceOf(worker1);
            uint256 feeReceived = w1BalAfter - w1BalBefore;
            assertEq(feeReceived, JOB_DEPOSIT / 3, "Worker should receive full round fee");

            currentOldLo = newLo;
            currentOldHi = newHi;
        }
    }

    // ============ Round Expiry When Threshold Not Met ============

    function test_RoundExpiry_RefundsNothingButExpires() public {
        uint256 initialCommitment = uint256(keccak256(abi.encodePacked(uint256(555), uint256(666))));
        uint256 modelId = coordinator.registerModel("ExpiryModel", "Expiry test", "QmExpiry", initialCommitment, 4, 8, 2, 2, 0);

        // Start round requiring 3 participants but don't submit any proofs
        coordinator.startRoundWithThreshold(modelId, ROUND_DURATION, 3);

        // Warp past deadline
        vm.warp(block.timestamp + ROUND_DURATION + 1);

        // Expire the round
        coordinator.expireRound(modelId, 1);

        // Round should be marked completed
        (,,, bool isCompleted,) = coordinator.rounds(modelId, 1);
        assertTrue(isCompleted, "Expired round should be marked completed");
    }

    // ============ Invalid Proof Slashing in Lifecycle ============

    function test_InvalidProof_SlashesWorker() public {
        uint256 oldHashLo = 777;
        uint256 oldHashHi = 888;
        uint256 initialCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        uint256 modelId = coordinator.registerModel("SlashModel", "Slash test", "QmSlash", initialCommitment, 4, 8, 2, 2, 0);
        coordinator.startRound(modelId, ROUND_DURATION);

        uint256[] memory publicInputs = _buildPublicInputs(
            oldHashLo, oldHashHi, 11111, 22222, 100, 10, 1, modelId
        );
        bytes memory proof = ProofFixtureHardcoded.createValidProof();

        // Make verifier reject
        mockVerifier.setAccept(false);

        uint256 stakeBefore = _getStakeAmount(worker1);

        vm.prank(worker1);
        coordinator.submitProof(modelId, 1, proof, publicInputs);

        uint256 stakeAfter = _getStakeAmount(worker1);
        assertLt(stakeAfter, stakeBefore, "Worker should be slashed for invalid proof");

        // Slashing record should exist
        assertEq(coordinator.getSlashingRecordCount(), 1);
    }

    // ============ Reward Pool Info Verification ============

    function test_RewardPoolInfo() public view {
        (uint256 total, uint256 distributed, uint256 remaining, uint256 perRound, bool isActive) = rewards.getRewardPoolInfo();
        assertEq(total, REWARD_POOL);
        assertEq(distributed, 0);
        assertEq(remaining, REWARD_POOL);
        assertEq(perRound, REWARD_PER_ROUND);
        assertTrue(isActive);
    }

    // ============ Worker Stats After Multiple Rounds ============

    function test_WorkerStats_AccumulateAcrossRounds() public {
        uint256 oldHashLo = 1000;
        uint256 oldHashHi = 2000;
        uint256 initialCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        uint256 modelId = coordinator.registerModel("StatsModel", "Stats test", "QmStats", initialCommitment, 4, 8, 2, 2, 0);

        uint256 currentOldLo = oldHashLo;
        uint256 currentOldHi = oldHashHi;

        for (uint256 r = 1; r <= 2; r++) {
            coordinator.startRound(modelId, ROUND_DURATION);

            uint256 newLo = uint256(keccak256(abi.encodePacked("stats_lo", r)));
            uint256 newHi = uint256(keccak256(abi.encodePacked("stats_hi", r)));

            uint256[] memory publicInputs = _buildPublicInputs(
                currentOldLo, currentOldHi, newLo, newHi, 100, 10, r, modelId
            );

            bytes memory proof = abi.encodePacked(
                ProofFixtureHardcoded.createValidProof(),
                bytes32(uint256(r)),
                bytes32(uint256(0x5747))
            );

            vm.prank(worker1);
            coordinator.submitProof(modelId, r, proof, publicInputs);

            currentOldLo = newLo;
            currentOldHi = newHi;
        }

        // Check participant stats
        (uint256 totalEarned,, uint256 roundsParticipated,) = rewards.getParticipantStats(worker1);
        assertEq(roundsParticipated, 2, "Worker should have participated in 2 rounds");
        assertGt(totalEarned, 0, "Worker should have earned rewards");
    }

    // ============ Cancel Training Job ============

    function test_CancelTrainingJob_RefundsRemaining() public {
        uint256 initialCommitment = uint256(keccak256(abi.encodePacked(uint256(9000), uint256(9001))));
        uint256 modelId = coordinator.registerModel("CancelModel", "Cancel test", "QmCancel", initialCommitment, 4, 8, 2, 2, 0);

        uint256 ownerBalBefore = token.balanceOf(modelOwner);
        coordinator.createTrainingJob(modelId, 10, 1000e18);
        uint256 ownerBalAfterCreate = token.balanceOf(modelOwner);
        assertEq(ownerBalBefore - ownerBalAfterCreate, 1000e18, "Deposit should be deducted");

        // Cancel job
        coordinator.cancelTrainingJob(1);

        uint256 ownerBalAfterCancel = token.balanceOf(modelOwner);
        assertEq(ownerBalAfterCancel - ownerBalAfterCreate, 1000e18, "Full deposit should be refunded");
    }

    // ============ Helpers ============

    function _stakeWorker(address worker) internal {
        vm.startPrank(worker);
        token.approve(address(staking), type(uint256).max);
        staking.stake(STAKE_AMOUNT);
        vm.stopPrank();
    }

    function _buildPublicInputs(
        uint256 oldHashLo, uint256 oldHashHi,
        uint256 newHashLo, uint256 newHashHi,
        uint256 loss, uint256 errorBound,
        uint256 step, uint256 modelId
    ) internal view returns (uint256[] memory) {
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = oldHashLo;
        inputs[1] = oldHashHi;
        inputs[2] = newHashLo;
        inputs[3] = newHashHi;
        inputs[4] = loss;
        inputs[5] = errorBound;
        inputs[6] = step;
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(errorBound, step, modelId, coordinator.maxErrorBound());
        return inputs;
    }

    function _getStakeAmount(address staker) internal view returns (uint256) {
        (uint256 amount,,,) = staking.getStakeInfo(staker);
        return amount;
    }
}

/// @notice Mock verifier for lifecycle tests
contract MockVerifierForLifecycle {
    bool public shouldAccept = true;

    function verifyProof(bytes memory, uint256[] memory) external view returns (bool) {
        return shouldAccept;
    }

    function setAccept(bool _accept) external {
        shouldAccept = _accept;
    }
}
