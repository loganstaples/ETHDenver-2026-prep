// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../src/core/HelixCoordinatorV4.sol";

/// @title HelixCoordinatorV4Test
/// @notice Comprehensive tests for the MPC-primary HelixCoordinatorV4 contract
contract HelixCoordinatorV4Test is Test {
    HelixCoordinatorV4 public coordinator;
    MockVerifierForV4Test public mockVerifier;

    address public treasuryAddr;
    address public jobOwner;

    // Worker accounts (with known private keys for signing)
    uint256 constant WORKER1_PK = 0x1;
    uint256 constant WORKER2_PK = 0x2;
    uint256 constant WORKER3_PK = 0x3;
    uint256 constant OUTSIDER_PK = 0x4;

    address public worker1;
    address public worker2;
    address public worker3;
    address public outsider;

    uint256 constant STAKE_AMOUNT = 1 ether;
    uint256 constant PAYMENT_AMOUNT = 10 ether;
    uint256 constant CHECKPOINT_FREQ = 100;
    uint256 constant NUM_ROUNDS = 500;
    bytes32 constant ARCH_HASH = keccak256("784,32,10");

    event JobRegistered(uint256 indexed jobId, address indexed owner, bytes32 architectureHash, uint256 checkpointFreq, uint256 numRounds, uint256 paymentAmount);
    event WorkerJoined(uint256 indexed jobId, address indexed worker, uint256 stakeAmount);
    event CheckpointSubmitted(uint256 indexed jobId, uint256 indexed stepNumber, bytes32 weightCommitment, uint256 loss, uint256 signerCount);
    event WorkerSlashed(uint256 indexed jobId, address indexed cheater, uint256 stepNumber, uint256 slashedAmount, uint256 bountyAmount, uint256 reporterCount);
    event TrainingCompleted(uint256 indexed jobId, bytes32 finalCommitment, uint256 totalSteps, uint256 activeWorkerCount);
    event PaymentDistributed(uint256 indexed jobId, address indexed worker, uint256 amount);

    function setUp() public {
        treasuryAddr = makeAddr("treasury");
        jobOwner = makeAddr("jobOwner");

        worker1 = vm.addr(WORKER1_PK);
        worker2 = vm.addr(WORKER2_PK);
        worker3 = vm.addr(WORKER3_PK);
        outsider = vm.addr(OUTSIDER_PK);

        // Deploy mock verifier
        mockVerifier = new MockVerifierForV4Test();

        // Deploy coordinator
        coordinator = new HelixCoordinatorV4(treasuryAddr, address(mockVerifier));

        // Fund accounts
        vm.deal(jobOwner, 100 ether);
        vm.deal(worker1, 10 ether);
        vm.deal(worker2, 10 ether);
        vm.deal(worker3, 10 ether);
        vm.deal(outsider, 10 ether);
    }

    // ============ Helper Functions ============

    /// @dev Register a job and have 3 workers join
    function _setupJobWith3Workers() internal returns (uint256 jobId) {
        vm.prank(jobOwner);
        jobId = coordinator.registerTrainingJob{value: PAYMENT_AMOUNT}(
            ARCH_HASH, CHECKPOINT_FREQ, NUM_ROUNDS, PAYMENT_AMOUNT, false, 0, false, 0, address(0)
        );

        vm.prank(worker1);
        coordinator.stakeAndJoin{value: STAKE_AMOUNT}(jobId);

        vm.prank(worker2);
        coordinator.stakeAndJoin{value: STAKE_AMOUNT}(jobId);

        vm.prank(worker3);
        coordinator.stakeAndJoin{value: STAKE_AMOUNT}(jobId);
    }

    /// @dev Build checkpoint message hash
    function _buildCheckpointMessage(
        uint256 jobId,
        uint256 stepNumber,
        bytes32 weightCommitment,
        uint256 loss
    ) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked(
            "HELIX_CHECKPOINT",
            jobId,
            stepNumber,
            weightCommitment,
            loss
        ));
    }

    /// @dev Build MAC failure message hash
    function _buildMACFailureMessage(
        uint256 jobId,
        uint256 stepNumber,
        address cheater,
        bytes memory evidence
    ) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked(
            "HELIX_MAC_FAILURE",
            jobId,
            stepNumber,
            cheater,
            evidence
        ));
    }

    /// @dev Build completion message hash
    function _buildCompletionMessage(
        uint256 jobId,
        bytes32 finalCommitment
    ) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked(
            "HELIX_COMPLETE",
            jobId,
            finalCommitment
        ));
    }

    /// @dev Sign a message hash with a private key (EIP-191 eth_sign)
    function _sign(uint256 pk, bytes32 messageHash) internal pure returns (bytes memory) {
        bytes32 ethSignedHash = MessageHashUtils.toEthSignedMessageHash(messageHash);
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(pk, ethSignedHash);
        return abi.encodePacked(r, s, v);
    }

    /// @dev Create signatures for a checkpoint from workers 1, 2, 3
    function _signCheckpoint(
        uint256 jobId,
        uint256 step,
        bytes32 commitment,
        uint256 loss
    ) internal pure returns (bytes[] memory) {
        bytes32 message = keccak256(abi.encodePacked(
            "HELIX_CHECKPOINT", jobId, step, commitment, loss
        ));
        bytes[] memory sigs = new bytes[](3);
        sigs[0] = _sign(WORKER1_PK, message);
        sigs[1] = _sign(WORKER2_PK, message);
        sigs[2] = _sign(WORKER3_PK, message);
        return sigs;
    }

    /// @dev Create signatures for a checkpoint from specific worker PKs
    function _signCheckpointWith(
        uint256 jobId,
        uint256 step,
        bytes32 commitment,
        uint256 loss,
        uint256[] memory pks
    ) internal pure returns (bytes[] memory) {
        bytes32 message = keccak256(abi.encodePacked(
            "HELIX_CHECKPOINT", jobId, step, commitment, loss
        ));
        bytes[] memory sigs = new bytes[](pks.length);
        for (uint256 i = 0; i < pks.length; i++) {
            sigs[i] = _sign(pks[i], message);
        }
        return sigs;
    }

    /// @dev Create MAC failure report signatures
    function _signMACFailure(
        uint256 jobId,
        uint256 step,
        address cheater,
        bytes memory evidence,
        uint256[] memory pks
    ) internal pure returns (bytes[] memory) {
        bytes32 message = keccak256(abi.encodePacked(
            "HELIX_MAC_FAILURE", jobId, step, cheater, evidence
        ));
        bytes[] memory sigs = new bytes[](pks.length);
        for (uint256 i = 0; i < pks.length; i++) {
            sigs[i] = _sign(pks[i], message);
        }
        return sigs;
    }

    /// @dev Create completion signatures from specific worker PKs
    function _signCompletion(
        uint256 jobId,
        bytes32 finalCommitment,
        uint256[] memory pks
    ) internal pure returns (bytes[] memory) {
        bytes32 message = keccak256(abi.encodePacked(
            "HELIX_COMPLETE", jobId, finalCommitment
        ));
        bytes[] memory sigs = new bytes[](pks.length);
        for (uint256 i = 0; i < pks.length; i++) {
            sigs[i] = _sign(pks[i], message);
        }
        return sigs;
    }

    // ============ Constructor Tests ============

    function test_Constructor_SetsAddresses() public view {
        assertEq(coordinator.treasury(), treasuryAddr);
        assertEq(coordinator.owner(), address(this));
        assertEq(address(coordinator.verifier()), address(mockVerifier));
    }

    function test_Constructor_RejectsZeroTreasury() public {
        vm.expectRevert(HelixCoordinatorV4.InvalidTreasury.selector);
        new HelixCoordinatorV4(address(0), address(mockVerifier));
    }

    function test_Constructor_AllowsZeroVerifier() public {
        HelixCoordinatorV4 c = new HelixCoordinatorV4(treasuryAddr, address(0));
        assertEq(address(c.verifier()), address(0));
    }

    // ============ Job Registration Tests ============

    function test_RegisterTrainingJob() public {
        vm.prank(jobOwner);
        uint256 jobId = coordinator.registerTrainingJob{value: PAYMENT_AMOUNT}(
            ARCH_HASH, CHECKPOINT_FREQ, NUM_ROUNDS, PAYMENT_AMOUNT, false, 0, false, 0, address(0)
        );
        assertEq(jobId, 0);

        (address jOwner, uint256 step, uint256 rounds, uint256 payment, uint256 activeCount, bool active, bool completed,,) =
            coordinator.getJobSummary(jobId);
        assertEq(jOwner, jobOwner);
        assertEq(step, 0);
        assertEq(rounds, NUM_ROUNDS);
        assertEq(payment, PAYMENT_AMOUNT);
        assertEq(activeCount, 0);
        assertTrue(active);
        assertFalse(completed);
    }

    function test_RegisterTrainingJob_RejectsZeroPayment() public {
        vm.prank(jobOwner);
        vm.expectRevert(HelixCoordinatorV4.InvalidPayment.selector);
        coordinator.registerTrainingJob{value: 0}(ARCH_HASH, CHECKPOINT_FREQ, NUM_ROUNDS, 0, false, 0, false, 0, address(0));
    }

    function test_RegisterTrainingJob_RejectsMismatchedPayment() public {
        vm.prank(jobOwner);
        vm.expectRevert(HelixCoordinatorV4.InvalidPayment.selector);
        coordinator.registerTrainingJob{value: 1 ether}(ARCH_HASH, CHECKPOINT_FREQ, NUM_ROUNDS, 2 ether, false, 0, false, 0, address(0));
    }

    function test_RegisterTrainingJob_RejectsZeroCheckpointFreq() public {
        vm.prank(jobOwner);
        vm.expectRevert(HelixCoordinatorV4.InvalidCheckpointFreq.selector);
        coordinator.registerTrainingJob{value: PAYMENT_AMOUNT}(ARCH_HASH, 0, NUM_ROUNDS, PAYMENT_AMOUNT, false, 0, false, 0, address(0));
    }

    function test_RegisterTrainingJob_RejectsZeroRounds() public {
        vm.prank(jobOwner);
        vm.expectRevert(HelixCoordinatorV4.InvalidNumRounds.selector);
        coordinator.registerTrainingJob{value: PAYMENT_AMOUNT}(ARCH_HASH, CHECKPOINT_FREQ, 0, PAYMENT_AMOUNT, false, 0, false, 0, address(0));
    }

    function test_RegisterMultipleJobs() public {
        vm.startPrank(jobOwner);
        uint256 j1 = coordinator.registerTrainingJob{value: 1 ether}(ARCH_HASH, 10, 100, 1 ether, false, 0, false, 0, address(0));
        uint256 j2 = coordinator.registerTrainingJob{value: 2 ether}(ARCH_HASH, 20, 200, 2 ether, false, 0, false, 0, address(0));
        vm.stopPrank();

        assertEq(j1, 0);
        assertEq(j2, 1);
    }

    // ============ Worker Registration Tests ============

    function test_StakeAndJoin() public {
        vm.prank(jobOwner);
        uint256 jobId = coordinator.registerTrainingJob{value: PAYMENT_AMOUNT}(
            ARCH_HASH, CHECKPOINT_FREQ, NUM_ROUNDS, PAYMENT_AMOUNT, false, 0, false, 0, address(0)
        );

        vm.prank(worker1);
        coordinator.stakeAndJoin{value: STAKE_AMOUNT}(jobId);

        (uint256 stakeAmt,, , bool registered, bool slashed) = coordinator.getWorkerInfo(jobId, worker1);
        assertEq(stakeAmt, STAKE_AMOUNT);
        assertTrue(registered);
        assertFalse(slashed);
        assertEq(coordinator.getActiveWorkerCount(jobId), 1);
    }

    function test_StakeAndJoin_RejectsInsufficientStake() public {
        vm.prank(jobOwner);
        uint256 jobId = coordinator.registerTrainingJob{value: PAYMENT_AMOUNT}(
            ARCH_HASH, CHECKPOINT_FREQ, NUM_ROUNDS, PAYMENT_AMOUNT, false, 0, false, 0, address(0)
        );

        vm.prank(worker1);
        vm.expectRevert(HelixCoordinatorV4.InsufficientStake.selector);
        coordinator.stakeAndJoin{value: 0.0001 ether}(jobId); // Below 0.001 ether min
    }

    function test_StakeAndJoin_RejectsDuplicateRegistration() public {
        vm.prank(jobOwner);
        uint256 jobId = coordinator.registerTrainingJob{value: PAYMENT_AMOUNT}(
            ARCH_HASH, CHECKPOINT_FREQ, NUM_ROUNDS, PAYMENT_AMOUNT, false, 0, false, 0, address(0)
        );

        vm.prank(worker1);
        coordinator.stakeAndJoin{value: STAKE_AMOUNT}(jobId);

        vm.prank(worker1);
        vm.expectRevert(HelixCoordinatorV4.AlreadyRegistered.selector);
        coordinator.stakeAndJoin{value: STAKE_AMOUNT}(jobId);
    }

    // ============ TEST 1: Full Job Lifecycle ============

    function test_full_job_lifecycle() public {
        // Register job with 3 workers
        uint256 jobId = _setupJobWith3Workers();

        // Verify initial state
        assertEq(coordinator.getActiveWorkerCount(jobId), 3);

        // Submit 5 checkpoints with valid signatures
        for (uint256 step = 100; step <= 500; step += 100) {
            bytes32 commitment = keccak256(abi.encodePacked("weights_at_step_", step));
            uint256 loss = 1000 - step; // Decreasing loss
            bytes[] memory sigs = _signCheckpoint(jobId, step, commitment, loss);
            coordinator.submitCheckpoint(jobId, step, commitment, loss, sigs);
        }

        // Verify checkpoints stored
        assertEq(coordinator.getCheckpointCount(jobId), 5);

        // Verify last checkpoint data
        (uint256 cpStep, bytes32 cpCommitment, uint256 cpLoss, , uint256 cpSigners) =
            coordinator.getCheckpoint(jobId, 4);
        assertEq(cpStep, 500);
        assertEq(cpCommitment, keccak256(abi.encodePacked("weights_at_step_", uint256(500))));
        assertEq(cpLoss, 500);
        assertEq(cpSigners, 3);

        // Complete training
        bytes32 finalCommitment = keccak256("final_weights");
        uint256[] memory completionPKs = new uint256[](3);
        completionPKs[0] = WORKER1_PK;
        completionPKs[1] = WORKER2_PK;
        completionPKs[2] = WORKER3_PK;
        bytes[] memory completionSigs = _signCompletion(jobId, finalCommitment, completionPKs);

        // Track balances before completion
        uint256 w1BalBefore = worker1.balance;
        uint256 w2BalBefore = worker2.balance;
        uint256 w3BalBefore = worker3.balance;

        coordinator.completeTraining(jobId, finalCommitment, completionSigs);

        // Verify job is completed
        (, , , , , bool active, bool completed,,) = coordinator.getJobSummary(jobId);
        assertFalse(active);
        assertTrue(completed);

        // Verify payment was distributed (all workers participated equally)
        uint256 w1Payment = worker1.balance - w1BalBefore;
        uint256 w2Payment = worker2.balance - w2BalBefore;
        uint256 w3Payment = worker3.balance - w3BalBefore;
        uint256 totalPaid = w1Payment + w2Payment + w3Payment;
        assertEq(totalPaid, PAYMENT_AMOUNT);

        // Each worker should get approximately 1/3 (equal participation)
        // Allow for small rounding differences
        assertApproxEqAbs(w1Payment, PAYMENT_AMOUNT / 3, 2);
        assertApproxEqAbs(w2Payment, PAYMENT_AMOUNT / 3, 2);
        assertApproxEqAbs(w3Payment, PAYMENT_AMOUNT / 3, 2);
    }

    // ============ TEST 2: MAC Failure Slashing ============

    function test_mac_failure_slashing() public {
        uint256 jobId = _setupJobWith3Workers();

        // Submit a checkpoint first so there's some state
        bytes32 commitment = keccak256("weights_100");
        bytes[] memory cpSigs = _signCheckpoint(jobId, 100, commitment, 500);
        coordinator.submitCheckpoint(jobId, 100, commitment, 500, cpSigs);

        // Workers 1 and 3 report MAC failure against worker 2
        bytes memory evidence = abi.encode("sigma_values", "pairwise_check_failure");
        uint256[] memory reporterPKs = new uint256[](2);
        reporterPKs[0] = WORKER1_PK;
        reporterPKs[1] = WORKER3_PK;
        bytes[] memory reportSigs = _signMACFailure(jobId, 150, worker2, evidence, reporterPKs);

        uint256 w1BalBefore = worker1.balance;
        uint256 w3BalBefore = worker3.balance;
        uint256 treasuryBalBefore = treasuryAddr.balance;

        coordinator.reportMACFailure(jobId, 150, worker2, evidence, reportSigs);

        // Verify worker2 is slashed
        (, , , , bool slashed) = coordinator.getWorkerInfo(jobId, worker2);
        assertTrue(slashed);

        // Verify worker2 removed from active list
        assertEq(coordinator.getActiveWorkerCount(jobId), 2);
        assertFalse(coordinator.isActiveWorker(jobId, worker2));

        // Verify bounty distributed (10% to reporters)
        uint256 expectedBountyTotal = (STAKE_AMOUNT * 1000) / 10000; // 0.1 ether
        uint256 bountyPerReporter = expectedBountyTotal / 2;
        uint256 actualBountyDistributed = bountyPerReporter * 2;

        assertEq(worker1.balance - w1BalBefore, bountyPerReporter);
        assertEq(worker3.balance - w3BalBefore, bountyPerReporter);

        // Treasury gets the rest
        uint256 expectedTreasury = STAKE_AMOUNT - actualBountyDistributed;
        assertEq(treasuryAddr.balance - treasuryBalBefore, expectedTreasury);

        // Verify MAC failure report stored
        assertEq(coordinator.getMACFailureReportCount(jobId), 1);
        (uint256 rStep, address rCheater, , uint256 rSlashed, uint256 rReporters) =
            coordinator.getMACFailureReport(jobId, 0);
        assertEq(rStep, 150);
        assertEq(rCheater, worker2);
        assertEq(rSlashed, STAKE_AMOUNT);
        assertEq(rReporters, 2);
    }

    // ============ TEST 3: Insufficient Signatures Rejected ============

    function test_insufficient_signatures_rejected() public {
        uint256 jobId = _setupJobWith3Workers();

        bytes32 commitment = keccak256("weights_100");
        uint256 loss = 500;

        // Only 1 signature out of 3 required
        uint256[] memory onePK = new uint256[](1);
        onePK[0] = WORKER1_PK;
        bytes[] memory oneSig = _signCheckpointWith(jobId, 100, commitment, loss, onePK);

        vm.expectRevert(HelixCoordinatorV4.InvalidSignatureCount.selector);
        coordinator.submitCheckpoint(jobId, 100, commitment, loss, oneSig);
    }

    function test_two_of_three_signatures_rejected() public {
        uint256 jobId = _setupJobWith3Workers();

        bytes32 commitment = keccak256("weights_100");
        uint256 loss = 500;

        // 2 signatures out of 3 required
        uint256[] memory twoPKs = new uint256[](2);
        twoPKs[0] = WORKER1_PK;
        twoPKs[1] = WORKER2_PK;
        bytes[] memory twoSigs = _signCheckpointWith(jobId, 100, commitment, loss, twoPKs);

        vm.expectRevert(HelixCoordinatorV4.InvalidSignatureCount.selector);
        coordinator.submitCheckpoint(jobId, 100, commitment, loss, twoSigs);
    }

    // ============ TEST 4: Fake Reporter Rejected ============

    function test_fake_reporter_rejected() public {
        uint256 jobId = _setupJobWith3Workers();

        // Try to report with a non-registered worker (outsider) signing
        bytes memory evidence = abi.encode("fake_evidence");
        uint256[] memory fakePKs = new uint256[](2);
        fakePKs[0] = WORKER1_PK;
        fakePKs[1] = OUTSIDER_PK; // Not a registered worker!
        bytes[] memory reportSigs = _signMACFailure(jobId, 100, worker2, evidence, fakePKs);

        vm.expectRevert(HelixCoordinatorV4.InvalidSigner.selector);
        coordinator.reportMACFailure(jobId, 100, worker2, evidence, reportSigs);
    }

    function test_cheater_cannot_sign_own_report() public {
        uint256 jobId = _setupJobWith3Workers();

        // Worker2 (the cheater) tries to sign the report against themselves
        bytes memory evidence = abi.encode("evidence");
        uint256[] memory pks = new uint256[](2);
        pks[0] = WORKER1_PK;
        pks[1] = WORKER2_PK; // Worker2 is the cheater!
        bytes[] memory reportSigs = _signMACFailure(jobId, 100, worker2, evidence, pks);

        vm.expectRevert(HelixCoordinatorV4.InvalidSigner.selector);
        coordinator.reportMACFailure(jobId, 100, worker2, evidence, reportSigs);
    }

    // ============ TEST 5: Payment Proportional ============

    function test_payment_proportional() public {
        uint256 jobId = _setupJobWith3Workers();

        // Submit checkpoint at step 50 with all 3 workers
        bytes32 commitment50 = keccak256("weights_50");
        bytes[] memory sigs50 = _signCheckpoint(jobId, 50, commitment50, 800);
        coordinator.submitCheckpoint(jobId, 50, commitment50, 800, sigs50);

        // Slash worker2 at step 50
        bytes memory evidence = abi.encode("mac_failure");
        uint256[] memory reporterPKs = new uint256[](2);
        reporterPKs[0] = WORKER1_PK;
        reporterPKs[1] = WORKER3_PK;
        bytes[] memory reportSigs = _signMACFailure(jobId, 50, worker2, evidence, reporterPKs);
        coordinator.reportMACFailure(jobId, 50, worker2, evidence, reportSigs);

        // Continue with 2 workers, submit more checkpoints
        uint256[] memory twoPKs = new uint256[](2);
        twoPKs[0] = WORKER1_PK;
        twoPKs[1] = WORKER3_PK;

        bytes32 commitment100 = keccak256("weights_100");
        bytes[] memory sigs100 = _signCheckpointWith(jobId, 100, commitment100, 400, twoPKs);
        coordinator.submitCheckpoint(jobId, 100, commitment100, 400, sigs100);

        // Complete at step 100 with 2 remaining workers
        bytes32 finalCommitment = keccak256("final");
        bytes[] memory completionSigs = _signCompletion(jobId, finalCommitment, twoPKs);

        uint256 w1BalBefore = worker1.balance;
        uint256 w3BalBefore = worker3.balance;

        coordinator.completeTraining(jobId, finalCommitment, completionSigs);

        uint256 w1Payment = worker1.balance - w1BalBefore;
        uint256 w3Payment = worker3.balance - w3BalBefore;

        // Both worker1 and worker3 participated from step 0 to step 100 (same weight)
        // Worker2 was slashed and gets nothing
        // Payment should be split equally between worker1 and worker3
        assertEq(w1Payment + w3Payment, PAYMENT_AMOUNT);
        assertApproxEqAbs(w1Payment, PAYMENT_AMOUNT / 2, 1);
        assertApproxEqAbs(w3Payment, PAYMENT_AMOUNT / 2, 1);
    }

    // ============ TEST 6: Double Slash Prevention ============

    function test_double_slash_prevention() public {
        uint256 jobId = _setupJobWith3Workers();

        // First MAC failure report succeeds
        bytes memory evidence = abi.encode("evidence1");
        uint256[] memory reporterPKs = new uint256[](2);
        reporterPKs[0] = WORKER1_PK;
        reporterPKs[1] = WORKER3_PK;
        bytes[] memory reportSigs = _signMACFailure(jobId, 100, worker2, evidence, reporterPKs);
        coordinator.reportMACFailure(jobId, 100, worker2, evidence, reportSigs);

        // Second report for same cheater at same step should fail
        bytes memory evidence2 = abi.encode("evidence2");
        bytes[] memory reportSigs2 = _signMACFailure(jobId, 100, worker2, evidence2, reporterPKs);
        vm.expectRevert(HelixCoordinatorV4.CheaterAlreadySlashed.selector);
        coordinator.reportMACFailure(jobId, 100, worker2, evidence2, reportSigs2);

        // Report at a different step should also fail (worker already slashed globally)
        bytes[] memory reportSigs3 = _signMACFailure(jobId, 200, worker2, evidence2, reporterPKs);
        vm.expectRevert(HelixCoordinatorV4.CheaterAlreadySlashed.selector);
        coordinator.reportMACFailure(jobId, 200, worker2, evidence2, reportSigs3);
    }

    // ============ Additional Edge Case Tests ============

    function test_checkpoint_with_zk_proof() public {
        uint256 jobId = _setupJobWith3Workers();

        bytes memory proof = hex"deadbeef";
        uint256[] memory publicInputs = new uint256[](6);
        publicInputs[0] = 0; // old_hash_lo
        publicInputs[1] = 0; // old_hash_hi
        publicInputs[2] = 1; // new_hash_lo
        publicInputs[3] = 2; // new_hash_hi
        publicInputs[4] = 3; // delta_hash
        publicInputs[5] = 4; // error_bound

        bytes32 commitment = keccak256("zk_weights");
        coordinator.submitCheckpointWithProof(jobId, 100, commitment, 500, proof, publicInputs);

        assertEq(coordinator.getCheckpointCount(jobId), 1);
        (uint256 cpStep, bytes32 cpCommitment, uint256 cpLoss, , uint256 cpSigners) =
            coordinator.getCheckpoint(jobId, 0);
        assertEq(cpStep, 100);
        assertEq(cpCommitment, commitment);
        assertEq(cpLoss, 500);
        assertEq(cpSigners, 0); // ZK proof, not signatures
    }

    function test_checkpoint_with_zk_proof_rejects_invalid() public {
        uint256 jobId = _setupJobWith3Workers();

        mockVerifier.setAccept(false);

        bytes memory proof = hex"deadbeef";
        uint256[] memory publicInputs = new uint256[](6);
        for (uint i = 0; i < 6; i++) publicInputs[i] = i;

        vm.expectRevert(HelixCoordinatorV4.InvalidProof.selector);
        coordinator.submitCheckpointWithProof(jobId, 100, keccak256("x"), 500, proof, publicInputs);
    }

    function test_checkpoint_with_zk_proof_requires_verifier() public {
        // Deploy coordinator without verifier
        HelixCoordinatorV4 noVerifier = new HelixCoordinatorV4(treasuryAddr, address(0));

        vm.prank(jobOwner);
        uint256 jobId = noVerifier.registerTrainingJob{value: 1 ether}(ARCH_HASH, 10, 100, 1 ether, false, 0, false, 0, address(0));

        bytes memory proof = hex"deadbeef";
        uint256[] memory publicInputs = new uint256[](6);
        for (uint i = 0; i < 6; i++) publicInputs[i] = i;

        vm.expectRevert(HelixCoordinatorV4.VerifierNotSet.selector);
        noVerifier.submitCheckpointWithProof(jobId, 100, keccak256("x"), 500, proof, publicInputs);
    }

    function test_stake_withdrawal_after_cooldown() public {
        uint256 jobId = _setupJobWith3Workers();

        // Complete training
        bytes32 finalCommitment = keccak256("final");
        uint256[] memory allPKs = new uint256[](3);
        allPKs[0] = WORKER1_PK;
        allPKs[1] = WORKER2_PK;
        allPKs[2] = WORKER3_PK;
        bytes[] memory sigs = _signCompletion(jobId, finalCommitment, allPKs);
        coordinator.completeTraining(jobId, finalCommitment, sigs);

        // Can't withdraw before cooldown
        vm.prank(worker1);
        vm.expectRevert(HelixCoordinatorV4.CooldownNotExpired.selector);
        coordinator.withdrawStake(jobId);

        // Fast forward past cooldown
        vm.warp(block.timestamp + 7 days + 1);

        uint256 balBefore = worker1.balance;
        vm.prank(worker1);
        coordinator.withdrawStake(jobId);

        assertEq(worker1.balance - balBefore, STAKE_AMOUNT);

        // Can't withdraw twice
        vm.prank(worker1);
        vm.expectRevert(HelixCoordinatorV4.AlreadyWithdrawn.selector);
        coordinator.withdrawStake(jobId);
    }

    function test_slashed_worker_cannot_withdraw_stake() public {
        uint256 jobId = _setupJobWith3Workers();

        // Slash worker2
        bytes memory evidence = abi.encode("evidence");
        uint256[] memory reporterPKs = new uint256[](2);
        reporterPKs[0] = WORKER1_PK;
        reporterPKs[1] = WORKER3_PK;
        bytes[] memory reportSigs = _signMACFailure(jobId, 100, worker2, evidence, reporterPKs);
        coordinator.reportMACFailure(jobId, 100, worker2, evidence, reportSigs);

        // Complete with remaining workers
        bytes32 finalCommitment = keccak256("final");
        uint256[] memory remainPKs = new uint256[](2);
        remainPKs[0] = WORKER1_PK;
        remainPKs[1] = WORKER3_PK;
        bytes[] memory completionSigs = _signCompletion(jobId, finalCommitment, remainPKs);
        coordinator.completeTraining(jobId, finalCommitment, completionSigs);

        // Fast forward
        vm.warp(block.timestamp + 7 days + 1);

        // Slashed worker can't withdraw
        vm.prank(worker2);
        vm.expectRevert(HelixCoordinatorV4.WorkerSlashedError.selector);
        coordinator.withdrawStake(jobId);
    }

    function test_mac_failure_insufficient_reporters() public {
        uint256 jobId = _setupJobWith3Workers();

        // Only 1 reporter (need majority = 2 for 3 workers)
        bytes memory evidence = abi.encode("evidence");
        uint256[] memory onePK = new uint256[](1);
        onePK[0] = WORKER1_PK;
        bytes[] memory reportSigs = _signMACFailure(jobId, 100, worker2, evidence, onePK);

        vm.expectRevert(HelixCoordinatorV4.InsufficientReporters.selector);
        coordinator.reportMACFailure(jobId, 100, worker2, evidence, reportSigs);
    }

    function test_duplicate_checkpoint_signer_rejected() public {
        uint256 jobId = _setupJobWith3Workers();

        bytes32 commitment = keccak256("weights_100");
        uint256 loss = 500;

        // Sign with worker1 twice and worker2 once (3 sigs but duplicate)
        bytes32 message = keccak256(abi.encodePacked(
            "HELIX_CHECKPOINT", jobId, uint256(100), commitment, loss
        ));
        bytes[] memory sigs = new bytes[](3);
        sigs[0] = _sign(WORKER1_PK, message);
        sigs[1] = _sign(WORKER1_PK, message); // duplicate!
        sigs[2] = _sign(WORKER3_PK, message);

        vm.expectRevert(HelixCoordinatorV4.DuplicateSigner.selector);
        coordinator.submitCheckpoint(jobId, 100, commitment, loss, sigs);
    }

    function test_non_registered_signer_rejected() public {
        uint256 jobId = _setupJobWith3Workers();

        bytes32 commitment = keccak256("weights_100");
        uint256 loss = 500;

        // Sign with outsider instead of a registered worker
        bytes32 message = keccak256(abi.encodePacked(
            "HELIX_CHECKPOINT", jobId, uint256(100), commitment, loss
        ));
        bytes[] memory sigs = new bytes[](3);
        sigs[0] = _sign(WORKER1_PK, message);
        sigs[1] = _sign(WORKER2_PK, message);
        sigs[2] = _sign(OUTSIDER_PK, message); // not registered!

        vm.expectRevert(HelixCoordinatorV4.InvalidSigner.selector);
        coordinator.submitCheckpoint(jobId, 100, commitment, loss, sigs);
    }

    function test_complete_training_distributes_proportionally_with_late_joiner() public {
        // Register job
        vm.prank(jobOwner);
        uint256 jobId = coordinator.registerTrainingJob{value: PAYMENT_AMOUNT}(
            ARCH_HASH, CHECKPOINT_FREQ, NUM_ROUNDS, PAYMENT_AMOUNT, false, 0, false, 0, address(0)
        );

        // Worker1 and Worker2 join at start
        vm.prank(worker1);
        coordinator.stakeAndJoin{value: STAKE_AMOUNT}(jobId);
        vm.prank(worker2);
        coordinator.stakeAndJoin{value: STAKE_AMOUNT}(jobId);

        // Checkpoint at step 50 (2 workers)
        bytes32 commitment50 = keccak256("weights_50");
        uint256[] memory twoPKs = new uint256[](2);
        twoPKs[0] = WORKER1_PK;
        twoPKs[1] = WORKER2_PK;
        bytes[] memory sigs50 = _signCheckpointWith(jobId, 50, commitment50, 800, twoPKs);
        coordinator.submitCheckpoint(jobId, 50, commitment50, 800, sigs50);

        // Worker3 joins at step 50
        vm.prank(worker3);
        coordinator.stakeAndJoin{value: STAKE_AMOUNT}(jobId);

        // Checkpoint at step 100 (3 workers)
        bytes32 commitment100 = keccak256("weights_100");
        bytes[] memory sigs100 = _signCheckpoint(jobId, 100, commitment100, 400);
        coordinator.submitCheckpoint(jobId, 100, commitment100, 400, sigs100);

        // Complete at step 100
        bytes32 finalCommitment = keccak256("final");
        uint256[] memory allPKs = new uint256[](3);
        allPKs[0] = WORKER1_PK;
        allPKs[1] = WORKER2_PK;
        allPKs[2] = WORKER3_PK;
        bytes[] memory completionSigs = _signCompletion(jobId, finalCommitment, allPKs);

        uint256 w1Before = worker1.balance;
        uint256 w2Before = worker2.balance;
        uint256 w3Before = worker3.balance;

        coordinator.completeTraining(jobId, finalCommitment, completionSigs);

        uint256 w1Payment = worker1.balance - w1Before;
        uint256 w2Payment = worker2.balance - w2Before;
        uint256 w3Payment = worker3.balance - w3Before;

        // Worker1 and Worker2 participated from 0-100 (100 steps)
        // Worker3 participated from 50-100 (50 steps)
        // Total weight: 100 + 100 + 50 = 250
        // Worker1: 100/250 = 40%, Worker2: 100/250 = 40%, Worker3: 50/250 = 20%
        uint256 totalPaid = w1Payment + w2Payment + w3Payment;
        assertEq(totalPaid, PAYMENT_AMOUNT);

        // Worker1 and Worker2 should get more than Worker3
        assertGt(w1Payment, w3Payment);
        assertGt(w2Payment, w3Payment);
        assertApproxEqAbs(w1Payment, w2Payment, 1); // w1 and w2 equal
    }

    function test_cannot_submit_checkpoint_to_nonexistent_job() public {
        bytes[] memory sigs = new bytes[](0);
        vm.expectRevert(HelixCoordinatorV4.JobNotFound.selector);
        coordinator.submitCheckpoint(999, 100, keccak256("x"), 500, sigs);
    }

    function test_cannot_join_completed_job() public {
        uint256 jobId = _setupJobWith3Workers();

        // Complete training
        bytes32 finalCommitment = keccak256("final");
        uint256[] memory allPKs = new uint256[](3);
        allPKs[0] = WORKER1_PK;
        allPKs[1] = WORKER2_PK;
        allPKs[2] = WORKER3_PK;
        bytes[] memory sigs = _signCompletion(jobId, finalCommitment, allPKs);
        coordinator.completeTraining(jobId, finalCommitment, sigs);

        vm.prank(outsider);
        vm.expectRevert(HelixCoordinatorV4.JobNotActive.selector);
        coordinator.stakeAndJoin{value: STAKE_AMOUNT}(jobId);
    }

    function test_setTreasury() public {
        address newTreasury = makeAddr("newTreasury");
        coordinator.setTreasury(newTreasury);
        assertEq(coordinator.treasury(), newTreasury);
    }

    function test_setTreasury_rejectsZero() public {
        vm.expectRevert(HelixCoordinatorV4.InvalidTreasury.selector);
        coordinator.setTreasury(address(0));
    }

    function test_setTreasury_onlyOwner() public {
        vm.prank(worker1);
        vm.expectRevert(HelixCoordinatorV4.OnlyOwner.selector);
        coordinator.setTreasury(makeAddr("x"));
    }

    function test_setVerifier() public {
        address newVerifier = makeAddr("newVerifier");
        coordinator.setVerifier(newVerifier);
        assertEq(address(coordinator.verifier()), newVerifier);
    }

    function test_getActiveWorkers() public {
        uint256 jobId = _setupJobWith3Workers();

        address[] memory activeList = coordinator.getActiveWorkers(jobId);
        assertEq(activeList.length, 3);
        assertEq(activeList[0], worker1);
        assertEq(activeList[1], worker2);
        assertEq(activeList[2], worker3);
    }

    function test_mac_failure_report_with_unregistered_cheater() public {
        uint256 jobId = _setupJobWith3Workers();

        bytes memory evidence = abi.encode("evidence");
        uint256[] memory reporterPKs = new uint256[](2);
        reporterPKs[0] = WORKER1_PK;
        reporterPKs[1] = WORKER3_PK;
        bytes[] memory reportSigs = _signMACFailure(jobId, 100, outsider, evidence, reporterPKs);

        vm.expectRevert(HelixCoordinatorV4.CheaterNotRegistered.selector);
        coordinator.reportMACFailure(jobId, 100, outsider, evidence, reportSigs);
    }

    function test_checkpoint_no_workers_reverts() public {
        vm.prank(jobOwner);
        uint256 jobId = coordinator.registerTrainingJob{value: PAYMENT_AMOUNT}(
            ARCH_HASH, CHECKPOINT_FREQ, NUM_ROUNDS, PAYMENT_AMOUNT, false, 0, false, 0, address(0)
        );

        bytes[] memory sigs = new bytes[](0);
        vm.expectRevert(HelixCoordinatorV4.NoWorkersJoined.selector);
        coordinator.submitCheckpoint(jobId, 100, keccak256("x"), 500, sigs);
    }

    function test_complete_training_no_workers_reverts() public {
        vm.prank(jobOwner);
        uint256 jobId = coordinator.registerTrainingJob{value: PAYMENT_AMOUNT}(
            ARCH_HASH, CHECKPOINT_FREQ, NUM_ROUNDS, PAYMENT_AMOUNT, false, 0, false, 0, address(0)
        );

        bytes[] memory sigs = new bytes[](0);
        vm.expectRevert(HelixCoordinatorV4.NoWorkersJoined.selector);
        coordinator.completeTraining(jobId, keccak256("final"), sigs);
    }

    /// @dev Integration: slash, continue, complete, and verify the whole flow
    function test_full_flow_with_slash_and_completion() public {
        uint256 jobId = _setupJobWith3Workers();

        // Checkpoint 1: all 3 workers
        bytes32 cp1 = keccak256("cp1");
        bytes[] memory sigs1 = _signCheckpoint(jobId, 100, cp1, 800);
        coordinator.submitCheckpoint(jobId, 100, cp1, 800, sigs1);

        // Slash worker2
        bytes memory evidence = abi.encode("mac_fail");
        uint256[] memory reporterPKs = new uint256[](2);
        reporterPKs[0] = WORKER1_PK;
        reporterPKs[1] = WORKER3_PK;
        bytes[] memory reportSigs = _signMACFailure(jobId, 120, worker2, evidence, reporterPKs);
        coordinator.reportMACFailure(jobId, 120, worker2, evidence, reportSigs);

        // Checkpoint 2: only workers 1 and 3
        uint256[] memory remainPKs = new uint256[](2);
        remainPKs[0] = WORKER1_PK;
        remainPKs[1] = WORKER3_PK;
        bytes32 cp2 = keccak256("cp2");
        bytes[] memory sigs2 = _signCheckpointWith(jobId, 200, cp2, 500, remainPKs);
        coordinator.submitCheckpoint(jobId, 200, cp2, 500, sigs2);

        // Complete
        bytes32 finalCommitment = keccak256("final");
        bytes[] memory completionSigs = _signCompletion(jobId, finalCommitment, remainPKs);
        coordinator.completeTraining(jobId, finalCommitment, completionSigs);

        // Verify final state
        (, , , , , bool active, bool completed,,) = coordinator.getJobSummary(jobId);
        assertFalse(active);
        assertTrue(completed);
        assertEq(coordinator.getActiveWorkerCount(jobId), 2);
        assertEq(coordinator.getCheckpointCount(jobId), 2);
        assertEq(coordinator.getMACFailureReportCount(jobId), 1);
    }

    function test_receive_ether() public {
        // Contract should be able to receive ETH
        (bool success, ) = address(coordinator).call{value: 1 ether}("");
        assertTrue(success);
    }

    // ============ ZK Proof Hash Chain Tests ============

    function test_zkProof_firstProofNoChainCheck() public {
        uint256 jobId = _setupJobWith3Workers();

        uint256[] memory pi = new uint256[](6);
        pi[0] = 0; // old_hash_lo (no predecessor)
        pi[1] = 0; // old_hash_hi
        pi[2] = 123; // new_hash_lo
        pi[3] = 456; // new_hash_hi
        pi[4] = 789; // delta_hash
        pi[5] = 100; // error_bound

        vm.prank(jobOwner);
        coordinator.submitCheckpointWithProof(
            jobId, 100, keccak256("weights1"), 500, hex"aabb", pi
        );

        // Verify hash was stored
        assertEq(coordinator.zkWeightHash(jobId, 0), 123);
        assertEq(coordinator.zkWeightHash(jobId, 1), 456);
    }

    function test_zkProof_validHashChain() public {
        uint256 jobId = _setupJobWith3Workers();

        // First proof
        uint256[] memory pi1 = new uint256[](6);
        pi1[0] = 0; pi1[1] = 0;
        pi1[2] = 100; pi1[3] = 200;
        pi1[4] = 300; pi1[5] = 50;

        vm.prank(jobOwner);
        coordinator.submitCheckpointWithProof(
            jobId, 100, keccak256("w1"), 500, hex"aabb", pi1
        );

        // Second proof — old_hash matches first proof's new_hash
        uint256[] memory pi2 = new uint256[](6);
        pi2[0] = 100; pi2[1] = 200; // Must match pi1[2], pi1[3]
        pi2[2] = 400; pi2[3] = 500;
        pi2[4] = 600; pi2[5] = 75;

        vm.prank(jobOwner);
        coordinator.submitCheckpointWithProof(
            jobId, 200, keccak256("w2"), 300, hex"ccdd", pi2
        );

        // Verify updated hash
        assertEq(coordinator.zkWeightHash(jobId, 0), 400);
        assertEq(coordinator.zkWeightHash(jobId, 1), 500);
    }

    function test_zkProof_revert_hashChainMismatch() public {
        uint256 jobId = _setupJobWith3Workers();

        // First proof
        uint256[] memory pi1 = new uint256[](6);
        pi1[0] = 0; pi1[1] = 0;
        pi1[2] = 100; pi1[3] = 200;
        pi1[4] = 300; pi1[5] = 50;

        vm.prank(jobOwner);
        coordinator.submitCheckpointWithProof(
            jobId, 100, keccak256("w1"), 500, hex"aabb", pi1
        );

        // Second proof with WRONG old_hash (doesn't match stored new_hash)
        uint256[] memory pi2 = new uint256[](6);
        pi2[0] = 999; pi2[1] = 888; // Wrong! Should be 100, 200
        pi2[2] = 400; pi2[3] = 500;
        pi2[4] = 600; pi2[5] = 75;

        vm.prank(jobOwner);
        vm.expectRevert(HelixCoordinatorV4.ZkHashChainMismatch.selector);
        coordinator.submitCheckpointWithProof(
            jobId, 200, keccak256("w2"), 300, hex"ccdd", pi2
        );
    }

    function test_zkProof_revert_wrongPublicInputsLength() public {
        uint256 jobId = _setupJobWith3Workers();

        uint256[] memory pi = new uint256[](8); // Wrong! Should be 6
        for (uint i = 0; i < 8; i++) pi[i] = i;

        vm.prank(jobOwner);
        vm.expectRevert(HelixCoordinatorV4.InvalidPublicInputsLength.selector);
        coordinator.submitCheckpointWithProof(
            jobId, 100, keccak256("w1"), 500, hex"aabb", pi
        );
    }
}

/// @notice Mock verifier for V4 tests
contract MockVerifierForV4Test {
    bool public shouldAccept = true;

    function verifyProof(bytes memory, uint256[] memory) external view returns (bool) {
        return shouldAccept;
    }

    function setAccept(bool _accept) external {
        shouldAccept = _accept;
    }
}
