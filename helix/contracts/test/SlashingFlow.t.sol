// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import "../src/core/HelixCoordinatorV2.sol";
import "../src/verification/SlashingEvidence.sol";
import "../src/token/Staking.sol";
import "../src/mocks/MockVerifier.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "./ProofFixtures.t.sol";

/// @title MockHelixToken
/// @notice Simple ERC20 for testing
contract MockHelixToken is ERC20 {
    constructor() ERC20("HELIX", "HLX") {
        _mint(msg.sender, 1_000_000 ether);
    }

    function mint(address to, uint256 amount) external {
        _mint(to, amount);
    }
}

/// @title TreasuryReceiver
/// @notice Simple contract that can receive ETH for testing
contract TreasuryReceiver {
    receive() external payable {}
}

/// @title SlashingFlowTest
/// @notice Complete end-to-end tests for the adversarial slashing flow
/// @dev Tests: invalid submission -> detection -> slash -> evidence -> reward
contract SlashingFlowTest is Test {
    // ============ Contracts ============
    HelixCoordinatorV2 public coordinator;
    SlashingEvidence public slashingEvidence;
    MockVerifier public mockVerifier;
    MockHelixToken public helixToken;
    Staking public staking;
    TreasuryReceiver public treasuryContract;

    // ============ Actors ============
    address public owner;
    address public treasury;
    address public modelOwner;
    address public honestWorker;
    address public byzantineWorker;
    address public challenger;
    address public arbitrator;

    // ============ Constants ============
    uint256 constant MIN_STAKE = 0.1 ether;
    uint256 constant LARGE_STAKE = 2 ether;
    uint256 constant ROUND_DURATION = 1 hours;

    // ============ Events ============
    event Slashed(address indexed prover, uint256 indexed modelId, uint256 roundId, uint256 amount, uint256 remainingStake, string reason);
    event InvalidProofDetected(uint256 indexed modelId, uint256 indexed roundId, address indexed prover, bytes32 proofHash);
    event EvidenceSubmitted(
        uint256 indexed evidenceId,
        address indexed prover,
        uint64 indexed modelId,
        SlashingEvidence.ViolationType violationType,
        SlashingEvidence.ErrorCode errorCode,
        SlashingEvidence.SeverityLevel severity,
        uint128 slashedAmount
    );
    event ChallengerRewarded(
        uint256 indexed evidenceId,
        address indexed challenger,
        uint128 rewardAmount,
        uint128 protocolFee
    );

    function setUp() public {
        owner = address(this);
        modelOwner = makeAddr("modelOwner");
        honestWorker = makeAddr("honestWorker");
        byzantineWorker = makeAddr("byzantineWorker");
        challenger = makeAddr("challenger");
        arbitrator = makeAddr("arbitrator");

        // Deploy treasury receiver contract so it can receive ETH
        treasuryContract = new TreasuryReceiver();
        treasury = address(treasuryContract);

        // Deploy contracts
        mockVerifier = new MockVerifier();
        coordinator = new HelixCoordinatorV2(address(mockVerifier), treasury);
        slashingEvidence = new SlashingEvidence(address(coordinator));

        // Configure coordinator with slashing evidence
        coordinator.setSlashingEvidenceContract(address(slashingEvidence));

        // Deploy token and staking for ERC20-based tests
        helixToken = new MockHelixToken();
        staking = new Staking(
            address(helixToken),
            0.1 ether,  // minStake
            7 days,     // unbondingPeriod
            5000        // 50% slashing rate
        );

        // Fund actors
        vm.deal(modelOwner, 100 ether);
        vm.deal(honestWorker, 100 ether);
        vm.deal(byzantineWorker, 100 ether);
        vm.deal(challenger, 100 ether);

        // Give tokens to staking actors
        helixToken.mint(honestWorker, 100 ether);
        helixToken.mint(byzantineWorker, 100 ether);

        // Add arbitrator
        slashingEvidence.addArbitrator(arbitrator);

        // Set treasury in slashing evidence
        slashingEvidence.setTreasury(treasury);
    }

    // ============ Helper Functions ============

    function _setupModelAndRound() internal returns (uint256 modelId, uint256 hashLo, uint256 hashHi) {
        hashLo = 12345;
        hashHi = 67890;
        uint256 commitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));

        vm.prank(modelOwner);
        modelId = coordinator.registerModel("QmTestModel", commitment, MIN_STAKE);

        vm.prank(modelOwner);
        coordinator.startRound(modelId, ROUND_DURATION);
    }

    function _createValidPublicInputs(
        uint256 oldLo, uint256 oldHi,
        uint256 newLo, uint256 newHi,
        uint256 modelId
    ) internal pure returns (uint256[] memory) {
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = oldLo;
        inputs[1] = oldHi;
        inputs[2] = newLo;
        inputs[3] = newHi;
        inputs[4] = 100;  // loss
        inputs[5] = 10;   // error bound
        inputs[6] = 1;    // step
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(inputs[5], inputs[6], modelId, 1e18);
        return inputs;
    }

    // ============ Complete Flow Tests ============

    /// @notice Test complete flow: invalid submission -> detection -> slash -> evidence -> reward
    function test_CompleteSlashingFlow() public {
        // Setup model and round
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        // Byzantine worker stakes
        vm.prank(byzantineWorker);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        // Configure mock to fail verification
        mockVerifier.setShouldPass(false);

        // Prepare invalid proof
        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);
        bytes memory proof = new bytes(320);

        // Record initial state
        uint256 treasuryBefore = treasury.balance;
        (uint256 stakeBefore,,) = coordinator.getStake(byzantineWorker, modelId);

        // Step 1: Submit invalid proof (gets detected and slashed)
        vm.prank(byzantineWorker);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Step 2: Verify slashing occurred
        (uint256 stakeAfter,, bool slashed) = coordinator.getStake(byzantineWorker, modelId);
        assertTrue(slashed, "Worker should be marked as slashed");
        assertEq(stakeAfter, stakeBefore / 2, "Half stake should remain");
        assertEq(treasury.balance, treasuryBefore + stakeBefore / 2, "Treasury should receive slashed amount");

        // Step 3: Verify slashing record was created
        assertEq(coordinator.getSlashingRecordCount(), 1, "Should have 1 slashing record");
    }

    /// @notice Test complete flow with challenger reward distribution
    function test_CompleteFlowWithChallengerReward() public {
        // Setup model and round
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        // Both workers stake
        vm.prank(byzantineWorker);
        coordinator.stake{value: LARGE_STAKE}(modelId);
        vm.prank(honestWorker);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        // Byzantine worker submits a valid-looking proof initially
        mockVerifier.setShouldPass(true);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);
        bytes memory proof = new bytes(320);

        vm.prank(byzantineWorker);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Later, someone discovers the proof was fraudulent
        // Set verifier to fail for challenge
        mockVerifier.setShouldPass(false);

        // Challenger challenges the proof
        vm.prank(challenger);
        coordinator.challengeProof(modelId, 1, proof, inputs);

        // Verify byzantine worker was slashed
        (,, bool slashed) = coordinator.getStake(byzantineWorker, modelId);
        assertTrue(slashed, "Byzantine worker should be slashed after challenge");
    }

    /// @notice Test multi-step slashing with gradual severity escalation
    function test_GradualSlashingEscalation() public {
        // Setup model
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        // Byzantine worker stakes
        vm.prank(byzantineWorker);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        mockVerifier.setShouldPass(false);

        // Submit invalid proof and get slashed
        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);
        bytes memory proof = new bytes(320);

        vm.prank(byzantineWorker);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Verify first slashing
        (uint256 stakeAfter,, bool slashed) = coordinator.getStake(byzantineWorker, modelId);
        assertTrue(slashed);
        assertEq(stakeAfter, LARGE_STAKE / 2);

        // Cannot submit again after being slashed
        vm.prank(byzantineWorker);
        vm.expectRevert(HelixCoordinatorV2.StakeSlashed.selector);
        coordinator.submitProof(modelId, 1, proof, inputs);
    }

    /// @notice Test slashing with evidence contract integration
    function test_SlashingEvidenceIntegration() public {
        // Set slashingEvidence as coordinator for evidence submission
        slashingEvidence.setCoordinator(address(this));

        // Simulate evidence submission
        uint256 evidenceId = slashingEvidence.submitEvidence(
            byzantineWorker,
            uint64(1),         // modelId
            uint32(1),         // roundId
            SlashingEvidence.ViolationType.InvalidProof,
            SlashingEvidence.ErrorCode.PROOF_VERIFICATION_FAILED,
            uint128(1 ether),  // slashedAmount
            uint128(1 ether),  // remainingStake
            keccak256("proof"),
            challenger,
            "Invalid proof verification"
        );

        // Verify evidence was recorded
        SlashingEvidence.Evidence memory evidence = slashingEvidence.getEvidence(evidenceId);
        assertEq(evidence.prover, byzantineWorker);
        assertEq(uint8(evidence.violationType), uint8(SlashingEvidence.ViolationType.InvalidProof));
        assertEq(evidence.challenger, challenger);

        // Verify warning record was updated
        SlashingEvidence.WarningRecord memory warning = slashingEvidence.getWarningRecord(byzantineWorker);
        assertEq(warning.warningCount, 1);
    }

    /// @notice Test complete dispute flow
    function test_CompleteDisputeFlow() public {
        slashingEvidence.setCoordinator(address(this));

        // Submit evidence
        uint256 evidenceId = slashingEvidence.submitEvidence(
            byzantineWorker,
            uint64(1),
            uint32(1),
            SlashingEvidence.ViolationType.InvalidProof,
            SlashingEvidence.ErrorCode.PROOF_VERIFICATION_FAILED,
            uint128(1 ether),
            uint128(1 ether),
            keccak256("proof"),
            address(0),
            "Test slashing"
        );

        // Byzantine worker files dispute
        uint256 disputeStake = slashingEvidence.disputeStakeRequired();
        vm.deal(byzantineWorker, 10 ether);

        vm.prank(byzantineWorker);
        uint256 disputeId = slashingEvidence.fileDispute{value: disputeStake}(
            evidenceId,
            keccak256("counter-evidence"),
            "I am innocent!"
        );

        // Verify dispute was filed
        SlashingEvidence.Dispute memory dispute = slashingEvidence.getDispute(disputeId);
        assertEq(dispute.evidenceId, evidenceId);
        assertEq(dispute.disputer, byzantineWorker);

        // Evidence should be disputed
        SlashingEvidence.Evidence memory evidence = slashingEvidence.getEvidence(evidenceId);
        assertEq(uint8(evidence.status), uint8(SlashingEvidence.EvidenceStatus.Disputed));

        // Owner resolves dispute against prover
        slashingEvidence.resolveDispute(disputeId, false);

        // Evidence should be resolved
        evidence = slashingEvidence.getEvidence(evidenceId);
        assertEq(uint8(evidence.status), uint8(SlashingEvidence.EvidenceStatus.Resolved));
    }

    /// @notice Test complete appeal flow
    function test_CompleteAppealFlow() public {
        slashingEvidence.setCoordinator(address(this));

        // Submit evidence
        uint256 evidenceId = slashingEvidence.submitEvidence(
            byzantineWorker,
            uint64(1),
            uint32(1),
            SlashingEvidence.ViolationType.InvalidProof,
            SlashingEvidence.ErrorCode.PROOF_VERIFICATION_FAILED,
            uint128(1 ether),
            uint128(1 ether),
            keccak256("proof"),
            address(0),
            "Test slashing"
        );

        // File and resolve dispute against prover
        uint256 disputeStake = slashingEvidence.disputeStakeRequired();
        vm.deal(byzantineWorker, 10 ether);

        vm.prank(byzantineWorker);
        uint256 disputeId = slashingEvidence.fileDispute{value: disputeStake}(
            evidenceId,
            keccak256("counter-evidence"),
            "I am innocent!"
        );

        slashingEvidence.resolveDispute(disputeId, false);

        // File appeal
        uint256 appealStake = slashingEvidence.appealStakeRequired();

        vm.prank(byzantineWorker);
        uint256 appealId = slashingEvidence.fileAppeal{value: appealStake}(
            evidenceId,
            disputeId,
            keccak256("appeal-evidence"),
            "New evidence shows innocence"
        );

        // Evidence should be appealed
        SlashingEvidence.Evidence memory evidence = slashingEvidence.getEvidence(evidenceId);
        assertEq(uint8(evidence.status), uint8(SlashingEvidence.EvidenceStatus.Appealed));

        // Arbitrator resolves appeal in favor of prover
        uint256 byzantineBalanceBefore = byzantineWorker.balance;

        vm.prank(arbitrator);
        slashingEvidence.resolveAppeal(appealId, true);

        // Appeal stake should be returned
        assertEq(byzantineWorker.balance, byzantineBalanceBefore + appealStake);

        // Evidence should be rejected (appeal successful)
        evidence = slashingEvidence.getEvidence(evidenceId);
        assertEq(uint8(evidence.status), uint8(SlashingEvidence.EvidenceStatus.Rejected));
    }

    /// @notice Test challenger reward distribution
    function test_ChallengerRewardDistribution() public {
        slashingEvidence.setCoordinator(address(this));

        // Fund the evidence contract with rewards
        vm.deal(address(slashingEvidence), 10 ether);

        // Submit evidence with challenger
        uint256 evidenceId = slashingEvidence.submitEvidence(
            byzantineWorker,
            uint64(1),
            uint32(1),
            SlashingEvidence.ViolationType.InvalidProof,
            SlashingEvidence.ErrorCode.PROOF_VERIFICATION_FAILED,
            uint128(1 ether),
            uint128(1 ether),
            keccak256("proof"),
            challenger,
            "Test slashing"
        );

        // Fast forward past dispute period
        vm.warp(block.timestamp + 8 days);

        // Verify evidence
        slashingEvidence.verifyEvidence(evidenceId);

        // Get challenger reward details
        SlashingEvidence.Evidence memory evidence = slashingEvidence.getEvidence(evidenceId);
        uint128 expectedReward = evidence.challengerReward;

        // Record challenger balance
        uint256 challengerBalanceBefore = challenger.balance;

        // Distribute reward
        slashingEvidence.distributesChallengerReward(evidenceId);

        // Verify challenger received reward
        assertEq(challenger.balance, challengerBalanceBefore + expectedReward);

        // Verify can't distribute twice
        vm.expectRevert("Already rewarded");
        slashingEvidence.distributesChallengerReward(evidenceId);
    }

    /// @notice Test permanent ban after repeated violations
    function test_PermanentBanAfterRepeatedViolations() public {
        slashingEvidence.setCoordinator(address(this));

        // Submit multiple violations to trigger ban
        for (uint256 i = 0; i < 6; i++) {
            slashingEvidence.submitEvidence(
                byzantineWorker,
                uint64(1),
                uint32(i + 1),
                SlashingEvidence.ViolationType.InvalidProof,
                SlashingEvidence.ErrorCode.PROOF_VERIFICATION_FAILED,
                uint128(0.1 ether),
                uint128(0.9 ether - i * 0.1 ether),
                keccak256(abi.encodePacked("proof", i)),
                address(0),
                "Repeated violation"
            );
        }

        // Check if prover is banned
        assertTrue(slashingEvidence.isProverBanned(byzantineWorker));

        // Verify cannot submit more evidence for banned prover
        vm.expectRevert(abi.encodeWithSelector(
            SlashingEvidence.ProverPermanentlyBanned.selector,
            byzantineWorker
        ));
        slashingEvidence.submitEvidence(
            byzantineWorker,
            uint64(1),
            uint32(7),
            SlashingEvidence.ViolationType.InvalidProof,
            SlashingEvidence.ErrorCode.PROOF_VERIFICATION_FAILED,
            uint128(0.1 ether),
            uint128(0.3 ether),
            keccak256("proof7"),
            address(0),
            "Should fail - banned"
        );
    }

    /// @notice Test severity escalation
    function test_SeverityEscalation() public {
        slashingEvidence.setCoordinator(address(this));

        // First violation - should be warning for minor violation (first offense)
        slashingEvidence.submitEvidence(
            byzantineWorker,
            uint64(1),
            uint32(1),
            SlashingEvidence.ViolationType.TimeoutViolation,  // Minor
            SlashingEvidence.ErrorCode.SUBMISSION_TOO_LATE,
            uint128(0),  // No slash for warning
            uint128(1 ether),
            keccak256("proof1"),
            address(0),
            "First timeout"
        );

        SlashingEvidence.Evidence memory evidence1 = slashingEvidence.getEvidence(0);
        assertEq(uint8(evidence1.severity), uint8(SlashingEvidence.SeverityLevel.Warning), "First offense should be warning");

        // Verify warning record after first violation
        SlashingEvidence.WarningRecord memory record1 = slashingEvidence.getWarningRecord(byzantineWorker);
        assertEq(record1.warningCount, 1, "Should have 1 warning after first violation");

        // Second violation - threshold[Minor]=2, so with warningCount=1 we still get Warning
        vm.warp(block.timestamp + 1);
        slashingEvidence.submitEvidence(
            byzantineWorker,
            uint64(1),
            uint32(2),
            SlashingEvidence.ViolationType.TimeoutViolation,
            SlashingEvidence.ErrorCode.SUBMISSION_TOO_LATE,
            uint128(0),  // Still warning level
            uint128(1 ether),
            keccak256("proof2"),
            address(0),
            "Second timeout"
        );

        SlashingEvidence.Evidence memory evidence2 = slashingEvidence.getEvidence(1);
        // With warningCount=1 before this, escalation thresholds are: Minor >= 2
        // So this is still Warning level (but warningCount is now 2)
        assertEq(uint8(evidence2.severity), uint8(SlashingEvidence.SeverityLevel.Warning), "Second still warning");

        // Third violation - now warningCount=2 >= threshold[Minor]=2
        vm.warp(block.timestamp + 2);
        slashingEvidence.submitEvidence(
            byzantineWorker,
            uint64(1),
            uint32(3),
            SlashingEvidence.ViolationType.TimeoutViolation,
            SlashingEvidence.ErrorCode.SUBMISSION_TOO_LATE,
            uint128(0.1 ether),  // Now actually slashed
            uint128(0.9 ether),
            keccak256("proof3"),
            address(0),
            "Third timeout"
        );

        SlashingEvidence.Evidence memory evidence3 = slashingEvidence.getEvidence(2);
        // With warningCount=2 before this, we should get Minor (2 >= threshold[Minor]=2)
        assertEq(uint8(evidence3.severity), uint8(SlashingEvidence.SeverityLevel.Minor), "Third should be Minor");

        // Check warning record
        SlashingEvidence.WarningRecord memory record = slashingEvidence.getWarningRecord(byzantineWorker);
        assertEq(record.warningCount, 3, "Should have 3 warnings total");
    }

    /// @notice Test critical violation skips warning
    function test_CriticalViolationSkipsWarning() public {
        slashingEvidence.setCoordinator(address(this));

        // Critical violation (MaliciousGradient) should not get warning
        slashingEvidence.submitEvidence(
            byzantineWorker,
            uint64(1),
            uint32(1),
            SlashingEvidence.ViolationType.MaliciousGradient,  // Critical
            SlashingEvidence.ErrorCode.GRADIENT_POISONING_DETECTED,
            uint128(0.5 ether),  // Direct major slash
            uint128(0.5 ether),
            keccak256("proof"),
            address(0),
            "Gradient poisoning detected"
        );

        SlashingEvidence.Evidence memory evidence = slashingEvidence.getEvidence(0);
        // Should be Major severity, not Warning
        assertEq(uint8(evidence.severity), uint8(SlashingEvidence.SeverityLevel.Major));
    }

    /// @notice Test crypto evidence storage
    function test_CryptoEvidenceStorage() public {
        slashingEvidence.setCoordinator(address(this));

        // Submit evidence
        uint256 evidenceId = slashingEvidence.submitEvidence(
            byzantineWorker,
            uint64(1),
            uint32(1),
            SlashingEvidence.ViolationType.CommitmentMismatch,
            SlashingEvidence.ErrorCode.OLD_COMMITMENT_WRONG,
            uint128(1 ether),
            uint128(1 ether),
            keccak256("proof"),
            address(0),
            "Commitment mismatch"
        );

        // Store crypto evidence
        bytes memory invalidProof = new bytes(320);
        uint256[] memory publicInputs = new uint256[](8);
        for (uint i = 0; i < 7; i++) publicInputs[i] = i + 1;
        publicInputs[7] = 0;

        slashingEvidence.storeCryptoEvidence(
            evidenceId,
            invalidProof,
            publicInputs,
            bytes32(uint256(123)),  // expected
            bytes32(uint256(456)),  // actual
            10,                      // claimed error bound
            100,                     // actual error bound
            bytes32(uint256(789)),  // data commitment
            bytes32(uint256(101)),  // gradient commitment
            ""                       // additional data
        );

        // Retrieve and verify
        SlashingEvidence.CryptoEvidence memory crypto = slashingEvidence.getCryptoEvidence(evidenceId);
        assertEq(crypto.expectedCommitment, bytes32(uint256(123)));
        assertEq(crypto.actualCommitment, bytes32(uint256(456)));
        assertEq(crypto.claimedErrorBound, 10);
        assertEq(crypto.actualErrorBound, 100);
    }

    /// @notice Test evidence deduplication
    function test_EvidenceDeduplication() public {
        slashingEvidence.setCoordinator(address(this));

        // Submit first evidence
        slashingEvidence.submitEvidence(
            byzantineWorker,
            uint64(1),
            uint32(1),
            SlashingEvidence.ViolationType.InvalidProof,
            SlashingEvidence.ErrorCode.PROOF_VERIFICATION_FAILED,
            uint128(1 ether),
            uint128(1 ether),
            keccak256("proof"),
            address(0),
            "First submission"
        );

        // Try to submit duplicate (same params in same block) - should fail
        // Note: timestamp is part of hash, so same block = duplicate
        vm.expectRevert(abi.encodeWithSelector(
            SlashingEvidence.DuplicateEvidence.selector,
            keccak256(abi.encodePacked(
                byzantineWorker,
                uint64(1),
                uint32(1),
                SlashingEvidence.ViolationType.InvalidProof,
                SlashingEvidence.ErrorCode.PROOF_VERIFICATION_FAILED,
                keccak256("proof"),
                block.timestamp
            ))
        ));
        slashingEvidence.submitEvidence(
            byzantineWorker,
            uint64(1),
            uint32(1),
            SlashingEvidence.ViolationType.InvalidProof,
            SlashingEvidence.ErrorCode.PROOF_VERIFICATION_FAILED,
            uint128(1 ether),
            uint128(1 ether),
            keccak256("proof"),
            address(0),
            "Duplicate"
        );
    }

    /// @notice Test batch reward distribution
    function test_BatchRewardDistribution() public {
        slashingEvidence.setCoordinator(address(this));
        vm.deal(address(slashingEvidence), 100 ether);

        address[] memory challengers = new address[](3);
        uint256[] memory evidenceIds = new uint256[](3);

        // Create multiple evidences with different challengers
        for (uint256 i = 0; i < 3; i++) {
            challengers[i] = makeAddr(string(abi.encodePacked("challenger", i)));

            vm.warp(block.timestamp + i + 1);  // Different timestamps

            evidenceIds[i] = slashingEvidence.submitEvidence(
                byzantineWorker,
                uint64(1),
                uint32(i + 1),
                SlashingEvidence.ViolationType.InvalidProof,
                SlashingEvidence.ErrorCode.PROOF_VERIFICATION_FAILED,
                uint128(1 ether),
                uint128(1 ether),
                keccak256(abi.encodePacked("proof", i)),
                challengers[i],
                "Test"
            );
        }

        // Fast forward and verify all
        vm.warp(block.timestamp + 8 days);
        slashingEvidence.batchVerifyEvidence(evidenceIds);

        // Batch distribute rewards
        slashingEvidence.batchDistributeRewards(evidenceIds);

        // All challengers should have received rewards
        for (uint256 i = 0; i < 3; i++) {
            assertTrue(challengers[i].balance > 0, "Challenger should have received reward");
        }
    }

    /// @notice Test ERC20-based slashing flow
    function test_ERC20StakingSlashingFlow() public {
        // Set staking operator
        staking.setOperator(address(this));

        // Byzantine worker stakes tokens
        vm.startPrank(byzantineWorker);
        helixToken.approve(address(staking), 10 ether);
        staking.stake(10 ether);
        vm.stopPrank();

        // Verify stake
        (uint256 amount, bool isActive,,) = staking.getStakeInfo(byzantineWorker);
        assertEq(amount, 10 ether);
        assertTrue(isActive);

        // Slash worker
        staking.slash(byzantineWorker, "Invalid proof detected");

        // Verify slashing
        (uint256 amountAfter, bool isActiveAfter,,) = staking.getStakeInfo(byzantineWorker);
        assertEq(amountAfter, 5 ether);  // 50% slashed
        assertTrue(isActiveAfter);  // Still active above minimum
    }

    /// @notice Test multi-round slashing accumulation
    function test_MultiRoundSlashingAccumulation() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        // Multiple workers stake
        address[] memory workers = new address[](3);
        for (uint256 i = 0; i < 3; i++) {
            workers[i] = makeAddr(string(abi.encodePacked("worker", i)));
            vm.deal(workers[i], 10 ether);
            vm.prank(workers[i]);
            coordinator.stake{value: LARGE_STAKE}(modelId);
        }

        // First worker submits valid proof
        mockVerifier.setShouldPass(true);
        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);
        bytes memory proof = new bytes(320);

        vm.prank(workers[0]);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Start second round
        vm.prank(modelOwner);
        coordinator.startRound(modelId, ROUND_DURATION);

        // Second worker submits invalid proof
        mockVerifier.setShouldPass(false);
        uint256[] memory inputs2 = _createValidPublicInputs(1111, 2222, 3333, 4444, modelId);

        vm.prank(workers[1]);
        coordinator.submitProof(modelId, 2, proof, inputs2);

        // Verify first worker not slashed, second slashed
        (,, bool slashed0) = coordinator.getStake(workers[0], modelId);
        (,, bool slashed1) = coordinator.getStake(workers[1], modelId);

        assertFalse(slashed0, "Valid worker should not be slashed");
        assertTrue(slashed1, "Invalid worker should be slashed");
    }

    /// @notice Test slashing record immutability
    function test_SlashingRecordImmutability() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(byzantineWorker);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        mockVerifier.setShouldPass(false);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);
        bytes memory proof = new bytes(320);

        vm.prank(byzantineWorker);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Get slashing record
        uint256 recordCount = coordinator.getSlashingRecordCount();
        assertEq(recordCount, 1);

        // Verify record details via public getter
        (address prover, uint64 recordModelId, uint32 roundId, uint128 amount,, uint40 timestamp) =
            coordinator.slashingRecords(0);

        assertEq(prover, byzantineWorker);
        assertEq(recordModelId, modelId);
        assertEq(roundId, 1);
        assertEq(amount, LARGE_STAKE / 2);
        assertTrue(timestamp > 0);
    }

    /// @notice Test that deploying with zero treasury reverts
    function test_ZeroTreasuryReverts() public {
        vm.expectRevert(HelixCoordinatorV2.InvalidTreasury.selector);
        new HelixCoordinatorV2(
            address(mockVerifier),
            address(0)
        );
    }
}
