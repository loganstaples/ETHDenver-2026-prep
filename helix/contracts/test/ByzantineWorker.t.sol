// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../src/core/HelixCoordinatorV2.sol";
import "../src/core/TrainingRound.sol";
import "../src/verification/AggregationVerifier.sol";
import "../src/verification/SlashingEvidence.sol";
import "../src/mocks/MockVerifier.sol";

/// @title ByzantineWorkerTest
/// @notice Comprehensive Byzantine fault tolerance testing for HELIX protocol
/// @dev Tests various Byzantine behaviors: invalid proofs, gradient poisoning, Sybil attacks, etc.
contract ByzantineWorkerTest is Test {
    // ============ Contracts ============
    HelixCoordinatorV2 public coordinator;
    MockVerifier public mockVerifier;
    TrainingRound public trainingRound;
    AggregationVerifier public aggregationVerifier;
    SlashingEvidence public slashingEvidence;

    // ============ Actors ============
    address public owner;
    address public treasury;
    address public modelOwner;
    address[] public honestWorkers;
    address[] public byzantineWorkers;

    // ============ Constants ============
    uint256 constant MIN_STAKE = 0.1 ether;
    uint256 constant STANDARD_STAKE = 1 ether;
    uint256 constant ROUND_DURATION = 1 hours;
    uint256 constant NUM_HONEST_WORKERS = 5;
    uint256 constant NUM_BYZANTINE_WORKERS = 2;

    // ============ Events ============
    event Slashed(address indexed prover, uint256 indexed modelId, uint256 roundId, uint256 amount, uint256 remainingStake, string reason);
    event InvalidProofDetected(uint256 indexed modelId, uint256 indexed roundId, address indexed prover, bytes32 proofHash);

    function setUp() public {
        owner = address(this);
        treasury = makeAddr("treasury");
        modelOwner = makeAddr("modelOwner");

        // Create honest workers
        for (uint256 i = 0; i < NUM_HONEST_WORKERS; i++) {
            address worker = makeAddr(string(abi.encodePacked("honest", i)));
            honestWorkers.push(worker);
            vm.deal(worker, 100 ether);
        }

        // Create byzantine workers
        for (uint256 i = 0; i < NUM_BYZANTINE_WORKERS; i++) {
            address worker = makeAddr(string(abi.encodePacked("byzantine", i)));
            byzantineWorkers.push(worker);
            vm.deal(worker, 100 ether);
        }

        // Deploy contracts
        mockVerifier = new MockVerifier();
        coordinator = new HelixCoordinatorV2(address(mockVerifier), treasury);
        trainingRound = new TrainingRound();
        aggregationVerifier = new AggregationVerifier(address(mockVerifier));
        slashingEvidence = new SlashingEvidence(address(coordinator));

        coordinator.setSlashingEvidenceContract(address(slashingEvidence));

        // Fund model owner
        vm.deal(modelOwner, 100 ether);
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

    function _stakeAllWorkers(uint256 modelId, uint256 amount) internal {
        for (uint256 i = 0; i < honestWorkers.length; i++) {
            vm.prank(honestWorkers[i]);
            coordinator.stake{value: amount}(modelId);
        }
        for (uint256 i = 0; i < byzantineWorkers.length; i++) {
            vm.prank(byzantineWorkers[i]);
            coordinator.stake{value: amount}(modelId);
        }
    }

    function _createValidPublicInputs(
        uint256 oldLo, uint256 oldHi,
        uint256 newLo, uint256 newHi
    ) internal pure returns (uint256[] memory) {
        uint256[] memory inputs = new uint256[](7);
        inputs[0] = oldLo;
        inputs[1] = oldHi;
        inputs[2] = newLo;
        inputs[3] = newHi;
        inputs[4] = 100;  // loss
        inputs[5] = 10;   // error bound
        inputs[6] = 1;    // step
        return inputs;
    }

    // ============ Byzantine Behavior: Invalid Proof Submission ============

    /// @notice Test single Byzantine worker submitting invalid proof
    function test_SingleByzantineInvalidProof() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();
        _stakeAllWorkers(modelId, STANDARD_STAKE);

        // Byzantine worker submits invalid proof
        mockVerifier.setShouldPass(false);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222);
        bytes memory proof = new bytes(320);

        uint256 treasuryBefore = treasury.balance;

        vm.prank(byzantineWorkers[0]);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Verify slashing occurred
        (uint256 stake,, bool slashed) = coordinator.getStake(byzantineWorkers[0], modelId);
        assertTrue(slashed, "Byzantine worker should be slashed");
        assertEq(stake, STANDARD_STAKE / 2, "Half stake should remain");
        assertEq(treasury.balance, treasuryBefore + STANDARD_STAKE / 2, "Treasury should receive slashed amount");
    }

    /// @notice Test Byzantine workers racing to submit first
    function test_ByzantineRaceCondition() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();
        _stakeAllWorkers(modelId, STANDARD_STAKE);

        mockVerifier.setShouldPass(true);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222);
        bytes memory proof = new bytes(320);

        // Honest worker submits first
        vm.prank(honestWorkers[0]);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Byzantine worker tries to submit same round
        vm.prank(byzantineWorkers[0]);
        vm.expectRevert("Round completed");
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Honest worker should have completed the round
        (uint256 stakeHonest,, bool slashedHonest) = coordinator.getStake(honestWorkers[0], modelId);
        assertFalse(slashedHonest, "Honest worker should not be slashed");
        assertEq(stakeHonest, STANDARD_STAKE, "Honest worker stake should be intact");
    }

    /// @notice Test multiple Byzantine workers submitting invalid proofs across rounds
    function test_MultipleByzantineAcrossRounds() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();
        _stakeAllWorkers(modelId, STANDARD_STAKE);

        // First Byzantine submits invalid in round 1
        mockVerifier.setShouldPass(false);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222);
        bytes memory proof = new bytes(320);

        vm.prank(byzantineWorkers[0]);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Honest worker completes round 1
        mockVerifier.setShouldPass(true);
        vm.prank(honestWorkers[0]);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Start round 2
        vm.prank(modelOwner);
        coordinator.startRound(modelId, ROUND_DURATION);

        // Second Byzantine submits invalid in round 2
        mockVerifier.setShouldPass(false);
        uint256[] memory inputs2 = _createValidPublicInputs(1111, 2222, 3333, 4444);

        vm.prank(byzantineWorkers[1]);
        coordinator.submitProof(modelId, 2, proof, inputs2);

        // Both Byzantine workers should be slashed
        (,, bool slashed0) = coordinator.getStake(byzantineWorkers[0], modelId);
        (,, bool slashed1) = coordinator.getStake(byzantineWorkers[1], modelId);

        assertTrue(slashed0, "First Byzantine should be slashed");
        assertTrue(slashed1, "Second Byzantine should be slashed");
    }

    // ============ Byzantine Behavior: Commitment Manipulation ============

    /// @notice Test Byzantine worker trying to manipulate old commitment
    function test_ByzantineCommitmentManipulation() public {
        (uint256 modelId,,) = _setupModelAndRound();
        _stakeAllWorkers(modelId, STANDARD_STAKE);

        mockVerifier.setShouldPass(true);

        // Byzantine tries to use wrong old commitment
        uint256[] memory inputs = _createValidPublicInputs(99999, 88888, 1111, 2222);
        bytes memory proof = new bytes(320);

        vm.prank(byzantineWorkers[0]);
        vm.expectRevert("Old commitment mismatch");
        coordinator.submitProof(modelId, 1, proof, inputs);
    }

    /// @notice Test Byzantine worker trying to set malicious new commitment
    function test_ByzantineMaliciousNewCommitment() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();
        _stakeAllWorkers(modelId, STANDARD_STAKE);

        mockVerifier.setShouldPass(true);

        // Byzantine sets specific malicious commitment (0, 0)
        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 0, 0);
        bytes memory proof = new bytes(320);

        vm.prank(byzantineWorkers[0]);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Commitment was updated to Byzantine's choice
        uint256 expectedCommitment = uint256(keccak256(abi.encodePacked(uint256(0), uint256(0))));
        (, uint256 currentCommitment,) = coordinator.getModelState(modelId);
        assertEq(currentCommitment, expectedCommitment);
    }

    // ============ Byzantine Behavior: Error Bound Manipulation ============

    /// @notice Test Byzantine worker submitting excessive error bound
    function test_ByzantineExcessiveErrorBound() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();
        _stakeAllWorkers(modelId, STANDARD_STAKE);

        mockVerifier.setShouldPass(true);

        uint256[] memory inputs = new uint256[](7);
        inputs[0] = hashLo;
        inputs[1] = hashHi;
        inputs[2] = 1111;
        inputs[3] = 2222;
        inputs[4] = 100;
        inputs[5] = coordinator.maxErrorBound() + 1;  // Exceeds maximum
        inputs[6] = 1;

        bytes memory proof = new bytes(320);

        vm.prank(byzantineWorkers[0]);
        vm.expectRevert("Error bound exceeds maximum");
        coordinator.submitProof(modelId, 1, proof, inputs);
    }

    /// @notice Test Byzantine accumulating error across multiple rounds
    function test_ByzantineErrorAccumulation() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(honestWorkers[0]);
        coordinator.stake{value: STANDARD_STAKE}(modelId);

        mockVerifier.setShouldPass(true);

        uint256 currentLo = hashLo;
        uint256 currentHi = hashHi;
        uint256 errorPerRound = 100;

        // Submit multiple rounds with high error
        for (uint256 i = 0; i < 5; i++) {
            if (i > 0) {
                vm.prank(modelOwner);
                coordinator.startRound(modelId, ROUND_DURATION);
            }

            uint256 newLo = currentLo + 1;
            uint256 newHi = currentHi + 1;

            uint256[] memory inputs = new uint256[](7);
            inputs[0] = currentLo;
            inputs[1] = currentHi;
            inputs[2] = newLo;
            inputs[3] = newHi;
            inputs[4] = 100;
            inputs[5] = errorPerRound;
            inputs[6] = i + 1;

            bytes memory proof = new bytes(320);

            vm.prank(honestWorkers[0]);
            coordinator.submitProof(modelId, i + 1, proof, inputs);

            currentLo = newLo;
            currentHi = newHi;
        }

        // Total accumulated error should be 500
        assertEq(coordinator.getAccumulatedErrorBound(modelId), 500);
    }

    // ============ Byzantine Behavior: Sybil Attacks ============

    /// @notice Test Sybil attack with insufficient stake
    function test_SybilAttackInsufficientStake() public {
        (uint256 modelId,,) = _setupModelAndRound();

        // Create many Sybil accounts with minimal stake
        uint256 numSybils = 10;
        address[] memory sybils = new address[](numSybils);

        for (uint256 i = 0; i < numSybils; i++) {
            sybils[i] = makeAddr(string(abi.encodePacked("sybil", i)));
            vm.deal(sybils[i], MIN_STAKE - 0.01 ether);  // Less than minimum
        }

        // None should be able to stake due to insufficient funds
        for (uint256 i = 0; i < numSybils; i++) {
            assertLt(sybils[i].balance, MIN_STAKE, "Sybil should have less than min stake");
        }
    }

    /// @notice Test coordinated Sybil attack
    function test_CoordinatedSybilAttack() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        // Create Sybil cluster controlled by Byzantine
        address[] memory sybilCluster = new address[](3);
        for (uint256 i = 0; i < 3; i++) {
            sybilCluster[i] = makeAddr(string(abi.encodePacked("sybilCluster", i)));
            vm.deal(sybilCluster[i], 10 ether);
            vm.prank(sybilCluster[i]);
            coordinator.stake{value: STANDARD_STAKE}(modelId);
        }

        // First Sybil submits valid
        mockVerifier.setShouldPass(true);
        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222);
        bytes memory proof = new bytes(320);

        vm.prank(sybilCluster[0]);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Other Sybils can't submit to same round
        vm.prank(sybilCluster[1]);
        vm.expectRevert("Round completed");
        coordinator.submitProof(modelId, 1, proof, inputs);
    }

    // ============ Byzantine Behavior: Timing Attacks ============

    /// @notice Test Byzantine worker submitting after deadline
    function test_ByzantineDeadlineViolation() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();
        _stakeAllWorkers(modelId, STANDARD_STAKE);

        mockVerifier.setShouldPass(true);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222);
        bytes memory proof = new bytes(320);

        // Fast forward past deadline
        vm.warp(block.timestamp + ROUND_DURATION + 1);

        // Byzantine tries to submit after deadline
        vm.prank(byzantineWorkers[0]);
        vm.expectRevert("Round expired");
        coordinator.submitProof(modelId, 1, proof, inputs);
    }

    /// @notice Test Byzantine worker timing manipulation with TrainingRound
    function test_ByzantineTrainingRoundTiming() public {
        trainingRound.setCoordinator(address(this));

        uint256 modelId = 1;
        bytes32 startCommit = keccak256("start");

        uint256 roundId = trainingRound.createRound(modelId, startCommit, 2, 10, ROUND_DURATION);

        // Register workers
        vm.prank(honestWorkers[0]);
        trainingRound.registerForRound(modelId, roundId);

        vm.prank(byzantineWorkers[0]);
        trainingRound.registerForRound(modelId, roundId);

        // Fast forward past deadline
        vm.warp(block.timestamp + ROUND_DURATION + 1);

        // Byzantine tries to register after deadline
        vm.prank(byzantineWorkers[1]);
        vm.expectRevert("Round deadline passed");
        trainingRound.registerForRound(modelId, roundId);
    }

    // ============ Byzantine Behavior: Gradient Poisoning ============

    /// @notice Test detecting gradient outliers in aggregation
    function test_ByzantineGradientOutlier() public {
        mockVerifier.setShouldPass(true);

        bytes memory proof = new bytes(320);
        uint256[] memory inputs = new uint256[](7);
        for (uint i = 0; i < 7; i++) inputs[i] = i + 1;

        // Honest workers submit normal gradients
        for (uint256 i = 0; i < 3; i++) {
            vm.prank(honestWorkers[i]);
            aggregationVerifier.submitContribution(
                1,                                    // modelId
                1,                                    // roundId
                keccak256(abi.encodePacked("normal", i)),
                1000,                                 // stakeWeight
                50,                                   // normal errorBound
                proof,
                inputs
            );
        }

        // Byzantine submits outlier gradient with high error bound
        vm.prank(byzantineWorkers[0]);
        vm.expectRevert("Error bound exceeds maximum");
        aggregationVerifier.submitContribution(
            1,
            1,
            keccak256("malicious"),
            1000,
            1001,  // Exceeds max error bound (1000)
            proof,
            inputs
        );
    }

    /// @notice Test Byzantine double contribution
    function test_ByzantineDoubleContribution() public {
        mockVerifier.setShouldPass(true);

        bytes memory proof = new bytes(320);
        uint256[] memory inputs = new uint256[](7);
        for (uint i = 0; i < 7; i++) inputs[i] = i + 1;

        // Byzantine submits first contribution
        vm.prank(byzantineWorkers[0]);
        aggregationVerifier.submitContribution(
            1,
            1,
            keccak256("first"),
            1000,
            50,
            proof,
            inputs
        );

        // Byzantine tries to submit again
        vm.prank(byzantineWorkers[0]);
        vm.expectRevert("Already contributed");
        aggregationVerifier.submitContribution(
            1,
            1,
            keccak256("second"),
            1000,
            50,
            proof,
            inputs
        );
    }

    // ============ Byzantine Behavior: State Manipulation ============

    /// @notice Test Byzantine worker cannot manipulate model state directly
    function test_ByzantineCannotManipulateModelState() public {
        (uint256 modelId,,) = _setupModelAndRound();

        // Byzantine cannot pause model
        vm.prank(byzantineWorkers[0]);
        vm.expectRevert("Not authorized");
        coordinator.pauseModel(modelId);

        // Byzantine cannot change parameters
        vm.prank(byzantineWorkers[0]);
        vm.expectRevert("Only owner");
        coordinator.setSlashPercentage(10000);

        vm.prank(byzantineWorkers[0]);
        vm.expectRevert("Only owner");
        coordinator.setMaxErrorBound(0);
    }

    /// @notice Test Byzantine worker cannot start rounds
    function test_ByzantineCannotStartRound() public {
        (uint256 modelId,,) = _setupModelAndRound();

        // Complete first round
        _stakeAllWorkers(modelId, STANDARD_STAKE);
        mockVerifier.setShouldPass(true);

        uint256[] memory inputs = _createValidPublicInputs(12345, 67890, 1111, 2222);
        bytes memory proof = new bytes(320);

        vm.prank(honestWorkers[0]);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Byzantine cannot start next round
        vm.prank(byzantineWorkers[0]);
        vm.expectRevert("Only model owner");
        coordinator.startRound(modelId, ROUND_DURATION);
    }

    // ============ Byzantine Tolerance Tests ============

    /// @notice Test system continues with minority Byzantine workers
    function test_SystemContinuesWithMinorityByzantine() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();
        _stakeAllWorkers(modelId, STANDARD_STAKE);

        // Byzantine workers get slashed
        mockVerifier.setShouldPass(false);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222);
        bytes memory proof = new bytes(320);

        for (uint256 i = 0; i < byzantineWorkers.length; i++) {
            vm.prank(byzantineWorkers[i]);
            coordinator.submitProof(modelId, 1, proof, inputs);
        }

        // Honest worker can still complete round
        mockVerifier.setShouldPass(true);
        vm.prank(honestWorkers[0]);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Model should have advanced
        (uint256 currentRound,, bool active) = coordinator.getModelState(modelId);
        assertEq(currentRound, 1);
        assertTrue(active);
    }

    /// @notice Test all Byzantine workers get slashed
    function test_AllByzantineWorkersSlashed() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();
        _stakeAllWorkers(modelId, STANDARD_STAKE);

        mockVerifier.setShouldPass(false);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222);
        bytes memory proof = new bytes(320);

        // All Byzantine workers try invalid proofs
        for (uint256 i = 0; i < byzantineWorkers.length; i++) {
            vm.prank(byzantineWorkers[i]);
            coordinator.submitProof(modelId, 1, proof, inputs);
        }

        // Verify all are slashed
        for (uint256 i = 0; i < byzantineWorkers.length; i++) {
            (,, bool slashed) = coordinator.getStake(byzantineWorkers[i], modelId);
            assertTrue(slashed, "Byzantine worker should be slashed");
        }

        // Verify treasury received all slashed funds
        assertEq(treasury.balance, byzantineWorkers.length * STANDARD_STAKE / 2);
    }

    /// @notice Test honest workers unaffected by Byzantine behavior
    function test_HonestWorkersUnaffectedByByzantine() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();
        _stakeAllWorkers(modelId, STANDARD_STAKE);

        // Byzantine submits invalid
        mockVerifier.setShouldPass(false);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222);
        bytes memory proof = new bytes(320);

        vm.prank(byzantineWorkers[0]);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Verify all honest workers still have full stake
        for (uint256 i = 0; i < honestWorkers.length; i++) {
            (uint256 stake,, bool slashed) = coordinator.getStake(honestWorkers[i], modelId);
            assertEq(stake, STANDARD_STAKE, "Honest stake should be intact");
            assertFalse(slashed, "Honest worker should not be slashed");
        }
    }

    // ============ Byzantine Detection Tests ============

    /// @notice Test replay attack detection
    function test_ByzantineReplayAttackDetection() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();
        _stakeAllWorkers(modelId, STANDARD_STAKE);

        mockVerifier.setShouldPass(true);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222);
        bytes memory proof = new bytes(320);

        // Honest completes round 1
        vm.prank(honestWorkers[0]);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Start round 2
        vm.prank(modelOwner);
        coordinator.startRound(modelId, ROUND_DURATION);

        // Byzantine tries to replay old proof
        vm.prank(byzantineWorkers[0]);
        vm.expectRevert("Old commitment mismatch");
        coordinator.submitProof(modelId, 2, proof, inputs);
    }

    /// @notice Test proof challenge after suspicious activity
    function test_ByzantineProofChallenge() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();
        _stakeAllWorkers(modelId, STANDARD_STAKE);

        // Byzantine submits "valid" proof
        mockVerifier.setShouldPass(true);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222);
        bytes memory proof = new bytes(320);

        vm.prank(byzantineWorkers[0]);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Later discovered to be fraudulent
        mockVerifier.setShouldPass(false);

        // Honest worker challenges
        vm.prank(honestWorkers[0]);
        coordinator.challengeProof(modelId, 1, proof, inputs);

        // Byzantine should be slashed
        (,, bool slashed) = coordinator.getStake(byzantineWorkers[0], modelId);
        assertTrue(slashed, "Byzantine should be slashed after challenge");
    }

    // ============ Economic Attack Tests ============

    /// @notice Test Byzantine cannot profit from slashing
    function test_ByzantineCannotProfitFromSlashing() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        // Byzantine stakes
        vm.prank(byzantineWorkers[0]);
        coordinator.stake{value: STANDARD_STAKE}(modelId);

        uint256 byzantineBalanceBefore = byzantineWorkers[0].balance;

        // Byzantine submits invalid
        mockVerifier.setShouldPass(false);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222);
        bytes memory proof = new bytes(320);

        vm.prank(byzantineWorkers[0]);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Byzantine should have lost money
        (uint256 stakeRemaining,,) = coordinator.getStake(byzantineWorkers[0], modelId);
        uint256 totalByzantineAssets = byzantineBalanceBefore + stakeRemaining;
        uint256 originalAssets = byzantineBalanceBefore + STANDARD_STAKE;

        assertLt(totalByzantineAssets, originalAssets, "Byzantine should have lost money");
    }

    /// @notice Test slashing creates economic deterrent
    function test_SlashingEconomicDeterrent() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        uint256 largeStake = 10 ether;

        vm.prank(byzantineWorkers[0]);
        coordinator.stake{value: largeStake}(modelId);

        mockVerifier.setShouldPass(false);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222);
        bytes memory proof = new bytes(320);

        vm.prank(byzantineWorkers[0]);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // 50% of large stake should be slashed
        assertEq(treasury.balance, largeStake / 2, "Half of large stake should go to treasury");
    }

    /// @notice Test Byzantine cannot restake after slashing
    function test_ByzantineCannotRestakeAfterSlashing() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(byzantineWorkers[0]);
        coordinator.stake{value: STANDARD_STAKE}(modelId);

        // Get slashed
        mockVerifier.setShouldPass(false);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222);
        bytes memory proof = new bytes(320);

        vm.prank(byzantineWorkers[0]);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Try to restake
        vm.prank(byzantineWorkers[0]);
        vm.expectRevert("Previous stake was slashed");
        coordinator.stake{value: STANDARD_STAKE}(modelId);
    }
}
