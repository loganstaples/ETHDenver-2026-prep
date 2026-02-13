// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import "../src/core/HelixCoordinatorV2.sol";
import "../src/core/HelixCoordinatorV3.sol";
import "../src/core/ModelRegistry.sol";
import "../src/token/HelixToken.sol";
import "../src/token/Staking.sol";
import "../src/token/Rewards.sol";
import "../src/mocks/MockVerifier.sol";
import "./ProofFixtures.t.sol";

/// @title ETH-receivable treasury for tests
contract OptTestTreasury {
    receive() external payable {}
}

// ============================================================================
// 1. Immutable Verifier Tests
// ============================================================================

/// @title ImmutableVerifierTest
/// @notice Verifies that the verifier is set at construction and cannot be changed
contract ImmutableVerifierTest is Test {
    HelixCoordinatorV2 public coordinator;
    MockVerifier public mockVerifier;
    OptTestTreasury public treasuryContract;

    function setUp() public {
        treasuryContract = new OptTestTreasury();
        mockVerifier = new MockVerifier();
        coordinator = new HelixCoordinatorV2(address(mockVerifier), address(treasuryContract));
    }

    /// @notice Test: verifier is set at construction
    function test_Verifier_SetAtConstruction() public view {
        assertEq(address(coordinator.verifier()), address(mockVerifier), "Verifier should be set at construction");
    }

    /// @notice Test: verifier is the same address across calls (immutable)
    function test_Verifier_IsImmutable() public view {
        IHelixVerifier v1 = coordinator.verifier();
        IHelixVerifier v2 = coordinator.verifier();
        assertEq(address(v1), address(v2), "Verifier should always return the same address");
    }
}

// ============================================================================
// 2. Named Constants Tests
// ============================================================================

/// @title NamedConstantsTest
/// @notice Verifies all constant values match expected defaults
contract NamedConstantsTest is Test {
    HelixCoordinatorV2 public v2;
    HelixCoordinatorV3 public v3;
    MockVerifier public mockVerifier;
    OptTestTreasury public treasuryContract;

    // V3 dependencies
    HelixToken public token;
    Staking public staking;
    Rewards public rewards;
    ModelRegistry public registry;

    function setUp() public {
        treasuryContract = new OptTestTreasury();
        mockVerifier = new MockVerifier();
        v2 = new HelixCoordinatorV2(address(mockVerifier), address(treasuryContract));

        // Deploy V3 dependencies
        token = new HelixToken(address(treasuryContract));
        staking = new Staking(address(token), 1000 ether, 7 days, 5000);
        rewards = new Rewards(address(token));
        registry = new ModelRegistry();

        v3 = new HelixCoordinatorV3(
            address(mockVerifier),
            address(staking),
            address(rewards),
            address(registry),
            address(treasuryContract)
        );
    }

    // ---- V2 Constants ----

    function test_V2_DefaultSlashPercentage() public view {
        assertEq(v2.DEFAULT_SLASH_PERCENTAGE(), 5000, "DEFAULT_SLASH_PERCENTAGE should be 5000");
        assertEq(v2.slashPercentage(), v2.DEFAULT_SLASH_PERCENTAGE(), "Constructor should use named constant");
    }

    function test_V2_DefaultStakeLockDays() public view {
        assertEq(v2.DEFAULT_STAKE_LOCK_DAYS(), 7, "DEFAULT_STAKE_LOCK_DAYS should be 7");
        assertEq(v2.stakeLockDays(), v2.DEFAULT_STAKE_LOCK_DAYS(), "Constructor should use named constant");
    }

    function test_V2_DefaultMinStake() public view {
        assertEq(v2.DEFAULT_MIN_STAKE(), 0.1 ether, "DEFAULT_MIN_STAKE should be 0.1 ether");
        assertEq(v2.defaultMinStake(), v2.DEFAULT_MIN_STAKE(), "Constructor should use named constant");
    }

    function test_V2_DefaultMaxErrorBound() public view {
        assertEq(v2.DEFAULT_MAX_ERROR_BOUND(), 1e18, "DEFAULT_MAX_ERROR_BOUND should be 1e18");
        assertEq(v2.maxErrorBound(), v2.DEFAULT_MAX_ERROR_BOUND(), "Constructor should use named constant");
    }

    function test_V2_DefaultRequiredGuardians() public view {
        assertEq(v2.DEFAULT_REQUIRED_GUARDIANS(), 2, "DEFAULT_REQUIRED_GUARDIANS should be 2");
        assertEq(v2.requiredGuardians(), v2.DEFAULT_REQUIRED_GUARDIANS(), "Constructor should use named constant");
    }

    function test_V2_DefaultRecoveryTimelock() public view {
        assertEq(v2.DEFAULT_RECOVERY_TIMELOCK(), 48 hours, "DEFAULT_RECOVERY_TIMELOCK should be 48 hours");
        assertEq(v2.recoveryTimeLock(), v2.DEFAULT_RECOVERY_TIMELOCK(), "Constructor should use named constant");
    }

    function test_V2_DefaultChallengerRewardPct() public view {
        assertEq(v2.DEFAULT_CHALLENGER_REWARD_PCT(), 1000, "DEFAULT_CHALLENGER_REWARD_PCT should be 1000");
        assertEq(v2.challengerRewardPercentage(), v2.DEFAULT_CHALLENGER_REWARD_PCT(), "Constructor should use named constant");
    }

    function test_V2_DefaultMinChallengerReward() public view {
        assertEq(v2.DEFAULT_MIN_CHALLENGER_REWARD(), 0.001 ether);
        assertEq(v2.minChallengerReward(), v2.DEFAULT_MIN_CHALLENGER_REWARD());
    }

    function test_V2_DefaultMaxChallengerReward() public view {
        assertEq(v2.DEFAULT_MAX_CHALLENGER_REWARD(), 10 ether);
        assertEq(v2.maxChallengerReward(), v2.DEFAULT_MAX_CHALLENGER_REWARD());
    }

    function test_V2_MaxPercentage() public view {
        assertEq(v2.MAX_PERCENTAGE(), 10000, "MAX_PERCENTAGE should be 10000");
    }

    function test_V2_MaxRewardPercentage() public view {
        assertEq(v2.MAX_REWARD_PERCENTAGE(), 5000, "MAX_REWARD_PERCENTAGE should be 5000");
    }

    function test_V2_ExpectedPublicInputs() public view {
        assertEq(v2.EXPECTED_PUBLIC_INPUTS(), 8, "EXPECTED_PUBLIC_INPUTS should be 8");
    }

    function test_V2_RecoveryTimelockBounds() public view {
        assertEq(v2.MIN_RECOVERY_TIMELOCK(), 1 hours, "MIN_RECOVERY_TIMELOCK should be 1 hour");
        assertEq(v2.MAX_RECOVERY_TIMELOCK(), 30 days, "MAX_RECOVERY_TIMELOCK should be 30 days");
    }

    // ---- V3 Constants ----

    function test_V3_DefaultMaxErrorBound() public view {
        assertEq(v3.DEFAULT_MAX_ERROR_BOUND(), 1e18);
        assertEq(v3.maxErrorBound(), v3.DEFAULT_MAX_ERROR_BOUND());
    }

    function test_V3_DefaultRequiredGuardians() public view {
        assertEq(v3.DEFAULT_REQUIRED_GUARDIANS(), 2);
        assertEq(v3.requiredGuardians(), v3.DEFAULT_REQUIRED_GUARDIANS());
    }

    function test_V3_DefaultRecoveryTimelock() public view {
        assertEq(v3.DEFAULT_RECOVERY_TIMELOCK(), 48 hours);
        assertEq(v3.recoveryTimeLock(), v3.DEFAULT_RECOVERY_TIMELOCK());
    }

    function test_V3_DefaultChallengerRewardPct() public view {
        assertEq(v3.DEFAULT_CHALLENGER_REWARD_PCT(), 1000);
        assertEq(v3.challengerRewardPercentage(), v3.DEFAULT_CHALLENGER_REWARD_PCT());
    }

    function test_V3_ExpectedPublicInputs() public view {
        assertEq(v3.EXPECTED_PUBLIC_INPUTS(), 8);
    }

    function test_V3_RecoveryTimelockBounds() public view {
        assertEq(v3.MIN_RECOVERY_TIMELOCK(), 1 hours);
        assertEq(v3.MAX_RECOVERY_TIMELOCK(), 30 days);
    }
}

// ============================================================================
// 3. Batch Submission Tests (V2)
// ============================================================================

/// @title BatchSubmissionTest
/// @notice Tests batch proof submission functionality in V2
contract BatchSubmissionTest is Test {
    HelixCoordinatorV2 public coordinator;
    MockVerifier public mockVerifier;
    OptTestTreasury public treasuryContract;

    address public prover;
    uint256 constant ROUND_DURATION = 1 hours;

    event BatchProofSubmitted(address indexed submitter, uint256 totalProofs);
    event ProofSubmitted(uint256 indexed modelId, uint256 indexed roundId, address indexed prover, uint256 newCommitment, uint256 errorBound);

    function setUp() public {
        treasuryContract = new OptTestTreasury();
        mockVerifier = new MockVerifier();
        coordinator = new HelixCoordinatorV2(address(mockVerifier), address(treasuryContract));
        prover = makeAddr("prover");
        vm.deal(prover, 100 ether);
    }

    /// @notice Helper to set up a model with a matching commitment
    function _setupModel(uint256 hashLo, uint256 hashHi) internal returns (uint256 modelId) {
        uint256 commitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));
        modelId = coordinator.registerModel("hash", commitment, 0.1 ether, 4, 8, 2, 2, 0);
        coordinator.startRound(modelId, ROUND_DURATION);
    }

    /// @notice Helper to create valid public inputs
    function _createPublicInputs(
        uint256 hashLo,
        uint256 hashHi,
        uint256 newLo,
        uint256 newHi,
        uint256 modelId
    ) internal view returns (uint256[] memory) {
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = hashLo;
        inputs[1] = hashHi;
        inputs[2] = newLo;
        inputs[3] = newHi;
        inputs[4] = 100;     // loss
        inputs[5] = 10;      // error bound
        inputs[6] = 1;       // step
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(10, 1, modelId, coordinator.maxErrorBound());
        return inputs;
    }

    /// @notice Test: single proof batch succeeds
    function test_BatchSubmit_SingleProof() public {
        uint256 hashLo = 1000;
        uint256 hashHi = 2000;
        uint256 modelId = _setupModel(hashLo, hashHi);

        vm.prank(prover);
        coordinator.stake{value: 1 ether}(modelId);

        uint256[] memory inputs = _createPublicInputs(hashLo, hashHi, 3000, 4000, modelId);
        bytes memory proof = new bytes(320);

        HelixCoordinatorV2.ProofSubmissionData[] memory submissions = new HelixCoordinatorV2.ProofSubmissionData[](1);
        submissions[0] = HelixCoordinatorV2.ProofSubmissionData({
            modelId: modelId,
            roundId: 1,
            proof: proof,
            publicInputs: inputs
        });

        vm.expectEmit(true, false, false, true);
        emit BatchProofSubmitted(prover, 1);

        vm.prank(prover);
        coordinator.submitProofBatch(submissions);

        // Verify round completed
        (, uint256 commitment, ) = coordinator.getModelState(modelId);
        uint256 expectedCommitment = uint256(keccak256(abi.encodePacked(uint256(3000), uint256(4000))));
        assertEq(commitment, expectedCommitment, "Model commitment should be updated");
    }

    /// @notice Test: multi-proof batch across different models
    function test_BatchSubmit_MultipleProofs() public {
        // Set up 2 models with different commitments
        uint256 hashLo1 = 1100;
        uint256 hashHi1 = 2200;
        uint256 modelId1 = _setupModel(hashLo1, hashHi1);

        uint256 hashLo2 = 3300;
        uint256 hashHi2 = 4400;
        uint256 modelId2 = _setupModel(hashLo2, hashHi2);

        // Stake on both models
        vm.startPrank(prover);
        coordinator.stake{value: 1 ether}(modelId1);
        coordinator.stake{value: 1 ether}(modelId2);
        vm.stopPrank();

        uint256[] memory inputs1 = _createPublicInputs(hashLo1, hashHi1, 5000, 6000, modelId1);
        uint256[] memory inputs2 = _createPublicInputs(hashLo2, hashHi2, 7000, 8000, modelId2);

        HelixCoordinatorV2.ProofSubmissionData[] memory submissions = new HelixCoordinatorV2.ProofSubmissionData[](2);
        submissions[0] = HelixCoordinatorV2.ProofSubmissionData({
            modelId: modelId1,
            roundId: 1,
            proof: new bytes(320),
            publicInputs: inputs1
        });
        // Use a different proof to avoid replay
        bytes memory proof2 = new bytes(320);
        proof2[0] = 0x01;
        submissions[1] = HelixCoordinatorV2.ProofSubmissionData({
            modelId: modelId2,
            roundId: 1,
            proof: proof2,
            publicInputs: inputs2
        });

        vm.expectEmit(true, false, false, true);
        emit BatchProofSubmitted(prover, 2);

        vm.prank(prover);
        coordinator.submitProofBatch(submissions);

        // Verify both rounds completed
        (, uint256 commitment1, ) = coordinator.getModelState(modelId1);
        (, uint256 commitment2, ) = coordinator.getModelState(modelId2);
        assertEq(commitment1, uint256(keccak256(abi.encodePacked(uint256(5000), uint256(6000)))));
        assertEq(commitment2, uint256(keccak256(abi.encodePacked(uint256(7000), uint256(8000)))));
    }

    /// @notice Test: empty batch reverts
    function test_BatchSubmit_EmptyReverts() public {
        HelixCoordinatorV2.ProofSubmissionData[] memory submissions = new HelixCoordinatorV2.ProofSubmissionData[](0);

        vm.prank(prover);
        vm.expectRevert(HelixCoordinatorV2.EmptyBatch.selector);
        coordinator.submitProofBatch(submissions);
    }

    /// @notice Test: invalid proof within batch still slashes
    function test_BatchSubmit_InvalidProofSlashesWithinBatch() public {
        uint256 hashLo = 5500;
        uint256 hashHi = 6600;
        uint256 modelId = _setupModel(hashLo, hashHi);

        vm.prank(prover);
        coordinator.stake{value: 1 ether}(modelId);

        // Set verifier to reject
        mockVerifier.setShouldPass(false);

        uint256[] memory inputs = _createPublicInputs(hashLo, hashHi, 7700, 8800, modelId);

        HelixCoordinatorV2.ProofSubmissionData[] memory submissions = new HelixCoordinatorV2.ProofSubmissionData[](1);
        submissions[0] = HelixCoordinatorV2.ProofSubmissionData({
            modelId: modelId,
            roundId: 1,
            proof: hex"deadbeef",
            publicInputs: inputs
        });

        vm.prank(prover);
        coordinator.submitProofBatch(submissions);

        // Verify prover was slashed
        (, , bool slashed) = coordinator.getStake(prover, modelId);
        assertTrue(slashed, "Prover should be slashed for invalid proof in batch");
    }
}

// ============================================================================
// 4. ModelRegistry Integration Tests (V2)
// ============================================================================

/// @title ModelRegistryIntegrationTest
/// @notice Tests optional ModelRegistry integration in V2
contract ModelRegistryIntegrationTest is Test {
    HelixCoordinatorV2 public coordinator;
    MockVerifier public mockVerifier;
    ModelRegistry public registry;
    OptTestTreasury public treasuryContract;

    address public prover;
    uint256 constant ROUND_DURATION = 1 hours;

    event ModelRegistryUpdated(address indexed oldRegistry, address indexed newRegistry);

    function setUp() public {
        treasuryContract = new OptTestTreasury();
        mockVerifier = new MockVerifier();
        coordinator = new HelixCoordinatorV2(address(mockVerifier), address(treasuryContract));
        registry = new ModelRegistry();
        prover = makeAddr("prover");
        vm.deal(prover, 100 ether);

        // Set coordinator as registry's coordinator so it can call updateModel
        registry.setCoordinator(address(coordinator));
    }

    /// @notice Test: registerModel creates checkpoint when registry is set
    function test_RegisterModel_CreatesCheckpoint() public {
        coordinator.setModelRegistry(address(registry));

        uint256 initialCommitment = 12345;
        coordinator.registerModel("ipfs://test", initialCommitment, 0.1 ether, 4, 8, 2, 2, 0);

        // Registry should have 1 model with initial checkpoint
        assertEq(registry.getCheckpointCount(0), 1, "Should have 1 checkpoint");
    }

    /// @notice Test: submitProof updates registry when set
    function test_SubmitProof_UpdatesRegistry() public {
        coordinator.setModelRegistry(address(registry));

        uint256 hashLo = 9900;
        uint256 hashHi = 9901;
        uint256 commitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));

        uint256 modelId = coordinator.registerModel("hash", commitment, 0.1 ether, 4, 8, 2, 2, 0);
        coordinator.startRound(modelId, ROUND_DURATION);

        vm.prank(prover);
        coordinator.stake{value: 1 ether}(modelId);

        uint256[] memory inputs = new uint256[](8);
        inputs[0] = hashLo;
        inputs[1] = hashHi;
        inputs[2] = 1111;
        inputs[3] = 2222;
        inputs[4] = 100;
        inputs[5] = 10;
        inputs[6] = 1;
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(10, 1, modelId, coordinator.maxErrorBound());

        vm.prank(prover);
        coordinator.submitProof(modelId, 1, new bytes(320), inputs);

        // Registry should have 2 checkpoints: initial + proof update
        assertEq(registry.getCheckpointCount(0), 2, "Should have 2 checkpoints after proof submission");
    }

    /// @notice Test: works without registry (null address)
    function test_WorksWithoutRegistry() public {
        // Don't set registry - should still work
        assertEq(address(coordinator.modelRegistry()), address(0), "Registry should be null");

        uint256 hashLo = 8800;
        uint256 hashHi = 8801;
        uint256 commitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));

        uint256 modelId = coordinator.registerModel("hash", commitment, 0.1 ether, 4, 8, 2, 2, 0);
        coordinator.startRound(modelId, ROUND_DURATION);

        vm.prank(prover);
        coordinator.stake{value: 1 ether}(modelId);

        uint256[] memory inputs = new uint256[](8);
        inputs[0] = hashLo;
        inputs[1] = hashHi;
        inputs[2] = 3333;
        inputs[3] = 4444;
        inputs[4] = 100;
        inputs[5] = 10;
        inputs[6] = 1;
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(10, 1, modelId, coordinator.maxErrorBound());

        vm.prank(prover);
        coordinator.submitProof(modelId, 1, new bytes(320), inputs);

        // Verify round completed normally without registry
        (, uint256 newCommitment,) = coordinator.getModelState(modelId);
        assertEq(newCommitment, uint256(keccak256(abi.encodePacked(uint256(3333), uint256(4444)))));
    }

    /// @notice Test: setModelRegistry only owner
    function test_SetModelRegistry_OnlyOwner() public {
        address nonOwner = makeAddr("nonOwner");
        vm.prank(nonOwner);
        vm.expectRevert(HelixCoordinatorV2.OnlyOwner.selector);
        coordinator.setModelRegistry(address(registry));
    }

    /// @notice Test: setModelRegistry emits event
    function test_SetModelRegistry_EmitsEvent() public {
        vm.expectEmit(true, true, false, false);
        emit ModelRegistryUpdated(address(0), address(registry));
        coordinator.setModelRegistry(address(registry));
    }
}

// ============================================================================
// 5. V3 Batch Submission Tests
// ============================================================================

/// @title V3BatchSubmissionTest
/// @notice Tests batch proof submission functionality in V3
contract V3BatchSubmissionTest is Test {
    HelixCoordinatorV3 public coordinator;
    MockVerifier public mockVerifier;
    HelixToken public token;
    Staking public staking;
    Rewards public rewards;
    ModelRegistry public registry;
    OptTestTreasury public treasuryContract;

    address public prover;
    uint256 constant ROUND_DURATION = 1 hours;
    uint256 constant STAKE_AMOUNT = 10_000 ether;

    event BatchProofSubmitted(address indexed submitter, uint256 totalProofs);

    function setUp() public {
        treasuryContract = new OptTestTreasury();
        mockVerifier = new MockVerifier();
        token = new HelixToken(address(treasuryContract));
        staking = new Staking(address(token), 1000 ether, 7 days, 5000);
        rewards = new Rewards(address(token));
        registry = new ModelRegistry();
        prover = makeAddr("prover");

        coordinator = new HelixCoordinatorV3(
            address(mockVerifier),
            address(staking),
            address(rewards),
            address(registry),
            address(treasuryContract)
        );

        // Set coordinator on dependencies
        registry.setCoordinator(address(coordinator));
        rewards.setCoordinator(address(coordinator));
        staking.setOperator(address(coordinator));

        // Fund and stake for prover
        token.mint(prover, STAKE_AMOUNT);
        vm.startPrank(prover);
        token.approve(address(staking), STAKE_AMOUNT);
        staking.stake(STAKE_AMOUNT);
        vm.stopPrank();
    }

    /// @notice Helper to set up a model and round
    function _setupV3Model(uint256 hashLo, uint256 hashHi) internal returns (uint256 modelId) {
        uint256 commitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));
        modelId = coordinator.registerModel("Model", "desc", "hash", commitment, 4, 8, 2, 2, 0);
        coordinator.startRound(modelId, ROUND_DURATION);
    }

    /// @notice Test: single proof batch succeeds in V3
    function test_V3_BatchSubmit_SingleProof() public {
        uint256 hashLo = 1000;
        uint256 hashHi = 2000;
        uint256 modelId = _setupV3Model(hashLo, hashHi);

        uint256[] memory inputs = new uint256[](8);
        inputs[0] = hashLo;
        inputs[1] = hashHi;
        inputs[2] = 3000;
        inputs[3] = 4000;
        inputs[4] = 100;
        inputs[5] = 10;
        inputs[6] = 1;
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(10, 1, modelId, coordinator.maxErrorBound());

        HelixCoordinatorV3.ProofSubmissionData[] memory submissions = new HelixCoordinatorV3.ProofSubmissionData[](1);
        submissions[0] = HelixCoordinatorV3.ProofSubmissionData({
            modelId: modelId,
            roundId: 1,
            proof: new bytes(320),
            publicInputs: inputs
        });

        vm.expectEmit(true, false, false, true);
        emit BatchProofSubmitted(prover, 1);

        vm.prank(prover);
        coordinator.submitProofBatch(submissions);

        // Verify round completed
        (, uint256 commitment, ) = coordinator.getModelState(modelId);
        assertEq(commitment, uint256(keccak256(abi.encodePacked(uint256(3000), uint256(4000)))));
    }

    /// @notice Test: empty batch reverts in V3
    function test_V3_BatchSubmit_EmptyReverts() public {
        HelixCoordinatorV3.ProofSubmissionData[] memory submissions = new HelixCoordinatorV3.ProofSubmissionData[](0);

        vm.prank(prover);
        vm.expectRevert("Empty batch");
        coordinator.submitProofBatch(submissions);
    }
}
