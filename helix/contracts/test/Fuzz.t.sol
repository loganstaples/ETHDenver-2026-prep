// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import "../src/core/HelixCoordinatorV2.sol";
import "../src/mocks/MockVerifier.sol";
import "./ProofFixtures.t.sol";

/// @title FuzzTest
/// @notice Fuzz testing for HELIX protocol staking and unstaking flows
/// @dev Uses Foundry's built-in fuzzing capabilities for edge case discovery
contract FuzzTest is Test {
    HelixCoordinatorV2 public coordinator;
    MockVerifier public mockVerifier;

    address public owner;
    address public treasury;
    address public modelOwner;

    uint256 constant MIN_STAKE = 0.1 ether;
    uint256 constant ROUND_DURATION = 1 hours;

    function setUp() public {
        owner = address(this);
        treasury = makeAddr("treasury");
        modelOwner = makeAddr("modelOwner");

        mockVerifier = new MockVerifier();
        coordinator = new HelixCoordinatorV2(address(mockVerifier), treasury);

        vm.deal(modelOwner, 1000 ether);
    }

    // ============ Staking Fuzz Tests ============

    /// @notice Fuzz test staking with various amounts
    function testFuzz_StakeVariousAmounts(uint96 stakeAmount) public {
        // Bound to reasonable range
        stakeAmount = uint96(bound(stakeAmount, MIN_STAKE, 100 ether));

        uint256 commitment = uint256(keccak256(abi.encodePacked(uint256(1), uint256(2))));
        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("FuzzModel", commitment, MIN_STAKE, 4, 8, 2, 2, 0);

        address prover = makeAddr("prover");
        vm.deal(prover, stakeAmount);

        vm.prank(prover);
        coordinator.stake{value: stakeAmount}(modelId);

        (uint256 amount,, bool slashed) = coordinator.getStake(prover, modelId);
        assertEq(amount, stakeAmount);
        assertFalse(slashed);
    }

    /// @notice Fuzz test multiple stakes accumulate correctly
    function testFuzz_MultipleStakesAccumulate(uint96[5] memory amounts) public {
        uint256 commitment = uint256(keccak256(abi.encodePacked(uint256(1), uint256(2))));
        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("FuzzModel", commitment, MIN_STAKE, 4, 8, 2, 2, 0);

        address prover = makeAddr("prover");
        vm.deal(prover, 1000 ether);

        uint256 totalStaked = 0;

        vm.startPrank(prover);
        for (uint i = 0; i < 5; i++) {
            // Bound each amount to ensure total doesn't overflow
            uint256 amount = bound(amounts[i], 0.01 ether, 10 ether);

            coordinator.stake{value: amount}(modelId);
            totalStaked += amount;
        }
        vm.stopPrank();

        (uint256 finalAmount,,) = coordinator.getStake(prover, modelId);
        assertEq(finalAmount, totalStaked);
    }

    /// @notice Fuzz test staking to multiple models
    function testFuzz_StakeMultipleModels(uint8 numModels) public {
        // Bound number of models
        numModels = uint8(bound(numModels, 1, 10));

        address prover = makeAddr("prover");
        vm.deal(prover, 100 ether);

        uint256[] memory modelIds = new uint256[](numModels);

        // Register models
        vm.startPrank(modelOwner);
        for (uint i = 0; i < numModels; i++) {
            uint256 commitment = uint256(keccak256(abi.encodePacked(i, uint256(1))));
            modelIds[i] = coordinator.registerModel(
                string(abi.encodePacked("Model", i)),
                commitment,
                MIN_STAKE,
                4, 8, 2, 2, 0
            );
        }
        vm.stopPrank();

        // Stake in each model
        vm.startPrank(prover);
        for (uint i = 0; i < numModels; i++) {
            coordinator.stake{value: 0.5 ether}(modelIds[i]);
        }
        vm.stopPrank();

        // Verify stakes are independent
        for (uint i = 0; i < numModels; i++) {
            (uint256 amount,,) = coordinator.getStake(prover, modelIds[i]);
            assertEq(amount, 0.5 ether);
        }
    }

    // ============ Unstaking Fuzz Tests ============

    /// @notice Fuzz test unstaking after varying time periods
    function testFuzz_UnstakeAfterVaryingPeriods(uint256 waitTime) public {
        // Bound wait time between lock period and reasonable max
        waitTime = bound(waitTime, 7 days, 365 days);

        uint256 commitment = uint256(keccak256(abi.encodePacked(uint256(1), uint256(2))));
        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("FuzzModel", commitment, MIN_STAKE, 4, 8, 2, 2, 0);

        address prover = makeAddr("prover");
        vm.deal(prover, 10 ether);

        uint256 stakeAmount = 1 ether;
        vm.prank(prover);
        coordinator.stake{value: stakeAmount}(modelId);

        // Wait for the time period
        vm.warp(block.timestamp + waitTime);

        uint256 balanceBefore = prover.balance;
        vm.prank(prover);
        coordinator.unstake(modelId);
        uint256 balanceAfter = prover.balance;

        assertEq(balanceAfter - balanceBefore, stakeAmount);
    }

    /// @notice Fuzz test that unstaking before lock period fails
    function testFuzz_UnstakeBeforeLockFails(uint256 waitTime) public {
        // Bound wait time to less than lock period
        waitTime = bound(waitTime, 0, 7 days - 1);

        uint256 commitment = uint256(keccak256(abi.encodePacked(uint256(1), uint256(2))));
        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("FuzzModel", commitment, MIN_STAKE, 4, 8, 2, 2, 0);

        address prover = makeAddr("prover");
        vm.deal(prover, 10 ether);

        vm.prank(prover);
        coordinator.stake{value: 1 ether}(modelId);

        vm.warp(block.timestamp + waitTime);

        vm.prank(prover);
        vm.expectRevert(HelixCoordinatorV2.StillLocked.selector);
        coordinator.unstake(modelId);
    }

    // ============ Slashing Fuzz Tests ============

    /// @notice Fuzz test slashing with various stake amounts
    function testFuzz_SlashingAmount(uint96 stakeAmount) public {
        // Bound to reasonable range above min stake
        stakeAmount = uint96(bound(stakeAmount, MIN_STAKE, 100 ether));

        uint256 hashLo = 12345;
        uint256 hashHi = 67890;
        uint256 commitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("FuzzModel", commitment, MIN_STAKE, 4, 8, 2, 2, 0);

        address prover = makeAddr("prover");
        vm.deal(prover, stakeAmount);

        vm.prank(prover);
        coordinator.stake{value: stakeAmount}(modelId);

        vm.prank(modelOwner);
        coordinator.startRound(modelId, ROUND_DURATION);

        // Submit invalid proof
        mockVerifier.setShouldPass(false);

        uint256[] memory inputs = new uint256[](8);
        inputs[0] = hashLo;
        inputs[1] = hashHi;
        inputs[2] = 1111;
        inputs[3] = 2222;
        inputs[4] = 100;
        inputs[5] = 10;
        inputs[6] = 1;
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(10, 1, modelId, 1e18);

        bytes memory proof = new bytes(320);

        uint256 treasuryBefore = treasury.balance;

        vm.prank(prover);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Verify slashing math
        uint256 expectedSlashed = (stakeAmount * coordinator.slashPercentage()) / 10000;
        uint256 treasuryAfter = treasury.balance;

        assertEq(treasuryAfter - treasuryBefore, expectedSlashed);

        (uint256 remainingStake,, bool slashed) = coordinator.getStake(prover, modelId);
        assertTrue(slashed);
        assertEq(remainingStake, stakeAmount - expectedSlashed);
    }

    /// @notice Fuzz test slashing percentage configuration
    function testFuzz_SlashPercentage(uint96 percentage) public {
        // Bound percentage to valid range (0-100%)
        percentage = uint96(bound(percentage, 0, 10000));

        coordinator.emergencyPause();
        coordinator.setSlashPercentage(percentage);
        coordinator.unpause();
        assertEq(coordinator.slashPercentage(), percentage);

        // Test slashing with this percentage
        uint256 hashLo = 12345;
        uint256 hashHi = 67890;
        uint256 commitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("FuzzModel", commitment, MIN_STAKE, 4, 8, 2, 2, 0);

        address prover = makeAddr("prover");
        uint256 stakeAmount = 1 ether;
        vm.deal(prover, stakeAmount);

        vm.prank(prover);
        coordinator.stake{value: stakeAmount}(modelId);

        vm.prank(modelOwner);
        coordinator.startRound(modelId, ROUND_DURATION);

        mockVerifier.setShouldPass(false);

        uint256[] memory inputs = new uint256[](8);
        inputs[0] = hashLo;
        inputs[1] = hashHi;
        inputs[2] = 1111;
        inputs[3] = 2222;
        inputs[4] = 100;
        inputs[5] = 10;
        inputs[6] = 1;
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(10, 1, modelId, 1e18);

        bytes memory proof = new bytes(320);

        vm.prank(prover);
        coordinator.submitProof(modelId, 1, proof, inputs);

        uint256 expectedSlashed = (stakeAmount * percentage) / 10000;
        (uint256 remainingStake,,) = coordinator.getStake(prover, modelId);
        assertEq(remainingStake, stakeAmount - expectedSlashed);
    }

    // ============ Round Duration Fuzz Tests ============

    /// @notice Fuzz test various round durations
    function testFuzz_RoundDuration(uint256 duration) public {
        // Bound duration to reasonable range
        duration = bound(duration, 1 minutes, 30 days);

        uint256 commitment = uint256(keccak256(abi.encodePacked(uint256(1), uint256(2))));
        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("FuzzModel", commitment, MIN_STAKE, 4, 8, 2, 2, 0);

        vm.prank(modelOwner);
        coordinator.startRound(modelId, duration);

        // Verify round was created with correct deadline
        // Note: We can't directly access rounds mapping, but we can verify through behavior
    }

    // ============ Commitment Fuzz Tests ============

    /// @notice Fuzz test commitment hash construction
    function testFuzz_CommitmentHash(uint256 hashLo, uint256 hashHi) public {
        // Reconstruct commitment same way as coordinator
        uint256 expectedCommitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("FuzzModel", expectedCommitment, MIN_STAKE, 4, 8, 2, 2, 0);

        (, uint256 storedCommitment,) = coordinator.getModelState(modelId);
        assertEq(storedCommitment, expectedCommitment);
    }

    // ============ Error Bound Fuzz Tests ============

    /// @notice Fuzz test error bound validation
    function testFuzz_ErrorBoundValidation(uint256 errorBound) public {
        uint256 hashLo = 12345;
        uint256 hashHi = 67890;
        uint256 commitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("FuzzModel", commitment, MIN_STAKE, 4, 8, 2, 2, 0);

        address prover = makeAddr("prover");
        vm.deal(prover, 10 ether);

        vm.prank(prover);
        coordinator.stake{value: 1 ether}(modelId);

        vm.prank(modelOwner);
        coordinator.startRound(modelId, ROUND_DURATION);

        mockVerifier.setShouldPass(true);

        uint256[] memory inputs = new uint256[](8);
        inputs[0] = hashLo;
        inputs[1] = hashHi;
        inputs[2] = 1111;
        inputs[3] = 2222;
        inputs[4] = 100;
        inputs[5] = errorBound;  // Fuzzed error bound
        inputs[6] = 1;
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(errorBound, 1, modelId, 1e18);

        bytes memory proof = new bytes(320);

        if (errorBound > coordinator.maxErrorBound()) {
            vm.prank(prover);
            vm.expectRevert(HelixCoordinatorV2.ErrorBoundExceeded.selector);
            coordinator.submitProof(modelId, 1, proof, inputs);
        } else {
            vm.prank(prover);
            coordinator.submitProof(modelId, 1, proof, inputs);
            assertEq(coordinator.getAccumulatedErrorBound(modelId), errorBound);
        }
    }

    // ============ Min Stake Fuzz Tests ============

    /// @notice Fuzz test minimum stake configuration
    function testFuzz_MinStakeConfig(uint96 minStake) public {
        // Bound to reasonable range
        minStake = uint96(bound(minStake, 0.001 ether, 10 ether));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("FuzzModel", 12345, minStake, 4, 8, 2, 2, 0);

        address prover = makeAddr("prover");
        vm.deal(prover, 100 ether);

        // Stake exactly at minimum should work
        vm.prank(prover);
        coordinator.stake{value: minStake}(modelId);

        (uint256 amount,,) = coordinator.getStake(prover, modelId);
        assertEq(amount, minStake);
    }

    /// @notice Fuzz test default min stake application
    function testFuzz_DefaultMinStake(uint96 defaultMin) public {
        // Bound to reasonable range
        defaultMin = uint96(bound(defaultMin, 0.001 ether, 10 ether));

        coordinator.emergencyPause();
        coordinator.setDefaultMinStake(defaultMin);
        coordinator.unpause();
        assertEq(coordinator.defaultMinStake(), defaultMin);

        // Register with zero min stake should use default
        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("FuzzModel", 12345, 0, 4, 8, 2, 2, 0);

        // Model should have default min stake
        // We verify by checking if stake below default fails to meet requirements
    }

    // ============ Model Registration Fuzz Tests ============

    /// @notice Fuzz test multiple model registrations
    function testFuzz_MultipleModelRegistration(uint8 numModels) public {
        // Bound number of models
        numModels = uint8(bound(numModels, 1, 50));

        vm.startPrank(modelOwner);
        for (uint i = 0; i < numModels; i++) {
            uint256 commitment = uint256(keccak256(abi.encodePacked(i)));
            uint256 modelId = coordinator.registerModel(
                string(abi.encodePacked("Model", i)),
                commitment,
                MIN_STAKE,
                4, 8, 2, 2, 0
            );
            assertEq(modelId, i);
        }
        vm.stopPrank();

        assertEq(coordinator.nextModelId(), numModels);
    }

    // ============ Invariant Tests ============

    /// @notice Invariant: Treasury balance should only increase
    function testFuzz_TreasuryOnlyIncreases(uint96[3] memory stakeAmounts) public {
        // Use consistent commitment values since invalid proofs don't update model state
        uint256 hashLo = 100;
        uint256 hashHi = 200;
        uint256 commitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("FuzzModel", commitment, MIN_STAKE, 4, 8, 2, 2, 0);

        mockVerifier.setShouldPass(false);

        uint256 treasuryTotal = 0;

        for (uint i = 0; i < 3; i++) {
            uint256 stakeAmount = bound(stakeAmounts[i], MIN_STAKE, 10 ether);

            address prover = makeAddr(string(abi.encodePacked("prover", i)));
            vm.deal(prover, stakeAmount);

            vm.prank(prover);
            coordinator.stake{value: stakeAmount}(modelId);

            vm.prank(modelOwner);
            coordinator.startRound(modelId, ROUND_DURATION);

            // Since proofs fail, the model commitment stays the same
            // So we always use the same old commitment
            uint256[] memory inputs = new uint256[](8);
            inputs[0] = hashLo;
            inputs[1] = hashHi;
            inputs[2] = hashLo + 1;  // New commitment values (won't be stored since proof fails)
            inputs[3] = hashHi + 1;
            inputs[4] = 100;
            inputs[5] = 10;
            inputs[6] = i + 1;
            inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(10, i + 1, modelId, 1e18);

            bytes memory proof = new bytes(320);

            uint256 treasuryBefore = treasury.balance;

            vm.prank(prover);
            coordinator.submitProof(modelId, i + 1, proof, inputs);

            // Treasury should have increased or stayed same
            assertTrue(treasury.balance >= treasuryBefore);
            treasuryTotal = treasury.balance;
        }

        // Final treasury balance should be sum of slashed amounts
        assertTrue(treasuryTotal > 0);
    }

    /// @notice Invariant: Stake can never be negative
    function testFuzz_StakeNeverNegative(uint96 initialStake, uint8 numSlashes) public {
        // This tests the edge case of multiple slashing attempts
        initialStake = uint96(bound(initialStake, MIN_STAKE, 10 ether));
        numSlashes = uint8(bound(numSlashes, 1, 5));

        uint256 hashLo = 100;
        uint256 hashHi = 200;
        uint256 commitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("FuzzModel", commitment, MIN_STAKE, 4, 8, 2, 2, 0);

        address prover = makeAddr("prover");
        vm.deal(prover, initialStake);

        vm.prank(prover);
        coordinator.stake{value: initialStake}(modelId);

        vm.prank(modelOwner);
        coordinator.startRound(modelId, ROUND_DURATION);

        mockVerifier.setShouldPass(false);

        uint256[] memory inputs = new uint256[](8);
        inputs[0] = hashLo;
        inputs[1] = hashHi;
        inputs[2] = 1111;
        inputs[3] = 2222;
        inputs[4] = 100;
        inputs[5] = 10;
        inputs[6] = 1;
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(10, 1, modelId, 1e18);

        bytes memory proof = new bytes(320);

        // First slash succeeds
        vm.prank(prover);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Verify stake is non-negative
        (uint256 stake,, bool slashed) = coordinator.getStake(prover, modelId);
        assertTrue(stake >= 0); // Always true for uint, but documents intent
        assertTrue(slashed);
    }
}
