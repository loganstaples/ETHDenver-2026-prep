// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../src/core/HelixCoordinatorV2.sol";
import "../src/verification/Halo2Verifier.sol";

/// @title HelixCoordinatorV2Test
/// @notice Integration tests for HelixCoordinatorV2 with real Halo2Verifier
contract HelixCoordinatorV2Test is Test {
    HelixCoordinatorV2 public coordinator;
    Halo2Verifier public verifier;

    address public owner;
    address public treasury;
    address public prover1;
    address public prover2;

    uint256 constant MIN_STAKE = 0.1 ether;
    uint256 constant ROUND_DURATION = 1 hours;

    event ModelRegistered(uint256 indexed modelId, address indexed owner, uint256 initialCommitment, string ipfsHash);
    event RoundStarted(uint256 indexed modelId, uint256 indexed roundId, uint256 deadline);
    event Staked(address indexed prover, uint256 indexed modelId, uint256 amount);
    event Slashed(address indexed prover, uint256 indexed modelId, uint256 amount, string reason);

    function setUp() public {
        owner = address(this);
        treasury = makeAddr("treasury");
        prover1 = makeAddr("prover1");
        prover2 = makeAddr("prover2");

        // Deploy verifier and coordinator
        verifier = new Halo2Verifier();
        coordinator = new HelixCoordinatorV2(address(verifier), treasury);

        // Fund provers
        vm.deal(prover1, 10 ether);
        vm.deal(prover2, 10 ether);
    }

    // ============ Model Registration Tests ============

    function test_RegisterModel() public {
        string memory ipfsHash = "QmTestHash123";
        uint256 initialCommitment = 12345;

        vm.expectEmit(true, true, false, true);
        emit ModelRegistered(0, owner, initialCommitment, ipfsHash);

        uint256 modelId = coordinator.registerModel(ipfsHash, initialCommitment, MIN_STAKE);

        assertEq(modelId, 0);

        (uint256 round, uint256 commitment, bool active) = coordinator.getModelState(modelId);
        assertEq(round, 0);
        assertEq(commitment, initialCommitment);
        assertTrue(active);
    }

    function test_RegisterMultipleModels() public {
        uint256 model1 = coordinator.registerModel("hash1", 100, MIN_STAKE);
        uint256 model2 = coordinator.registerModel("hash2", 200, MIN_STAKE);
        uint256 model3 = coordinator.registerModel("hash3", 300, MIN_STAKE);

        assertEq(model1, 0);
        assertEq(model2, 1);
        assertEq(model3, 2);
    }

    // ============ Staking Tests ============

    function test_Stake() public {
        uint256 modelId = coordinator.registerModel("hash", 100, MIN_STAKE);

        vm.prank(prover1);
        vm.expectEmit(true, true, false, true);
        emit Staked(prover1, modelId, 0.5 ether);
        coordinator.stake{value: 0.5 ether}(modelId);

        (uint256 amount, uint256 lockedUntil, bool slashed) = coordinator.getStake(prover1, modelId);
        assertEq(amount, 0.5 ether);
        assertGt(lockedUntil, block.timestamp);
        assertFalse(slashed);
    }

    function test_StakeMultipleTimes() public {
        uint256 modelId = coordinator.registerModel("hash", 100, MIN_STAKE);

        vm.startPrank(prover1);
        coordinator.stake{value: 0.3 ether}(modelId);
        coordinator.stake{value: 0.2 ether}(modelId);
        vm.stopPrank();

        (uint256 amount,,) = coordinator.getStake(prover1, modelId);
        assertEq(amount, 0.5 ether);
    }

    function test_Unstake_AfterLockPeriod() public {
        uint256 modelId = coordinator.registerModel("hash", 100, MIN_STAKE);

        vm.prank(prover1);
        coordinator.stake{value: 0.5 ether}(modelId);

        // Fast forward past lock period
        vm.warp(block.timestamp + 8 days);

        uint256 balanceBefore = prover1.balance;
        vm.prank(prover1);
        coordinator.unstake(modelId);
        uint256 balanceAfter = prover1.balance;

        assertEq(balanceAfter - balanceBefore, 0.5 ether);
    }

    function test_Unstake_RevertIfStillLocked() public {
        uint256 modelId = coordinator.registerModel("hash", 100, MIN_STAKE);

        vm.prank(prover1);
        coordinator.stake{value: 0.5 ether}(modelId);

        // Try to unstake immediately
        vm.prank(prover1);
        vm.expectRevert("Still locked");
        coordinator.unstake(modelId);
    }

    // ============ Round Tests ============

    function test_StartRound() public {
        uint256 modelId = coordinator.registerModel("hash", 100, MIN_STAKE);

        vm.expectEmit(true, true, false, true);
        emit RoundStarted(modelId, 1, block.timestamp + ROUND_DURATION);
        coordinator.startRound(modelId, ROUND_DURATION);

        (uint256 round,,) = coordinator.getModelState(modelId);
        assertEq(round, 1);
    }

    function test_StartRound_OnlyOwner() public {
        uint256 modelId = coordinator.registerModel("hash", 100, MIN_STAKE);

        vm.prank(prover1);
        vm.expectRevert("Only model owner");
        coordinator.startRound(modelId, ROUND_DURATION);
    }

    // ============ Proof Submission Tests ============

    function test_SubmitProof_InvalidProofSlashes() public {
        // Set up matching commitment
        uint256 oldHashLo = 12345;
        uint256 oldHashHi = 67890;
        uint256 correctCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        uint256 modelId = coordinator.registerModel("hash", correctCommitment, MIN_STAKE);
        coordinator.startRound(modelId, ROUND_DURATION);

        // Prover stakes
        vm.prank(prover1);
        coordinator.stake{value: 1 ether}(modelId);

        // Create invalid proof (too short)
        bytes memory invalidProof = hex"deadbeef";

        uint256[] memory publicInputs = new uint256[](7);
        publicInputs[0] = oldHashLo; // old_hash_lo
        publicInputs[1] = oldHashHi; // old_hash_hi
        publicInputs[2] = 3; // new_hash_lo
        publicInputs[3] = 4; // new_hash_hi
        publicInputs[4] = 100; // loss
        publicInputs[5] = 10; // error_bound
        publicInputs[6] = 1; // step

        uint256 treasuryBefore = treasury.balance;

        // Submit invalid proof - this will slash but NOT revert
        // (reverting would roll back the slashing)
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, invalidProof, publicInputs);

        // Verify stake was slashed
        (uint256 amount,, bool slashed) = coordinator.getStake(prover1, modelId);
        assertTrue(slashed);
        assertEq(amount, 0.5 ether); // 50% slashed

        // Verify treasury received slashed funds
        assertEq(treasury.balance - treasuryBefore, 0.5 ether);
    }

    function test_SubmitProof_InsufficientStake() public {
        uint256 modelId = coordinator.registerModel("hash", 100, MIN_STAKE);
        coordinator.startRound(modelId, ROUND_DURATION);

        // No stake
        bytes memory proof = new bytes(320);
        uint256[] memory publicInputs = new uint256[](7);

        vm.prank(prover1);
        vm.expectRevert("Insufficient stake");
        coordinator.submitProof(modelId, 1, proof, publicInputs);
    }

    function test_SubmitProof_WrongRound() public {
        uint256 modelId = coordinator.registerModel("hash", 100, MIN_STAKE);
        coordinator.startRound(modelId, ROUND_DURATION);

        vm.prank(prover1);
        coordinator.stake{value: 1 ether}(modelId);

        bytes memory proof = new bytes(320);
        uint256[] memory publicInputs = new uint256[](7);

        vm.prank(prover1);
        vm.expectRevert("Invalid round");
        coordinator.submitProof(modelId, 999, proof, publicInputs); // Wrong round
    }

    function test_SubmitProof_ExpiredRound() public {
        uint256 modelId = coordinator.registerModel("hash", 100, MIN_STAKE);
        coordinator.startRound(modelId, ROUND_DURATION);

        vm.prank(prover1);
        coordinator.stake{value: 1 ether}(modelId);

        // Fast forward past deadline
        vm.warp(block.timestamp + ROUND_DURATION + 1);

        bytes memory proof = new bytes(320);
        uint256[] memory publicInputs = new uint256[](7);

        vm.prank(prover1);
        vm.expectRevert("Round expired");
        coordinator.submitProof(modelId, 1, proof, publicInputs);
    }

    // ============ Slashing Tests ============

    function test_SlashPercentage() public {
        assertEq(coordinator.slashPercentage(), 5000); // 50%

        coordinator.setSlashPercentage(2500); // 25%
        assertEq(coordinator.slashPercentage(), 2500);
    }

    function test_SlashedProverCannotSubmit() public {
        // Set up matching commitment
        uint256 oldHashLo = 111;
        uint256 oldHashHi = 222;
        uint256 correctCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        uint256 modelId = coordinator.registerModel("hash", correctCommitment, MIN_STAKE);
        coordinator.startRound(modelId, ROUND_DURATION);

        vm.prank(prover1);
        coordinator.stake{value: 1 ether}(modelId);

        // Submit invalid proof to get slashed (no revert expected)
        bytes memory invalidProof = hex"deadbeef";
        uint256[] memory publicInputs = new uint256[](7);
        publicInputs[0] = oldHashLo;
        publicInputs[1] = oldHashHi;
        publicInputs[2] = 3;
        publicInputs[3] = 4;
        publicInputs[4] = 100;
        publicInputs[5] = 10;
        publicInputs[6] = 1;

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, invalidProof, publicInputs);

        // Verify prover is slashed
        (,, bool slashed) = coordinator.getStake(prover1, modelId);
        assertTrue(slashed);

        // Try to stake again - should fail
        vm.prank(prover1);
        vm.expectRevert("Previous stake was slashed");
        coordinator.stake{value: 1 ether}(modelId);
    }

    // ============ Admin Tests ============

    function test_PauseModel() public {
        uint256 modelId = coordinator.registerModel("hash", 100, MIN_STAKE);

        coordinator.pauseModel(modelId);
        (,, bool active) = coordinator.getModelState(modelId);
        assertFalse(active);

        coordinator.resumeModel(modelId);
        (,, active) = coordinator.getModelState(modelId);
        assertTrue(active);
    }

    function test_SetVerifier() public {
        Halo2Verifier newVerifier = new Halo2Verifier();
        coordinator.setVerifier(address(newVerifier));
        assertEq(address(coordinator.verifier()), address(newVerifier));
    }

    function test_SetTreasury() public {
        address newTreasury = makeAddr("newTreasury");
        coordinator.setTreasury(newTreasury);
        assertEq(coordinator.treasury(), newTreasury);
    }

    // ============ View Function Tests ============

    function test_GetSlashingRecordCount() public {
        // Set up matching commitment
        uint256 oldHashLo = 333;
        uint256 oldHashHi = 444;
        uint256 correctCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        uint256 modelId = coordinator.registerModel("hash", correctCommitment, MIN_STAKE);
        coordinator.startRound(modelId, ROUND_DURATION);

        vm.prank(prover1);
        coordinator.stake{value: 1 ether}(modelId);

        assertEq(coordinator.getSlashingRecordCount(), 0);

        // Trigger slashing with matching commitment (no revert expected)
        bytes memory invalidProof = hex"deadbeef";
        uint256[] memory publicInputs = new uint256[](7);
        publicInputs[0] = oldHashLo;
        publicInputs[1] = oldHashHi;
        publicInputs[2] = 3;
        publicInputs[3] = 4;
        publicInputs[4] = 100;
        publicInputs[5] = 10;
        publicInputs[6] = 1;

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, invalidProof, publicInputs);

        assertEq(coordinator.getSlashingRecordCount(), 1);
    }

    // ============ Receive ETH Test ============

    function test_ReceiveETH() public {
        (bool success,) = address(coordinator).call{value: 1 ether}("");
        assertTrue(success);
        assertEq(address(coordinator).balance, 1 ether);
    }
}
