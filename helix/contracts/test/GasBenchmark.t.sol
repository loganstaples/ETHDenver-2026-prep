// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../src/core/HelixCoordinatorV2.sol";
import "../src/verification/Halo2Verifier.sol";
import "../src/mocks/MockVerifier.sol";
import "./ProofFixtures.t.sol";

/// @title GasBenchmark
/// @notice Comprehensive gas benchmarking for all HELIX contract operations
/// @dev Run with `forge test --match-contract GasBenchmark -vvv` for detailed gas reports
contract GasBenchmark is Test {
    HelixCoordinatorV2 public coordinator;
    Halo2Verifier public halo2Verifier;
    MockVerifier public mockVerifier;

    address public owner;
    address public prover1;
    address public prover2;
    address public prover3;
    address public treasury;

    uint256 public modelId;

    // BN254 field order
    uint256 constant R = 21888242871839275222246405745257275088548364400416034343698204186575808495617;

    function setUp() public {
        owner = address(this);
        prover1 = makeAddr("prover1");
        prover2 = makeAddr("prover2");
        prover3 = makeAddr("prover3");
        treasury = makeAddr("treasury");

        // Deploy contracts
        mockVerifier = new MockVerifier();
        coordinator = new HelixCoordinatorV2(address(mockVerifier), treasury);
        halo2Verifier = new Halo2Verifier();

        // Fund provers
        vm.deal(prover1, 100 ether);
        vm.deal(prover2, 100 ether);
        vm.deal(prover3, 100 ether);

        // Register a model for testing
        // Initial commitment = keccak256(abi.encodePacked(1, 2)) to match _createValidPublicInputs
        uint256 initialCommitment = uint256(keccak256(abi.encodePacked(uint256(1), uint256(2))));
        modelId = coordinator.registerModel("benchmark-model", initialCommitment, 0.1 ether, 4, 8, 2, 2, 0);
    }

    // ============ Coordinator Operations ============

    function test_GasBenchmark_RegisterModel() public {
        uint256 gasBefore = gasleft();
        coordinator.registerModel("new-model", 99999, 0.5 ether, 4, 8, 2, 2, 0);
        uint256 gasUsed = gasBefore - gasleft();

        emit log_named_uint("RegisterModel gas", gasUsed);
    }

    function test_GasBenchmark_Stake() public {
        vm.prank(prover1);
        uint256 gasBefore = gasleft();
        coordinator.stake{value: 1 ether}(modelId);
        uint256 gasUsed = gasBefore - gasleft();

        emit log_named_uint("Stake gas", gasUsed);
    }

    function test_GasBenchmark_StakeSecondTime() public {
        // First stake
        vm.prank(prover1);
        coordinator.stake{value: 1 ether}(modelId);

        // Second stake (should be cheaper - storage already initialized)
        vm.prank(prover1);
        uint256 gasBefore = gasleft();
        coordinator.stake{value: 0.5 ether}(modelId);
        uint256 gasUsed = gasBefore - gasleft();

        emit log_named_uint("Stake (second time) gas", gasUsed);
    }

    function test_GasBenchmark_Unstake() public {
        // Setup: stake first
        vm.prank(prover1);
        coordinator.stake{value: 1 ether}(modelId);

        // Fast forward past lock period
        vm.warp(block.timestamp + 8 days);

        vm.prank(prover1);
        uint256 gasBefore = gasleft();
        coordinator.unstake(modelId);
        uint256 gasUsed = gasBefore - gasleft();

        emit log_named_uint("Unstake gas", gasUsed);
    }

    function test_GasBenchmark_StartRound() public {
        uint256 gasBefore = gasleft();
        coordinator.startRound(modelId, 1 hours);
        uint256 gasUsed = gasBefore - gasleft();

        emit log_named_uint("StartRound gas", gasUsed);
    }

    function test_GasBenchmark_SubmitProof() public {
        // Setup
        vm.prank(prover1);
        coordinator.stake{value: 1 ether}(modelId);
        coordinator.startRound(modelId, 1 hours);

        bytes memory proof = _createValidProof();
        uint256[] memory inputs = _createValidPublicInputs();
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(inputs[5], inputs[6], modelId, 1e18);

        vm.prank(prover1);
        uint256 gasBefore = gasleft();
        coordinator.submitProof(modelId, 1, proof, inputs);
        uint256 gasUsed = gasBefore - gasleft();

        emit log_named_uint("SubmitProof gas", gasUsed);
    }

    function test_GasBenchmark_SubmitProofInvalidSlash() public {
        // Setup
        vm.prank(prover1);
        coordinator.stake{value: 1 ether}(modelId);
        coordinator.startRound(modelId, 1 hours);

        // Make verifier reject proofs
        mockVerifier.setShouldPass(false);

        bytes memory proof = _createValidProof();
        uint256[] memory inputs = _createValidPublicInputs();
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(inputs[5], inputs[6], modelId, 1e18);

        vm.prank(prover1);
        uint256 gasBefore = gasleft();
        try coordinator.submitProof(modelId, 1, proof, inputs) {
            // If it doesn't revert, measure gas
        } catch {
            // Expected - slashing occurred
        }
        uint256 gasUsed = gasBefore - gasleft();

        emit log_named_uint("SubmitProof (invalid, slash) gas", gasUsed);
    }

    function test_GasBenchmark_PauseModel() public {
        uint256 gasBefore = gasleft();
        coordinator.pauseModel(modelId);
        uint256 gasUsed = gasBefore - gasleft();

        emit log_named_uint("PauseModel gas", gasUsed);
    }

    // ============ Verifier Operations ============

    function test_GasBenchmark_Halo2VerifyProof() public {
        bytes memory proof = new bytes(320);
        uint256[] memory inputs = _createValidPublicInputs();

        uint256 gasBefore = gasleft();
        halo2Verifier.verifyProof(proof, inputs);
        uint256 gasUsed = gasBefore - gasleft();

        emit log_named_uint("Halo2Verifier.verifyProof gas", gasUsed);
    }

    function test_GasBenchmark_Halo2BatchVerify_2() public {
        bytes[] memory proofs = new bytes[](2);
        uint256[][] memory inputs = new uint256[][](2);

        for (uint i = 0; i < 2; i++) {
            proofs[i] = new bytes(320);
            inputs[i] = _createValidPublicInputs();
            inputs[i][0] = i + 1;
        }

        uint256 gasBefore = gasleft();
        halo2Verifier.batchVerify(proofs, inputs);
        uint256 gasUsed = gasBefore - gasleft();

        emit log_named_uint("Halo2Verifier.batchVerify (2 proofs) gas", gasUsed);
        emit log_named_uint("Per-proof gas (batch of 2)", gasUsed / 2);
    }

    function test_GasBenchmark_Halo2BatchVerify_5() public {
        bytes[] memory proofs = new bytes[](5);
        uint256[][] memory inputs = new uint256[][](5);

        for (uint i = 0; i < 5; i++) {
            proofs[i] = new bytes(320);
            inputs[i] = _createValidPublicInputs();
            inputs[i][0] = i + 1;
        }

        uint256 gasBefore = gasleft();
        halo2Verifier.batchVerify(proofs, inputs);
        uint256 gasUsed = gasBefore - gasleft();

        emit log_named_uint("Halo2Verifier.batchVerify (5 proofs) gas", gasUsed);
        emit log_named_uint("Per-proof gas (batch of 5)", gasUsed / 5);
    }

    function test_GasBenchmark_Halo2BatchVerify_10() public {
        bytes[] memory proofs = new bytes[](10);
        uint256[][] memory inputs = new uint256[][](10);

        for (uint i = 0; i < 10; i++) {
            proofs[i] = new bytes(320);
            inputs[i] = _createValidPublicInputs();
            inputs[i][0] = i + 1;
        }

        uint256 gasBefore = gasleft();
        halo2Verifier.batchVerify(proofs, inputs);
        uint256 gasUsed = gasBefore - gasleft();

        emit log_named_uint("Halo2Verifier.batchVerify (10 proofs) gas", gasUsed);
        emit log_named_uint("Per-proof gas (batch of 10)", gasUsed / 10);
    }

    function test_GasBenchmark_Halo2VerifyAndRecord() public {
        bytes memory proof = new bytes(320);
        uint256[] memory inputs = _createValidPublicInputs();

        uint256 gasBefore = gasleft();
        halo2Verifier.verifyAndRecord(proof, inputs);
        uint256 gasUsed = gasBefore - gasleft();

        emit log_named_uint("Halo2Verifier.verifyAndRecord gas", gasUsed);
    }

    function test_GasBenchmark_MockVerifyProof() public {
        bytes memory proof = new bytes(256);
        uint256[] memory inputs = _createValidPublicInputs();

        uint256 gasBefore = gasleft();
        mockVerifier.verifyProof(proof, inputs);
        uint256 gasUsed = gasBefore - gasleft();

        emit log_named_uint("MockVerifier.verifyProof gas", gasUsed);
    }

    // ============ View Functions ============

    function test_GasBenchmark_GetModelState() public {
        // Setup
        vm.prank(prover1);
        coordinator.stake{value: 1 ether}(modelId);
        coordinator.startRound(modelId, 1 hours);

        uint256 gasBefore = gasleft();
        coordinator.getModelState(modelId);
        uint256 gasUsed = gasBefore - gasleft();

        emit log_named_uint("getModelState gas", gasUsed);
    }

    function test_GasBenchmark_GetStake() public {
        vm.prank(prover1);
        coordinator.stake{value: 1 ether}(modelId);

        uint256 gasBefore = gasleft();
        coordinator.getStake(prover1, modelId);
        uint256 gasUsed = gasBefore - gasleft();

        emit log_named_uint("getStake gas", gasUsed);
    }

    // ============ Comparative Analysis ============

    function test_GasBenchmark_SingleVsBatchVerification() public {
        bytes[] memory proofs = new bytes[](5);
        uint256[][] memory inputs = new uint256[][](5);

        for (uint i = 0; i < 5; i++) {
            proofs[i] = new bytes(320);
            inputs[i] = _createValidPublicInputs();
            inputs[i][0] = i + 1;
        }

        // Measure 5 individual verifications
        uint256 totalSingleGas = 0;
        for (uint i = 0; i < 5; i++) {
            uint256 singleGasBefore = gasleft();
            halo2Verifier.verifyProof(proofs[i], inputs[i]);
            totalSingleGas += singleGasBefore - gasleft();
        }

        // Measure batch verification
        uint256 batchGasBefore = gasleft();
        halo2Verifier.batchVerify(proofs, inputs);
        uint256 batchGas = batchGasBefore - gasleft();

        emit log_named_uint("5 individual verifications total gas", totalSingleGas);
        emit log_named_uint("Batch verification (5 proofs) gas", batchGas);

        if (batchGas < totalSingleGas) {
            uint256 savings = totalSingleGas - batchGas;
            emit log_named_uint("Gas savings with batch", savings);
            uint256 savingsPercent = (savings * 100) / totalSingleGas;
            emit log_named_uint("Savings percentage", savingsPercent);
        }
    }

    function test_GasBenchmark_StorageWarmVsCold() public {
        // First stake (cold storage)
        vm.prank(prover1);
        uint256 coldGasBefore = gasleft();
        coordinator.stake{value: 1 ether}(modelId);
        uint256 coldGas = coldGasBefore - gasleft();

        // Second stake (warm storage)
        vm.prank(prover1);
        uint256 warmGasBefore = gasleft();
        coordinator.stake{value: 0.5 ether}(modelId);
        uint256 warmGas = warmGasBefore - gasleft();

        emit log_named_uint("Stake (cold storage) gas", coldGas);
        emit log_named_uint("Stake (warm storage) gas", warmGas);
        emit log_named_uint("Gas savings (warm)", coldGas - warmGas);
    }

    // ============ Full Workflow Benchmark ============

    function test_GasBenchmark_FullTrainingRound() public {
        uint256 totalGas = 0;
        uint256 gasBefore;
        uint256 gasUsed;

        emit log("=== Full Training Round Gas Breakdown ===");

        // 1. Register model with commitment matching our test inputs
        uint256 newCommitment = uint256(keccak256(abi.encodePacked(uint256(100), uint256(200))));
        gasBefore = gasleft();
        uint256 newModelId = coordinator.registerModel("benchmark", newCommitment, 0.5 ether, 4, 8, 2, 2, 0);
        gasUsed = gasBefore - gasleft();
        totalGas += gasUsed;
        emit log_named_uint("1. RegisterModel", gasUsed);

        // 2. Prover stakes
        vm.prank(prover1);
        gasBefore = gasleft();
        coordinator.stake{value: 1 ether}(newModelId);
        gasUsed = gasBefore - gasleft();
        totalGas += gasUsed;
        emit log_named_uint("2. Stake", gasUsed);

        // 3. Start round
        gasBefore = gasleft();
        coordinator.startRound(newModelId, 1 hours);
        gasUsed = gasBefore - gasleft();
        totalGas += gasUsed;
        emit log_named_uint("3. StartRound", gasUsed);

        // 4. Submit proof
        bytes memory proof = _createValidProof();
        uint256[] memory inputs = _createValidPublicInputsForCommitment(100, 200);
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(inputs[5], inputs[6], newModelId, 1e18);

        vm.prank(prover1);
        gasBefore = gasleft();
        coordinator.submitProof(newModelId, 1, proof, inputs);
        gasUsed = gasBefore - gasleft();
        totalGas += gasUsed;
        emit log_named_uint("4. SubmitProof", gasUsed);

        // 5. Unstake (after lock period)
        vm.warp(block.timestamp + 8 days);
        vm.prank(prover1);
        gasBefore = gasleft();
        coordinator.unstake(newModelId);
        gasUsed = gasBefore - gasleft();
        totalGas += gasUsed;
        emit log_named_uint("5. Unstake", gasUsed);

        emit log("========================================");
        emit log_named_uint("TOTAL GAS for full workflow", totalGas);
    }

    // ============ Summary Report ============

    function test_GasBenchmark_Summary() public pure {
        // This test just prints a summary of expected gas ranges
        // Actual values are measured in other tests
    }

    // ============ Helpers ============

    function _createValidProof() internal pure returns (bytes memory) {
        return new bytes(320);
    }

    /// @notice Compute commitment hash the same way coordinator does
    function _computeCommitment(uint256 lo, uint256 hi) internal pure returns (uint256) {
        return uint256(keccak256(abi.encodePacked(lo, hi)));
    }

    /// @notice Create valid public inputs that match the model's initial commitment
    /// @dev The initial commitment (12345) was registered in setUp
    function _createValidPublicInputs() internal pure returns (uint256[] memory) {
        uint256[] memory inputs = new uint256[](8);
        // Use lo=1, hi=2 as our "old" values
        // The model must be registered with commitment = keccak256(1, 2)
        inputs[0] = 1;         // oldHashLo
        inputs[1] = 2;         // oldHashHi
        inputs[2] = 3;         // newHashLo
        inputs[3] = 4;         // newHashHi
        inputs[4] = 100;       // loss
        inputs[5] = 10;        // errorBound
        inputs[6] = 1;         // stepNumber
        inputs[7] = 0;         // errorChecksum (placeholder for verifier-only tests)
        return inputs;
    }

    /// @notice Create public inputs for a model with specific lo/hi commitment values
    function _createValidPublicInputsForCommitment(uint256 lo, uint256 hi) internal pure returns (uint256[] memory) {
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = lo;        // oldHashLo
        inputs[1] = hi;        // oldHashHi
        inputs[2] = 3;         // newHashLo
        inputs[3] = 4;         // newHashHi
        inputs[4] = 100;       // loss
        inputs[5] = 10;        // errorBound
        inputs[6] = 1;         // stepNumber
        inputs[7] = 0;         // errorChecksum (placeholder, set by caller for coordinator tests)
        return inputs;
    }
}
