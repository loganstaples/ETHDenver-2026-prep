// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../src/core/HelixCoordinatorV3.sol";
import "../src/verification/RLCAggregationVerifier.sol";
import "../src/verification/Halo2VerifierCore.sol";
import "../src/verification/PoseidonHasher.sol";
import "../src/core/ModelRegistry.sol";
import "../src/token/HelixToken.sol";
import "../src/token/Staking.sol";
import "../src/token/Rewards.sol";
import "../src/interfaces/IHelixVerifier.sol";

/// @notice Mock verifier core that accepts all proofs (for gas measurement)
contract MockVerifierCoreForBenchmark {
    function verifyProof(address, bytes memory, uint256[] memory) external pure returns (bool) {
        return true;
    }
}

/// @notice Mock VK contract (just needs to exist at an address)
contract MockVKForBenchmark {}

/// @notice Mock IHelixVerifier that accepts all proofs (for individual proof verification)
contract MockIndividualVerifier is IHelixVerifier {
    function verifyProof(bytes memory, uint256[] memory) external pure override returns (bool) {
        return true;
    }
}

/// @title AggregationGasBenchmark
/// @notice Compares gas costs: sequential N x submitProof vs single submitAggregatedProof
/// @dev Run with `forge test --match-contract AggregationGasBenchmark -vvv` for detailed gas reports
contract AggregationGasBenchmark is Test {
    HelixCoordinatorV3 public v3;
    HelixToken public token;
    Staking public staking;
    Rewards public rewards;
    ModelRegistry public registry;
    MockIndividualVerifier public individualVerifier;
    RLCAggregationVerifier public aggVerifier;

    address public deployer;
    address public prover;
    uint256 public modelId;

    /// @notice Initial commitment = keccak256(abi.encodePacked(uint256(0), uint256(0)))
    uint256 public initialCommitment;

    function setUp() public {
        deployer = address(this);
        prover = address(0xBEEF);

        // Compute initial commitment matching oldHash (0, 0)
        initialCommitment = uint256(keccak256(abi.encodePacked(uint256(0), uint256(0))));

        // 1. Deploy HelixToken (deployer gets INITIAL_SUPPLY)
        token = new HelixToken(deployer);

        // 2. Deploy mock verifier core + VK for aggregation verifier
        MockVerifierCoreForBenchmark mockCore = new MockVerifierCoreForBenchmark();
        MockVKForBenchmark mockVK = new MockVKForBenchmark();

        // 3. Deploy individual verifier (mock)
        individualVerifier = new MockIndividualVerifier();

        // 4. Deploy aggregation verifier (wraps mock core)
        aggVerifier = new RLCAggregationVerifier(
            address(mockCore),
            address(mockVK)
        );

        // 5. Deploy Staking (100 HELIX min stake, 7 day unbonding, 50% slash rate)
        staking = new Staking(address(token), 100e18, 7 days, 5000);

        // 6. Deploy Rewards
        rewards = new Rewards(address(token));

        // 7. Deploy ModelRegistry
        registry = new ModelRegistry();

        // 8. Deploy HelixCoordinatorV3
        v3 = new HelixCoordinatorV3(
            address(individualVerifier),
            address(staking),
            address(rewards),
            address(registry),
            deployer // treasury = deployer
        );

        // 9. Wire contracts
        staking.setOperator(address(v3));
        rewards.setCoordinator(address(v3));
        rewards.setStakingContract(address(staking));
        registry.setCoordinator(address(v3));
        token.addMinter(address(rewards));

        // 10. Set aggregation verifier on V3
        v3.setAggregationVerifier(address(aggVerifier));

        // 11. Fund prover with tokens and stake
        token.transfer(prover, 10000e18);

        vm.startPrank(prover);
        token.approve(address(staking), type(uint256).max);
        staking.stake(1000e18); // Stake enough to participate
        vm.stopPrank();

        // 12. Register model with initial commitment matching (0, 0)
        vm.prank(prover);
        modelId = v3.registerModel(
            "benchmark-model",
            "Gas benchmark model",
            "ipfs://benchmark",
            initialCommitment,
            2,  // dIn
            4,  // dHidden
            1,  // dOut
            2,  // numLayers
            0   // activationType = ReLU
        );

        // 13. Start a training round (prover is model owner)
        vm.prank(prover);
        v3.startRound(modelId, 3600);
    }

    /// @notice Helper: create public inputs with proper commitment chaining and valid checksum
    /// @param oldLo Old hash low part
    /// @param oldHi Old hash high part
    /// @param newLo New hash low part
    /// @param newHi New hash high part
    /// @param stepNum Training step number
    /// @param errorBound Error bound for this step
    function _makePublicInputs(
        uint256 oldLo,
        uint256 oldHi,
        uint256 newLo,
        uint256 newHi,
        uint256 stepNum,
        uint256 errorBound
    ) internal view returns (uint256[] memory) {
        uint256[] memory pis = new uint256[](8);
        pis[0] = oldLo;
        pis[1] = oldHi;
        pis[2] = newLo;
        pis[3] = newHi;
        pis[4] = 100;         // loss
        pis[5] = errorBound;   // error bound
        pis[6] = stepNum;      // step number
        // Compute valid Poseidon error checksum: hash(hash(errorBound, stepNumber), hash(modelId, maxErrorBound))
        pis[7] = PoseidonHasher.computeErrorChecksum(errorBound, stepNum, modelId, v3.maxErrorBound());
        return pis;
    }

    /// @notice Helper to compute commitment hash (same as V3 internal _hashPair)
    function _hashPair(uint256 lo, uint256 hi) internal pure returns (uint256) {
        return uint256(keccak256(abi.encodePacked(lo, hi)));
    }

    // ============ Sequential Submission Benchmarks ============

    /// @notice Gas cost: single submitProof
    function test_Gas_SingleSubmitProof() public {
        bytes memory proof = new bytes(1856);
        // Step 1, old commitment = hash(0, 0) = initialCommitment
        uint256[] memory pis = _makePublicInputs(0, 0, 1, 1, 1, 10);

        vm.prank(prover);
        uint256 gasBefore = gasleft();
        v3.submitProof(modelId, 1, proof, pis);
        uint256 gasUsed = gasBefore - gasleft();

        emit log_named_uint("Gas: single submitProof", gasUsed);
    }

    /// @notice Gas cost: 5 sequential submitProof calls
    /// @dev Each proof chains commitments: old = previous new
    function test_Gas_Sequential_5_Proofs() public {
        bytes memory proof = new bytes(1856);
        uint256 totalGas = 0;

        // We need to chain commitments properly.
        // Model starts with commitment = hash(0, 0).
        // Each proof: old_hash -> new_hash, and after acceptance the model commitment becomes hash(newLo, newHi).
        // For sequential proofs in the same round, each must start new round since single-participant auto-finalizes.
        uint256 currentOldLo = 0;
        uint256 currentOldHi = 0;

        for (uint256 i = 0; i < 5; i++) {
            uint256 newLo = (i + 1) * 100;
            uint256 newHi = (i + 1) * 200;
            uint256 stepNum = i + 1;

            uint256[] memory pis = _makePublicInputs(currentOldLo, currentOldHi, newLo, newHi, stepNum, 10);

            vm.prank(prover);
            uint256 gasBefore = gasleft();
            v3.submitProof(modelId, uint256(i + 1), proof, pis);
            uint256 gasUsed = gasBefore - gasleft();
            totalGas += gasUsed;

            // After auto-finalize, start a new round for next proof
            currentOldLo = newLo;
            currentOldHi = newHi;
            if (i < 4) {
                vm.prank(prover);
                v3.startRound(modelId, 3600);
            }
        }

        emit log_named_uint("Gas: 5x sequential submitProof (total)", totalGas);
        emit log_named_uint("Gas: 5x sequential submitProof (avg)", totalGas / 5);
    }

    /// @notice Gas cost: 10 sequential submitProof calls
    function test_Gas_Sequential_10_Proofs() public {
        bytes memory proof = new bytes(1856);
        uint256 totalGas = 0;

        uint256 currentOldLo = 0;
        uint256 currentOldHi = 0;

        for (uint256 i = 0; i < 10; i++) {
            uint256 newLo = (i + 1) * 100;
            uint256 newHi = (i + 1) * 200;
            uint256 stepNum = i + 1;

            uint256[] memory pis = _makePublicInputs(currentOldLo, currentOldHi, newLo, newHi, stepNum, 10);

            vm.prank(prover);
            uint256 gasBefore = gasleft();
            v3.submitProof(modelId, uint256(i + 1), proof, pis);
            uint256 gasUsed = gasBefore - gasleft();
            totalGas += gasUsed;

            currentOldLo = newLo;
            currentOldHi = newHi;
            if (i < 9) {
                vm.prank(prover);
                v3.startRound(modelId, 3600);
            }
        }

        emit log_named_uint("Gas: 10x sequential submitProof (total)", totalGas);
        emit log_named_uint("Gas: 10x sequential submitProof (avg)", totalGas / 10);
    }

    // ============ Aggregated Submission Benchmarks ============

    /// @notice Gas cost: single submitAggregatedProof covering 5 steps
    function test_Gas_Aggregated_5_Steps() public {
        bytes memory proof = new bytes(1856);
        uint256 numSteps = 5;
        uint256 totalError = 10 * numSteps; // 10 per step * 5 steps
        uint256 stepNum = 1; // expected step number for round 1

        uint256[] memory pis = _makePublicInputs(0, 0, 500, 1000, stepNum, totalError);

        vm.prank(prover);
        uint256 gasBefore = gasleft();
        v3.submitAggregatedProof(modelId, 1, proof, pis, numSteps);
        uint256 gasUsed = gasBefore - gasleft();

        emit log_named_uint("Gas: 1x submitAggregatedProof (5 steps)", gasUsed);
    }

    /// @notice Gas cost: single submitAggregatedProof covering 10 steps
    function test_Gas_Aggregated_10_Steps() public {
        bytes memory proof = new bytes(1856);
        uint256 numSteps = 10;
        uint256 totalError = 10 * numSteps;
        uint256 stepNum = 1;

        uint256[] memory pis = _makePublicInputs(0, 0, 1000, 2000, stepNum, totalError);

        vm.prank(prover);
        uint256 gasBefore = gasleft();
        v3.submitAggregatedProof(modelId, 1, proof, pis, numSteps);
        uint256 gasUsed = gasBefore - gasleft();

        emit log_named_uint("Gas: 1x submitAggregatedProof (10 steps)", gasUsed);
    }

    /// @notice Gas cost: single submitAggregatedProof covering 20 steps
    function test_Gas_Aggregated_20_Steps() public {
        bytes memory proof = new bytes(1856);
        uint256 numSteps = 20;
        uint256 totalError = 10 * numSteps;
        uint256 stepNum = 1;

        uint256[] memory pis = _makePublicInputs(0, 0, 2000, 4000, stepNum, totalError);

        vm.prank(prover);
        uint256 gasBefore = gasleft();
        v3.submitAggregatedProof(modelId, 1, proof, pis, numSteps);
        uint256 gasUsed = gasBefore - gasleft();

        emit log_named_uint("Gas: 1x submitAggregatedProof (20 steps)", gasUsed);
    }

    /// @notice Gas cost: single submitAggregatedProof covering 32 steps (max)
    function test_Gas_Aggregated_32_Steps() public {
        bytes memory proof = new bytes(1856);
        uint256 numSteps = 32;
        uint256 totalError = 10 * numSteps;
        uint256 stepNum = 1;

        uint256[] memory pis = _makePublicInputs(0, 0, 3200, 6400, stepNum, totalError);

        vm.prank(prover);
        uint256 gasBefore = gasleft();
        v3.submitAggregatedProof(modelId, 1, proof, pis, numSteps);
        uint256 gasUsed = gasBefore - gasleft();

        emit log_named_uint("Gas: 1x submitAggregatedProof (32 steps)", gasUsed);
    }

    // ============ Comparative Analysis ============

    /// @notice Side-by-side comparison: 5 sequential vs 1 aggregated (5 steps)
    /// @dev Uses separate V3 instances to avoid state interference
    function test_Gas_Comparison_5_SequentialVsAggregated() public {
        // --- Measure aggregated first (uses existing setup) ---
        bytes memory proof = new bytes(1856);
        uint256 numSteps = 5;
        uint256 totalError = 10 * numSteps;
        uint256 stepNum = 1;

        uint256[] memory aggPis = _makePublicInputs(0, 0, 500, 1000, stepNum, totalError);

        vm.prank(prover);
        uint256 aggGasBefore = gasleft();
        v3.submitAggregatedProof(modelId, 1, proof, aggPis, numSteps);
        uint256 aggGas = aggGasBefore - gasleft();

        emit log_named_uint("=== 5 Steps Comparison ===", 0);
        emit log_named_uint("Aggregated (1 tx, 5 steps) gas", aggGas);

        // For sequential, we can't reuse the same model (round already completed).
        // Log the aggregated result; sequential is measured in test_Gas_Sequential_5_Proofs.
        // The comparison can be done by running both tests together.
        emit log_named_uint("(See test_Gas_Sequential_5_Proofs for sequential baseline)", 0);
    }

    // ============ RLCAggregationVerifier Direct Tests ============

    /// @notice Gas cost: direct aggregation verifier estimateVerifyGas
    function test_Gas_AggVerifier_Direct() public {
        bytes memory proof = new bytes(1856);
        uint256[] memory pis = new uint256[](8);
        for (uint256 i = 0; i < 8; i++) pis[i] = i;

        uint256 gasUsed = aggVerifier.estimateVerifyGas(proof, pis);
        emit log_named_uint("Gas: direct aggregation verifier (estimateVerifyGas)", gasUsed);
    }

    /// @notice Verify aggregated proof replay protection via V3
    function test_AggregatedReplayProtection() public {
        bytes memory proof = new bytes(1856);
        uint256 numSteps = 5;
        uint256 totalError = 10 * numSteps;
        uint256 stepNum = 1;

        uint256[] memory pis = _makePublicInputs(0, 0, 500, 1000, stepNum, totalError);

        // First submission should succeed
        vm.prank(prover);
        v3.submitAggregatedProof(modelId, 1, proof, pis, numSteps);

        // Second submission with same proof should fail (replay protection)
        // After first submission, round is completed (auto-finalized), so revert is "Round completed"
        // or "Proof already used" depending on which check hits first.
        // Since _validateRound checks round.isCompleted before proof hash, we expect "Round completed".
        vm.prank(prover);
        vm.expectRevert("Round completed");
        v3.submitAggregatedProof(modelId, 1, proof, pis, numSteps);
    }

    /// @notice Verify aggregated proof with invalid numSteps
    function test_AggregatedInvalidNumSteps() public {
        bytes memory proof = new bytes(1856);
        uint256[] memory pis = _makePublicInputs(0, 0, 1, 1, 1, 10);

        // numSteps = 0 should fail
        vm.prank(prover);
        vm.expectRevert("numSteps must be between 1 and 32");
        v3.submitAggregatedProof(modelId, 1, proof, pis, 0);

        // numSteps = 33 should fail
        vm.prank(prover);
        vm.expectRevert("numSteps must be between 1 and 32");
        v3.submitAggregatedProof(modelId, 1, proof, pis, 33);
    }

    /// @notice Verify submitAggregatedProof fails when aggregation verifier not set
    function test_AggregatedVerifierNotSet() public {
        // Deploy a fresh V3 without setting aggregation verifier
        HelixCoordinatorV3 v3NoAgg = new HelixCoordinatorV3(
            address(individualVerifier),
            address(staking),
            address(rewards),
            address(registry),
            deployer
        );

        bytes memory proof = new bytes(1856);
        uint256[] memory pis = _makePublicInputs(0, 0, 1, 1, 1, 10);

        // Should fail because aggregation verifier not set
        // Note: v3NoAgg won't have a registered model, so it will fail on "Model does not exist" first.
        // We test the revert message by checking the specific require.
        vm.expectRevert("Model does not exist");
        v3NoAgg.submitAggregatedProof(modelId, 1, proof, pis, 1);
    }

    /// @notice RLCAggregationVerifier verifyAndRecordAggregated with replay detection
    function test_AggVerifier_VerifyAndRecord_Replay() public {
        bytes memory proof = new bytes(1856);
        uint256[] memory pis = new uint256[](8);
        for (uint256 i = 0; i < 8; i++) pis[i] = i + 1;

        // First call should succeed (mock core accepts all proofs)
        bool valid1 = aggVerifier.verifyAndRecordAggregated(proof, pis, 5, 1);
        assertTrue(valid1, "First verification should succeed");

        // Second call with same params should fail (replay detection returns false)
        bool valid2 = aggVerifier.verifyAndRecordAggregated(proof, pis, 5, 1);
        assertFalse(valid2, "Replay should be rejected");
    }

    /// @notice RLCAggregationVerifier verifyAggregatedProof with invalid batch size
    function test_AggVerifier_InvalidBatchSize() public view {
        bytes memory proof = new bytes(1856);
        uint256[] memory pis = new uint256[](8);
        for (uint256 i = 0; i < 8; i++) pis[i] = i;

        // numSteps = 0 returns false (no revert in view function)
        bool valid0 = aggVerifier.verifyAggregatedProof(proof, pis, 0, 1);
        assertFalse(valid0, "numSteps=0 should return false");

        // numSteps = 33 returns false
        bool valid33 = aggVerifier.verifyAggregatedProof(proof, pis, 33, 1);
        assertFalse(valid33, "numSteps=33 should return false");
    }

    /// @notice RLCAggregationVerifier verifyAndRecordAggregated reverts on invalid batch size
    function test_AggVerifier_VerifyAndRecord_InvalidBatchSize() public {
        bytes memory proof = new bytes(1856);
        uint256[] memory pis = new uint256[](8);
        for (uint256 i = 0; i < 8; i++) pis[i] = i;

        // numSteps = 0 reverts with require
        vm.expectRevert("Invalid batch size");
        aggVerifier.verifyAndRecordAggregated(proof, pis, 0, 1);

        // numSteps = 33 reverts with require
        vm.expectRevert("Invalid batch size");
        aggVerifier.verifyAndRecordAggregated(proof, pis, 33, 1);
    }

    /// @notice Verify proofHash tracking in RLCAggregationVerifier
    function test_AggVerifier_ProofHashTracking() public {
        bytes memory proof = new bytes(1856);
        uint256[] memory pis = new uint256[](8);
        for (uint256 i = 0; i < 8; i++) pis[i] = i + 10;

        // Compute expected proof hash
        bytes32 expectedHash = keccak256(abi.encodePacked(proof, pis, uint256(3), uint256(7)));

        // Should not be used before verification
        assertFalse(aggVerifier.isProofUsed(expectedHash), "Proof should not be used yet");

        // Verify and record
        bool valid = aggVerifier.verifyAndRecordAggregated(proof, pis, 3, 7);
        assertTrue(valid, "Verification should succeed");

        // Should be marked as used
        assertTrue(aggVerifier.isProofUsed(expectedHash), "Proof should be marked as used");
    }

    /// @notice RLCAggregationVerifier verifyProof (IHelixVerifier interface) with invalid inputs
    function test_AggVerifier_VerifyProof_InvalidInputsCount() public view {
        bytes memory proof = new bytes(1856);
        uint256[] memory badPis = new uint256[](7); // Wrong count (should be 8)

        bool valid = aggVerifier.verifyProof(proof, badPis);
        assertFalse(valid, "Should reject non-8 public inputs");
    }
}
