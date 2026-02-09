// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import "../src/core/HelixCoordinatorV2.sol";
import "../src/core/DataCommitment.sol";
import "../src/verification/Halo2Verifier.sol";
import "../src/verification/DataVerifier.sol";
import "../src/verification/SlashingEvidence.sol";
import "../src/mocks/MockVerifier.sol";
import "./ProofFixtures.t.sol";

/// @title CompleteFlowTest
/// @notice Tests the complete flow: register → commit data → train → verify
/// @dev Validates all B4 requirements including data commitment and slashing evidence
contract CompleteFlowTest is Test {
    // ============ Contracts ============
    HelixCoordinatorV2 public coordinator;
    MockVerifier public mockVerifier;
    DataCommitment public dataCommitment;
    DataVerifier public dataVerifier;
    SlashingEvidence public slashingEvidence;

    // ============ Actors ============
    address public owner;
    address public treasury;
    address public modelOwner;
    address public prover;
    address public attacker;

    // ============ Constants ============
    uint256 constant MIN_STAKE = 0.1 ether;
    uint256 constant PROVER_STAKE = 1 ether;
    uint256 constant ROUND_DURATION = 1 hours;

    // ============ Events ============
    event DatasetCommitted(uint256 indexed commitmentId, address indexed owner, bytes32 merkleRoot, uint64 datasetSize, string ipfsHash);
    event ModelDatasetLinked(uint256 indexed modelId, uint256 indexed commitmentId);
    event DataCommitmentSet(uint256 indexed modelId, bytes32 indexed dataRoot, address indexed setBy);
    event RoundDataCommitted(uint256 indexed modelId, uint256 indexed roundId, bytes32 dataRoot);
    event EmergencyPauseChanged(bool isPaused, address indexed changedBy);
    event EvidenceSubmitted(
        uint256 indexed evidenceId,
        address indexed prover,
        uint64 indexed modelId,
        SlashingEvidence.ViolationType violationType,
        SlashingEvidence.ErrorCode errorCode,
        SlashingEvidence.SeverityLevel severity,
        uint128 slashedAmount
    );

    function setUp() public {
        owner = address(this);
        treasury = makeAddr("treasury");
        modelOwner = makeAddr("modelOwner");
        prover = makeAddr("prover");
        attacker = makeAddr("attacker");

        // Deploy contracts
        mockVerifier = new MockVerifier();
        coordinator = new HelixCoordinatorV2(address(mockVerifier), treasury);
        dataCommitment = new DataCommitment(address(coordinator));
        dataVerifier = new DataVerifier();
        slashingEvidence = new SlashingEvidence(address(coordinator));

        // Configure coordinator
        coordinator.setDataCommitmentContract(address(dataCommitment));
        coordinator.setSlashingEvidenceContract(address(slashingEvidence));

        // Fund actors
        vm.deal(modelOwner, 100 ether);
        vm.deal(prover, 100 ether);
        vm.deal(attacker, 100 ether);
    }

    // ============ Complete Flow Test ============

    /// @notice Test complete flow: register → commit data → stake → train → verify
    function test_CompleteFlow() public {
        // ========== Step 1: Register Model ==========
        uint256 hashLo = 12345;
        uint256 hashHi = 67890;
        uint256 initialCommitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("QmTestModel", initialCommitment, MIN_STAKE);

        assertEq(modelId, 0);
        (uint256 currentRound, uint256 commitment, bool active) = coordinator.getModelState(modelId);
        assertEq(currentRound, 0);
        assertEq(commitment, initialCommitment);
        assertTrue(active);

        // ========== Step 2: Commit Dataset ==========
        bytes32 datasetRoot = keccak256("training-data-merkle-root");

        vm.prank(modelOwner);
        uint256 datasetId = dataCommitment.commitDataset(
            datasetRoot,
            10000, // 10k samples
            "QmDatasetMetadata"
        );

        assertEq(datasetId, 0);
        (bytes32 storedRoot, uint64 size,,,, string memory ipfs) = dataCommitment.getDatasetCommitment(datasetId);
        assertEq(storedRoot, datasetRoot);
        assertEq(size, 10000);

        // ========== Step 3: Link Dataset to Model ==========
        vm.prank(modelOwner);
        coordinator.setModelDataCommitment(modelId, datasetRoot);

        assertEq(coordinator.getModelDataCommitment(modelId), datasetRoot);
        assertTrue(coordinator.hasDataCommitment(modelId));

        // ========== Step 4: Prover Stakes ==========
        vm.prank(prover);
        coordinator.stake{value: PROVER_STAKE}(modelId);

        (uint256 stakeAmount,, bool slashed) = coordinator.getStake(prover, modelId);
        assertEq(stakeAmount, PROVER_STAKE);
        assertFalse(slashed);

        // ========== Step 5: Start Training Round ==========
        vm.prank(modelOwner);
        coordinator.startRound(modelId, ROUND_DURATION);

        (currentRound,,) = coordinator.getModelState(modelId);
        assertEq(currentRound, 1);

        // ========== Step 6: Commit Round Data ==========
        bytes32 roundDataRoot = keccak256("round-1-batch-data");

        vm.prank(modelOwner);
        coordinator.commitRoundData(modelId, 1, roundDataRoot);

        assertEq(coordinator.getRoundDataRoot(modelId, 1), roundDataRoot);

        // ========== Step 7: Submit Valid Proof ==========
        mockVerifier.setShouldPass(true);

        uint256 newHashLo = 11111;
        uint256 newHashHi = 22222;
        uint256[] memory publicInputs = new uint256[](8);
        publicInputs[0] = hashLo;
        publicInputs[1] = hashHi;
        publicInputs[2] = newHashLo;
        publicInputs[3] = newHashHi;
        publicInputs[4] = 100;  // loss
        publicInputs[5] = 10;   // error bound
        publicInputs[6] = 1;    // step
        publicInputs[7] = ProofFixtureHardcoded.computeErrorChecksum(publicInputs[5], publicInputs[6], modelId, 1e18);

        bytes memory proof = new bytes(320);

        vm.prank(prover);
        coordinator.submitProof(modelId, 1, proof, publicInputs);

        // ========== Step 8: Verify State Updates ==========
        uint256 expectedNewCommitment = uint256(keccak256(abi.encodePacked(newHashLo, newHashHi)));
        (, commitment,) = coordinator.getModelState(modelId);
        assertEq(commitment, expectedNewCommitment);

        uint256 accumulatedError = coordinator.getAccumulatedErrorBound(modelId);
        assertEq(accumulatedError, 10);

        // ========== Step 9: Verify Data Commitment ==========
        assertTrue(coordinator.verifyRoundDataCommitment(modelId, 1));
    }

    // ============ Data Commitment Tests ============

    /// @notice Test dataset commitment and verification
    function test_DatasetCommitment() public {
        bytes32 root = keccak256("test-data");

        vm.expectEmit(true, true, false, true);
        emit DatasetCommitted(0, modelOwner, root, 5000, "QmTest");

        vm.prank(modelOwner);
        uint256 id = dataCommitment.commitDataset(root, 5000, "QmTest");

        assertEq(id, 0);

        // Verify commitment
        uint256 lookupId = dataCommitment.commitmentIdByRoot(root);
        assertEq(lookupId, 0);
    }

    /// @notice Test Merkle proof verification
    function test_MerkleProofVerification() public {
        // Create simple Merkle tree
        bytes32 leaf1 = keccak256("data1");
        bytes32 leaf2 = keccak256("data2");
        bytes32 leaf3 = keccak256("data3");
        bytes32 leaf4 = keccak256("data4");

        bytes32 branch1 = keccak256(abi.encodePacked(leaf1, leaf2));
        bytes32 branch2 = keccak256(abi.encodePacked(leaf3, leaf4));
        bytes32 root = keccak256(abi.encodePacked(branch1, branch2));

        // Verify leaf1 inclusion
        bytes32[] memory proof = new bytes32[](2);
        proof[0] = leaf2;
        proof[1] = branch2;

        bool valid = dataVerifier.verifyProof(root, leaf1, 0, proof);
        assertTrue(valid);

        // Verify leaf4 inclusion
        bytes32[] memory proof4 = new bytes32[](2);
        proof4[0] = leaf3;
        proof4[1] = branch1;

        valid = dataVerifier.verifyProof(root, leaf4, 3, proof4);
        assertTrue(valid);
    }

    /// @notice Test batch data commitment
    function test_BatchDataCommitment() public {
        uint256 hashLo = 100;
        uint256 hashHi = 200;
        uint256 commitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("Test", commitment, MIN_STAKE);

        // Commit batch data
        bytes32 batchRoot = keccak256("batch-1");

        vm.prank(prover);
        dataCommitment.commitBatch(modelId, 1, batchRoot, 0, 100);

        DataCommitment.BatchCommitment[] memory batches = dataCommitment.getBatchCommitments(modelId, 1);
        assertEq(batches.length, 1);
        assertEq(batches[0].batchRoot, batchRoot);
        assertEq(batches[0].batchSize, 100);
    }

    // ============ Emergency Pause Tests ============

    /// @notice Test emergency pause mechanism
    function test_EmergencyPause() public {
        // Pause the contract
        vm.expectEmit(false, true, false, true);
        emit EmergencyPauseChanged(true, owner);
        coordinator.emergencyPause();

        assertTrue(coordinator.paused());

        // Cannot register model while paused
        uint256 commitment = uint256(keccak256(abi.encodePacked(uint256(1), uint256(2))));
        vm.prank(modelOwner);
        vm.expectRevert(HelixCoordinatorV2.ContractPaused.selector);
        coordinator.registerModel("Test", commitment, MIN_STAKE);

        // Unpause
        coordinator.unpause();
        assertFalse(coordinator.paused());

        // Now can register
        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("Test", commitment, MIN_STAKE);
        assertEq(modelId, 0);
    }

    /// @notice Test only owner can pause/unpause
    function test_OnlyOwnerCanPause() public {
        vm.prank(attacker);
        vm.expectRevert(HelixCoordinatorV2.OnlyOwner.selector);
        coordinator.emergencyPause();
    }

    // ============ Slashing Evidence Tests ============

    /// @notice Test slashing evidence submission
    function test_SlashingEvidenceSubmission() public {
        // Setup
        uint256 hashLo = 12345;
        uint256 hashHi = 67890;
        uint256 commitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("Test", commitment, MIN_STAKE);

        vm.prank(modelOwner);
        coordinator.startRound(modelId, ROUND_DURATION);

        vm.prank(attacker);
        coordinator.stake{value: PROVER_STAKE}(modelId);

        // Submit invalid proof (should slash)
        mockVerifier.setShouldPass(false);

        uint256[] memory inputs = new uint256[](8);
        inputs[0] = hashLo;
        inputs[1] = hashHi;
        inputs[2] = 1111;
        inputs[3] = 2222;
        inputs[4] = 100;
        inputs[5] = 10;
        inputs[6] = 1;
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(inputs[5], inputs[6], modelId, 1e18);

        bytes memory proof = new bytes(320);

        vm.prank(attacker);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Verify slashing occurred
        (uint256 stake,, bool slashed) = coordinator.getStake(attacker, modelId);
        assertTrue(slashed);
        assertEq(stake, PROVER_STAKE / 2); // 50% slashed

        // Verify slashing record
        assertEq(coordinator.getSlashingRecordCount(), 1);
    }

    /// @notice Test slashing evidence contract
    function test_SlashingEvidenceContract() public {
        // The coordinator is set to address(coordinator), so we need to set
        // slashingEvidence's coordinator to this test contract for testing
        slashingEvidence.setCoordinator(address(this));

        // Submit evidence directly to slashing evidence contract
        uint256 evidenceId = slashingEvidence.submitEvidence(
            attacker,
            0, // modelId
            1, // roundId
            SlashingEvidence.ViolationType.InvalidProof,
            SlashingEvidence.ErrorCode.PROOF_VERIFICATION_FAILED,
            0.5 ether, // slashed
            0.5 ether, // remaining
            keccak256("proof"),
            address(0), // challenger
            "Test slashing"
        );

        assertEq(evidenceId, 0);

        // Get evidence
        SlashingEvidence.Evidence memory evidence = slashingEvidence.getEvidence(evidenceId);
        assertEq(evidence.prover, attacker);
        assertEq(uint8(evidence.violationType), uint8(SlashingEvidence.ViolationType.InvalidProof));
        assertEq(evidence.slashedAmount, 0.5 ether);

        // Check prover evidence
        uint256[] memory proverEvidence = slashingEvidence.getProverEvidence(attacker);
        assertEq(proverEvidence.length, 1);
        assertEq(proverEvidence[0], 0);
    }

    /// @notice Test dispute filing
    function test_DisputeFiling() public {
        // Set coordinator to this test contract for testing
        slashingEvidence.setCoordinator(address(this));

        // Submit evidence
        uint256 evidenceId = slashingEvidence.submitEvidence(
            prover,
            0, 1,
            SlashingEvidence.ViolationType.InvalidProof,
            SlashingEvidence.ErrorCode.PROOF_VERIFICATION_FAILED,
            0.5 ether, 0.5 ether,
            keccak256("proof"),
            address(0), // challenger
            "Test"
        );

        // File dispute (requires stake)
        uint256 disputeStake = slashingEvidence.disputeStakeRequired();
        vm.deal(prover, disputeStake);
        vm.prank(prover);
        uint256 disputeId = slashingEvidence.fileDispute{value: disputeStake}(
            evidenceId,
            keccak256("counter-evidence"),
            "I believe the proof was valid"
        );

        assertEq(disputeId, 0);

        // Check evidence status
        SlashingEvidence.Evidence memory evidence = slashingEvidence.getEvidence(evidenceId);
        assertEq(uint8(evidence.status), uint8(SlashingEvidence.EvidenceStatus.Disputed));

        // Resolve dispute
        slashingEvidence.resolveDispute(disputeId, true); // In favor of prover

        evidence = slashingEvidence.getEvidence(evidenceId);
        assertEq(uint8(evidence.status), uint8(SlashingEvidence.EvidenceStatus.Rejected));
    }

    // ============ Gas Optimization Verification ============

    /// @notice Test gas usage for verification is under 200k
    function test_VerificationGasUnder200k() public {
        // Setup
        uint256 hashLo = 12345;
        uint256 hashHi = 67890;
        uint256 commitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("Test", commitment, MIN_STAKE);

        vm.prank(prover);
        coordinator.stake{value: PROVER_STAKE}(modelId);

        vm.prank(modelOwner);
        coordinator.startRound(modelId, ROUND_DURATION);

        mockVerifier.setShouldPass(true);

        uint256[] memory inputs = new uint256[](8);
        inputs[0] = hashLo;
        inputs[1] = hashHi;
        inputs[2] = 1111;
        inputs[3] = 2222;
        inputs[4] = 100;
        inputs[5] = 10;
        inputs[6] = 1;
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(inputs[5], inputs[6], modelId, 1e18);

        bytes memory proof = new bytes(320);

        // Measure gas for proof submission
        uint256 gasBefore = gasleft();
        vm.prank(prover);
        coordinator.submitProof(modelId, 1, proof, inputs);
        uint256 gasUsed = gasBefore - gasleft();

        console.log("Gas used for submitProof:", gasUsed);
        assertLt(gasUsed, 200_000, "Gas usage exceeds 200k target");
    }

    /// @notice Test gas for Merkle verification
    function test_MerkleVerificationGas() public {
        bytes32 root = keccak256("root");
        bytes32 leaf = keccak256("leaf");
        bytes32[] memory proof = new bytes32[](10); // 10-level tree
        for (uint i = 0; i < 10; i++) {
            proof[i] = keccak256(abi.encodePacked("sibling", i));
        }

        uint256 gasBefore = gasleft();
        dataVerifier.verifyProof(root, leaf, 0, proof);
        uint256 gasUsed = gasBefore - gasleft();

        console.log("Gas for Merkle verification (10 levels):", gasUsed);
        assertLt(gasUsed, 50_000, "Merkle verification gas too high");
    }

    // ============ Edge Cases ============

    /// @notice Test model without data commitment
    function test_ModelWithoutDataCommitment() public {
        uint256 commitment = uint256(keccak256(abi.encodePacked(uint256(1), uint256(2))));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("Test", commitment, MIN_STAKE);

        // No data commitment set
        assertFalse(coordinator.hasDataCommitment(modelId));

        // Verification should still pass (no requirement)
        assertTrue(coordinator.verifyRoundDataCommitment(modelId, 1));
    }

    /// @notice Test cannot start round while paused
    function test_CannotStartRoundWhilePaused() public {
        uint256 commitment = uint256(keccak256(abi.encodePacked(uint256(1), uint256(2))));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("Test", commitment, MIN_STAKE);

        // Pause
        coordinator.emergencyPause();

        // Cannot start round
        vm.prank(modelOwner);
        vm.expectRevert(HelixCoordinatorV2.ContractPaused.selector);
        coordinator.startRound(modelId, ROUND_DURATION);
    }
}
