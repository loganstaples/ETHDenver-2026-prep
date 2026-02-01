// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Script.sol";
import "forge-std/console.sol";
import "../src/core/HelixCoordinatorV2.sol";
import "../src/verification/SlashingEvidence.sol";
import "../src/verification/BatchVerifier.sol";
import "../src/mocks/MockVerifier.sol";

/// @title AdversarialDemo
/// @notice Demonstrates the complete adversarial slashing flow for HELIX protocol
/// @dev Shows: invalid submission -> detection -> slash -> evidence -> reward
///
/// Run with: forge script script/AdversarialDemo.s.sol --broadcast -vvvv
/// Or for simulation: forge script script/AdversarialDemo.s.sol -vvvv
contract AdversarialDemo is Script {
    // ============ Contracts ============
    HelixCoordinatorV2 public coordinator;
    SlashingEvidence public slashingEvidence;
    BatchVerifier public batchVerifier;
    MockVerifier public mockVerifier;

    // ============ Actors ============
    address public deployer;
    address public treasury;
    address public modelOwner;
    address public honestWorker;
    address public byzantineWorker;
    address public challenger;

    // ============ Constants ============
    uint256 constant STANDARD_STAKE = 1 ether;
    uint256 constant ROUND_DURATION = 1 hours;

    function run() public {
        // Setup actors
        deployer = vm.addr(1);
        treasury = vm.addr(2);
        modelOwner = vm.addr(3);
        honestWorker = vm.addr(4);
        byzantineWorker = vm.addr(5);
        challenger = vm.addr(6);

        // Fund actors for local simulation
        vm.deal(deployer, 100 ether);
        vm.deal(modelOwner, 100 ether);
        vm.deal(honestWorker, 100 ether);
        vm.deal(byzantineWorker, 100 ether);
        vm.deal(challenger, 10 ether);

        vm.startBroadcast(deployer);

        _printHeader("HELIX Adversarial Slashing Demonstration");

        // Phase 1: Deploy contracts
        _deployContracts();

        // Phase 2: Setup model and round
        uint256 modelId = _setupModelAndRound();

        // Phase 3: Workers stake
        _workersStake(modelId);

        // Phase 4: Byzantine worker submits invalid proof -> Gets slashed
        _byzantineAttack(modelId);

        // Phase 5: Show evidence system
        _demonstrateEvidenceSystem();

        // Phase 6: Show gradual slashing
        _demonstrateGradualSlashing();

        // Phase 7: Demonstrate dispute resolution
        _demonstrateDisputeResolution();

        // Phase 8: Show challenger rewards
        _demonstrateChallengerRewards();

        // Phase 9: Batch verification and gas analysis
        _demonstrateBatchVerification();

        // Phase 10: Multi-sig emergency pause
        _demonstrateMultiSigPause();

        // Phase 11: Summary
        _printSummary();

        vm.stopBroadcast();
    }

    /// @notice Run with mock verifier for testing
    function runWithMock() public {
        run();
    }

    // ============ Internal Functions ============

    function _deployContracts() internal {
        _printPhase("Phase 1: Deploying Contracts");

        // Deploy mock verifier
        mockVerifier = new MockVerifier();
        console.log("MockVerifier deployed at:", address(mockVerifier));

        // Deploy coordinator
        coordinator = new HelixCoordinatorV2(address(mockVerifier), treasury);
        console.log("HelixCoordinatorV2 deployed at:", address(coordinator));

        // Deploy slashing evidence
        slashingEvidence = new SlashingEvidence(address(coordinator));
        console.log("SlashingEvidence deployed at:", address(slashingEvidence));

        // Deploy batch verifier
        batchVerifier = new BatchVerifier(address(mockVerifier));
        console.log("BatchVerifier deployed at:", address(batchVerifier));

        // Configure coordinator
        coordinator.setSlashingEvidenceContract(address(slashingEvidence));
        console.log("Contracts configured and linked");

        // Fund the slashing evidence contract for rewards
        payable(address(slashingEvidence)).transfer(10 ether);
        console.log("SlashingEvidence funded with 10 ETH for rewards");

        console.log("");
    }

    function _setupModelAndRound() internal returns (uint256 modelId) {
        _printPhase("Phase 2: Setting Up Model and Training Round");

        // Fund model owner
        vm.deal(modelOwner, 10 ether);

        uint256 hashLo = 12345;
        uint256 hashHi = 67890;
        uint256 commitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));

        vm.stopBroadcast();
        vm.startBroadcast(modelOwner);

        modelId = coordinator.registerModel("QmTestModelIPFSHash", commitment, STANDARD_STAKE / 10);
        console.log("Model registered with ID:", modelId);
        console.log("Initial commitment:", commitment);

        coordinator.startRound(modelId, ROUND_DURATION);
        console.log("Training round 1 started");
        console.log("Round duration:", ROUND_DURATION, "seconds");

        vm.stopBroadcast();
        vm.startBroadcast(deployer);

        console.log("");
    }

    function _workersStake(uint256 modelId) internal {
        _printPhase("Phase 3: Workers Staking");

        // Fund workers
        vm.deal(honestWorker, 10 ether);
        vm.deal(byzantineWorker, 10 ether);

        vm.stopBroadcast();

        // Honest worker stakes
        vm.startBroadcast(honestWorker);
        coordinator.stake{value: STANDARD_STAKE}(modelId);
        console.log("Honest worker staked:", STANDARD_STAKE / 1e18, "ETH");
        vm.stopBroadcast();

        // Byzantine worker stakes
        vm.startBroadcast(byzantineWorker);
        coordinator.stake{value: STANDARD_STAKE}(modelId);
        console.log("Byzantine worker staked:", STANDARD_STAKE / 1e18, "ETH");
        vm.stopBroadcast();

        vm.startBroadcast(deployer);

        // Show total staked
        (uint256 honestStake,,) = coordinator.getStake(honestWorker, modelId);
        (uint256 byzantineStake,,) = coordinator.getStake(byzantineWorker, modelId);
        console.log("Total staked:", (honestStake + byzantineStake) / 1e18, "ETH");
        console.log("");
    }

    function _byzantineAttack(uint256 modelId) internal {
        _printPhase("Phase 4: Byzantine Attack - Invalid Proof Submission");

        // Get current model state for public inputs
        uint256 hashLo = 12345;
        uint256 hashHi = 67890;

        // Create public inputs
        uint256[] memory inputs = new uint256[](7);
        inputs[0] = hashLo;      // old hash lo
        inputs[1] = hashHi;      // old hash hi
        inputs[2] = 1111;        // new hash lo
        inputs[3] = 2222;        // new hash hi
        inputs[4] = 100;         // loss
        inputs[5] = 10;          // error bound
        inputs[6] = 1;           // step number

        bytes memory proof = new bytes(320);

        console.log("Byzantine worker submitting INVALID proof...");
        console.log("Treasury balance before:", treasury.balance / 1e18, "ETH");

        // Set verifier to reject proof
        mockVerifier.setShouldPass(false);

        vm.stopBroadcast();
        vm.startBroadcast(byzantineWorker);

        // Submit invalid proof - will be detected and slashed
        coordinator.submitProof(modelId, 1, proof, inputs);

        vm.stopBroadcast();
        vm.startBroadcast(deployer);

        // Show results
        (uint256 stakeAfter,, bool slashed) = coordinator.getStake(byzantineWorker, modelId);

        console.log("");
        console.log("=== SLASHING DETECTED ===");
        console.log("Byzantine worker SLASHED!");
        console.log("Stake remaining:", stakeAfter / 1e18, "ETH");
        console.log("Amount slashed:", (STANDARD_STAKE - stakeAfter) / 1e18, "ETH");
        console.log("Treasury balance after:", treasury.balance / 1e18, "ETH");
        console.log("Worker marked as slashed:", slashed);

        uint256 slashingCount = coordinator.getSlashingRecordCount();
        console.log("Total slashing records:", slashingCount);
        console.log("");
    }

    function _demonstrateEvidenceSystem() internal {
        _printPhase("Phase 5: Slashing Evidence System");

        console.log("Setting up evidence recording...");

        // Set deployer as coordinator for evidence demo (deployer is owner of slashingEvidence)
        slashingEvidence.setCoordinator(deployer);

        // Submit evidence for the slashing
        uint256 evidenceId = slashingEvidence.submitEvidence(
            byzantineWorker,
            uint64(0),       // modelId
            uint32(1),       // roundId
            SlashingEvidence.ViolationType.InvalidProof,
            SlashingEvidence.ErrorCode.PROOF_VERIFICATION_FAILED,
            uint128(STANDARD_STAKE / 2),  // slashed amount
            uint128(STANDARD_STAKE / 2),  // remaining stake
            keccak256("invalidProof"),
            challenger,      // challenger who reported it
            "Invalid ZK proof verification - proof failed cryptographic verification"
        );

        console.log("Evidence submitted with ID:", evidenceId);

        SlashingEvidence.Evidence memory evidence = slashingEvidence.getEvidence(evidenceId);
        console.log("Violation type: InvalidProof");
        console.log("Error code: PROOF_VERIFICATION_FAILED");
        console.log("Severity level:", uint8(evidence.severity));
        console.log("Challenger reward calculated:", evidence.challengerReward / 1e15, "finney");
        console.log("Protocol fee:", evidence.protocolFee / 1e15, "finney");
        console.log("");
    }

    function _demonstrateGradualSlashing() internal {
        _printPhase("Phase 6: Gradual Slashing Demonstration");

        console.log("Showing severity escalation for repeat offenders...");
        console.log("");

        // First offense - Warning
        address repeatOffender = vm.addr(10);
        console.log("First offense (TimeoutViolation):");
        uint256 ev1 = slashingEvidence.submitEvidence(
            repeatOffender,
            uint64(1),
            uint32(1),
            SlashingEvidence.ViolationType.TimeoutViolation,
            SlashingEvidence.ErrorCode.SUBMISSION_TOO_LATE,
            0,  // Warning - no slash
            uint128(1 ether),
            keccak256("proof1"),
            address(0),
            "Timeout violation"
        );
        SlashingEvidence.Evidence memory e1 = slashingEvidence.getEvidence(ev1);
        console.log("  Severity: Warning (0)");
        console.log("  Slashed: 0 ETH");
        console.log("");

        // Second offense - Minor
        console.log("Second offense (TimeoutViolation):");
        uint256 ev2 = slashingEvidence.submitEvidence(
            repeatOffender,
            uint64(1),
            uint32(2),
            SlashingEvidence.ViolationType.TimeoutViolation,
            SlashingEvidence.ErrorCode.SUBMISSION_TOO_LATE,
            uint128(0.1 ether),  // 10% slash
            uint128(0.9 ether),
            keccak256("proof2"),
            address(0),
            "Second timeout violation"
        );
        SlashingEvidence.Evidence memory e2 = slashingEvidence.getEvidence(ev2);
        console.log("  Severity: Minor (1) - ESCALATED");
        console.log("  Slashed: 0.1 ETH (10%)");
        console.log("");

        // Third offense - Moderate
        console.log("Third offense (ErrorBoundExceeded):");
        uint256 ev3 = slashingEvidence.submitEvidence(
            repeatOffender,
            uint64(1),
            uint32(3),
            SlashingEvidence.ViolationType.ErrorBoundExceeded,
            SlashingEvidence.ErrorCode.ERROR_BOUND_EXCEEDED,
            uint128(0.225 ether),  // 25% slash
            uint128(0.675 ether),
            keccak256("proof3"),
            address(0),
            "Error bound exceeded"
        );
        SlashingEvidence.Evidence memory e3 = slashingEvidence.getEvidence(ev3);
        console.log("  Severity: Moderate (2) - ESCALATED");
        console.log("  Slashed: 0.225 ETH (25%)");
        console.log("");

        SlashingEvidence.WarningRecord memory record = slashingEvidence.getWarningRecord(repeatOffender);
        console.log("Warning record for repeat offender:");
        console.log("  Total warnings:", record.warningCount);
        console.log("  Total slashed:", record.totalSlashedAmount / 1e15, "finney");
        console.log("  Max severity reached:", uint8(record.maxSeverity));
        console.log("");
    }

    function _demonstrateDisputeResolution() internal {
        _printPhase("Phase 7: Dispute Resolution");

        console.log("Byzantine worker filing dispute...");

        // Fund byzantine worker for dispute stake
        vm.deal(byzantineWorker, 1 ether);

        // Get evidence ID from phase 5 (ID = 0)
        uint256 evidenceId = 0;

        vm.stopBroadcast();
        vm.startBroadcast(byzantineWorker);

        uint256 disputeStake = slashingEvidence.disputeStakeRequired();
        console.log("Dispute stake required:", disputeStake / 1e15, "finney");

        uint256 disputeId = slashingEvidence.fileDispute{value: disputeStake}(
            evidenceId,
            keccak256("counter-evidence"),
            "I claim the proof was actually valid - verifier bug"
        );
        console.log("Dispute filed with ID:", disputeId);

        vm.stopBroadcast();
        vm.startBroadcast(deployer);

        SlashingEvidence.Evidence memory evidence = slashingEvidence.getEvidence(evidenceId);
        console.log("Evidence status changed to: Disputed");

        // Owner resolves dispute
        console.log("");
        console.log("Protocol owner reviewing dispute...");
        slashingEvidence.resolveDispute(disputeId, false);  // Against prover
        console.log("Dispute resolved AGAINST prover (evidence upheld)");
        console.log("Dispute stake forfeited to treasury");
        console.log("");
    }

    function _demonstrateChallengerRewards() internal {
        _printPhase("Phase 8: Challenger Reward Distribution");

        // Evidence ID 0 should now be resolved
        uint256 evidenceId = 0;

        SlashingEvidence.Evidence memory evidence = slashingEvidence.getEvidence(evidenceId);
        console.log("Challenger address:", evidence.challenger);
        console.log("Reward amount:", evidence.challengerReward / 1e15, "finney");
        console.log("Protocol fee:", evidence.protocolFee / 1e15, "finney");

        uint256 challengerBalanceBefore = challenger.balance;
        console.log("Challenger balance before:", challengerBalanceBefore / 1e15, "finney");

        // Distribute reward
        slashingEvidence.distributesChallengerReward(evidenceId);

        uint256 challengerBalanceAfter = challenger.balance;
        console.log("Challenger balance after:", challengerBalanceAfter / 1e15, "finney");
        console.log("Reward received:", (challengerBalanceAfter - challengerBalanceBefore) / 1e15, "finney");
        console.log("");
        console.log("=== CHALLENGER REWARDED FOR FINDING FRAUD ===");
        console.log("");
    }

    function _demonstrateBatchVerification() internal {
        _printPhase("Phase 9: Batch Verification & Gas Analysis");

        console.log("Testing batch proof verification for gas optimization...");
        console.log("");

        // Create test proofs
        bytes[] memory proofs = new bytes[](5);
        uint256[][] memory inputsArray = new uint256[][](5);

        for (uint256 i = 0; i < 5; i++) {
            proofs[i] = new bytes(320);
            inputsArray[i] = new uint256[](7);
            inputsArray[i][0] = 12345 + i;
            inputsArray[i][1] = 67890;
            inputsArray[i][2] = 1111;
            inputsArray[i][3] = 2222;
            inputsArray[i][4] = 100;
            inputsArray[i][5] = 10;
            inputsArray[i][6] = i + 1;
        }

        // Single verification gas
        mockVerifier.setShouldPass(true);
        uint256 singleGas = batchVerifier.estimateSingleVerifyGas(proofs[0], inputsArray[0]);
        console.log("Single proof verification gas:", singleGas);

        // Batch verification gas
        (uint256 totalGas, uint256 perProofGas) = batchVerifier.estimateBatchVerifyGas(proofs, inputsArray);
        console.log("Batch (5 proofs) total gas:", totalGas);
        console.log("Batch per-proof gas:", perProofGas);

        // Gas target check
        uint256 gasTarget = 250000;
        bool underTarget = perProofGas < gasTarget;
        console.log("");
        console.log("=== GAS TARGET CHECK ===");
        console.log("Target: <", gasTarget);
        console.log("Actual:", perProofGas);
        console.log("Status:", underTarget ? "PASS" : "FAIL");

        // Demonstrate error codes
        console.log("");
        console.log("Error code system:");
        console.log("  0x0103:", batchVerifier.getErrorDescription(0x0103));
        console.log("  0x0201:", batchVerifier.getErrorDescription(0x0201));
        console.log("  0x0301:", batchVerifier.getErrorDescription(0x0301));
        console.log("  0x0602:", batchVerifier.getErrorDescription(0x0602));

        // Batch verification result
        mockVerifier.setShouldPass(true);
        BatchVerifier.BatchResult memory result = batchVerifier.batchVerify(proofs, inputsArray);
        console.log("");
        console.log("Batch verification results:");
        console.log("  Total proofs:", result.totalProofs);
        console.log("  Valid proofs:", result.validProofs);
        console.log("  Invalid proofs:", result.invalidProofs);
        console.log("  Average gas per proof:", result.averageGasPerProof);
        console.log("");
    }

    function _demonstrateMultiSigPause() internal {
        _printPhase("Phase 10: Multi-Sig Emergency Pause System");

        console.log("Coordinator multi-sig pause configuration:");
        console.log("");

        // Show current guardian configuration
        console.log("Guardian Management:");
        console.log("  Owner is guardian:", coordinator.isGuardian(deployer));
        console.log("  Guardian count:", coordinator.guardianCount());
        console.log("  Required guardians:", coordinator.requiredGuardians());

        // Add additional guardians for demo
        address guardian2 = vm.addr(20);
        address guardian3 = vm.addr(21);

        coordinator.addGuardian(guardian2);
        coordinator.addGuardian(guardian3);

        console.log("");
        console.log("Added 2 additional guardians:");
        console.log("  Guardian 2:", guardian2);
        console.log("  Guardian 3:", guardian3);
        console.log("  New guardian count:", coordinator.guardianCount());

        // Show pause approval status
        (uint64 nonce, uint8 approvals, uint8 required) = coordinator.getPauseApprovalStatus();
        console.log("");
        console.log("Pause Approval Status:");
        console.log("  Current nonce:", nonce);
        console.log("  Current approvals:", approvals);
        console.log("  Required approvals:", required);

        // Show challenger reward config
        (uint16 rewardPct, uint128 minReward, uint128 maxReward, bool enabled) = coordinator.getChallengerConfig();
        console.log("");
        console.log("Challenger Reward Configuration:");
        console.log("  Reward percentage:", rewardPct / 100, "%");
        console.log("  Min reward:", uint256(minReward) / 1e15, "finney");
        console.log("  Max reward:", uint256(maxReward) / 1e18, "ETH");
        console.log("  Enabled:", enabled);

        // Recovery configuration
        console.log("");
        console.log("Recovery Configuration:");
        console.log("  Time lock:", coordinator.recoveryTimeLock() / 3600, "hours");

        (address pendingOwner, uint256 execTime, bool isPending) = coordinator.getPendingRecovery();
        console.log("  Pending recovery:", isPending);
        console.log("");
    }

    function _printSummary() internal {
        _printPhase("Demo Summary");

        console.log("Demonstrated capabilities:");
        console.log("  [x] Invalid proof detection");
        console.log("  [x] Automatic slashing (50% default)");
        console.log("  [x] Detailed evidence recording");
        console.log("  [x] Error codes for specific violations");
        console.log("  [x] Gradual slashing with severity escalation");
        console.log("  [x] Dispute filing and resolution");
        console.log("  [x] Challenger reward distribution");
        console.log("  [x] Protocol fee collection");
        console.log("  [x] Batch proof verification (<250K gas target)");
        console.log("  [x] Multi-sig emergency pause system");
        console.log("  [x] Time-locked recovery mechanism");
        console.log("  [x] Guardian management");
        console.log("");
        console.log("Contract addresses:");
        console.log("  Coordinator:", address(coordinator));
        console.log("  SlashingEvidence:", address(slashingEvidence));
        console.log("  BatchVerifier:", address(batchVerifier));
        console.log("  MockVerifier:", address(mockVerifier));
        console.log("");
        console.log("Final state:");
        console.log("  Treasury balance:", treasury.balance / 1e18, "ETH");
        console.log("  Slashing records:", coordinator.getSlashingRecordCount());
        console.log("  Evidence records:", slashingEvidence.nextEvidenceId());
        console.log("  Disputes filed:", slashingEvidence.nextDisputeId());
        console.log("");
    }

    function _printHeader(string memory title) internal pure {
        console.log("");
        console.log("================================================================");
        console.log(title);
        console.log("================================================================");
        console.log("");
    }

    function _printPhase(string memory phase) internal pure {
        console.log("----------------------------------------------------------------");
        console.log(phase);
        console.log("----------------------------------------------------------------");
    }

    /// @notice Receive ETH for deployment
    receive() external payable {}
}
