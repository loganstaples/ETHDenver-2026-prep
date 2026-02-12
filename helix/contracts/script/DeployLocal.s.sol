// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Script.sol";
import "../src/core/HelixCoordinatorV2.sol";
import "../src/core/DataCommitment.sol";
import "../src/core/TrainingRound.sol";
import "../src/verification/Halo2Verifier.sol";
import "../src/verification/PoseidonHasher.sol";
import "../src/verification/DataVerifier.sol";
import "../src/verification/SlashingEvidence.sol";
import "../src/verification/AggregationVerifier.sol";
import "../src/token/HelixToken.sol";
import "../src/token/Staking.sol";
import "../src/token/Rewards.sol";

/// @title DeployLocalScript
/// @notice Comprehensive deployment script for local testnet with complete flow demonstration
/// @dev Deploys all HELIX contracts and demonstrates: register → commit data → train → verify
///
/// Usage:
///   forge script script/DeployLocal.s.sol --rpc-url http://localhost:8545 --broadcast -vvvv
///
/// Or with Anvil:
///   anvil &
///   forge script script/DeployLocal.s.sol --rpc-url http://localhost:8545 --broadcast -vvvv
contract DeployLocalScript is Script {
    // ============ Deployed Contracts ============
    HelixCoordinatorV2 public coordinator;
    Halo2Verifier public verifier;
    DataCommitment public dataCommitment;
    DataVerifier public dataVerifier;
    SlashingEvidence public slashingEvidence;
    TrainingRound public trainingRound;
    AggregationVerifier public aggregationVerifier;
    HelixToken public helixToken;
    Staking public staking;
    Rewards public rewards;

    // ============ Test Addresses ============
    address public deployer;
    address public treasury;
    address public modelOwner;
    address public prover1;

    // ============ Constants ============
    uint256 constant MIN_STAKE = 0.1 ether;
    uint256 constant PROVER_STAKE = 1 ether;
    uint256 constant ROUND_DURATION = 1 hours;

    /// @notice Main deployment function
    function run() public {
        // Get deployer private key (Anvil default account 0)
        uint256 deployerPrivateKey = vm.envOr(
            "PRIVATE_KEY",
            uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80)
        );
        deployer = vm.addr(deployerPrivateKey);
        treasury = address(uint160(uint256(keccak256("treasury"))));
        modelOwner = deployer; // For demo, deployer is also model owner
        prover1 = deployer;    // For demo, deployer is also prover

        console.log("\n========================================");
        console.log("    HELIX Protocol Local Deployment");
        console.log("========================================\n");

        vm.startBroadcast(deployerPrivateKey);

        // ========== Phase 1: Deploy Core Infrastructure ==========
        console.log("Phase 1: Deploying Core Infrastructure...");

        // Deploy Verifier
        verifier = new Halo2Verifier();
        console.log("  Halo2Verifier:", address(verifier));

        // Deploy Coordinator
        coordinator = new HelixCoordinatorV2(address(verifier), treasury);
        console.log("  HelixCoordinatorV2:", address(coordinator));

        // Deploy Data Commitment
        dataCommitment = new DataCommitment(address(coordinator));
        console.log("  DataCommitment:", address(dataCommitment));

        // Deploy Data Verifier
        dataVerifier = new DataVerifier();
        console.log("  DataVerifier:", address(dataVerifier));

        // Deploy Slashing Evidence
        slashingEvidence = new SlashingEvidence(address(coordinator));
        console.log("  SlashingEvidence:", address(slashingEvidence));

        // ========== Phase 2: Deploy Training Infrastructure ==========
        console.log("\nPhase 2: Deploying Training Infrastructure...");

        // Deploy Training Round
        trainingRound = new TrainingRound();
        trainingRound.setCoordinator(address(coordinator));
        console.log("  TrainingRound:", address(trainingRound));

        // Deploy Aggregation Verifier
        aggregationVerifier = new AggregationVerifier(address(verifier));
        console.log("  AggregationVerifier:", address(aggregationVerifier));

        // ========== Phase 3: Deploy Token Infrastructure ==========
        console.log("\nPhase 3: Deploying Token Infrastructure...");

        // Deploy HELIX Token
        helixToken = new HelixToken(deployer);
        console.log("  HelixToken:", address(helixToken));

        // Deploy Staking Contract (min stake: 0.1 ETH worth, unbonding: 7 days, slash: 50%)
        staking = new Staking(
            address(helixToken),
            0.1 ether,    // minStake
            7 days,       // unbondingPeriod
            5000          // slashingRate (50%)
        );
        console.log("  Staking:", address(staking));

        // Deploy Rewards Contract
        rewards = new Rewards(address(helixToken));
        console.log("  Rewards:", address(rewards));

        // ========== Phase 4: Configure Contracts ==========
        console.log("\nPhase 4: Configuring Contracts...");

        // Link coordinator to supporting contracts
        coordinator.setDataCommitmentContract(address(dataCommitment));
        coordinator.setSlashingEvidenceContract(address(slashingEvidence));
        console.log("  Linked DataCommitment and SlashingEvidence to Coordinator");

        // Grant minter role to Rewards for incentive distribution
        helixToken.grantRole(helixToken.MINTER_ROLE(), address(rewards));
        console.log("  Granted MINTER_ROLE to Rewards contract");

        vm.stopBroadcast();

        // ========== Summary ==========
        console.log("\n========================================");
        console.log("         Deployment Summary");
        console.log("========================================");
        console.log("Core:");
        console.log("  Coordinator:", address(coordinator));
        console.log("  Verifier:   ", address(verifier));
        console.log("  Treasury:   ", treasury);
        console.log("\nData:");
        console.log("  DataCommitment:", address(dataCommitment));
        console.log("  DataVerifier:  ", address(dataVerifier));
        console.log("\nSlashing:");
        console.log("  SlashingEvidence:", address(slashingEvidence));
        console.log("\nTraining:");
        console.log("  TrainingRound:     ", address(trainingRound));
        console.log("  AggregationVerifier:", address(aggregationVerifier));
        console.log("\nTokens:");
        console.log("  HelixToken:", address(helixToken));
        console.log("  Staking:   ", address(staking));
        console.log("  Rewards:   ", address(rewards));
        console.log("========================================\n");
    }

    /// @notice Demonstrate complete flow: register → commit data → train → verify
    function runWithDemo() public {
        // First deploy everything
        run();

        uint256 deployerPrivateKey = vm.envOr(
            "PRIVATE_KEY",
            uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80)
        );

        vm.startBroadcast(deployerPrivateKey);

        console.log("\n========================================");
        console.log("     Running Complete Demo Flow");
        console.log("========================================\n");

        // ========== Step 1: Register Model ==========
        console.log("Step 1: Registering Model...");
        uint256 hashLo = 12345;
        uint256 hashHi = 67890;
        uint256 initialCommitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));

        uint256 modelId = coordinator.registerModel("QmDemoModel", initialCommitment, MIN_STAKE);
        console.log("  Model ID:", modelId);
        console.log("  Initial Commitment:", initialCommitment);

        // ========== Step 2: Commit Dataset ==========
        console.log("\nStep 2: Committing Dataset...");
        bytes32 datasetRoot = keccak256("demo-training-data-merkle-root");

        uint256 datasetId = dataCommitment.commitDataset(
            datasetRoot,
            10000, // 10k samples
            "QmDatasetMetadata"
        );
        console.log("  Dataset ID:", datasetId);
        console.log("  Dataset Root:", vm.toString(datasetRoot));

        // Link model to dataset
        coordinator.setModelDataCommitment(modelId, datasetRoot);
        console.log("  Linked dataset to model");

        // ========== Step 3: Stake to Participate ==========
        console.log("\nStep 3: Staking to Participate...");
        coordinator.stake{value: PROVER_STAKE}(modelId);
        (uint256 stakeAmount,,) = coordinator.getStake(deployer, modelId);
        console.log("  Staked Amount:", stakeAmount / 1e18, "ETH");

        // ========== Step 4: Start Training Round ==========
        console.log("\nStep 4: Starting Training Round...");
        coordinator.startRound(modelId, ROUND_DURATION);
        (uint256 currentRound,,) = coordinator.getModelState(modelId);
        console.log("  Round ID:", currentRound);

        // ========== Step 5: Commit Round Data ==========
        console.log("\nStep 5: Committing Round Data...");
        bytes32 roundDataRoot = keccak256("round-1-batch-data");
        coordinator.commitRoundData(modelId, currentRound, roundDataRoot);
        console.log("  Round Data Root:", vm.toString(roundDataRoot));

        // ========== Summary ==========
        console.log("\n========================================");
        console.log("         Demo Complete!");
        console.log("========================================");
        console.log("Model registered, dataset committed,");
        console.log("stake deposited, round started.");
        console.log("");
        console.log("To submit proof, use SubmitProof.s.sol");
        console.log("========================================\n");

        vm.stopBroadcast();
    }

    /// @notice Deploy with mock verifier for testing (proofs always pass)
    function runWithMock() public {
        uint256 deployerPrivateKey = vm.envOr(
            "PRIVATE_KEY",
            uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80)
        );
        deployer = vm.addr(deployerPrivateKey);
        treasury = address(uint160(uint256(keccak256("treasury"))));

        console.log("\n========================================");
        console.log("  HELIX Mock Deployment (Testing)");
        console.log("========================================\n");

        vm.startBroadcast(deployerPrivateKey);

        // Deploy mock verifier
        MockVerifierLocal mockVerifier = new MockVerifierLocal();
        console.log("MockVerifier:", address(mockVerifier));

        // Deploy Coordinator with mock
        coordinator = new HelixCoordinatorV2(address(mockVerifier), treasury);
        console.log("HelixCoordinatorV2:", address(coordinator));

        // Deploy supporting contracts
        dataCommitment = new DataCommitment(address(coordinator));
        console.log("DataCommitment:", address(dataCommitment));

        slashingEvidence = new SlashingEvidence(address(coordinator));
        console.log("SlashingEvidence:", address(slashingEvidence));

        // Configure
        coordinator.setDataCommitmentContract(address(dataCommitment));
        coordinator.setSlashingEvidenceContract(address(slashingEvidence));

        vm.stopBroadcast();

        console.log("\n========================================");
        console.log("  Mock deployment complete!");
        console.log("  All proofs will PASS by default.");
        console.log("========================================\n");
    }

}

/// @notice Mock verifier that accepts all proofs (for testing)
contract MockVerifierLocal {
    bool public shouldAccept = true;

    function verifyProof(bytes memory, uint256[] memory) external view returns (bool) {
        return shouldAccept;
    }

    function setAccept(bool _accept) external {
        shouldAccept = _accept;
    }
}

/// @title LocalIntegrationTest
/// @notice Script for running integration tests with real proof verification
/// @dev Used for testing Rust proof → Solidity verifier → Coordinator flow
///
/// Usage:
///   forge script script/DeployLocal.s.sol:LocalIntegrationTest --rpc-url http://localhost:8545 --broadcast -vvvv
///   forge script script/DeployLocal.s.sol:LocalIntegrationTest --sig "runProofVerification()" --broadcast -vvvv
contract LocalIntegrationTest is Script {
    // ============ Contracts ============
    HelixCoordinatorV2 public coordinator;
    Halo2Verifier public verifier;

    // ============ BN254 G1 Points ============
    // Generator point (valid on curve)
    uint256 constant G1_X = 1;
    uint256 constant G1_Y = 2;

    // Second valid point from SRS
    uint256 constant G1_2_X = 0x198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2;
    uint256 constant G1_2_Y = 0x1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed;

    // Third valid point
    uint256 constant G1_3_X = 0x090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b;
    uint256 constant G1_3_Y = 0x12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa;

    // ============ Constants ============
    uint256 constant MIN_STAKE = 0.1 ether;
    uint256 constant PROVER_STAKE = 1 ether;
    uint256 constant ROUND_DURATION = 1 hours;

    /// @notice Deploy and run proof verification integration test
    function run() public {
        uint256 deployerPrivateKey = vm.envOr(
            "PRIVATE_KEY",
            uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80)
        );
        address deployer = vm.addr(deployerPrivateKey);
        address treasury = address(uint160(uint256(keccak256("treasury"))));

        console.log("\n========================================");
        console.log("  HELIX Proof Integration Test");
        console.log("========================================\n");
        console.log("Deployer:", deployer);

        vm.startBroadcast(deployerPrivateKey);

        // 1. Deploy contracts
        console.log("\n[1/6] Deploying Verifier and Coordinator...");
        verifier = new Halo2Verifier();
        coordinator = new HelixCoordinatorV2(address(verifier), treasury);
        console.log("  Halo2Verifier:", address(verifier));
        console.log("  Coordinator:", address(coordinator));

        // 2. Register model
        console.log("\n[2/6] Registering Model...");
        uint256 hashLo = 0x3039;  // 12345
        uint256 hashHi = 0x3042;  // 12354
        uint256 initialCommitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));
        uint256 modelId = coordinator.registerModel("QmIntegrationTestModel", initialCommitment, MIN_STAKE);
        console.log("  Model ID:", modelId);
        console.log("  Initial Commitment:", initialCommitment);

        // 3. Stake
        console.log("\n[3/6] Staking...");
        coordinator.stake{value: PROVER_STAKE}(modelId);
        (uint256 stakeAmount,,) = coordinator.getStake(deployer, modelId);
        console.log("  Staked:", stakeAmount / 1e18, "ETH");

        // 4. Start round
        console.log("\n[4/6] Starting Training Round...");
        coordinator.startRound(modelId, ROUND_DURATION);
        (uint256 currentRound,,) = coordinator.getModelState(modelId);
        console.log("  Round ID:", currentRound);

        // 5. Create and submit proof
        console.log("\n[5/6] Submitting Real Proof...");
        bytes memory proof = _createValidProof();
        uint256[] memory publicInputs = _createPublicInputs(modelId);

        console.log("  Proof length:", proof.length, "bytes");
        console.log("  Public inputs:", publicInputs.length);

        // Note: This proof has valid curve points but isn't cryptographically valid
        // So it will fail verification and trigger slashing
        coordinator.submitProof(modelId, 1, proof, publicInputs);

        // 6. Check results
        console.log("\n[6/6] Checking Results...");
        (uint256 finalStake,, bool slashed) = coordinator.getStake(deployer, modelId);

        if (slashed) {
            console.log("  [Expected] Proof failed verification, prover slashed");
            console.log("  Remaining stake:", finalStake / 1e18, "ETH");
            console.log("  Slashed amount:", (PROVER_STAKE - finalStake) / 1e18, "ETH");
        } else {
            console.log("  [Unexpected] Proof passed verification");
        }

        vm.stopBroadcast();

        // Summary
        console.log("\n========================================");
        console.log("  Integration Test Complete");
        console.log("========================================");
        console.log("The test demonstrates:");
        console.log("- Real proof bytes with valid BN254 G1 points");
        console.log("- Proper public input encoding");
        console.log("- End-to-end coordinator flow");
        console.log("- Slashing on invalid proof (expected)");
        console.log("========================================\n");
    }

    /// @notice Test only proof verification (no coordinator)
    function runProofVerification() public {
        uint256 deployerPrivateKey = vm.envOr(
            "PRIVATE_KEY",
            uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80)
        );

        console.log("\n========================================");
        console.log("  Proof Verification Test");
        console.log("========================================\n");

        vm.startBroadcast(deployerPrivateKey);

        verifier = new Halo2Verifier();
        console.log("Halo2Verifier deployed:", address(verifier));

        bytes memory validProof = _createValidProof();
        bytes memory invalidProof = _createInvalidProof();
        uint256[] memory inputs = _createPublicInputs();

        console.log("\nTesting Valid Curve Points Proof:");
        uint256 gasBefore = gasleft();
        bool validResult = verifier.verifyProof(validProof, inputs);
        uint256 gasUsed = gasBefore - gasleft();
        console.log("  Result:", validResult ? "PASS" : "FAIL");
        console.log("  Gas used:", gasUsed);

        console.log("\nTesting Invalid Curve Points Proof:");
        gasBefore = gasleft();
        bool invalidResult = verifier.verifyProof(invalidProof, inputs);
        gasUsed = gasBefore - gasleft();
        console.log("  Result:", invalidResult ? "PASS" : "FAIL");
        console.log("  Gas used:", gasUsed);

        console.log("\nTesting Batch Verification (3 proofs):");
        bytes[] memory proofs = new bytes[](3);
        uint256[][] memory inputsArray = new uint256[][](3);
        for (uint i = 0; i < 3; i++) {
            proofs[i] = _createValidProof();
            inputsArray[i] = _createPublicInputs();
            inputsArray[i][6] = i + 1; // Different step numbers
            // Recompute checksum for the updated step number
            inputsArray[i][7] = PoseidonHasher.computeErrorChecksum(10, i + 1, 1, 1e18);
        }
        gasBefore = gasleft();
        bool batchResult = verifier.batchVerify(proofs, inputsArray);
        gasUsed = gasBefore - gasleft();
        console.log("  Result:", batchResult ? "PASS" : "FAIL");
        console.log("  Total gas:", gasUsed);
        console.log("  Per proof:", gasUsed / 3);

        vm.stopBroadcast();

        console.log("\n========================================");
        console.log("  Verification Test Complete");
        console.log("========================================\n");
    }

    /// @notice Test adversarial scenarios
    function runAdversarialTest() public {
        uint256 deployerPrivateKey = vm.envOr(
            "PRIVATE_KEY",
            uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80)
        );
        address deployer = vm.addr(deployerPrivateKey);
        address treasury = address(uint160(uint256(keccak256("treasury"))));

        console.log("\n========================================");
        console.log("  Adversarial Proof Test");
        console.log("========================================\n");

        vm.startBroadcast(deployerPrivateKey);

        verifier = new Halo2Verifier();
        coordinator = new HelixCoordinatorV2(address(verifier), treasury);

        // Setup
        uint256 initialCommitment = uint256(keccak256(abi.encodePacked(uint256(0x3039), uint256(0x3042))));
        uint256 modelId = coordinator.registerModel("QmAdversarialTest", initialCommitment, MIN_STAKE);
        coordinator.stake{value: PROVER_STAKE}(modelId);
        coordinator.startRound(modelId, ROUND_DURATION);

        (uint256 stakeBefore,,) = coordinator.getStake(deployer, modelId);
        console.log("Stake before:", stakeBefore / 1e18, "ETH");

        // Submit corrupted proof
        console.log("\nSubmitting adversarial proof (corrupted bytes)...");
        bytes memory corruptedProof = _createInvalidProof();
        uint256[] memory inputs = _createPublicInputs(modelId);

        coordinator.submitProof(modelId, 1, corruptedProof, inputs);

        (uint256 stakeAfter,, bool slashed) = coordinator.getStake(deployer, modelId);
        console.log("\nResults:");
        console.log("  Slashed:", slashed ? "YES" : "NO");
        console.log("  Stake after:", stakeAfter / 1e18, "ETH");
        console.log("  Amount slashed:", (stakeBefore - stakeAfter) / 1e18, "ETH");

        // Check slashing record
        (address slashedProver,, uint32 roundId, uint128 amount, string memory reason,) =
            coordinator.slashingRecords(0);
        console.log("\nSlashing Record:");
        console.log("  Prover:", slashedProver);
        console.log("  Round:", roundId);
        console.log("  Amount:", uint256(amount) / 1e18, "ETH");
        console.log("  Reason:", reason);

        vm.stopBroadcast();

        console.log("\n========================================");
        console.log("  Adversarial Test Complete");
        console.log("========================================\n");
    }

    // ============ Helper Functions ============

    /// @notice Creates a valid 320-byte proof with real BN254 curve points
    function _createValidProof() internal pure returns (bytes memory proof) {
        proof = new bytes(320);
        assembly {
            // Point 1: Generator (1, 2)
            mstore(add(proof, 32), G1_X)
            mstore(add(proof, 64), G1_Y)
            // Point 2: Valid SRS point
            mstore(add(proof, 96), G1_2_X)
            mstore(add(proof, 128), G1_2_Y)
            // Point 3: Another valid point
            mstore(add(proof, 160), G1_3_X)
            mstore(add(proof, 192), G1_3_Y)
            // W point: Generator
            mstore(add(proof, 224), G1_X)
            mstore(add(proof, 256), G1_Y)
            // W' point: Generator
            mstore(add(proof, 288), G1_X)
            mstore(add(proof, 320), G1_Y)
        }
    }

    /// @notice Creates an invalid proof (point not on curve)
    function _createInvalidProof() internal pure returns (bytes memory proof) {
        proof = new bytes(320);
        assembly {
            // Invalid point: x=1, y=3 is NOT on the curve y^2 = x^3 + 3
            mstore(add(proof, 32), 1)
            mstore(add(proof, 64), 3) // 3^2 = 9 != 1^3 + 3 = 4
            // Rest valid
            mstore(add(proof, 96), G1_X)
            mstore(add(proof, 128), G1_Y)
            mstore(add(proof, 160), G1_X)
            mstore(add(proof, 192), G1_Y)
            mstore(add(proof, 224), G1_X)
            mstore(add(proof, 256), G1_Y)
            mstore(add(proof, 288), G1_X)
            mstore(add(proof, 320), G1_Y)
        }
    }

    /// @notice Creates public inputs matching Rust circuit format
    /// @param modelId The registered model ID (needed for Poseidon checksum)
    function _createPublicInputs(uint256 modelId) internal pure returns (uint256[] memory inputs) {
        inputs = new uint256[](8);
        inputs[0] = 0x3039;  // oldHashLo
        inputs[1] = 0x3042;  // oldHashHi
        inputs[2] = 0x7b16;  // newHashLo
        inputs[3] = 0x7b32;  // newHashHi
        inputs[4] = 1000;    // loss
        inputs[5] = 10;      // errorBound
        inputs[6] = 1;       // stepNumber
        // Compute real Poseidon checksum matching coordinator's _computeErrorChecksum
        inputs[7] = PoseidonHasher.computeErrorChecksum(10, 1, modelId, 1e18);
    }

    /// @notice Creates public inputs for standalone verifier tests (no coordinator)
    function _createPublicInputs() internal pure returns (uint256[] memory inputs) {
        return _createPublicInputs(1);
    }
}
