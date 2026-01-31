// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Script.sol";
import "../src/core/HelixCoordinatorV2.sol";
import "../src/core/DataCommitment.sol";
import "../src/core/TrainingRound.sol";
import "../src/verification/Halo2Verifier.sol";
import "../src/verification/DataVerifier.sol";
import "../src/verification/SlashingEvidence.sol";
import "../src/verification/AggregationVerifier.sol";
import "../src/verification/BoundsChecker.sol";
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
    BoundsChecker public boundsChecker;
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

        // Deploy Bounds Checker (max error 1000)
        boundsChecker = new BoundsChecker(1000);
        console.log("  BoundsChecker:", address(boundsChecker));

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
        console.log("  BoundsChecker:     ", address(boundsChecker));
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
