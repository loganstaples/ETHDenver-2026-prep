// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Script.sol";
import "../src/core/HelixCoordinatorV2.sol";

/// @title SubmitProofScript
/// @notice Submits a training proof to the HelixCoordinatorV2 contract
/// @dev Usage: forge script script/SubmitProof.s.sol --rpc-url http://localhost:8545 --broadcast
contract SubmitProofScript is Script {
    // Configuration
    address payable public coordinator;
    uint256 public modelId;
    uint256 public roundId;

    function run() public {
        // Load configuration from environment
        coordinator = payable(vm.envAddress("COORDINATOR"));
        modelId = vm.envUint("MODEL_ID");
        roundId = vm.envUint("ROUND_ID");

        // Load proof from file or environment
        bytes memory proof = vm.envOr("PROOF_HEX", bytes(""));
        require(proof.length > 0, "PROOF_HEX environment variable required");

        // Load public inputs (7 values for MLTrainingStepCircuit)
        uint256[] memory publicInputs = new uint256[](7);
        publicInputs[0] = vm.envUint("PI_OLD_HASH_LO");   // old_state_hash_lo
        publicInputs[1] = vm.envUint("PI_OLD_HASH_HI");   // old_state_hash_hi
        publicInputs[2] = vm.envUint("PI_NEW_HASH_LO");   // new_state_hash_lo
        publicInputs[3] = vm.envUint("PI_NEW_HASH_HI");   // new_state_hash_hi
        publicInputs[4] = vm.envUint("PI_LOSS");          // loss
        publicInputs[5] = vm.envUint("PI_ERROR_BOUND");   // total_error_bound
        publicInputs[6] = vm.envUint("PI_STEP");          // step_number

        uint256 deployerPrivateKey = vm.envOr("PRIVATE_KEY", uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80));

        vm.startBroadcast(deployerPrivateKey);

        HelixCoordinatorV2(payable(coordinator)).submitProof(
            modelId,
            roundId,
            proof,
            publicInputs
        );

        vm.stopBroadcast();

        console.log("=== Proof Submitted ===");
        console.log("Model ID:", modelId);
        console.log("Round ID:", roundId);
        console.log("Proof length:", proof.length);
        console.log("=======================");
    }

    /// @notice Submits a proof directly with provided parameters
    function submitProof(
        address payable _coordinator,
        uint256 _modelId,
        uint256 _roundId,
        bytes memory _proof,
        uint256[] memory _publicInputs
    ) public {
        uint256 deployerPrivateKey = vm.envOr("PRIVATE_KEY", uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80));

        vm.startBroadcast(deployerPrivateKey);

        HelixCoordinatorV2(_coordinator).submitProof(
            _modelId,
            _roundId,
            _proof,
            _publicInputs
        );

        vm.stopBroadcast();
    }

    /// @notice Stakes ETH for a model
    function stake(address payable _coordinator, uint256 _modelId, uint256 _amount) public {
        uint256 deployerPrivateKey = vm.envOr("PRIVATE_KEY", uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80));

        vm.startBroadcast(deployerPrivateKey);

        HelixCoordinatorV2(_coordinator).stake{value: _amount}(_modelId);

        vm.stopBroadcast();

        console.log("Staked", _amount, "wei for model", _modelId);
    }

    /// @notice Registers a new model
    function registerModel(
        address payable _coordinator,
        string memory _ipfsHash,
        uint256 _initialCommitment,
        uint256 _minStake
    ) public returns (uint256) {
        uint256 deployerPrivateKey = vm.envOr("PRIVATE_KEY", uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80));

        vm.startBroadcast(deployerPrivateKey);

        uint256 newModelId = HelixCoordinatorV2(_coordinator).registerModel(
            _ipfsHash,
            _initialCommitment,
            _minStake,
            4, 8, 2, 2, 0
        );

        vm.stopBroadcast();

        console.log("Registered model with ID:", newModelId);
        return newModelId;
    }

    /// @notice Starts a new training round
    function startRound(
        address payable _coordinator,
        uint256 _modelId,
        uint256 _duration
    ) public {
        uint256 deployerPrivateKey = vm.envOr("PRIVATE_KEY", uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80));

        vm.startBroadcast(deployerPrivateKey);

        HelixCoordinatorV2(_coordinator).startRound(_modelId, _duration);

        vm.stopBroadcast();

        console.log("Started round for model", _modelId);
    }
}
