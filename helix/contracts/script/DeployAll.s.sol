// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Script.sol";
import "../src/core/HelixCoordinatorV4.sol";
import "../src/core/HelixModelStore.sol";
import "../src/token/HelixToken.sol";
import "../src/verification/Halo2Verifier.sol";

/// @title DeployAll
/// @notice Unified deployment: HelixCoordinatorV4 + HelixModelStore + HelixToken
///
/// Usage (local Anvil):
///   forge script script/DeployAll.s.sol --rpc-url http://localhost:8545 --broadcast
///
/// Usage (ADI Chain testnet):
///   source .env
///   forge script script/DeployAll.s.sol --rpc-url $RPC_URL --broadcast --verify
///
/// Usage (with mock verifier, no real ZK verification):
///   forge script script/DeployAll.s.sol --sig "runWithMock()" --rpc-url $RPC_URL --broadcast
contract DeployAll is Script {
    function run() public {
        uint256 pk = vm.envOr(
            "PRIVATE_KEY",
            uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80)
        );
        address deployer = vm.addr(pk);
        address treasury = vm.envOr("TREASURY", deployer);

        console.log("========================================");
        console.log("  HELIX Unified Deployment");
        console.log("========================================");
        console.log("  Deployer:", deployer);
        console.log("  Treasury:", treasury);
        console.log("");

        vm.startBroadcast(pk);

        // 1. Deploy Halo2Verifier (real BN254 pairing-based KZG verification)
        Halo2Verifier verifier = new Halo2Verifier();
        console.log("  Halo2Verifier:", address(verifier));

        // 2. Deploy HelixCoordinatorV4
        HelixCoordinatorV4 coordinator = new HelixCoordinatorV4(treasury, address(verifier));
        console.log("  HelixCoordinatorV4:", address(coordinator));

        // 3. Deploy HelixModelStore (ERC-721)
        HelixModelStore modelStore = new HelixModelStore();
        console.log("  HelixModelStore:", address(modelStore));

        // 4. Deploy HelixToken (ERC-20 governance)
        HelixToken token = new HelixToken(treasury);
        console.log("  HelixToken:", address(token));

        vm.stopBroadcast();

        console.log("");
        console.log("========================================");
        console.log("  Deployment Complete");
        console.log("========================================");
        console.log("  Coordinator: ", address(coordinator));
        console.log("  Verifier:    ", address(verifier));
        console.log("  ModelStore:  ", address(modelStore));
        console.log("  Token:       ", address(token));
        console.log("  Treasury:    ", treasury);
        console.log("========================================");
    }

    function runWithMock() public {
        uint256 pk = vm.envOr(
            "PRIVATE_KEY",
            uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80)
        );
        address deployer = vm.addr(pk);
        address treasury = vm.envOr("TREASURY", deployer);

        console.log("========================================");
        console.log("  HELIX Unified Deployment (Mock ZK)");
        console.log("========================================");

        vm.startBroadcast(pk);

        // Deploy mock verifier (all proofs pass)
        MockVerifierForDeployAll mockVerifier = new MockVerifierForDeployAll();
        console.log("  MockVerifier:", address(mockVerifier));

        HelixCoordinatorV4 coordinator = new HelixCoordinatorV4(treasury, address(mockVerifier));
        console.log("  HelixCoordinatorV4:", address(coordinator));

        HelixModelStore modelStore = new HelixModelStore();
        console.log("  HelixModelStore:", address(modelStore));

        HelixToken token = new HelixToken(treasury);
        console.log("  HelixToken:", address(token));

        vm.stopBroadcast();

        console.log("");
        console.log("  Coordinator: ", address(coordinator));
        console.log("  Verifier:    ", address(mockVerifier), "(MOCK)");
        console.log("  ModelStore:  ", address(modelStore));
        console.log("  Token:       ", address(token));
        console.log("========================================");
    }
}

/// @notice Mock verifier that accepts all proofs (for testing/demo)
contract MockVerifierForDeployAll {
    function verifyProof(bytes memory, uint256[] memory) external pure returns (bool) {
        return true;
    }
}
