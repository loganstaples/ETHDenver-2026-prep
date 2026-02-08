// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Script.sol";
import "../src/verification/Halo2Verifier.sol";
import "../src/core/HelixCoordinatorV2.sol";

/// @title DeployScript
/// @notice Deploys the HELIX verification infrastructure to Anvil/testnet
/// @dev Usage:
///   Default (G2 generator):  forge script script/Deploy.s.sol --broadcast
///   With real VK:            forge script script/Deploy.s.sol --sig "runWithVK()" --broadcast
///   With mock verifier:      forge script script/Deploy.s.sol --sig "runWithMock()" --broadcast
contract DeployScript is Script {
    // Deployed contract addresses (set after deployment)
    address public verifier;
    address public coordinator;
    address public treasury;

    /// @notice Deploy with default SRS (G2 generator, s=1)
    function run() public returns (address, address) {
        return deployWithVK(Halo2VKDefaults.g2Generator());
    }

    /// @notice Deploy with VK parameters from environment variables
    /// @dev Set VK_S_G2_X0, VK_S_G2_X1, VK_S_G2_Y0, VK_S_G2_Y1 env vars
    function runWithVK() public returns (address, address) {
        uint256[4] memory sG2;
        sG2[0] = vm.envUint("VK_S_G2_X0");
        sG2[1] = vm.envUint("VK_S_G2_X1");
        sG2[2] = vm.envUint("VK_S_G2_Y0");
        sG2[3] = vm.envUint("VK_S_G2_Y1");

        return deployWithVK(sG2);
    }

    /// @notice Deploy with explicit VK parameters
    /// @param sG2 The SRS [s]₂ G2 point coordinates: [x0, x1, y0, y1]
    function deployWithVK(uint256[4] memory sG2) public returns (address, address) {
        // Get deployer private key from environment
        uint256 deployerPrivateKey = vm.envOr("PRIVATE_KEY", uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80));

        // Treasury address (defaults to deployer)
        treasury = vm.envOr("TREASURY", vm.addr(deployerPrivateKey));

        vm.startBroadcast(deployerPrivateKey);

        // Deploy Halo2Verifier with SRS parameters
        Halo2Verifier halo2Verifier = new Halo2Verifier(sG2);
        verifier = address(halo2Verifier);

        // Deploy HelixCoordinatorV2 with verifier and treasury
        HelixCoordinatorV2 helixCoordinator = new HelixCoordinatorV2(verifier, treasury);
        coordinator = address(helixCoordinator);

        vm.stopBroadcast();

        // Log deployment addresses
        console.log("=== HELIX Deployment Complete ===");
        console.log("Halo2Verifier:", verifier);
        console.log("  VK_S_G2_X0:", sG2[0]);
        console.log("  VK_S_G2_X1:", sG2[1]);
        console.log("  VK_S_G2_Y0:", sG2[2]);
        console.log("  VK_S_G2_Y1:", sG2[3]);
        console.log("HelixCoordinatorV2:", coordinator);
        console.log("Treasury:", treasury);
        console.log("================================");

        return (verifier, coordinator);
    }

    /// @notice Deploy with a mock verifier for testing
    function runWithMock() public returns (address, address) {
        uint256 deployerPrivateKey = vm.envOr("PRIVATE_KEY", uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80));
        treasury = vm.envOr("TREASURY", vm.addr(deployerPrivateKey));

        vm.startBroadcast(deployerPrivateKey);

        // Deploy MockVerifier for integration testing
        MockVerifierForDeploy mockVerifier = new MockVerifierForDeploy();
        verifier = address(mockVerifier);

        HelixCoordinatorV2 helixCoordinator = new HelixCoordinatorV2(verifier, treasury);
        coordinator = address(helixCoordinator);

        vm.stopBroadcast();

        console.log("=== HELIX Mock Deployment ===");
        console.log("MockVerifier:", verifier);
        console.log("HelixCoordinatorV2:", coordinator);
        console.log("=============================");

        return (verifier, coordinator);
    }
}

/// @notice Simple mock verifier that accepts all proofs (for testing)
contract MockVerifierForDeploy {
    bool public shouldAccept = true;

    function verifyProof(bytes memory, uint256[] memory) external view returns (bool) {
        return shouldAccept;
    }

    function setAccept(bool _accept) external {
        shouldAccept = _accept;
    }
}
