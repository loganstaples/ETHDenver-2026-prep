// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Script.sol";
import "../src/verification/Halo2Verifier.sol";
import "../src/core/HelixCoordinatorV2.sol";
import "../src/core/HelixCoordinatorV3.sol";
import "../src/core/ModelRegistry.sol";
import "../src/token/HelixToken.sol";
import "../src/token/Staking.sol";
import "../src/token/Rewards.sol";

/// @title DeployScript
/// @notice Deploys the HELIX verification infrastructure to Anvil/testnet
/// @dev Usage:
///   V2 Default:                  forge script script/Deploy.s.sol --broadcast
///   V2 With mock verifier:       forge script script/Deploy.s.sol --sig "runWithMock()" --broadcast
///   V3 Full stack:               forge script script/Deploy.s.sol --sig "deployV3()" --broadcast
///   V3 With mock:                forge script script/Deploy.s.sol --sig "deployV3WithMock()" --broadcast
contract DeployScript is Script {
    // Deployed contract addresses (set after deployment)
    address public verifier;
    address public coordinator;
    address public treasury;
    address public helixToken;
    address public stakingAddr;
    address public rewardsAddr;
    address public registryAddr;

    // ============ V2 Deployment ============

    /// @notice Deploy V2 with real PSE-generated Halo2Verifier
    /// @dev VK is embedded in the Halo2VerifyingKey contract (circuit-specific).
    ///      To update the VK, regenerate Halo2VerifierCore.sol and Halo2VerifyingKey.sol
    ///      from the Rust pipeline and redeploy.
    function run() public returns (address, address) {
        uint256 deployerPrivateKey = vm.envOr("PRIVATE_KEY", uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80));
        treasury = vm.envOr("TREASURY", vm.addr(deployerPrivateKey));

        vm.startBroadcast(deployerPrivateKey);

        Halo2Verifier halo2Verifier = new Halo2Verifier();
        verifier = address(halo2Verifier);

        HelixCoordinatorV2 helixCoordinator = new HelixCoordinatorV2(verifier, treasury);
        coordinator = address(helixCoordinator);

        vm.stopBroadcast();

        console.log("=== HELIX V2 Deployment Complete ===");
        console.log("Halo2Verifier:", verifier);
        console.log("  Core:", address(halo2Verifier.core()));
        console.log("  VK:", halo2Verifier.vk());
        console.log("HelixCoordinatorV2:", coordinator);
        console.log("Treasury:", treasury);
        console.log("====================================");

        return (verifier, coordinator);
    }

    /// @notice Deploy V2 with a mock verifier for testing
    function runWithMock() public returns (address, address) {
        uint256 deployerPrivateKey = vm.envOr("PRIVATE_KEY", uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80));
        treasury = vm.envOr("TREASURY", vm.addr(deployerPrivateKey));

        vm.startBroadcast(deployerPrivateKey);

        MockVerifierForDeploy mockVerifier = new MockVerifierForDeploy();
        verifier = address(mockVerifier);

        HelixCoordinatorV2 helixCoordinator = new HelixCoordinatorV2(verifier, treasury);
        coordinator = address(helixCoordinator);

        vm.stopBroadcast();

        console.log("=== HELIX V2 Mock Deployment ===");
        console.log("MockVerifier:", verifier);
        console.log("HelixCoordinatorV2:", coordinator);
        console.log("================================");

        return (verifier, coordinator);
    }

    // ============ V3 Full Stack Deployment ============

    /// @notice Deploy V3 full stack with real PSE-generated Halo2Verifier
    function deployV3() public {
        uint256 deployerPrivateKey = vm.envOr("PRIVATE_KEY", uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80));
        address deployer = vm.addr(deployerPrivateKey);
        treasury = vm.envOr("TREASURY", deployer);

        vm.startBroadcast(deployerPrivateKey);

        // 1. Deploy HelixToken
        HelixToken token = new HelixToken(treasury);
        helixToken = address(token);

        // 2. Deploy Halo2Verifier (deploys core verifier + VK internally)
        Halo2Verifier halo2Verifier = new Halo2Verifier();
        verifier = address(halo2Verifier);

        // 3. Deploy Staking (100 HELIX min stake, 7 day unbonding, 50% slash rate)
        Staking staking = new Staking(helixToken, 100e18, 7 days, 5000);
        stakingAddr = address(staking);

        // 4. Deploy Rewards
        Rewards rewards = new Rewards(helixToken);
        rewardsAddr = address(rewards);

        // 5. Deploy ModelRegistry
        ModelRegistry registry = new ModelRegistry();
        registryAddr = address(registry);

        // 6. Deploy HelixCoordinatorV3
        HelixCoordinatorV3 v3 = new HelixCoordinatorV3(
            verifier,
            stakingAddr,
            rewardsAddr,
            registryAddr,
            treasury
        );
        coordinator = address(v3);

        // 7. Wire contracts: set coordinator as operator/coordinator
        staking.setOperator(coordinator);
        rewards.setCoordinator(coordinator);
        rewards.setStakingContract(stakingAddr);
        registry.setCoordinator(coordinator);

        // 8. Grant minter role to Rewards for token distribution
        token.addMinter(rewardsAddr);

        vm.stopBroadcast();

        _logV3Deployment("Halo2");
    }

    /// @notice Deploy V3 full stack with mock verifier for testing
    function deployV3WithMock() public {
        uint256 deployerPrivateKey = vm.envOr("PRIVATE_KEY", uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80));
        address deployer = vm.addr(deployerPrivateKey);
        treasury = vm.envOr("TREASURY", deployer);

        vm.startBroadcast(deployerPrivateKey);

        // 1. Deploy HelixToken
        HelixToken token = new HelixToken(treasury);
        helixToken = address(token);

        // 2. Deploy MockVerifier
        MockVerifierForDeploy mockVerifier = new MockVerifierForDeploy();
        verifier = address(mockVerifier);

        // 3. Deploy Staking (100 HELIX min stake, 7 day unbonding, 50% slash rate)
        Staking staking = new Staking(helixToken, 100e18, 7 days, 5000);
        stakingAddr = address(staking);

        // 4. Deploy Rewards
        Rewards rewards = new Rewards(helixToken);
        rewardsAddr = address(rewards);

        // 5. Deploy ModelRegistry
        ModelRegistry registry = new ModelRegistry();
        registryAddr = address(registry);

        // 6. Deploy HelixCoordinatorV3
        HelixCoordinatorV3 v3 = new HelixCoordinatorV3(
            verifier,
            stakingAddr,
            rewardsAddr,
            registryAddr,
            treasury
        );
        coordinator = address(v3);

        // 7. Wire contracts: set coordinator as operator/coordinator
        staking.setOperator(coordinator);
        rewards.setCoordinator(coordinator);
        rewards.setStakingContract(stakingAddr);
        registry.setCoordinator(coordinator);

        // 8. Grant minter role to Rewards for token distribution
        token.addMinter(rewardsAddr);

        vm.stopBroadcast();

        _logV3Deployment("Mock");
    }

    function _logV3Deployment(string memory verifierType) internal view {
        console.log("=== HELIX V3 Full Stack Deployment ===");
        console.log("Verifier (%s):", verifierType, verifier);
        console.log("HelixToken:", helixToken);
        console.log("Staking:", stakingAddr);
        console.log("Rewards:", rewardsAddr);
        console.log("ModelRegistry:", registryAddr);
        console.log("HelixCoordinatorV3:", coordinator);
        console.log("Treasury:", treasury);
        console.log("======================================");
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
