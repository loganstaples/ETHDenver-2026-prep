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
import "../src/verification/RLCAggregationVerifier.sol";
import "../src/governance/TrainingDAO.sol";

/// @title DeployScript
/// @notice Deploys the HELIX verification infrastructure to Anvil/testnet
/// @dev Default deployment (run()) now deploys V3 full stack.
///      V2 is deprecated — use deployV2() or deployV2WithMock() for legacy compatibility only.
///
///   V3 Default (production):       forge script script/Deploy.s.sol --broadcast
///   V3 With mock verifier:         forge script script/Deploy.s.sol --sig "runWithMock()" --broadcast
///   V2 Legacy:                     forge script script/Deploy.s.sol --sig "deployV2()" --broadcast
///   V2 Legacy with mock:           forge script script/Deploy.s.sol --sig "deployV2WithMock()" --broadcast
contract DeployScript is Script {
    // ============ Token Distribution Constants ============

    /// @notice Worker incentive pool: 15M HELIX (for staking rewards over time)
    uint256 public constant WORKER_INCENTIVE_POOL = 15_000_000 * 1e18;

    /// @notice Staking rewards pool: 10M HELIX (funded into Rewards contract)
    uint256 public constant STAKING_REWARDS_POOL = 10_000_000 * 1e18;

    /// @notice Rewards per training round: 100 HELIX
    uint256 public constant REWARDS_PER_ROUND = 100 * 1e18;

    /// @notice Reward pool duration: 365 days
    uint256 public constant REWARD_DURATION = 365 days;

    // Deployed contract addresses (set after deployment)
    address public verifier;
    address public coordinator;
    address public treasury;
    address public helixToken;
    address public stakingAddr;
    address public rewardsAddr;
    address public registryAddr;
    address public aggregationVerifierAddr;
    address public daoAddr;

    // ============ V3 Default Deployment ============

    /// @notice Default deployment: V3 full stack with real Halo2Verifier and token distribution
    function run() public {
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

        // 6b. Deploy RLCAggregationVerifier
        address coreAddr = address(halo2Verifier.core());
        address vkAddr = halo2Verifier.vk();
        RLCAggregationVerifier rlcAggVerifier = new RLCAggregationVerifier(coreAddr, vkAddr);
        aggregationVerifierAddr = address(rlcAggVerifier);

        // 7. Deploy TrainingDAO for governance
        TrainingDAO dao = new TrainingDAO(helixToken);
        daoAddr = address(dao);
        dao.setCoordinator(coordinator);

        // 8. Wire contracts
        staking.setOperator(coordinator);
        staking.setTreasury(treasury);
        rewards.setCoordinator(coordinator);
        rewards.setStakingContract(stakingAddr);
        registry.setCoordinator(coordinator);

        // 9. Grant minter role to Rewards for token distribution
        token.addMinter(rewardsAddr);

        // 10. Set aggregation verifier on V3
        v3.setAggregationVerifier(aggregationVerifierAddr);

        // 11. Initial token distribution
        _distributeInitialTokens(token, rewards, deployer);

        vm.stopBroadcast();

        _logV3Deployment("Halo2");
    }

    /// @notice Deploy V3 with mock verifier for testing
    function runWithMock() public {
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

        // 6b. Deploy mock aggregation verifier
        MockVerifierForDeploy mockAggVerifier = new MockVerifierForDeploy();
        aggregationVerifierAddr = address(mockAggVerifier);

        // 7. Deploy TrainingDAO for governance
        TrainingDAO dao = new TrainingDAO(helixToken);
        daoAddr = address(dao);
        dao.setCoordinator(coordinator);

        // 8. Wire contracts
        staking.setOperator(coordinator);
        staking.setTreasury(treasury);
        rewards.setCoordinator(coordinator);
        rewards.setStakingContract(stakingAddr);
        registry.setCoordinator(coordinator);

        // 9. Grant minter role to Rewards for token distribution
        token.addMinter(rewardsAddr);

        // 10. Set aggregation verifier on V3
        v3.setAggregationVerifier(aggregationVerifierAddr);

        // 11. Initial token distribution
        _distributeInitialTokens(token, rewards, deployer);

        vm.stopBroadcast();

        _logV3Deployment("Mock");
    }

    // ============ Token Distribution ============

    /// @notice Distributes initial tokens: treasury allocation, worker incentive pool, staking rewards
    /// @dev Deployer starts with 20M INITIAL_SUPPLY.
    ///      Treasury receives 30M via completeInitialDistribution().
    ///      Deployer transfers 15M to treasury as worker incentive pool.
    ///      Deployer funds reward contract with 10M for staking rewards.
    ///      Remaining 5M stays with deployer for operational needs.
    function _distributeInitialTokens(HelixToken token, Rewards rewards, address deployer) internal {
        // Step 1: Complete treasury distribution (mints 30M to treasury)
        token.completeInitialDistribution();

        // Step 2: Transfer worker incentive pool to treasury
        // These tokens will be distributed to workers over time via governance
        token.transfer(treasury, WORKER_INCENTIVE_POOL);

        // Step 3: Fund the rewards contract for staking rewards
        // Approve + fund in one step
        token.approve(address(rewards), STAKING_REWARDS_POOL);
        rewards.fundRewardPool(
            STAKING_REWARDS_POOL,
            REWARDS_PER_ROUND,
            REWARD_DURATION
        );

        console.log("=== Token Distribution Complete ===");
        console.log("Treasury (30M governance + 15M worker incentives):", treasury);
        console.log("Rewards contract (10M staking rewards):", address(rewards));
        console.log("Deployer retained (5M operational):", deployer);
        console.log("Rewards per round:", REWARDS_PER_ROUND / 1e18, "HELIX");
        console.log("===================================");
    }

    // ============ V2 Legacy Deployment ============

    /// @notice DEPRECATED: Deploy V2 with real Halo2Verifier (legacy compatibility only)
    /// @dev Use run() for V3 production deployment instead.
    function deployV2() public returns (address, address) {
        uint256 deployerPrivateKey = vm.envOr("PRIVATE_KEY", uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80));
        treasury = vm.envOr("TREASURY", vm.addr(deployerPrivateKey));

        vm.startBroadcast(deployerPrivateKey);

        Halo2Verifier halo2Verifier = new Halo2Verifier();
        verifier = address(halo2Verifier);

        HelixCoordinatorV2 helixCoordinator = new HelixCoordinatorV2(verifier, treasury);
        coordinator = address(helixCoordinator);

        vm.stopBroadcast();

        console.log("=== HELIX V2 Deployment (DEPRECATED) ===");
        console.log("WARNING: V2 is deprecated. Use V3 for new deployments.");
        console.log("Halo2Verifier:", verifier);
        console.log("HelixCoordinatorV2:", coordinator);
        console.log("Treasury:", treasury);
        console.log("========================================");

        return (verifier, coordinator);
    }

    /// @notice DEPRECATED: Deploy V2 with mock verifier (legacy compatibility only)
    function deployV2WithMock() public returns (address, address) {
        uint256 deployerPrivateKey = vm.envOr("PRIVATE_KEY", uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80));
        treasury = vm.envOr("TREASURY", vm.addr(deployerPrivateKey));

        vm.startBroadcast(deployerPrivateKey);

        MockVerifierForDeploy mockVerifier = new MockVerifierForDeploy();
        verifier = address(mockVerifier);

        HelixCoordinatorV2 helixCoordinator = new HelixCoordinatorV2(verifier, treasury);
        coordinator = address(helixCoordinator);

        vm.stopBroadcast();

        console.log("=== HELIX V2 Mock Deployment (DEPRECATED) ===");
        console.log("WARNING: V2 is deprecated. Use V3 for new deployments.");
        console.log("MockVerifier:", verifier);
        console.log("HelixCoordinatorV2:", coordinator);
        console.log("=============================================");

        return (verifier, coordinator);
    }

    // Keep old function names as aliases for backward compatibility
    function deployV3() public { run(); }
    function deployV3WithMock() public { runWithMock(); }

    function _logV3Deployment(string memory verifierType) internal view {
        console.log("=== HELIX V3 Full Stack Deployment ===");
        console.log("Verifier (%s):", verifierType, verifier);
        console.log("AggregationVerifier:", aggregationVerifierAddr);
        console.log("HelixToken:", helixToken);
        console.log("Staking:", stakingAddr);
        console.log("Rewards:", rewardsAddr);
        console.log("ModelRegistry:", registryAddr);
        console.log("TrainingDAO:", daoAddr);
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
