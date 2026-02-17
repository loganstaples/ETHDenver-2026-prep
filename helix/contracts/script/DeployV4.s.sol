// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Script.sol";
import "../src/core/HelixCoordinatorV4.sol";

/// @title DeployV4Script
/// @notice Deploy the V4 coordinator for the MPC-primary demo.
///
/// Usage:
///   PRIVATE_KEY=0xac09... forge script script/DeployV4.s.sol \
///       --rpc-url http://localhost:8545 --broadcast -q 2>/dev/null
///
/// The script prints ONLY the deployed address to stdout (for easy capture).
contract DeployV4Script is Script {
    function run() public {
        uint256 pk = vm.envOr(
            "PRIVATE_KEY",
            uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80)
        );
        address treasury = vm.envOr("TREASURY", vm.addr(pk));

        vm.startBroadcast(pk);
        HelixCoordinatorV4 coord = new HelixCoordinatorV4(treasury, address(0));
        vm.stopBroadcast();

        // Print address so the caller can capture it
        console.log(address(coord));
    }
}
