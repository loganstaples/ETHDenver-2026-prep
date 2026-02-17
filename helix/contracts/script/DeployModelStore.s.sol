// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Script.sol";
import "../src/core/HelixModelStore.sol";

contract DeployModelStore is Script {
    function run() external {
        uint256 deployerPrivateKey = vm.envOr(
            "PRIVATE_KEY",
            uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80)
        );

        vm.startBroadcast(deployerPrivateKey);

        HelixModelStore store = new HelixModelStore();
        console.log("HelixModelStore (ERC-721) deployed at:", address(store));
        console.log("  Name:", store.name());
        console.log("  Symbol:", store.symbol());

        vm.stopBroadcast();
    }
}
