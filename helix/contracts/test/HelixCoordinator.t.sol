// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../src/core/HelixCoordinator.sol";

contract HelixCoordinatorTest is Test {
    HelixCoordinator coordinator;

    function setUp() public {
        coordinator = new HelixCoordinator();
    }
}
