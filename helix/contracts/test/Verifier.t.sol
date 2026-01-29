// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../src/verification/HelixVerifier.sol";

contract VerifierTest is Test {
    HelixVerifier verifier;

    function setUp() public {
        verifier = new HelixVerifier();
    }
}
