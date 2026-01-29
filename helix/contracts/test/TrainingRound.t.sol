// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../src/core/TrainingRound.sol";

contract TrainingRoundTest is Test {
    TrainingRound round;

    function setUp() public {
        round = new TrainingRound();
    }
}
