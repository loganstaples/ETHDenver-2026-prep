// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../src/core/HelixCoordinator.sol";
import "../src/mocks/MockVerifier.sol";

contract HelixCoordinatorTest is Test {
    HelixCoordinator coordinator;
    MockVerifier verifier;
    
    address owner = address(0xABC);
    address prover = address(0xDEF);

    function setUp() public {
        verifier = new MockVerifier();
        coordinator = new HelixCoordinator(address(verifier));
    }

    function testRegisterModel() public {
        vm.prank(owner);
        coordinator.registerModel("ipfs://QmHash", 12345);
        
        (string memory ipfs, uint256 commit, uint256 round, address modelOwner) = coordinator.models(0);
        assertEq(ipfs, "ipfs://QmHash");
        assertEq(commit, 12345);
        assertEq(round, 0);
        assertEq(modelOwner, owner);
    }

    function testStartRound() public {
        vm.startPrank(owner);
        coordinator.registerModel("ipfs://QmHash", 100);
        coordinator.startRound(0);
        vm.stopPrank();

        (uint256 commit, bool completed) = coordinator.rounds(0, 1); // Round 1
        assertEq(commit, 100);
        assertEq(completed, false);
    }

    function testSubmitGradient() public {
        // Setup
        vm.startPrank(owner);
        coordinator.registerModel("ipfs://QmHash", 100);
        coordinator.startRound(0);
        vm.stopPrank();

        // Prepare inputs
        uint256[] memory inputs = new uint256[](3);
        inputs[0] = 100; // Old commitment
        inputs[1] = 200; // New commitment
        inputs[2] = 50;  // Gradient commitment
        bytes memory proof = hex"1234";

        vm.prank(prover);
        coordinator.submitGradient(0, 1, proof, inputs);

        // Verify state update
        (,, uint256 currentRound, ) = coordinator.models(0);
        assertEq(currentRound, 1);
        
        (uint256 commit, bool completed) = coordinator.rounds(0, 1);
        assertEq(completed, true);
        
        // Model commitment should update
        (, uint256 newCommit,,) = coordinator.models(0);
        assertEq(newCommit, 200);
    }

    function test_RevertIf_InvalidProof() public {
        verifier.setShouldPass(false); // Mock verify failure
        
        vm.startPrank(owner);
        coordinator.registerModel("ipfs://QmHash", 100);
        coordinator.startRound(0);
        vm.stopPrank();

        uint256[] memory inputs = new uint256[](2);
        inputs[0] = 100;
        inputs[1] = 200;
        
        vm.expectRevert("Invalid proof");
        coordinator.submitGradient(0, 1, hex"", inputs);
    }
}
