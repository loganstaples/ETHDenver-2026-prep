// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../src/governance/TrainingDAO.sol";
import "../src/token/HelixToken.sol";

contract DAOEncodingBugTest is Test {
    TrainingDAO public dao;
    HelixToken public token;
    address public proposer;

    function setUp() public {
        token = new HelixToken(address(this));
        dao = new TrainingDAO(address(token));

        proposer = makeAddr("proposer");
        token.mint(proposer, 10000e18);

        vm.prank(proposer);
        token.delegate(proposer);

        vm.roll(block.number + 1);
    }

    function test_CreateParameterProposal_EncodesCorrectId() public {
        TrainingDAO.ParameterProposal memory params = TrainingDAO.ParameterProposal({
            learningRate: 5e15,
            batchSize: 64,
            maxErrorBound: 500,
            minParticipants: 5,
            roundDuration: 2 hours
        });

        vm.prank(proposer);
        uint256 proposalId = dao.createParameterProposal("Update params", params);

        // The callData should encode the ACTUAL proposalId
        bytes memory expectedCallData = abi.encodeWithSignature("applyParameters(uint256)", proposalId);

        // Verify the proposal exists and has correct type
        (address prop_proposer, TrainingDAO.ProposalType pType,,,,,,) = dao.getProposalInfo(proposalId);
        assertEq(prop_proposer, proposer);
        assertTrue(pType == TrainingDAO.ProposalType.ParameterChange);

        // Verify params stored correctly
        (uint256 lr, uint256 bs, uint256 meb, uint256 mp, uint256 rd) = dao.parameterProposals(proposalId);
        assertEq(lr, 5e15);
        assertEq(bs, 64);
        assertEq(meb, 500);
        assertEq(mp, 5);
        assertEq(rd, 2 hours);
    }

    function test_CreateParameterProposal_MultipleProposals() public {
        TrainingDAO.ParameterProposal memory params1 = TrainingDAO.ParameterProposal({
            learningRate: 5e15, batchSize: 64, maxErrorBound: 500, minParticipants: 5, roundDuration: 2 hours
        });
        TrainingDAO.ParameterProposal memory params2 = TrainingDAO.ParameterProposal({
            learningRate: 1e16, batchSize: 128, maxErrorBound: 1000, minParticipants: 10, roundDuration: 4 hours
        });

        vm.startPrank(proposer);
        uint256 id1 = dao.createParameterProposal("First", params1);
        uint256 id2 = dao.createParameterProposal("Second", params2);
        vm.stopPrank();

        assertEq(id1, 1);
        assertEq(id2, 2);

        // Verify each proposal's params are stored under the correct ID
        (uint256 lr1,,,,) = dao.parameterProposals(id1);
        (uint256 lr2,,,,) = dao.parameterProposals(id2);
        assertEq(lr1, 5e15);
        assertEq(lr2, 1e16);
    }
}
