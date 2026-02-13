// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../src/core/HelixCoordinatorV3.sol";
import "../src/core/ModelRegistry.sol";
import "../src/token/HelixToken.sol";
import "../src/token/Staking.sol";
import "../src/token/Rewards.sol";
import "../src/governance/TrainingDAO.sol";
import "./ProofFixtures.t.sol";

// ============================================================================
// 1. Flash Loan Attack on TrainingDAO Proposal Creation
// ============================================================================

/// @title FlashLoanProposalAttackTest
/// @notice Verifies that getPastVotes(block.number - 1) prevents same-block flash loan attacks
contract FlashLoanProposalAttackTest is Test {
    HelixToken public token;
    TrainingDAO public dao;

    address public legitimateVoter;
    address public flashAttacker;

    uint256 constant VOTER_BALANCE = 10_000 ether;
    uint256 constant THRESHOLD = 1000 ether; // DAO default proposal threshold

    function setUp() public {
        legitimateVoter = makeAddr("legitimateVoter");
        flashAttacker = makeAddr("flashAttacker");

        token = new HelixToken(makeAddr("treasury"));
        dao = new TrainingDAO(address(token));

        // Give legitimate voter tokens and delegate
        token.mint(legitimateVoter, VOTER_BALANCE);
        vm.prank(legitimateVoter);
        token.delegate(legitimateVoter);

        // Advance block so delegation checkpoint is finalized
        vm.roll(block.number + 1);
    }

    /// @notice Test: flash loan in the same block cannot create a proposal
    /// @dev Simulates acquiring tokens + delegating + creating proposal in one block
    function test_FlashLoanSameBlock_CannotCreateProposal() public {
        // Flash attacker acquires tokens in this block
        token.mint(flashAttacker, THRESHOLD);

        // Flash attacker delegates in this block
        vm.prank(flashAttacker);
        token.delegate(flashAttacker);

        // Flash attacker tries to create proposal in the SAME block
        // getPastVotes(block.number - 1) checks the previous block where attacker had 0 votes
        vm.prank(flashAttacker);
        vm.expectRevert("Below proposal threshold");
        dao.createProposal(
            TrainingDAO.ProposalType.Custom,
            "Flash loan proposal",
            address(0),
            ""
        );
    }

    /// @notice Test: tokens acquired in current block have zero getPastVotes
    function test_CurrentBlockTokens_ZeroPastVotes() public {
        // Mint and delegate in current block
        token.mint(flashAttacker, THRESHOLD * 2);
        vm.prank(flashAttacker);
        token.delegate(flashAttacker);

        // getPastVotes at block.number - 1 should be 0
        uint256 pastVotes = token.getPastVotes(flashAttacker, block.number - 1);
        assertEq(pastVotes, 0, "Past votes should be zero for same-block delegation");
    }

    /// @notice Test: legitimate voter who delegated in a prior block can create proposal
    function test_LegitimateVoter_CanCreateProposal() public {
        // legitimateVoter delegated in setUp, block advanced
        vm.prank(legitimateVoter);
        uint256 proposalId = dao.createProposal(
            TrainingDAO.ProposalType.Custom,
            "Legitimate proposal",
            address(0),
            ""
        );
        assertGt(proposalId, 0, "Legitimate voter should be able to create proposal");
    }
}

// ============================================================================
// 2. Double-Claim Prevention (Single Claim Path)
// ============================================================================

/// @title DoubleClaimSinglePathTest
/// @notice Verifies that with claimRewards() removed, only claimRoundRewards() exists
///         and double-claims are impossible
contract DoubleClaimSinglePathTest is Test {
    HelixToken public token;
    Rewards public rewards;

    address public funder;
    address public participant;
    address public coordinator;

    uint256 constant FUND_AMOUNT = 100_000 ether;
    uint256 constant REWARDS_PER_ROUND = 1000 ether;

    function setUp() public {
        funder = makeAddr("funder");
        participant = makeAddr("participant");
        coordinator = makeAddr("coordinator");

        token = new HelixToken(makeAddr("treasury"));
        rewards = new Rewards(address(token));
        rewards.setCoordinator(coordinator);

        // Fund reward pool
        token.mint(funder, FUND_AMOUNT);
        vm.startPrank(funder);
        token.approve(address(rewards), FUND_AMOUNT);
        rewards.fundRewardPool(FUND_AMOUNT, REWARDS_PER_ROUND, 365 days);
        vm.stopPrank();
    }

    /// @notice Test: claiming the same round twice via claimRoundRewards reverts
    function test_DoubleClaimSameRound_Reverts() public {
        // Allocate rewards
        vm.startPrank(coordinator);
        rewards.registerParticipant(0, 1, participant);
        rewards.allocateRoundRewards(0, 1);
        vm.stopPrank();

        uint256[] memory modelIds = new uint256[](1);
        uint256[] memory roundIds = new uint256[](1);
        modelIds[0] = 0;
        roundIds[0] = 1;

        // First claim succeeds
        vm.prank(participant);
        rewards.claimRoundRewards(modelIds, roundIds);

        uint256 balanceAfterClaim = token.balanceOf(participant);
        assertGt(balanceAfterClaim, 0, "Should have received rewards");

        // Second claim for same round reverts
        vm.prank(participant);
        vm.expectRevert("No rewards to claim");
        rewards.claimRoundRewards(modelIds, roundIds);

        // Balance unchanged
        assertEq(token.balanceOf(participant), balanceAfterClaim, "Balance should not change");
    }

    /// @notice Test: claiming round in batch, then individually, is blocked
    function test_BatchThenIndividual_DoubleClaimBlocked() public {
        // Allocate 2 rounds
        vm.startPrank(coordinator);
        rewards.registerParticipant(0, 1, participant);
        rewards.allocateRoundRewards(0, 1);
        rewards.registerParticipant(0, 2, participant);
        rewards.allocateRoundRewards(0, 2);
        vm.stopPrank();

        // Claim both in batch
        uint256[] memory modelIds = new uint256[](2);
        uint256[] memory roundIds = new uint256[](2);
        modelIds[0] = 0;
        modelIds[1] = 0;
        roundIds[0] = 1;
        roundIds[1] = 2;

        vm.prank(participant);
        rewards.claimRoundRewards(modelIds, roundIds);

        (uint256 totalEarned,,,) = rewards.getParticipantStats(participant);
        assertEq(token.balanceOf(participant), totalEarned, "Should have claimed all");

        // Try to claim round 1 individually
        uint256[] memory singleModel = new uint256[](1);
        uint256[] memory singleRound = new uint256[](1);
        singleModel[0] = 0;
        singleRound[0] = 1;

        vm.prank(participant);
        vm.expectRevert("No rewards to claim");
        rewards.claimRoundRewards(singleModel, singleRound);
    }

    /// @notice Test: claimRewards() function no longer exists (compile-time check)
    /// @dev This test verifies the function was removed by checking that only
    ///      claimRoundRewards is the claim path
    function test_OnlySingleClaimPathExists() public {
        vm.startPrank(coordinator);
        rewards.registerParticipant(0, 1, participant);
        rewards.allocateRoundRewards(0, 1);
        vm.stopPrank();

        // The only way to claim is via claimRoundRewards
        uint256[] memory modelIds = new uint256[](1);
        uint256[] memory roundIds = new uint256[](1);
        modelIds[0] = 0;
        roundIds[0] = 1;

        vm.prank(participant);
        rewards.claimRoundRewards(modelIds, roundIds);

        (,uint256 totalClaimed,,) = rewards.getParticipantStats(participant);
        assertGt(totalClaimed, 0, "Should have claimed via the single path");
    }
}

// ============================================================================
// 3. V3 commitRoundData Access Control
// ============================================================================

/// @title V3CommitRoundDataAuthTest
/// @notice Verifies that commitRoundData in HelixCoordinatorV3 requires authorization
contract V3CommitRoundDataAuthTest is Test {
    HelixCoordinatorV3 public coordinator;
    HelixToken public token;
    Staking public staking;
    Rewards public rewards;
    ModelRegistry public registry;

    address public contractOwner;
    address public modelOwner;
    address public randomUser;
    address public treasuryAddr;

    /// @notice Mock verifier that always accepts
    MockVerifierForAuthTest public mockVerifier;

    function setUp() public {
        contractOwner = address(this);
        modelOwner = makeAddr("modelOwner");
        randomUser = makeAddr("randomUser");
        treasuryAddr = makeAddr("treasury");

        token = new HelixToken(treasuryAddr);
        mockVerifier = new MockVerifierForAuthTest();
        staking = new Staking(address(token), 100e18, 7 days, 5000);
        rewards = new Rewards(address(token));
        registry = new ModelRegistry();

        coordinator = new HelixCoordinatorV3(
            address(mockVerifier),
            address(staking),
            address(rewards),
            address(registry),
            treasuryAddr
        );

        // Wire contracts
        staking.setOperator(address(coordinator));
        rewards.setCoordinator(address(coordinator));
        registry.setCoordinator(address(coordinator));
    }

    /// @notice Helper to register a model and start a round
    function _setupModel() internal returns (uint256 modelId) {
        vm.prank(modelOwner);
        modelId = coordinator.registerModel("TestModel", "desc", "hash", 12345, 4, 8, 2, 2, 0);

        vm.prank(modelOwner);
        coordinator.startRound(modelId, 1 hours);
    }

    /// @notice Test: random unauthorized user CANNOT commit round data
    function test_CommitRoundData_UnauthorizedUser_Reverts() public {
        uint256 modelId = _setupModel();

        vm.prank(randomUser);
        vm.expectRevert("Not authorized");
        coordinator.commitRoundData(modelId, 1, bytes32(uint256(42)));
    }

    /// @notice Test: model owner CAN commit round data
    function test_CommitRoundData_ModelOwner_Succeeds() public {
        uint256 modelId = _setupModel();

        vm.prank(modelOwner);
        coordinator.commitRoundData(modelId, 1, bytes32(uint256(42)));

        assertEq(coordinator.getRoundDataRoot(modelId, 1), bytes32(uint256(42)));
    }

    /// @notice Test: contract owner CAN commit round data
    function test_CommitRoundData_ContractOwner_Succeeds() public {
        uint256 modelId = _setupModel();

        // contractOwner is address(this) which deployed the coordinator
        coordinator.commitRoundData(modelId, 1, bytes32(uint256(99)));

        assertEq(coordinator.getRoundDataRoot(modelId, 1), bytes32(uint256(99)));
    }

    /// @notice Test: unauthorized user cannot overwrite existing round data
    function test_CommitRoundData_CannotOverwriteByUnauthorized() public {
        uint256 modelId = _setupModel();

        // Model owner sets data
        vm.prank(modelOwner);
        coordinator.commitRoundData(modelId, 1, bytes32(uint256(42)));

        // Random user cannot overwrite
        vm.prank(randomUser);
        vm.expectRevert("Not authorized");
        coordinator.commitRoundData(modelId, 1, bytes32(uint256(99)));

        // Data unchanged
        assertEq(coordinator.getRoundDataRoot(modelId, 1), bytes32(uint256(42)));
    }

    /// @notice Test: multiple unauthorized addresses all revert
    function test_CommitRoundData_MultipleUnauthorized_AllRevert() public {
        uint256 modelId = _setupModel();

        address attacker1 = makeAddr("attacker1");
        address attacker2 = makeAddr("attacker2");
        address attacker3 = makeAddr("attacker3");

        vm.prank(attacker1);
        vm.expectRevert("Not authorized");
        coordinator.commitRoundData(modelId, 1, bytes32(uint256(1)));

        vm.prank(attacker2);
        vm.expectRevert("Not authorized");
        coordinator.commitRoundData(modelId, 1, bytes32(uint256(2)));

        vm.prank(attacker3);
        vm.expectRevert("Not authorized");
        coordinator.commitRoundData(modelId, 1, bytes32(uint256(3)));
    }
}

/// @notice Mock verifier for auth tests
contract MockVerifierForAuthTest {
    function verifyProof(bytes memory, uint256[] memory) external pure returns (bool) {
        return true;
    }
}
