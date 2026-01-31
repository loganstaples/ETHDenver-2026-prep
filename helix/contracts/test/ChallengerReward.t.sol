// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../src/verification/SlashingEvidence.sol";
import "../src/core/HelixCoordinatorV2.sol";
import "../src/mocks/MockVerifier.sol";

/// @title TreasuryReceiver
/// @notice Simple contract that can receive ETH for testing treasury functionality
contract TreasuryReceiver {
    receive() external payable {}
}

/// @title ChallengerRewardTest
/// @notice Comprehensive tests for challenger reward distribution in HELIX protocol
/// @dev Tests reward calculation, distribution, edge cases, and economic incentives
contract ChallengerRewardTest is Test {
    // ============ Contracts ============
    SlashingEvidence public slashingEvidence;
    HelixCoordinatorV2 public coordinator;
    MockVerifier public mockVerifier;

    // ============ Actors ============
    address public owner;
    address public treasury;
    address public modelOwner;
    address public byzantineWorker;
    address public challenger1;
    address public challenger2;
    address public challenger3;
    address public arbitrator;

    // ============ Constants ============
    uint256 constant STANDARD_STAKE = 1 ether;
    uint256 constant LARGE_STAKE = 10 ether;
    uint256 constant ROUND_DURATION = 1 hours;

    // ============ Events ============
    event ChallengerRewarded(
        uint256 indexed evidenceId,
        address indexed challenger,
        uint128 rewardAmount,
        uint128 protocolFee
    );
    event ChallengerConfigUpdated(SlashingEvidence.ChallengerConfig config);

    function setUp() public {
        owner = address(this);
        // Use a contract that can receive ETH for treasury
        TreasuryReceiver treasuryContract = new TreasuryReceiver();
        treasury = address(treasuryContract);
        modelOwner = makeAddr("modelOwner");
        byzantineWorker = makeAddr("byzantineWorker");
        challenger1 = makeAddr("challenger1");
        challenger2 = makeAddr("challenger2");
        challenger3 = makeAddr("challenger3");
        arbitrator = makeAddr("arbitrator");

        // Deploy contracts
        mockVerifier = new MockVerifier();
        coordinator = new HelixCoordinatorV2(address(mockVerifier), treasury);
        slashingEvidence = new SlashingEvidence(address(coordinator));

        // Configure coordinator
        coordinator.setSlashingEvidenceContract(address(slashingEvidence));

        // Set treasury on slashingEvidence (defaults to msg.sender in constructor)
        slashingEvidence.setTreasury(treasury);

        // Add arbitrator
        slashingEvidence.addArbitrator(arbitrator);

        // Fund contracts for rewards
        vm.deal(address(slashingEvidence), 100 ether);

        // Fund actors
        vm.deal(modelOwner, 100 ether);
        vm.deal(byzantineWorker, 100 ether);
        vm.deal(challenger1, 10 ether);
        vm.deal(challenger2, 10 ether);
        vm.deal(challenger3, 10 ether);
    }

    // ============ Helper Functions ============

    function _submitEvidence(
        address prover,
        uint128 slashedAmount,
        address challenger
    ) internal returns (uint256 evidenceId) {
        slashingEvidence.setCoordinator(address(this));

        uint128 remaining = slashedAmount < uint128(STANDARD_STAKE)
            ? uint128(STANDARD_STAKE) - slashedAmount
            : 0;

        evidenceId = slashingEvidence.submitEvidence(
            prover,
            uint64(1),  // modelId
            uint32(1),  // roundId
            SlashingEvidence.ViolationType.InvalidProof,
            SlashingEvidence.ErrorCode.PROOF_VERIFICATION_FAILED,
            slashedAmount,
            remaining,
            keccak256(abi.encodePacked("proof", block.timestamp)),
            challenger,
            "Invalid proof detected"
        );
    }

    function _submitEvidenceWithTimestamp(
        address prover,
        uint128 slashedAmount,
        address challenger,
        uint256 timestamp
    ) internal returns (uint256 evidenceId) {
        vm.warp(timestamp);
        return _submitEvidence(prover, slashedAmount, challenger);
    }

    function _verifyEvidence(uint256 evidenceId) internal {
        // Fast forward past dispute period
        vm.warp(block.timestamp + 8 days);
        slashingEvidence.verifyEvidence(evidenceId);
    }

    // ============ Reward Calculation Tests ============

    /// @notice Test basic reward calculation
    function test_BasicRewardCalculation() public {
        // Default config: 10% reward, 5% protocol fee
        uint128 slashedAmount = 1 ether;

        uint256 evidenceId = _submitEvidence(byzantineWorker, slashedAmount, challenger1);

        SlashingEvidence.Evidence memory evidence = slashingEvidence.getEvidence(evidenceId);

        // 10% of 1 ether = 0.1 ether reward
        assertEq(evidence.challengerReward, 0.1 ether, "Reward should be 10% of slashed amount");

        // 5% of 1 ether = 0.05 ether protocol fee
        assertEq(evidence.protocolFee, 0.05 ether, "Protocol fee should be 5% of slashed amount");
    }

    /// @notice Test reward with large slashed amount
    function test_LargeSlashedAmountReward() public {
        uint128 slashedAmount = 5 ether;

        uint256 evidenceId = _submitEvidence(byzantineWorker, slashedAmount, challenger1);

        SlashingEvidence.Evidence memory evidence = slashingEvidence.getEvidence(evidenceId);

        // 10% of 5 ether = 0.5 ether reward
        assertEq(evidence.challengerReward, 0.5 ether);
        // 5% of 5 ether = 0.25 ether protocol fee
        assertEq(evidence.protocolFee, 0.25 ether);
    }

    /// @notice Test minimum reward enforcement
    function test_MinimumRewardEnforcement() public {
        // Small slashed amount that would yield below minimum reward
        uint128 slashedAmount = 0.005 ether;  // 10% = 0.0005 ether, below 0.001 minimum

        uint256 evidenceId = _submitEvidence(byzantineWorker, slashedAmount, challenger1);

        SlashingEvidence.Evidence memory evidence = slashingEvidence.getEvidence(evidenceId);

        // Should be minimum reward (0.001 ether)
        assertEq(evidence.challengerReward, 0.001 ether, "Should enforce minimum reward");
    }

    /// @notice Test maximum reward cap
    function test_MaximumRewardCap() public {
        // Very large slashed amount
        uint128 slashedAmount = 200 ether;  // 10% = 20 ether, above 10 ether max

        uint256 evidenceId = _submitEvidence(byzantineWorker, slashedAmount, challenger1);

        SlashingEvidence.Evidence memory evidence = slashingEvidence.getEvidence(evidenceId);

        // Should be capped at maximum (10 ether)
        assertEq(evidence.challengerReward, 10 ether, "Should enforce maximum reward cap");
    }

    /// @notice Test zero slashed amount (warning only)
    function test_ZeroSlashedAmountNoReward() public {
        uint128 slashedAmount = 0;  // Warning - no actual slash

        uint256 evidenceId = _submitEvidence(byzantineWorker, slashedAmount, challenger1);

        SlashingEvidence.Evidence memory evidence = slashingEvidence.getEvidence(evidenceId);

        // No reward for zero slash
        assertEq(evidence.challengerReward, 0);
        assertEq(evidence.protocolFee, 0);
    }

    // ============ Reward Distribution Tests ============

    /// @notice Test successful reward distribution
    function test_SuccessfulRewardDistribution() public {
        uint256 evidenceId = _submitEvidence(byzantineWorker, uint128(1 ether), challenger1);

        _verifyEvidence(evidenceId);

        uint256 challengerBalanceBefore = challenger1.balance;
        uint256 treasuryBalanceBefore = treasury.balance;

        SlashingEvidence.Evidence memory evidence = slashingEvidence.getEvidence(evidenceId);
        uint128 expectedReward = evidence.challengerReward;
        uint128 expectedFee = evidence.protocolFee;

        vm.expectEmit(true, true, false, true);
        emit ChallengerRewarded(evidenceId, challenger1, expectedReward, expectedFee);

        slashingEvidence.distributesChallengerReward(evidenceId);

        // Check balances
        assertEq(challenger1.balance, challengerBalanceBefore + expectedReward);
        assertEq(treasury.balance, treasuryBalanceBefore + expectedFee);
    }

    /// @notice Test cannot distribute before verification
    function test_CannotDistributeBeforeVerification() public {
        uint256 evidenceId = _submitEvidence(byzantineWorker, uint128(1 ether), challenger1);

        // Evidence is still pending
        vm.expectRevert("Evidence not verified");
        slashingEvidence.distributesChallengerReward(evidenceId);
    }

    /// @notice Test cannot distribute twice
    function test_CannotDistributeTwice() public {
        uint256 evidenceId = _submitEvidence(byzantineWorker, uint128(1 ether), challenger1);
        _verifyEvidence(evidenceId);

        // First distribution succeeds
        slashingEvidence.distributesChallengerReward(evidenceId);

        // Second distribution fails
        vm.expectRevert("Already rewarded");
        slashingEvidence.distributesChallengerReward(evidenceId);
    }

    /// @notice Test no reward for no challenger
    function test_NoRewardForNoChallenger() public {
        // Submit evidence without challenger (system detected)
        uint256 evidenceId = _submitEvidence(byzantineWorker, uint128(1 ether), address(0));
        _verifyEvidence(evidenceId);

        vm.expectRevert("No challenger");
        slashingEvidence.distributesChallengerReward(evidenceId);
    }

    /// @notice Test reward distribution after dispute resolution
    function test_RewardAfterDisputeResolution() public {
        uint256 evidenceId = _submitEvidence(byzantineWorker, uint128(1 ether), challenger1);

        // File and resolve dispute against prover
        uint256 disputeStake = slashingEvidence.disputeStakeRequired();
        vm.deal(byzantineWorker, 10 ether);

        vm.prank(byzantineWorker);
        uint256 disputeId = slashingEvidence.fileDispute{value: disputeStake}(
            evidenceId,
            keccak256("counter-evidence"),
            "I am innocent!"
        );

        // Resolve against prover (evidence upheld)
        slashingEvidence.resolveDispute(disputeId, false);

        // Now challenger can get reward
        uint256 challengerBalanceBefore = challenger1.balance;
        slashingEvidence.distributesChallengerReward(evidenceId);
        assertTrue(challenger1.balance > challengerBalanceBefore);
    }

    /// @notice Test no reward if dispute favors prover
    function test_NoRewardIfDisputeFavorsProver() public {
        uint256 evidenceId = _submitEvidence(byzantineWorker, uint128(1 ether), challenger1);

        // File and resolve dispute in favor of prover
        uint256 disputeStake = slashingEvidence.disputeStakeRequired();
        vm.deal(byzantineWorker, 10 ether);

        vm.prank(byzantineWorker);
        uint256 disputeId = slashingEvidence.fileDispute{value: disputeStake}(
            evidenceId,
            keccak256("counter-evidence"),
            "I am innocent!"
        );

        slashingEvidence.resolveDispute(disputeId, true);  // In favor of prover

        // Evidence is now rejected, challenger can't get reward
        vm.expectRevert("Evidence not verified");
        slashingEvidence.distributesChallengerReward(evidenceId);
    }

    // ============ Batch Distribution Tests ============

    /// @notice Test batch reward distribution
    function test_BatchRewardDistribution() public {
        uint256[] memory evidenceIds = new uint256[](3);
        address[] memory challengers = new address[](3);
        challengers[0] = challenger1;
        challengers[1] = challenger2;
        challengers[2] = challenger3;

        // Create multiple evidence records with different timestamps
        for (uint256 i = 0; i < 3; i++) {
            evidenceIds[i] = _submitEvidenceWithTimestamp(
                byzantineWorker,
                uint128(1 ether),
                challengers[i],
                block.timestamp + i + 1
            );
        }

        // Verify all
        vm.warp(block.timestamp + 10 days);
        slashingEvidence.batchVerifyEvidence(evidenceIds);

        // Record balances
        uint256[] memory balancesBefore = new uint256[](3);
        for (uint256 i = 0; i < 3; i++) {
            balancesBefore[i] = challengers[i].balance;
        }

        // Batch distribute
        slashingEvidence.batchDistributeRewards(evidenceIds);

        // All challengers should have received rewards
        for (uint256 i = 0; i < 3; i++) {
            assertTrue(challengers[i].balance > balancesBefore[i], "Challenger should have received reward");
        }
    }

    /// @notice Test batch distribution handles failures gracefully
    function test_BatchDistributionHandlesFailures() public {
        // Create one valid and one without challenger
        uint256 validId = _submitEvidenceWithTimestamp(byzantineWorker, uint128(1 ether), challenger1, block.timestamp);
        uint256 noChallengerId = _submitEvidenceWithTimestamp(byzantineWorker, uint128(1 ether), address(0), block.timestamp + 1);

        vm.warp(block.timestamp + 10 days);
        slashingEvidence.batchVerifyEvidence(new uint256[](0));  // Empty array to force verification
        slashingEvidence.verifyEvidence(validId);
        slashingEvidence.verifyEvidence(noChallengerId);

        uint256[] memory evidenceIds = new uint256[](2);
        evidenceIds[0] = validId;
        evidenceIds[1] = noChallengerId;

        uint256 balanceBefore = challenger1.balance;

        // Batch distribute - should not revert even if one fails
        slashingEvidence.batchDistributeRewards(evidenceIds);

        // Valid challenger should have received reward
        assertTrue(challenger1.balance > balanceBefore);
    }

    // ============ Config Update Tests ============

    /// @notice Test updating challenger config
    function test_UpdateChallengerConfig() public {
        SlashingEvidence.ChallengerConfig memory newConfig = SlashingEvidence.ChallengerConfig({
            rewardPercentage: 2000,      // 20%
            protocolFeePercentage: 1000, // 10%
            minimumReward: 0.01 ether,
            maximumReward: 5 ether,
            enabled: true
        });

        vm.expectEmit(false, false, false, true);
        emit ChallengerConfigUpdated(newConfig);

        slashingEvidence.setChallengerConfig(newConfig);

        // Create new evidence with new config
        uint256 evidenceId = _submitEvidence(byzantineWorker, uint128(1 ether), challenger1);

        SlashingEvidence.Evidence memory evidence = slashingEvidence.getEvidence(evidenceId);

        // 20% of 1 ether = 0.2 ether reward
        assertEq(evidence.challengerReward, 0.2 ether);
        // 10% of 1 ether = 0.1 ether protocol fee
        assertEq(evidence.protocolFee, 0.1 ether);
    }

    /// @notice Test disabling challenger rewards
    function test_DisableChallengerRewards() public {
        SlashingEvidence.ChallengerConfig memory disabledConfig = SlashingEvidence.ChallengerConfig({
            rewardPercentage: 1000,
            protocolFeePercentage: 500,
            minimumReward: 0.001 ether,
            maximumReward: 10 ether,
            enabled: false  // Disabled
        });

        slashingEvidence.setChallengerConfig(disabledConfig);

        uint256 evidenceId = _submitEvidence(byzantineWorker, uint128(1 ether), challenger1);

        SlashingEvidence.Evidence memory evidence = slashingEvidence.getEvidence(evidenceId);

        // No rewards when disabled
        assertEq(evidence.challengerReward, 0);
        assertEq(evidence.protocolFee, 0);

        // Distribution should fail
        _verifyEvidence(evidenceId);
        vm.expectRevert("Rewards disabled");
        slashingEvidence.distributesChallengerReward(evidenceId);
    }

    /// @notice Test config validation - max reward percentage
    function test_ConfigValidation_MaxRewardPercentage() public {
        SlashingEvidence.ChallengerConfig memory invalidConfig = SlashingEvidence.ChallengerConfig({
            rewardPercentage: 6000,  // 60% - too high
            protocolFeePercentage: 500,
            minimumReward: 0.001 ether,
            maximumReward: 10 ether,
            enabled: true
        });

        vm.expectRevert("Max 50% reward");
        slashingEvidence.setChallengerConfig(invalidConfig);
    }

    /// @notice Test config validation - max protocol fee
    function test_ConfigValidation_MaxProtocolFee() public {
        SlashingEvidence.ChallengerConfig memory invalidConfig = SlashingEvidence.ChallengerConfig({
            rewardPercentage: 1000,
            protocolFeePercentage: 3000,  // 30% - too high
            minimumReward: 0.001 ether,
            maximumReward: 10 ether,
            enabled: true
        });

        vm.expectRevert("Max 20% fee");
        slashingEvidence.setChallengerConfig(invalidConfig);
    }

    // ============ Economic Incentive Tests ============

    /// @notice Test challenger has economic incentive
    function test_ChallengerEconomicIncentive() public {
        uint256 evidenceId = _submitEvidence(byzantineWorker, uint128(2 ether), challenger1);
        _verifyEvidence(evidenceId);

        uint256 challengerBalanceBefore = challenger1.balance;
        slashingEvidence.distributesChallengerReward(evidenceId);
        uint256 reward = challenger1.balance - challengerBalanceBefore;

        // 10% of 2 ether = 0.2 ether reward
        assertEq(reward, 0.2 ether);

        // This should be profitable (no gas cost simulation in this test)
        assertTrue(reward > 0, "Challenger should profit");
    }

    /// @notice Test protocol fee accumulation
    function test_ProtocolFeeAccumulation() public {
        uint256 treasuryBalanceBefore = treasury.balance;

        // Multiple evidence submissions
        for (uint256 i = 0; i < 5; i++) {
            uint256 evidenceId = _submitEvidenceWithTimestamp(
                byzantineWorker,
                uint128(1 ether),
                challenger1,
                block.timestamp + i + 1
            );
            _verifyEvidence(evidenceId);
            slashingEvidence.distributesChallengerReward(evidenceId);
        }

        uint256 totalFees = treasury.balance - treasuryBalanceBefore;
        // 5 * 0.05 ether = 0.25 ether total fees
        assertEq(totalFees, 0.25 ether);
    }

    /// @notice Test multiple challengers across different evidence
    function test_MultipleChallengersMultipleEvidence() public {
        address[] memory challengers = new address[](3);
        challengers[0] = challenger1;
        challengers[1] = challenger2;
        challengers[2] = challenger3;

        uint256[] memory rewards = new uint256[](3);

        for (uint256 i = 0; i < 3; i++) {
            uint256 evidenceId = _submitEvidenceWithTimestamp(
                byzantineWorker,
                uint128((i + 1) * 1 ether),  // Different amounts
                challengers[i],
                block.timestamp + i + 1
            );

            uint256 balanceBefore = challengers[i].balance;
            _verifyEvidence(evidenceId);
            slashingEvidence.distributesChallengerReward(evidenceId);
            rewards[i] = challengers[i].balance - balanceBefore;
        }

        // Verify rewards are proportional to slashed amounts
        assertEq(rewards[0], 0.1 ether);   // 10% of 1 ether
        assertEq(rewards[1], 0.2 ether);   // 10% of 2 ether
        assertEq(rewards[2], 0.3 ether);   // 10% of 3 ether
    }

    // ============ Edge Case Tests ============

    /// @notice Test reward when contract has insufficient funds
    function test_RewardWithInsufficientFunds() public {
        // Deploy new evidence contract with minimal funds
        SlashingEvidence poorEvidence = new SlashingEvidence(address(coordinator));
        poorEvidence.setCoordinator(address(this));
        vm.deal(address(poorEvidence), 0.01 ether);  // Minimal funds

        uint256 evidenceId = poorEvidence.submitEvidence(
            byzantineWorker,
            uint64(1),
            uint32(1),
            SlashingEvidence.ViolationType.InvalidProof,
            SlashingEvidence.ErrorCode.PROOF_VERIFICATION_FAILED,
            uint128(1 ether),  // Would need 0.1 ether reward
            uint128(0),
            keccak256("proof"),
            challenger1,
            "Test"
        );

        vm.warp(block.timestamp + 8 days);
        poorEvidence.verifyEvidence(evidenceId);

        // Distribution should fail due to insufficient funds
        vm.expectRevert("Reward transfer failed");
        poorEvidence.distributesChallengerReward(evidenceId);
    }

    /// @notice Test reward to contract challenger
    function test_RewardToContractChallenger() public {
        // Deploy a simple receiver contract
        ReceiverContract receiver = new ReceiverContract();

        uint256 evidenceId = _submitEvidence(byzantineWorker, uint128(1 ether), address(receiver));
        _verifyEvidence(evidenceId);

        uint256 receiverBalanceBefore = address(receiver).balance;
        slashingEvidence.distributesChallengerReward(evidenceId);

        assertEq(address(receiver).balance, receiverBalanceBefore + 0.1 ether);
    }

    /// @notice Test reward distribution under reentrancy attempt
    function test_RewardReentrancyProtection() public {
        // Deploy malicious receiver
        MaliciousReceiver malicious = new MaliciousReceiver(address(slashingEvidence));

        uint256 evidenceId = _submitEvidence(byzantineWorker, uint128(1 ether), address(malicious));
        _verifyEvidence(evidenceId);

        malicious.setEvidenceId(evidenceId);

        // Should succeed first time but malicious callback should fail
        slashingEvidence.distributesChallengerReward(evidenceId);

        // Malicious receiver only received once
        assertEq(address(malicious).balance, 0.1 ether);
    }

    /// @notice Test extremely small reward
    function test_ExtremelySmallReward() public {
        // Very small slashed amount
        uint128 slashedAmount = 100 wei;

        uint256 evidenceId = _submitEvidence(byzantineWorker, slashedAmount, challenger1);

        SlashingEvidence.Evidence memory evidence = slashingEvidence.getEvidence(evidenceId);

        // Should be minimum reward due to small calculation result
        assertEq(evidence.challengerReward, 0.001 ether);
    }

    /// @notice Test treasury at zero address
    function test_TreasuryAtZeroAddress() public {
        slashingEvidence.setTreasury(address(0));

        uint256 evidenceId = _submitEvidence(byzantineWorker, uint128(1 ether), challenger1);
        _verifyEvidence(evidenceId);

        uint256 challengerBalanceBefore = challenger1.balance;

        // Should still distribute challenger reward
        slashingEvidence.distributesChallengerReward(evidenceId);

        // Challenger gets reward
        assertEq(challenger1.balance, challengerBalanceBefore + 0.1 ether);

        // Protocol fee call should fail silently or be skipped
    }

    // ============ Appeal Tests ============

    /// @notice Test reward after successful appeal (should not get reward)
    function test_NoRewardAfterSuccessfulAppeal() public {
        uint256 evidenceId = _submitEvidence(byzantineWorker, uint128(1 ether), challenger1);

        // Dispute and resolve against prover
        uint256 disputeStake = slashingEvidence.disputeStakeRequired();
        vm.deal(byzantineWorker, 10 ether);

        vm.prank(byzantineWorker);
        uint256 disputeId = slashingEvidence.fileDispute{value: disputeStake}(
            evidenceId,
            keccak256("counter"),
            "Innocent"
        );

        slashingEvidence.resolveDispute(disputeId, false);  // Against prover

        // Prover appeals
        uint256 appealStake = slashingEvidence.appealStakeRequired();
        vm.prank(byzantineWorker);
        uint256 appealId = slashingEvidence.fileAppeal{value: appealStake}(
            evidenceId,
            disputeId,
            keccak256("appeal"),
            "New evidence"
        );

        // Appeal succeeds - prover was innocent
        vm.prank(arbitrator);
        slashingEvidence.resolveAppeal(appealId, true);

        // Evidence is now rejected, no reward for challenger
        vm.expectRevert("Evidence not verified");
        slashingEvidence.distributesChallengerReward(evidenceId);
    }

    /// @notice Test reward after failed appeal
    function test_RewardAfterFailedAppeal() public {
        uint256 evidenceId = _submitEvidence(byzantineWorker, uint128(1 ether), challenger1);

        // Dispute and resolve against prover
        uint256 disputeStake = slashingEvidence.disputeStakeRequired();
        vm.deal(byzantineWorker, 10 ether);

        vm.prank(byzantineWorker);
        uint256 disputeId = slashingEvidence.fileDispute{value: disputeStake}(
            evidenceId,
            keccak256("counter"),
            "Innocent"
        );

        slashingEvidence.resolveDispute(disputeId, false);

        // Prover appeals
        uint256 appealStake = slashingEvidence.appealStakeRequired();
        vm.prank(byzantineWorker);
        uint256 appealId = slashingEvidence.fileAppeal{value: appealStake}(
            evidenceId,
            disputeId,
            keccak256("appeal"),
            "New evidence"
        );

        // Appeal fails - prover was guilty
        vm.prank(arbitrator);
        slashingEvidence.resolveAppeal(appealId, false);

        // Challenger should get reward
        uint256 balanceBefore = challenger1.balance;
        slashingEvidence.distributesChallengerReward(evidenceId);
        assertTrue(challenger1.balance > balanceBefore);
    }
}

/// @title ReceiverContract
/// @notice Simple contract that can receive ETH
contract ReceiverContract {
    receive() external payable {}
}

/// @title MaliciousReceiver
/// @notice Contract that attempts reentrancy on reward distribution
contract MaliciousReceiver {
    SlashingEvidence public evidence;
    uint256 public evidenceId;
    uint256 public callCount;

    constructor(address _evidence) {
        evidence = SlashingEvidence(payable(_evidence));
    }

    function setEvidenceId(uint256 _id) external {
        evidenceId = _id;
    }

    receive() external payable {
        callCount++;
        if (callCount < 3) {
            // Try to call again (should fail due to already rewarded check)
            try evidence.distributesChallengerReward(evidenceId) {} catch {}
        }
    }
}
