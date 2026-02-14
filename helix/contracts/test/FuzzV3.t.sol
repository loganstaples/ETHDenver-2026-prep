// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import "../src/token/HelixToken.sol";
import "../src/token/Staking.sol";
import "../src/token/Rewards.sol";
import "../src/governance/TrainingDAO.sol";
import "../src/verification/Halo2Verifier.sol";
import "../src/mocks/MockVerifier.sol";

/// @title FuzzV3Test
/// @notice Comprehensive fuzz testing for the HELIX V3 protocol stack
/// @dev 22+ property-based fuzz tests covering Staking, Rewards, TrainingDAO,
///      HelixToken, and Halo2Verifier batch pairing verification.
contract FuzzV3Test is Test {
    HelixToken public token;
    Staking public staking;
    Rewards public rewards;
    TrainingDAO public dao;
    Halo2Verifier public halo2Verifier;
    MockVerifier public mockVerifier;

    address public deployer;
    address public treasury;

    uint256 constant MIN_STAKE = 100e18;
    uint256 constant UNBONDING_PERIOD = 7 days;
    uint256 constant SLASH_RATE = 5000; // 50%

    function setUp() public {
        deployer = address(this);
        treasury = makeAddr("treasury");

        // Deploy token
        token = new HelixToken(treasury);
        token.completeInitialDistribution();

        // Deploy staking
        staking = new Staking(address(token), MIN_STAKE, UNBONDING_PERIOD, SLASH_RATE);
        staking.setTreasury(treasury);

        // Deploy rewards
        rewards = new Rewards(address(token));

        // Deploy TrainingDAO
        dao = new TrainingDAO(address(token));

        // Deploy verifiers
        mockVerifier = new MockVerifier();
        halo2Verifier = new Halo2Verifier();
    }

    // ============ Helper Functions ============

    function _mintAndApprove(address user, uint256 amount) internal {
        token.mint(user, amount);
        vm.prank(user);
        token.approve(address(staking), amount);
    }

    function _stakeAs(address user, uint256 amount) internal {
        _mintAndApprove(user, amount);
        vm.prank(user);
        staking.stake(amount);
    }

    function _mintAndDelegate(address user, uint256 amount) internal {
        token.mint(user, amount);
        vm.prank(user);
        token.delegate(user);
        vm.roll(block.number + 1); // Advance so getPastVotes works
    }

    // ============ Staking Fuzz Tests ============

    /// @notice Fuzz: Staking various token amounts above minimum
    function testFuzz_StakeVariousTokenAmounts(uint96 amount) public {
        amount = uint96(bound(amount, MIN_STAKE, 10_000_000e18));

        address staker = makeAddr("staker");
        _mintAndApprove(staker, amount);

        vm.prank(staker);
        staking.stake(amount);

        (uint256 stakedAmount,,,,) = staking.stakes(staker);
        assertEq(stakedAmount, amount, "Stake amount mismatch");
        assertEq(staking.totalStaked(), amount, "Total staked mismatch");
    }

    /// @notice Fuzz: Multiple stakers with different amounts
    function testFuzz_MultipleStakers(uint96[5] memory amounts) public {
        uint256 expectedTotal = 0;

        for (uint256 i = 0; i < 5; i++) {
            uint256 amount = bound(amounts[i], MIN_STAKE, 1_000_000e18);
            address staker = makeAddr(string(abi.encodePacked("staker", i)));
            _stakeAs(staker, amount);
            expectedTotal += amount;
        }

        assertEq(staking.totalStaked(), expectedTotal, "Total staked should be sum of all stakes");
        assertEq(staking.activeStakersCount(), 5, "Should have 5 active stakers");
    }

    /// @notice Fuzz: Unbonding completes after varying wait periods
    function testFuzz_UnbondingAfterVaryingPeriods(uint256 waitTime) public {
        waitTime = bound(waitTime, UNBONDING_PERIOD, 365 days);

        address staker = makeAddr("staker");
        _stakeAs(staker, MIN_STAKE);

        vm.prank(staker);
        staking.startUnbonding();

        vm.warp(block.timestamp + waitTime);

        uint256 balBefore = token.balanceOf(staker);
        vm.prank(staker);
        staking.unstake();
        uint256 balAfter = token.balanceOf(staker);

        assertEq(balAfter - balBefore, MIN_STAKE, "Should receive full stake back");
        assertEq(staking.totalStaked(), 0, "Total staked should be 0");
    }

    /// @notice Fuzz: Unstaking before unbonding period fails
    function testFuzz_UnbondingBeforePeriodFails(uint256 waitTime) public {
        waitTime = bound(waitTime, 0, UNBONDING_PERIOD - 1);

        address staker = makeAddr("staker");
        _stakeAs(staker, MIN_STAKE);

        vm.prank(staker);
        staking.startUnbonding();

        vm.warp(block.timestamp + waitTime);

        vm.prank(staker);
        vm.expectRevert("Unbonding period not complete");
        staking.unstake();
    }

    /// @notice Fuzz: Slashing never exceeds stake (invariant)
    function testFuzz_SlashingNeverExceedsStake(uint96 stakeAmount, uint8 severityRaw) public {
        stakeAmount = uint96(bound(stakeAmount, MIN_STAKE, 10_000_000e18));
        uint256 severityIdx = bound(severityRaw, 0, 4); // Warning to Critical
        Staking.SeverityLevel severity = Staking.SeverityLevel(severityIdx);

        address staker = makeAddr("staker");
        _stakeAs(staker, stakeAmount);

        staking.slashWithSeverity(
            staker,
            severity,
            Staking.ViolationType.InvalidProof,
            address(0),
            "Fuzz test slash"
        );

        (uint256 remaining,,,,) = staking.stakes(staker);
        assertLe(remaining, stakeAmount, "Remaining should not exceed original stake");
    }

    /// @notice Fuzz: Severity escalation produces increasing slash percentages
    function testFuzz_SeverityEscalation(uint96 stakeAmount) public {
        stakeAmount = uint96(bound(stakeAmount, MIN_STAKE * 10, 10_000_000e18));

        address staker = makeAddr("staker");
        _stakeAs(staker, stakeAmount);

        // Get slash percentages for each severity
        uint256 warningPct = staking.severitySlashPercentage(Staking.SeverityLevel.Warning);
        uint256 minorPct = staking.severitySlashPercentage(Staking.SeverityLevel.Minor);
        uint256 moderatePct = staking.severitySlashPercentage(Staking.SeverityLevel.Moderate);
        uint256 majorPct = staking.severitySlashPercentage(Staking.SeverityLevel.Major);

        // Verify severity levels are non-decreasing
        assertLe(warningPct, minorPct, "Warning should slash <= Minor");
        assertLe(minorPct, moderatePct, "Minor should slash <= Moderate");
        assertLe(moderatePct, majorPct, "Moderate should slash <= Major");
    }

    /// @notice Fuzz: Staking below minimum fails
    function testFuzz_StakeBelowMinimumFails(uint96 amount) public {
        amount = uint96(bound(amount, 1, MIN_STAKE - 1));

        address staker = makeAddr("staker");
        _mintAndApprove(staker, amount);

        vm.prank(staker);
        vm.expectRevert("Below minimum stake");
        staking.stake(amount);
    }

    // ============ Rewards Fuzz Tests ============

    /// @notice Fuzz: Fund reward pool with various amounts
    function testFuzz_FundRewardPool(uint96 amount, uint96 perRound) public {
        uint256 deployerBalance = token.balanceOf(deployer);
        amount = uint96(bound(amount, 1e18, deployerBalance));
        perRound = uint96(bound(perRound, 1e18, amount));

        token.approve(address(rewards), amount);
        rewards.fundRewardPool(amount, perRound, 365 days);

        (uint256 totalRewards,,uint256 rewardsPerRound,,) = rewards.rewardPool();
        assertEq(totalRewards, amount, "Total rewards mismatch");
        assertEq(rewardsPerRound, perRound, "Per-round rewards mismatch");
    }

    /// @notice Fuzz: Pool split percentages always sum to 10000
    function testFuzz_PoolSplitInvariant(uint16 compute, uint16 quality) public {
        // Read current values to verify invariant
        uint256 computeBps = rewards.computePoolBps();
        uint256 qualityBps = rewards.qualityPoolBps();
        uint256 timelinessBps = rewards.timelinessPoolBps();

        assertEq(computeBps + qualityBps + timelinessBps, 10000, "Pool split must sum to 10000");
    }

    // ============ TrainingDAO Fuzz Tests ============

    /// @notice Fuzz: Parameter proposal validates bounds correctly
    function testFuzz_ParameterBoundsValidation(
        uint256 learningRate,
        uint256 batchSize,
        uint256 maxErrorBound,
        uint256 minParticipants,
        uint256 roundDuration
    ) public {
        learningRate = bound(learningRate, 0, 2e18);
        batchSize = bound(batchSize, 0, 20000);
        maxErrorBound = bound(maxErrorBound, 0, 1e20);
        minParticipants = bound(minParticipants, 0, 100);
        roundDuration = bound(roundDuration, 0, 60 days);

        // Give proposer enough tokens for threshold
        address proposer = makeAddr("proposer");
        _mintAndDelegate(proposer, 2000e18);

        TrainingDAO.ParameterProposal memory params = TrainingDAO.ParameterProposal({
            learningRate: learningRate,
            batchSize: batchSize,
            maxErrorBound: maxErrorBound,
            minParticipants: minParticipants,
            roundDuration: roundDuration
        });

        bool shouldRevert = (
            learningRate == 0 ||
            learningRate > 1e18 ||
            batchSize == 0 ||
            batchSize > 10000 ||
            maxErrorBound == 0 ||
            minParticipants == 0 ||
            roundDuration < 5 minutes ||
            roundDuration > 30 days
        );

        vm.prank(proposer);
        if (shouldRevert) {
            vm.expectRevert();
            dao.createParameterProposal("Fuzz params", params);
        } else {
            uint256 proposalId = dao.createParameterProposal("Fuzz params", params);
            assertGt(proposalId, 0, "Should create valid proposal");
        }
    }

    /// @notice Fuzz: Voting power matches token balance at snapshot
    function testFuzz_VotingPowerMatchesSnapshot(uint96 amount) public {
        amount = uint96(bound(amount, 1001e18, 10_000_000e18));

        address voter = makeAddr("voter");
        _mintAndDelegate(voter, amount);

        // Create proposal (need proposer with threshold)
        address proposer = makeAddr("proposer");
        _mintAndDelegate(proposer, 2000e18);

        vm.prank(proposer);
        uint256 proposalId = dao.createProposal(
            TrainingDAO.ProposalType.Custom,
            "Test proposal",
            address(0),
            ""
        );

        // Voting power should match delegated balance at snapshot block
        uint256 votingPower = dao.getVotingPower(voter, proposalId);
        assertEq(votingPower, amount, "Voting power should match token balance");
    }

    /// @notice Fuzz: Proposals below threshold fail
    function testFuzz_ProposalBelowThresholdFails(uint96 amount) public {
        // TrainingDAO threshold is 1000e18
        amount = uint96(bound(amount, 1, 999e18));

        address proposer = makeAddr("proposer");
        _mintAndDelegate(proposer, amount);

        vm.prank(proposer);
        vm.expectRevert("Below proposal threshold");
        dao.createProposal(
            TrainingDAO.ProposalType.Custom,
            "Underfunded proposal",
            address(0),
            ""
        );
    }

    /// @notice Fuzz: Casting vote correctly accumulates for/against votes
    function testFuzz_VoteCasting(uint96 forAmount, uint96 againstAmount, bool support) public {
        forAmount = uint96(bound(forAmount, 1001e18, 5_000_000e18));
        againstAmount = uint96(bound(againstAmount, 1001e18, 5_000_000e18));

        // Setup voters
        address voter1 = makeAddr("voter1");
        address voter2 = makeAddr("voter2");
        _mintAndDelegate(voter1, forAmount);
        _mintAndDelegate(voter2, againstAmount);

        // Create proposal
        address proposer = makeAddr("proposer");
        _mintAndDelegate(proposer, 2000e18);

        vm.prank(proposer);
        uint256 proposalId = dao.createProposal(
            TrainingDAO.ProposalType.Custom,
            "Vote test",
            address(0),
            ""
        );

        // Advance past voting delay
        (,uint256 votingDelay,,,) = dao.govConfig();
        vm.warp(block.timestamp + votingDelay + 1);

        // Cast votes
        vm.prank(voter1);
        dao.castVote(proposalId, true);

        vm.prank(voter2);
        dao.castVote(proposalId, false);

        // Verify vote tallies
        (,,,,,uint256 forVotes, uint256 againstVotes,) = dao.getProposalInfo(proposalId);
        assertEq(forVotes, forAmount, "For votes should match voter1 balance");
        assertEq(againstVotes, againstAmount, "Against votes should match voter2 balance");
    }

    /// @notice Fuzz: Double voting is prevented
    function testFuzz_DoubleVotingPrevented(bool firstSupport, bool secondSupport) public {
        address voter = makeAddr("voter");
        _mintAndDelegate(voter, 5000e18);

        address proposer = makeAddr("proposer");
        _mintAndDelegate(proposer, 2000e18);

        vm.prank(proposer);
        uint256 proposalId = dao.createProposal(
            TrainingDAO.ProposalType.Custom,
            "Double vote test",
            address(0),
            ""
        );

        (,uint256 votingDelay,,,) = dao.govConfig();
        vm.warp(block.timestamp + votingDelay + 1);

        vm.prank(voter);
        dao.castVote(proposalId, firstSupport);

        vm.prank(voter);
        vm.expectRevert("Already voted");
        dao.castVote(proposalId, secondSupport);
    }

    /// @notice Fuzz: Proposal cancellation by proposer always works
    function testFuzz_ProposerCanAlwaysCancel(uint96 amount) public {
        amount = uint96(bound(amount, 1001e18, 10_000_000e18));

        address proposer = makeAddr("proposer");
        _mintAndDelegate(proposer, amount);

        vm.prank(proposer);
        uint256 proposalId = dao.createProposal(
            TrainingDAO.ProposalType.Custom,
            "Cancel test",
            address(0),
            ""
        );

        vm.prank(proposer);
        dao.cancelProposal(proposalId);

        TrainingDAO.ProposalState state = dao.getProposalState(proposalId);
        assertEq(uint(state), uint(TrainingDAO.ProposalState.Cancelled), "Should be cancelled");
    }

    // ============ HelixToken Fuzz Tests ============

    /// @notice Fuzz: Minting respects max supply cap
    function testFuzz_MintWithinCap(uint96 amount) public {
        uint256 currentSupply = token.totalSupply();
        uint256 maxSupply = token.MAX_SUPPLY();
        uint256 remaining = maxSupply - currentSupply;

        amount = uint96(bound(amount, 1, type(uint96).max));

        address recipient = makeAddr("recipient");

        if (amount > remaining) {
            vm.expectRevert("Exceeds max supply");
            token.mint(recipient, amount);
        } else {
            token.mint(recipient, amount);
            assertEq(token.balanceOf(recipient), amount, "Should receive minted amount");
            assertLe(token.totalSupply(), maxSupply, "Total supply should not exceed max");
        }
    }

    /// @notice Fuzz: Token transfers preserve total supply
    function testFuzz_TransferPreservesTotalSupply(uint96 amount) public {
        uint256 totalBefore = token.totalSupply();

        amount = uint96(bound(amount, 1, token.balanceOf(deployer)));
        address recipient = makeAddr("recipient");

        token.transfer(recipient, amount);

        assertEq(token.totalSupply(), totalBefore, "Total supply must not change on transfer");
        assertEq(token.balanceOf(recipient), amount, "Recipient should receive tokens");
    }

    /// @notice Fuzz: Delegation activates voting power correctly
    function testFuzz_DelegationActivatesVotingPower(uint96 amount) public {
        amount = uint96(bound(amount, 1e18, 10_000_000e18));

        address user = makeAddr("user");
        token.mint(user, amount);

        // Before delegation: no voting power
        assertEq(token.getVotes(user), 0, "No votes before delegation");

        // After delegation: full voting power
        vm.prank(user);
        token.delegate(user);

        assertEq(token.getVotes(user), amount, "Votes should match balance after delegation");
    }

    // ============ Halo2Verifier Batch Pairing Fuzz Tests ============

    /// @notice Fuzz: Batch pairing rejects empty batches
    function testFuzz_BatchPairingEmptyBatch() public {
        bytes[] memory proofs = new bytes[](0);
        uint256[][] memory pis = new uint256[][](0);
        uint256[2][] memory lhs = new uint256[2][](0);
        uint256[2][] memory rhs = new uint256[2][](0);

        vm.expectRevert("Empty batch");
        halo2Verifier.batchVerifyPairing(proofs, pis, lhs, rhs);
    }

    /// @notice Fuzz: Batch pairing rejects mismatched lengths
    function testFuzz_BatchPairingLengthMismatch(uint8 nProofs, uint8 nLHS) public {
        nProofs = uint8(bound(nProofs, 2, 10));
        nLHS = uint8(bound(nLHS, 1, 10));
        vm.assume(nProofs != nLHS);

        bytes[] memory proofs = new bytes[](nProofs);
        uint256[][] memory pis = new uint256[][](nProofs);
        uint256[2][] memory lhs = new uint256[2][](nLHS);
        uint256[2][] memory rhs = new uint256[2][](nProofs);

        for (uint256 i = 0; i < nProofs; i++) {
            proofs[i] = new bytes(1856);
            pis[i] = new uint256[](8);
        }

        vm.expectRevert("Length mismatch");
        halo2Verifier.batchVerifyPairing(proofs, pis, lhs, rhs);
    }

    /// @notice Fuzz: Batch pairing rejects off-curve LHS points
    function testFuzz_BatchPairingRejectsOffCurvePoints(uint256 x, uint256 y) public {
        uint256 BN254_P = 21888242871839275222246405745257275088696311157297823662689037894645226208583;
        x = bound(x, 1, BN254_P - 1);
        y = bound(y, 1, BN254_P - 1);

        // Check if (x,y) is actually on the curve y^2 = x^3 + 3
        uint256 lhsCheck = mulmod(y, y, BN254_P);
        uint256 rhsCheck = addmod(mulmod(x, mulmod(x, x, BN254_P), BN254_P), 3, BN254_P);
        vm.assume(lhsCheck != rhsCheck); // Ensure point is NOT on curve

        bytes[] memory proofs = new bytes[](2);
        uint256[][] memory pis = new uint256[][](2);
        uint256[2][] memory lhs = new uint256[2][](2);
        uint256[2][] memory rhs = new uint256[2][](2);

        // Setup minimal proofs with enough length
        for (uint256 i = 0; i < 2; i++) {
            proofs[i] = new bytes(1856);
            pis[i] = new uint256[](8);
            // RHS = (0,0) point at infinity, matching last 64 bytes of zeroed proof
            rhs[i] = [uint256(0), uint256(0)];
        }

        // Set off-curve LHS point
        lhs[0] = [x, y];
        lhs[1] = [uint256(0), uint256(0)]; // point at infinity (valid)

        vm.expectRevert("LHS point not on curve");
        halo2Verifier.batchVerifyPairing(proofs, pis, lhs, rhs);
    }

    /// @notice Fuzz: Batch pairing rejects RHS that doesn't match proof W'
    function testFuzz_BatchPairingRHSMismatch(uint256 fakeX, uint256 fakeY) public {
        uint256 BN254_P = 21888242871839275222246405745257275088696311157297823662689037894645226208583;
        fakeX = bound(fakeX, 1, BN254_P - 1);
        fakeY = bound(fakeY, 1, BN254_P - 1);

        bytes[] memory proofs = new bytes[](2);
        uint256[][] memory pis = new uint256[][](2);
        uint256[2][] memory lhs = new uint256[2][](2);
        uint256[2][] memory rhs = new uint256[2][](2);

        for (uint256 i = 0; i < 2; i++) {
            proofs[i] = new bytes(1856); // All zeros
            pis[i] = new uint256[](8);
            lhs[i] = [uint256(0), uint256(0)];
        }

        // RHS[0] doesn't match last 64 bytes of proof (which are all zeros)
        rhs[0] = [fakeX, fakeY];
        rhs[1] = [uint256(0), uint256(0)]; // matches zeroed proof

        vm.expectRevert("RHS does not match W' in proof");
        halo2Verifier.batchVerifyPairing(proofs, pis, lhs, rhs);
    }

    /// @notice Fuzz: Batch pairing rejects short proofs
    function testFuzz_BatchPairingShortProof(uint8 proofLen) public {
        proofLen = uint8(bound(proofLen, 1, 127));

        bytes[] memory proofs = new bytes[](2);
        uint256[][] memory pis = new uint256[][](2);
        uint256[2][] memory lhs = new uint256[2][](2);
        uint256[2][] memory rhs = new uint256[2][](2);

        proofs[0] = new bytes(proofLen); // Too short
        proofs[1] = new bytes(1856);
        pis[0] = new uint256[](8);
        pis[1] = new uint256[](8);

        vm.expectRevert("Proof too short");
        halo2Verifier.batchVerifyPairing(proofs, pis, lhs, rhs);
    }

    // ============ Security Invariant Tests ============

    /// @notice Fuzz: Total staked always matches sum of individual stakes
    function testFuzz_TotalStakedInvariant(uint96[3] memory amounts) public {
        uint256 expectedTotal = 0;
        // Use explicit string names to avoid abi.encodePacked null-byte issues
        string[3] memory names = [string("inv_staker_a"), "inv_staker_b", "inv_staker_c"];

        for (uint256 i = 0; i < 3; i++) {
            uint256 amount = bound(amounts[i], MIN_STAKE, 1_000_000e18);
            address staker = makeAddr(names[i]);
            _stakeAs(staker, amount);
            expectedTotal += amount;
        }

        assertEq(staking.totalStaked(), expectedTotal, "Total staked invariant violated");

        // Slash first staker and verify invariant holds
        address firstStaker = makeAddr(names[0]);
        uint256 staker0Amount;
        (staker0Amount,,,,) = staking.stakes(firstStaker);
        uint256 majorPct = staking.severitySlashPercentage(Staking.SeverityLevel.Major);
        uint256 slashAmount = (staker0Amount * majorPct) / 10000;

        staking.slashWithSeverity(
            firstStaker,
            Staking.SeverityLevel.Major,
            Staking.ViolationType.InvalidProof,
            address(0),
            "test"
        );

        assertEq(
            staking.totalStaked(),
            expectedTotal - slashAmount,
            "Total staked should decrease by slash amount"
        );
    }

    /// @notice Fuzz: Emergency pause blocks staking
    function testFuzz_EmergencyPauseBlocksStaking(uint96 amount) public {
        amount = uint96(bound(amount, MIN_STAKE, 1_000_000e18));

        staking.emergencyPause();

        address staker = makeAddr("staker");
        _mintAndApprove(staker, amount);

        vm.prank(staker);
        vm.expectRevert("Contract is paused");
        staking.stake(amount);
    }

    /// @notice Fuzz: DAO emergency pause blocks proposals
    function testFuzz_DAOPauseBlocksProposals() public {
        address proposer = makeAddr("proposer");
        _mintAndDelegate(proposer, 2000e18);

        dao.emergencyPause();

        vm.prank(proposer);
        vm.expectRevert("Contract is paused");
        dao.createProposal(
            TrainingDAO.ProposalType.Custom,
            "Blocked proposal",
            address(0),
            ""
        );
    }
}
