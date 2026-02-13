// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/// @title Rewards
/// @notice Distributes HELIX token rewards to training participants
/// @dev Rewards are distributed based on proof submission and validation
contract Rewards is ReentrancyGuard {
    using SafeERC20 for IERC20;
    
    /// @notice The HELIX token
    IERC20 public immutable helixToken;
    
    /// @notice Staking contract for weight calculations
    address public stakingContract;
    
    /// @notice Coordinator contract for proof validation
    address public coordinator;
    
    /// @notice Owner for parameter updates
    address public owner;

    /// @notice Whether the contract is paused
    bool public paused;
    
    /// @notice Reward pool information
    struct RewardPool {
        uint256 totalRewards;      // Total rewards available
        uint256 distributedRewards; // Already distributed
        uint256 rewardsPerRound;   // Rewards to distribute per round
        uint256 startTime;         // When rewards started
        uint256 endTime;           // When rewards end
    }
    
    /// @notice Active reward pool
    RewardPool public rewardPool;
    
    /// @notice Reward claim information per address
    struct ClaimInfo {
        uint256 totalEarned;       // Total rewards earned
        uint256 totalClaimed;      // Total rewards claimed
        uint256 lastClaimTime;     // Last claim timestamp
        uint256 roundsParticipated; // Number of rounds participated
    }
    
    /// @notice Mapping of address to claim info
    mapping(address => ClaimInfo) public claimInfo;
    
    /// @notice Pending rewards per model per round per participant
    mapping(uint256 => mapping(uint256 => mapping(address => uint256))) public pendingRewards;
    
    /// @notice Whether rewards have been allocated for a round
    mapping(uint256 => mapping(uint256 => bool)) public roundRewardsAllocated;

    /// @notice Whether a participant has claimed rewards for a specific round
    /// @dev Unified tracker: prevents double-claim across claimRewards() and claimRoundRewards()
    mapping(uint256 => mapping(uint256 => mapping(address => bool))) public roundClaimed;

    /// @notice Participants in each round
    mapping(uint256 => mapping(uint256 => address[])) public roundParticipants;

    /// @notice Participant loss per round
    mapping(uint256 => mapping(uint256 => mapping(address => uint256))) public participantLoss;

    /// @notice Participant submission time
    mapping(uint256 => mapping(uint256 => mapping(address => uint40))) public participantSubmitTime;

    /// @notice Round timing info
    struct RoundTiming {
        uint40 startTime;
        uint40 deadline;
    }
    mapping(uint256 => mapping(uint256 => RoundTiming)) public roundTiming;

    /// @notice Per-participant proof count per round (for weighted reward distribution)
    mapping(uint256 => mapping(uint256 => mapping(address => uint256))) public participantProofCount;

    /// @notice Total proof count per round
    mapping(uint256 => mapping(uint256 => uint256)) public roundTotalProofCount;

    /// @notice Pool split configuration (basis points, must sum to 10000)
    uint16 public computePoolBps = 7000;  // 70%
    uint16 public qualityPoolBps = 2000;  // 20%
    uint16 public timelinessPoolBps = 1000; // 10%

    /// @notice Events
    event RewardPoolFunded(uint256 amount, uint256 rewardsPerRound, uint256 endTime);
    event RewardsAllocated(uint256 indexed modelId, uint256 indexed roundId, uint256 totalAmount, uint256 participantCount);
    event RewardsClaimed(address indexed claimer, uint256 amount);
    event ParticipantRegistered(uint256 indexed modelId, uint256 indexed roundId, address indexed participant);
    event BonusAwarded(address indexed recipient, uint256 amount, string reason);
    event EmergencyPauseChanged(bool isPaused, address indexed changedBy);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }
    
    modifier onlyCoordinator() {
        require(msg.sender == coordinator, "Only coordinator");
        _;
    }

    modifier whenNotPaused() {
        require(!paused, "Contract is paused");
        _;
    }
    
    constructor(address _helixToken) {
        require(_helixToken != address(0), "Invalid token");
        helixToken = IERC20(_helixToken);
        owner = msg.sender;
    }
    
    /// @notice Set the coordinator contract
    function setCoordinator(address _coordinator) external onlyOwner {
        require(_coordinator != address(0), "Invalid address");
        coordinator = _coordinator;
    }
    
    /// @notice Set the staking contract
    function setStakingContract(address _staking) external onlyOwner {
        require(_staking != address(0), "Invalid address");
        stakingContract = _staking;
    }
    
    /// @notice Fund the reward pool
    /// @param amount Total amount to add to reward pool
    /// @param rewardsPerRound Amount to distribute per training round
    /// @param duration How long the rewards should last (in seconds)
    function fundRewardPool(
        uint256 amount,
        uint256 rewardsPerRound,
        uint256 duration
    ) external nonReentrant whenNotPaused {
        require(amount > 0, "Amount must be positive");
        require(rewardsPerRound > 0, "Rewards per round must be positive");
        require(duration > 0, "Duration must be positive");
        
        // Transfer tokens to this contract
        helixToken.safeTransferFrom(msg.sender, address(this), amount);
        
        // Update reward pool
        rewardPool.totalRewards += amount;
        rewardPool.rewardsPerRound = rewardsPerRound;
        rewardPool.startTime = block.timestamp;
        rewardPool.endTime = block.timestamp + duration;
        
        emit RewardPoolFunded(amount, rewardsPerRound, rewardPool.endTime);
    }
    
    /// @notice Register a participant for a training round
    /// @dev Called by coordinator when a participant submits valid work
    function registerParticipant(
        uint256 modelId,
        uint256 roundId,
        address participant
    ) external onlyCoordinator {
        roundParticipants[modelId][roundId].push(participant);
        claimInfo[participant].roundsParticipated++;

        // Track proof count for weighted reward distribution
        participantProofCount[modelId][roundId][participant]++;
        roundTotalProofCount[modelId][roundId]++;

        emit ParticipantRegistered(modelId, roundId, participant);
    }

    /// @notice Register participant with loss and timing data for compute-first rewards
    function registerParticipantWithData(
        uint256 modelId,
        uint256 roundId,
        address participant,
        uint256 loss,
        uint40 submittedAt,
        uint40 roundStartTime,
        uint40 roundDeadline
    ) external onlyCoordinator {
        roundParticipants[modelId][roundId].push(participant);
        claimInfo[participant].roundsParticipated++;

        participantLoss[modelId][roundId][participant] = loss;
        participantSubmitTime[modelId][roundId][participant] = submittedAt;

        // Track proof count for weighted reward distribution
        participantProofCount[modelId][roundId][participant]++;
        roundTotalProofCount[modelId][roundId]++;

        // Only set timing once per round
        if (roundTiming[modelId][roundId].startTime == 0) {
            roundTiming[modelId][roundId] = RoundTiming(roundStartTime, roundDeadline);
        }

        emit ParticipantRegistered(modelId, roundId, participant);
    }

    /// @notice Allocate rewards for a completed round using 70/20/10 compute-first model
    /// @param modelId The model ID
    /// @param roundId The round ID
    function allocateRoundRewards(
        uint256 modelId,
        uint256 roundId
    ) external onlyCoordinator nonReentrant {
        require(!roundRewardsAllocated[modelId][roundId], "Already allocated");
        require(
            rewardPool.totalRewards - rewardPool.distributedRewards >= rewardPool.rewardsPerRound,
            "Insufficient reward pool"
        );

        address[] storage participants = roundParticipants[modelId][roundId];
        uint256 count = participants.length;
        require(count > 0, "No participants");

        roundRewardsAllocated[modelId][roundId] = true;

        uint256 totalRoundReward = rewardPool.rewardsPerRound;
        uint256 computePool = (totalRoundReward * computePoolBps) / 10000;
        uint256 qualityPool = (totalRoundReward * qualityPoolBps) / 10000;
        uint256 timelinessPool = totalRoundReward - computePool - qualityPool;

        // Compute pool: proportional to proof count (workers who submit more proofs get more)
        uint256 totalProofs = roundTotalProofCount[modelId][roundId];
        if (totalProofs == 0) totalProofs = count; // fallback to equal distribution

        // Quality pool: bonus for below-median loss
        uint256 medianLoss = _computeMedianLoss(modelId, roundId, participants);

        // First pass: compute quality improvements
        uint256[] memory improvements = new uint256[](count);
        uint256 qualityDenominator = 0;
        for (uint256 i = 0; i < count; i++) {
            uint256 loss = participantLoss[modelId][roundId][participants[i]];
            if (medianLoss > 0 && loss < medianLoss) {
                improvements[i] = medianLoss - loss;
                qualityDenominator += improvements[i];
            }
        }

        // Second pass: distribute all pools
        RoundTiming storage timing = roundTiming[modelId][roundId];
        uint256 totalDistributed = 0;
        for (uint256 i = 0; i < count; i++) {
            address participant = participants[i];
            // Compute reward proportional to proof count
            uint256 proofCount = participantProofCount[modelId][roundId][participant];
            if (proofCount == 0) proofCount = 1; // safety fallback
            uint256 reward = (computePool * proofCount) / totalProofs;

            // Quality bonus
            if (qualityDenominator > 0 && improvements[i] > 0) {
                reward += (qualityPool * improvements[i]) / qualityDenominator;
            }

            // Timeliness bonus
            reward += _computeTimelinessBonus(modelId, roundId, participant, timelinessPool / count, timing);

            pendingRewards[modelId][roundId][participant] = reward;
            claimInfo[participant].totalEarned += reward;
            totalDistributed += reward;
        }

        rewardPool.distributedRewards += totalDistributed;

        emit RewardsAllocated(modelId, roundId, totalDistributed, count);
    }

    /// @notice Compute the median loss for a set of participants
    function _computeMedianLoss(
        uint256 modelId, uint256 roundId, address[] storage participants
    ) internal view returns (uint256) {
        uint256 count = participants.length;
        if (count == 0) return 0;
        if (count == 1) return participantLoss[modelId][roundId][participants[0]];

        // Sort losses (insertion sort, O(n^2) fine for small n < 100)
        uint256[] memory losses = new uint256[](count);
        for (uint256 i = 0; i < count; i++) {
            losses[i] = participantLoss[modelId][roundId][participants[i]];
        }
        for (uint256 i = 1; i < count; i++) {
            uint256 key = losses[i];
            uint256 j = i;
            while (j > 0 && losses[j - 1] > key) {
                losses[j] = losses[j - 1];
                j--;
            }
            losses[j] = key;
        }
        return losses[count / 2];
    }

    /// @notice Compute timeliness bonus with linear decay after 75% of round duration
    function _computeTimelinessBonus(
        uint256 modelId, uint256 roundId, address participant,
        uint256 maxBonus, RoundTiming storage timing
    ) internal view returns (uint256) {
        uint40 submitTime = participantSubmitTime[modelId][roundId][participant];
        if (submitTime == 0 || timing.startTime == 0) return maxBonus;

        uint256 duration = uint256(timing.deadline) - uint256(timing.startTime);
        if (duration == 0) return maxBonus;

        uint256 elapsed = uint256(submitTime) - uint256(timing.startTime);
        uint256 progress = (elapsed * 10000) / duration;

        if (progress <= 7500) return maxBonus;
        if (progress >= 10000) return 0;

        uint256 decay = progress - 7500;
        return maxBonus * (2500 - decay) / 2500;
    }

    /// @notice Set the pool split configuration (owner only)
    /// @param _compute Compute pool basis points
    /// @param _quality Quality pool basis points
    /// @param _timeliness Timeliness pool basis points
    function setPoolSplit(uint16 _compute, uint16 _quality, uint16 _timeliness) external onlyOwner {
        require(_compute + _quality + _timeliness == 10000, "Must sum to 100%");
        computePoolBps = _compute;
        qualityPoolBps = _quality;
        timelinessPoolBps = _timeliness;
    }
    
    /// @notice Claim rewards for specific rounds (single claim path to prevent double-claims)
    /// @dev Each round can only be claimed once via the roundClaimed mapping.
    function claimRoundRewards(
        uint256[] calldata modelIds,
        uint256[] calldata roundIds
    ) external nonReentrant whenNotPaused {
        require(modelIds.length == roundIds.length, "Length mismatch");

        uint256 totalClaim = 0;

        for (uint i = 0; i < modelIds.length; i++) {
            uint256 modelId = modelIds[i];
            uint256 roundId = roundIds[i];

            // Skip rounds already claimed
            if (roundClaimed[modelId][roundId][msg.sender]) {
                continue;
            }

            uint256 reward = pendingRewards[modelId][roundId][msg.sender];
            if (reward > 0) {
                roundClaimed[modelId][roundId][msg.sender] = true;
                pendingRewards[modelId][roundId][msg.sender] = 0;
                totalClaim += reward;
            }
        }

        require(totalClaim > 0, "No rewards to claim");

        ClaimInfo storage info = claimInfo[msg.sender];

        info.totalClaimed += totalClaim;
        info.lastClaimTime = block.timestamp;

        helixToken.safeTransfer(msg.sender, totalClaim);

        emit RewardsClaimed(msg.sender, totalClaim);
    }
    
    /// @notice Award a bonus to a participant (e.g., for early adoption)
    /// @dev Transfers tokens directly to recipient to avoid unreachable pending rewards.
    function awardBonus(
        address recipient,
        uint256 amount,
        string calldata reason
    ) external onlyOwner nonReentrant {
        require(recipient != address(0), "Invalid recipient");
        require(amount > 0, "Amount must be positive");
        require(
            rewardPool.totalRewards - rewardPool.distributedRewards >= amount,
            "Insufficient reward pool"
        );

        rewardPool.distributedRewards += amount;
        claimInfo[recipient].totalEarned += amount;
        claimInfo[recipient].totalClaimed += amount;

        helixToken.safeTransfer(recipient, amount);

        emit BonusAwarded(recipient, amount, reason);
    }
    
    /// @notice Get claimable rewards for an address
    function getClaimableRewards(address account) external view returns (uint256) {
        ClaimInfo storage info = claimInfo[account];
        return info.totalEarned - info.totalClaimed;
    }
    
    /// @notice Get reward pool info
    function getRewardPoolInfo() external view returns (
        uint256 total,
        uint256 distributed,
        uint256 remaining,
        uint256 perRound,
        bool isActive
    ) {
        total = rewardPool.totalRewards;
        distributed = rewardPool.distributedRewards;
        remaining = total - distributed;
        perRound = rewardPool.rewardsPerRound;
        isActive = block.timestamp >= rewardPool.startTime && block.timestamp <= rewardPool.endTime;
    }
    
    /// @notice Get participant stats
    function getParticipantStats(address account) external view returns (
        uint256 totalEarned,
        uint256 totalClaimed,
        uint256 roundsParticipated,
        uint256 pendingAmount
    ) {
        ClaimInfo storage info = claimInfo[account];
        totalEarned = info.totalEarned;
        totalClaimed = info.totalClaimed;
        roundsParticipated = info.roundsParticipated;
        pendingAmount = totalEarned - totalClaimed;
    }
    
    /// @notice Transfer ownership
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid address");
        owner = newOwner;
    }
    
    /// @notice Emergency pause - stops funding and claims
    function emergencyPause() external onlyOwner {
        require(!paused, "Already paused");
        paused = true;
        emit EmergencyPauseChanged(true, msg.sender);
    }

    /// @notice Unpause the contract
    function unpause() external onlyOwner {
        require(paused, "Not paused");
        paused = false;
        emit EmergencyPauseChanged(false, msg.sender);
    }

    /// @notice Emergency withdraw (only excess tokens)
    function emergencyWithdraw(address to, uint256 amount) external onlyOwner {
        require(to != address(0), "Invalid address");
        uint256 balance = helixToken.balanceOf(address(this));
        uint256 locked = rewardPool.totalRewards - rewardPool.distributedRewards;
        require(amount <= balance - locked, "Cannot withdraw locked rewards");
        helixToken.safeTransfer(to, amount);
    }
}
