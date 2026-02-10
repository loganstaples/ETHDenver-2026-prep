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
    
    /// @notice Events
    event RewardPoolFunded(uint256 amount, uint256 rewardsPerRound, uint256 endTime);
    event RewardsAllocated(uint256 indexed modelId, uint256 indexed roundId, uint256 totalAmount, uint256 participantCount);
    event RewardsClaimed(address indexed claimer, uint256 amount);
    event ParticipantRegistered(uint256 indexed modelId, uint256 indexed roundId, address indexed participant);
    event BonusAwarded(address indexed recipient, uint256 amount, string reason);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }
    
    modifier onlyCoordinator() {
        require(msg.sender == coordinator, "Only coordinator");
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
    ) external nonReentrant {
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
        
        emit ParticipantRegistered(modelId, roundId, participant);
    }
    
    /// @notice Allocate rewards for a completed round
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
        uint256 participantCount = participants.length;
        
        require(participantCount > 0, "No participants");
        
        roundRewardsAllocated[modelId][roundId] = true;
        
        // Calculate individual rewards based on stake weight
        uint256 totalDistributed = 0;
        
        for (uint i = 0; i < participantCount; i++) {
            address participant = participants[i];
            uint256 reward = _calculateParticipantReward(participant, participantCount);
            
            pendingRewards[modelId][roundId][participant] = reward;
            claimInfo[participant].totalEarned += reward;
            totalDistributed += reward;
        }
        
        rewardPool.distributedRewards += totalDistributed;
        
        emit RewardsAllocated(modelId, roundId, totalDistributed, participantCount);
    }
    
    /// @notice Calculate reward for a participant based on stake weight
    function _calculateParticipantReward(
        address participant,
        uint256 totalParticipants
    ) internal view returns (uint256) {
        uint256 baseReward = rewardPool.rewardsPerRound / totalParticipants;
        
        // If staking contract is set, weight by stake
        if (stakingContract != address(0)) {
            // Get stake weight (would call staking contract)
            // For now, use equal distribution with potential bonus
            return baseReward;
        }
        
        return baseReward;
    }
    
    /// @notice Claim all pending rewards across all rounds
    /// @dev Uses the unified roundClaimed mapping to prevent double-claims
    function claimRewards() external nonReentrant {
        ClaimInfo storage info = claimInfo[msg.sender];

        uint256 claimable = info.totalEarned - info.totalClaimed;
        require(claimable > 0, "No rewards to claim");

        info.totalClaimed += claimable;
        info.lastClaimTime = block.timestamp;

        helixToken.safeTransfer(msg.sender, claimable);

        emit RewardsClaimed(msg.sender, claimable);
    }
    
    /// @notice Claim rewards for specific rounds
    /// @dev Uses unified roundClaimed mapping to prevent double-claim across both
    ///      claimRewards() and claimRoundRewards(). Each round can only be claimed once.
    function claimRoundRewards(
        uint256[] calldata modelIds,
        uint256[] calldata roundIds
    ) external nonReentrant {
        require(modelIds.length == roundIds.length, "Length mismatch");

        uint256 totalClaim = 0;

        for (uint i = 0; i < modelIds.length; i++) {
            uint256 modelId = modelIds[i];
            uint256 roundId = roundIds[i];

            // Skip rounds already claimed via either claim path
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

        // Cap transfer to what hasn't already been claimed via claimRewards()
        uint256 actualClaimable = info.totalEarned - info.totalClaimed;
        if (totalClaim > actualClaimable) {
            totalClaim = actualClaimable;
        }
        require(totalClaim > 0, "Already claimed");

        info.totalClaimed += totalClaim;
        info.lastClaimTime = block.timestamp;

        helixToken.safeTransfer(msg.sender, totalClaim);

        emit RewardsClaimed(msg.sender, totalClaim);
    }
    
    /// @notice Award a bonus to a participant (e.g., for early adoption)
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
    
    /// @notice Emergency withdraw (only excess tokens)
    function emergencyWithdraw(address to, uint256 amount) external onlyOwner {
        require(to != address(0), "Invalid address");
        uint256 balance = helixToken.balanceOf(address(this));
        uint256 locked = rewardPool.totalRewards - rewardPool.distributedRewards;
        require(amount <= balance - locked, "Cannot withdraw locked rewards");
        helixToken.safeTransfer(to, amount);
    }
}
