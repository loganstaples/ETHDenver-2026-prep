// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/// @title Staking
/// @notice Manages stake deposits for HELIX training nodes
/// @dev Stakers must deposit HELIX tokens to participate in training rounds
contract Staking is ReentrancyGuard {
    using SafeERC20 for IERC20;
    
    /// @notice The HELIX token
    IERC20 public immutable helixToken;
    
    /// @notice Minimum stake required to participate
    uint256 public minStake;
    
    /// @notice Unbonding period in seconds
    uint256 public unbondingPeriod;
    
    /// @notice Slashing rate (basis points, 100 = 1%)
    uint256 public slashingRate;
    
    /// @notice Stake information for each staker
    struct StakeInfo {
        uint256 amount;
        uint256 stakedAt;
        uint256 unbondingStartTime;
        bool isUnbonding;
        bool isActive; // Can participate in training
    }
    
    /// @notice Mapping of staker address to their stake info
    mapping(address => StakeInfo) public stakes;
    
    /// @notice Total staked amount
    uint256 public totalStaked;
    
    /// @notice Active stakers count
    uint256 public activeStakersCount;
    
    /// @notice Operator who can perform slashing (typically the coordinator)
    address public operator;
    
    /// @notice Owner for parameter updates
    address public owner;
    
    /// @notice Events
    event Staked(address indexed staker, uint256 amount, uint256 totalStake);
    event UnbondingStarted(address indexed staker, uint256 unbondingEndTime);
    event Unstaked(address indexed staker, uint256 amount);
    event Slashed(address indexed staker, uint256 amount, string reason);
    event StakeIncreased(address indexed staker, uint256 addedAmount, uint256 newTotal);
    event ParametersUpdated(uint256 minStake, uint256 unbondingPeriod, uint256 slashingRate);
    event OperatorUpdated(address indexed oldOperator, address indexed newOperator);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }
    
    modifier onlyOperator() {
        require(msg.sender == operator || msg.sender == owner, "Only operator");
        _;
    }
    
    constructor(
        address _helixToken,
        uint256 _minStake,
        uint256 _unbondingPeriod,
        uint256 _slashingRate
    ) {
        require(_helixToken != address(0), "Invalid token address");
        require(_slashingRate <= 10000, "Slashing rate too high");
        
        helixToken = IERC20(_helixToken);
        minStake = _minStake;
        unbondingPeriod = _unbondingPeriod;
        slashingRate = _slashingRate;
        owner = msg.sender;
        operator = msg.sender;
    }
    
    /// @notice Stake tokens to participate in training
    /// @param amount Amount of HELIX tokens to stake
    function stake(uint256 amount) external nonReentrant {
        require(amount > 0, "Amount must be positive");
        require(!stakes[msg.sender].isUnbonding, "Cannot stake while unbonding");
        
        uint256 newStake = stakes[msg.sender].amount + amount;
        require(newStake >= minStake, "Below minimum stake");
        
        // Transfer tokens to this contract
        helixToken.safeTransferFrom(msg.sender, address(this), amount);
        
        StakeInfo storage stakeInfo = stakes[msg.sender];
        
        bool wasActive = stakeInfo.isActive;
        
        stakeInfo.amount = newStake;
        stakeInfo.stakedAt = block.timestamp;
        stakeInfo.isActive = true;
        stakeInfo.isUnbonding = false;
        
        totalStaked += amount;
        
        if (!wasActive) {
            activeStakersCount++;
            emit Staked(msg.sender, amount, newStake);
        } else {
            emit StakeIncreased(msg.sender, amount, newStake);
        }
    }
    
    /// @notice Start the unbonding process to withdraw stake
    function startUnbonding() external nonReentrant {
        StakeInfo storage stakeInfo = stakes[msg.sender];
        
        require(stakeInfo.amount > 0, "No stake to unbond");
        require(!stakeInfo.isUnbonding, "Already unbonding");
        
        stakeInfo.isUnbonding = true;
        stakeInfo.unbondingStartTime = block.timestamp;
        stakeInfo.isActive = false;
        
        activeStakersCount--;
        
        emit UnbondingStarted(msg.sender, block.timestamp + unbondingPeriod);
    }
    
    /// @notice Complete unstaking after unbonding period
    function unstake() external nonReentrant {
        StakeInfo storage stakeInfo = stakes[msg.sender];
        
        require(stakeInfo.isUnbonding, "Not unbonding");
        require(
            block.timestamp >= stakeInfo.unbondingStartTime + unbondingPeriod,
            "Unbonding period not complete"
        );
        
        uint256 amount = stakeInfo.amount;
        require(amount > 0, "No stake to withdraw");
        
        totalStaked -= amount;
        stakeInfo.amount = 0;
        stakeInfo.isUnbonding = false;
        
        helixToken.safeTransfer(msg.sender, amount);
        
        emit Unstaked(msg.sender, amount);
    }
    
    /// @notice Slash a staker for protocol violation
    /// @param staker Address of the staker to slash
    /// @param reason Reason for slashing
    function slash(address staker, string calldata reason) external onlyOperator nonReentrant {
        StakeInfo storage stakeInfo = stakes[staker];
        
        require(stakeInfo.amount > 0, "No stake to slash");
        
        uint256 slashAmount = (stakeInfo.amount * slashingRate) / 10000;
        
        stakeInfo.amount -= slashAmount;
        totalStaked -= slashAmount;
        
        // If stake drops below minimum, deactivate
        if (stakeInfo.amount < minStake && stakeInfo.isActive) {
            stakeInfo.isActive = false;
            activeStakersCount--;
        }
        
        // Slashed tokens are burned (sent to zero address or kept in contract)
        // For simplicity, we keep them in the contract for potential redistribution
        
        emit Slashed(staker, slashAmount, reason);
    }
    
    /// @notice Get stake info for an address
    /// @param staker Address to query
    /// @return amount Current stake amount
    /// @return isActive Whether the staker is active
    /// @return isUnbonding Whether the staker is unbonding
    /// @return unbondingEndTime When unbonding will complete (0 if not unbonding)
    function getStakeInfo(address staker) external view returns (
        uint256 amount,
        bool isActive,
        bool isUnbonding,
        uint256 unbondingEndTime
    ) {
        StakeInfo storage info = stakes[staker];
        amount = info.amount;
        isActive = info.isActive;
        isUnbonding = info.isUnbonding;
        unbondingEndTime = info.isUnbonding ? info.unbondingStartTime + unbondingPeriod : 0;
    }
    
    /// @notice Check if an address has sufficient stake to participate
    /// @param staker Address to check
    /// @return Whether the staker can participate
    function canParticipate(address staker) external view returns (bool) {
        StakeInfo storage info = stakes[staker];
        return info.isActive && info.amount >= minStake && !info.isUnbonding;
    }
    
    /// @notice Get staker's voting weight (proportional to stake)
    /// @param staker Address to query
    /// @return weight Voting weight in basis points
    function getVotingWeight(address staker) external view returns (uint256 weight) {
        if (totalStaked == 0) return 0;
        return (stakes[staker].amount * 10000) / totalStaked;
    }
    
    /// @notice Get the aggregation weight for a staker (for weighted gradient aggregation)
    /// @param staker Address to query
    /// @return weight Weight as a fraction (multiply by 1e18 for precision)
    function getAggregationWeight(address staker) external view returns (uint256 weight) {
        if (totalStaked == 0) return 0;
        return (stakes[staker].amount * 1e18) / totalStaked;
    }
    
    /// @notice Update staking parameters
    function updateParameters(
        uint256 _minStake,
        uint256 _unbondingPeriod,
        uint256 _slashingRate
    ) external onlyOwner {
        require(_slashingRate <= 10000, "Slashing rate too high");
        
        minStake = _minStake;
        unbondingPeriod = _unbondingPeriod;
        slashingRate = _slashingRate;
        
        emit ParametersUpdated(_minStake, _unbondingPeriod, _slashingRate);
    }
    
    /// @notice Update the operator address
    function setOperator(address _operator) external onlyOwner {
        require(_operator != address(0), "Invalid address");
        emit OperatorUpdated(operator, _operator);
        operator = _operator;
    }
    
    /// @notice Transfer ownership
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid address");
        owner = newOwner;
    }
    
    /// @notice Emergency withdraw of slashed/excess tokens
    function emergencyWithdraw(address to, uint256 amount) external onlyOwner {
        require(to != address(0), "Invalid address");
        uint256 balance = helixToken.balanceOf(address(this));
        uint256 excess = balance > totalStaked ? balance - totalStaked : 0;
        require(amount <= excess, "Cannot withdraw staked tokens");
        helixToken.safeTransfer(to, amount);
    }
}
