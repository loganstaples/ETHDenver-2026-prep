// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/// @title Staking
/// @notice Production-grade stake management for HELIX training nodes
/// @dev Features: gradual slashing, evidence integration, challenger rewards, warnings
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

    /// @notice Severity levels for gradual slashing
    enum SeverityLevel {
        Warning,    // No slash, just warning
        Minor,      // 10% slash
        Moderate,   // 25% slash
        Major,      // 50% slash
        Critical,   // 75% slash
        Terminal    // 100% slash + permanent ban
    }

    /// @notice Violation types
    enum ViolationType {
        InvalidProof,
        CommitmentMismatch,
        ErrorBoundExceeded,
        TimeoutViolation,
        MaliciousGradient,
        DoubleSubmission,
        ProtocolViolation
    }

    /// @notice Stake information for each staker
    struct StakeInfo {
        uint256 amount;
        uint256 stakedAt;
        uint256 unbondingStartTime;
        bool isUnbonding;
        bool isActive; // Can participate in training
    }

    /// @notice Warning record for gradual slashing
    struct WarningRecord {
        uint256 warningCount;
        uint256 totalSlashed;
        uint40 firstWarningTime;
        uint40 lastWarningTime;
        SeverityLevel maxSeverity;
        bool permanentlyBanned;
    }

    /// @notice Slashing event record
    struct SlashingRecord {
        address staker;
        uint256 amount;
        uint40 timestamp;
        ViolationType violationType;
        SeverityLevel severity;
        address challenger;
        string reason;
    }

    /// @notice Challenger reward configuration
    struct ChallengerConfig {
        uint16 rewardPercentage;  // Basis points
        uint128 minimumReward;
        uint128 maximumReward;
        bool enabled;
    }

    /// @notice Mapping of staker address to their stake info
    mapping(address => StakeInfo) public stakes;

    /// @notice Warning records per staker
    mapping(address => WarningRecord) public warningRecords;

    /// @notice Severity level slashing percentages (basis points)
    mapping(SeverityLevel => uint16) public severitySlashPercentage;

    /// @notice Warning count thresholds for severity escalation
    mapping(SeverityLevel => uint256) public warningThreshold;

    /// @notice Total staked amount
    uint256 public totalStaked;

    /// @notice Active stakers count
    uint256 public activeStakersCount;

    /// @notice Operator who can perform slashing (typically the coordinator)
    address public operator;

    /// @notice Owner for parameter updates
    address public owner;

    /// @notice Treasury for slashed funds and protocol fees
    address public treasury;

    /// @notice Challenger reward configuration
    ChallengerConfig public challengerConfig;

    /// @notice Slashing records for audit trail
    SlashingRecord[] public slashingRecords;
    
    /// @notice Events
    event Staked(address indexed staker, uint256 amount, uint256 totalStake);
    event UnbondingStarted(address indexed staker, uint256 unbondingEndTime);
    event Unstaked(address indexed staker, uint256 amount);
    event Slashed(address indexed staker, uint256 amount, string reason);
    event SlashedWithEvidence(
        address indexed staker,
        uint256 amount,
        ViolationType violationType,
        SeverityLevel severity,
        address indexed challenger
    );
    event WarningIssued(address indexed staker, uint256 warningCount, SeverityLevel nextSeverity);
    event StakerBanned(address indexed staker, uint256 totalSlashed, string reason);
    event ChallengerRewarded(address indexed challenger, uint256 reward, address indexed slashedStaker);
    event StakeIncreased(address indexed staker, uint256 addedAmount, uint256 newTotal);
    event ParametersUpdated(uint256 minStake, uint256 unbondingPeriod, uint256 slashingRate);
    event OperatorUpdated(address indexed oldOperator, address indexed newOperator);
    event SeverityEscalated(address indexed staker, SeverityLevel oldLevel, SeverityLevel newLevel);
    event TreasuryUpdated(address indexed oldTreasury, address indexed newTreasury);
    event ChallengerConfigUpdated(ChallengerConfig config);
    
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
        treasury = msg.sender;

        // Initialize severity slashing percentages
        severitySlashPercentage[SeverityLevel.Warning] = 0;      // 0%
        severitySlashPercentage[SeverityLevel.Minor] = 1000;     // 10%
        severitySlashPercentage[SeverityLevel.Moderate] = 2500;  // 25%
        severitySlashPercentage[SeverityLevel.Major] = 5000;     // 50%
        severitySlashPercentage[SeverityLevel.Critical] = 7500;  // 75%
        severitySlashPercentage[SeverityLevel.Terminal] = 10000; // 100%

        // Initialize warning thresholds
        warningThreshold[SeverityLevel.Warning] = 1;
        warningThreshold[SeverityLevel.Minor] = 2;
        warningThreshold[SeverityLevel.Moderate] = 3;
        warningThreshold[SeverityLevel.Major] = 4;
        warningThreshold[SeverityLevel.Critical] = 5;
        warningThreshold[SeverityLevel.Terminal] = 6;

        // Initialize challenger config
        challengerConfig = ChallengerConfig({
            rewardPercentage: 1000,  // 10%
            minimumReward: 0,
            maximumReward: type(uint128).max,
            enabled: true
        });
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
    
    /// @notice Slash a staker for protocol violation (legacy interface)
    /// @param staker Address of the staker to slash
    /// @param reason Reason for slashing
    function slash(address staker, string calldata reason) external onlyOperator nonReentrant {
        _slashWithSeverity(staker, SeverityLevel.Major, ViolationType.ProtocolViolation, address(0), reason);
    }

    /// @notice Slash with full details and evidence
    /// @param staker Address of the staker to slash
    /// @param violationType Type of violation
    /// @param challenger Address of challenger (can be address(0))
    /// @param reason Detailed reason
    function slashWithEvidence(
        address staker,
        ViolationType violationType,
        address challenger,
        string calldata reason
    ) external onlyOperator nonReentrant {
        SeverityLevel severity = _calculateSeverity(staker, violationType);
        _slashWithSeverity(staker, severity, violationType, challenger, reason);
    }

    /// @notice Slash with explicit severity level
    /// @param staker Address of the staker to slash
    /// @param severity Severity level
    /// @param violationType Type of violation
    /// @param challenger Address of challenger
    /// @param reason Detailed reason
    function slashWithSeverity(
        address staker,
        SeverityLevel severity,
        ViolationType violationType,
        address challenger,
        string calldata reason
    ) external onlyOperator nonReentrant {
        _slashWithSeverity(staker, severity, violationType, challenger, reason);
    }

    /// @notice Internal slashing implementation with all features
    function _slashWithSeverity(
        address staker,
        SeverityLevel severity,
        ViolationType violationType,
        address challenger,
        string memory reason
    ) internal {
        StakeInfo storage stakeInfo = stakes[staker];
        WarningRecord storage warning = warningRecords[staker];

        require(stakeInfo.amount > 0, "No stake to slash");
        require(!warning.permanentlyBanned, "Staker already banned");

        // Calculate slash amount based on severity
        uint256 slashAmount = (stakeInfo.amount * severitySlashPercentage[severity]) / 10000;

        // Update warning record
        warning.warningCount++;
        warning.totalSlashed += slashAmount;
        warning.lastWarningTime = uint40(block.timestamp);
        if (warning.firstWarningTime == 0) {
            warning.firstWarningTime = uint40(block.timestamp);
        }

        // Check for severity escalation
        if (uint8(severity) > uint8(warning.maxSeverity)) {
            emit SeverityEscalated(staker, warning.maxSeverity, severity);
            warning.maxSeverity = severity;
        }

        // Handle warning only (no actual slash)
        if (severity == SeverityLevel.Warning) {
            emit WarningIssued(staker, warning.warningCount, _getNextSeverity(staker));
            return;
        }

        // Perform actual slashing
        stakeInfo.amount -= slashAmount;
        totalStaked -= slashAmount;

        // Handle challenger reward
        uint256 challengerReward = 0;
        if (challenger != address(0) && challengerConfig.enabled && slashAmount > 0) {
            challengerReward = _calculateAndDistributeReward(challenger, slashAmount, staker);
        }

        // Transfer remaining slashed amount to treasury
        uint256 toTreasury = slashAmount - challengerReward;
        if (toTreasury > 0 && treasury != address(0)) {
            helixToken.safeTransfer(treasury, toTreasury);
        }

        // If stake drops below minimum, deactivate
        if (stakeInfo.amount < minStake && stakeInfo.isActive) {
            stakeInfo.isActive = false;
            activeStakersCount--;
        }

        // Check for permanent ban
        if (severity == SeverityLevel.Terminal || warning.warningCount >= warningThreshold[SeverityLevel.Terminal]) {
            warning.permanentlyBanned = true;
            emit StakerBanned(staker, warning.totalSlashed, "Terminal severity or max warnings exceeded");
        }

        // Record slashing event
        slashingRecords.push(SlashingRecord({
            staker: staker,
            amount: slashAmount,
            timestamp: uint40(block.timestamp),
            violationType: violationType,
            severity: severity,
            challenger: challenger,
            reason: reason
        }));

        emit SlashedWithEvidence(staker, slashAmount, violationType, severity, challenger);
        emit Slashed(staker, slashAmount, reason);
    }

    /// @notice Calculate and distribute challenger reward
    function _calculateAndDistributeReward(
        address challenger,
        uint256 slashAmount,
        address staker
    ) internal returns (uint256 reward) {
        reward = (slashAmount * challengerConfig.rewardPercentage) / 10000;

        // Apply bounds
        if (reward < challengerConfig.minimumReward) {
            reward = challengerConfig.minimumReward;
        }
        if (reward > challengerConfig.maximumReward) {
            reward = challengerConfig.maximumReward;
        }

        // Cap to available slashed amount
        if (reward > slashAmount) {
            reward = slashAmount;
        }

        if (reward > 0) {
            helixToken.safeTransfer(challenger, reward);
            emit ChallengerRewarded(challenger, reward, staker);
        }
    }

    /// @notice Calculate severity based on violation type and history
    function _calculateSeverity(
        address staker,
        ViolationType violationType
    ) internal view returns (SeverityLevel) {
        WarningRecord storage record = warningRecords[staker];

        // First offense for minor violations starts as warning
        if (record.warningCount == 0 && _isMinorViolation(violationType)) {
            return SeverityLevel.Warning;
        }

        // Critical violations skip warning
        if (_isCriticalViolation(violationType)) {
            return SeverityLevel.Major;
        }

        // Escalate based on warning count
        if (record.warningCount >= warningThreshold[SeverityLevel.Terminal]) {
            return SeverityLevel.Terminal;
        }
        if (record.warningCount >= warningThreshold[SeverityLevel.Critical]) {
            return SeverityLevel.Critical;
        }
        if (record.warningCount >= warningThreshold[SeverityLevel.Major]) {
            return SeverityLevel.Major;
        }
        if (record.warningCount >= warningThreshold[SeverityLevel.Moderate]) {
            return SeverityLevel.Moderate;
        }
        if (record.warningCount >= warningThreshold[SeverityLevel.Minor]) {
            return SeverityLevel.Minor;
        }

        return SeverityLevel.Warning;
    }

    /// @notice Get next severity level for a staker
    function _getNextSeverity(address staker) internal view returns (SeverityLevel) {
        uint256 nextCount = warningRecords[staker].warningCount + 1;

        if (nextCount >= warningThreshold[SeverityLevel.Terminal]) return SeverityLevel.Terminal;
        if (nextCount >= warningThreshold[SeverityLevel.Critical]) return SeverityLevel.Critical;
        if (nextCount >= warningThreshold[SeverityLevel.Major]) return SeverityLevel.Major;
        if (nextCount >= warningThreshold[SeverityLevel.Moderate]) return SeverityLevel.Moderate;
        return SeverityLevel.Minor;
    }

    /// @notice Check if violation is minor
    function _isMinorViolation(ViolationType vType) internal pure returns (bool) {
        return vType == ViolationType.TimeoutViolation || vType == ViolationType.ErrorBoundExceeded;
    }

    /// @notice Check if violation is critical
    function _isCriticalViolation(ViolationType vType) internal pure returns (bool) {
        return vType == ViolationType.MaliciousGradient || vType == ViolationType.DoubleSubmission;
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

    // ============ View Functions ============

    /// @notice Get warning record for a staker
    function getWarningRecord(address staker) external view returns (WarningRecord memory) {
        return warningRecords[staker];
    }

    /// @notice Check if a staker is banned
    function isBanned(address staker) external view returns (bool) {
        return warningRecords[staker].permanentlyBanned;
    }

    /// @notice Get total slashing records count
    function getSlashingRecordCount() external view returns (uint256) {
        return slashingRecords.length;
    }

    /// @notice Get slash percentage for severity level
    function getSlashPercentage(SeverityLevel severity) external view returns (uint16) {
        return severitySlashPercentage[severity];
    }

    /// @notice Calculate expected slash amount
    function calculateSlashAmount(address staker, SeverityLevel severity) external view returns (uint256) {
        return (stakes[staker].amount * severitySlashPercentage[severity]) / 10000;
    }

    /// @notice Get current severity level for a staker
    function getCurrentSeverityLevel(address staker) external view returns (SeverityLevel) {
        return warningRecords[staker].maxSeverity;
    }

    // ============ Admin Functions ============

    /// @notice Set treasury address
    function setTreasury(address _treasury) external onlyOwner {
        emit TreasuryUpdated(treasury, _treasury);
        treasury = _treasury;
    }

    /// @notice Set challenger reward configuration
    function setChallengerConfig(ChallengerConfig calldata config) external onlyOwner {
        require(config.rewardPercentage <= 5000, "Max 50% reward");
        challengerConfig = config;
        emit ChallengerConfigUpdated(config);
    }

    /// @notice Set severity slash percentage
    function setSeveritySlashPercentage(SeverityLevel severity, uint16 percentage) external onlyOwner {
        require(percentage <= 10000, "Max 100%");
        severitySlashPercentage[severity] = percentage;
    }

    /// @notice Set warning threshold
    function setWarningThreshold(SeverityLevel severity, uint256 threshold) external onlyOwner {
        warningThreshold[severity] = threshold;
    }

    /// @notice Clear ban for a staker (emergency function)
    function clearBan(address staker) external onlyOwner {
        warningRecords[staker].permanentlyBanned = false;
    }

    /// @notice Reset warning record (emergency function)
    function resetWarningRecord(address staker) external onlyOwner {
        delete warningRecords[staker];
    }
}
