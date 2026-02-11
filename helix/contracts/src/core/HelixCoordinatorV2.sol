// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "../interfaces/IHelixVerifier.sol";
import "./ModelRegistry.sol";
import "../verification/PoseidonHasher.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/// @title HelixCoordinatorV2
/// @notice Gas-optimized coordinator for model registration, proof submission, staking, and slashing
/// @dev Supports real ZK proof verification with economic security
///      Storage layout optimized for gas efficiency with struct packing
///      Includes multi-sig emergency pause mechanism with time-locked recovery
///      and comprehensive challenger reward distribution
///      ReentrancyGuard added to protect external calls in slash() and unstake()
///      Admin timelocks: 48-hour delay for parameter changes
contract HelixCoordinatorV2 is ReentrancyGuard {
    // ============ Custom Errors (gas-optimized, ~200 gas savings per revert) ============

    error OnlyOwner();
    error ContractPaused();
    error ModelNotFound();
    error InsufficientStake();
    error StakeSlashed();
    error OnlyGuardian();
    error PauseInProgress();
    error InvalidTreasury();
    error InvalidAddress();
    error NotModelOwner();
    error ModelNotActive();
    error ZeroStake();
    error PreviousStakeSlashed();
    error NoStake();
    error StakeWasSlashed();
    error StillLocked();
    error TransferFailed();
    error InvalidRound();
    error RoundAlreadyCompleted();
    error RoundExpired();
    error InvalidPublicInputsCount();
    error OldCommitmentMismatch();
    error ErrorBoundExceeded();
    error ErrorChecksumMismatch();
    error NoStakeToSlash();
    error AlreadySlashed();
    error TreasuryTransferFailed();
    error AlreadyPaused();
    error NotPaused();
    error AlreadyApproved();
    error NoApprovalToCancel();
    error InvalidNewOwner();
    error RecoveryAlreadyPending();
    error ContractMustBePaused();
    error NoRecoveryPending();
    error TimeLockNotExpired();
    error NotAuthorized();
    error AlreadyGuardian();
    error NotGuardianRole();
    error BelowThreshold();
    error MinimumTimeLock();
    error MaximumTimeLock();
    error RequireAtLeastOne();
    error ExceedsGuardianCount();
    error MaxRewardPercentage();
    error MinExceedsMax();
    error InvalidDataRoot();
    error MaxPercentage();
    error RoundNotCompleted();
    error NoProverToChallenge();
    error CannotChallengeSelf();
    // Timelock errors
    error TimelockNotReady();
    error NoTimelockPending();
    error TimelockAlreadyPending();
    error PaginationOutOfBounds();
    // Proof replay error
    error ProofAlreadyUsed();
    // Batch submission error
    error EmptyBatch();

    // ============ Structs (Optimized for Storage Packing) ============

    /// @notice Model information - packed to minimize storage slots
    /// @dev Packed: owner(20) + currentRound(4) + active(1) + minStake(8) = 33 bytes (2 slots with commitment)
    struct Model {
        string ipfsHash;              // Dynamic - separate slot(s)
        uint256 currentCommitment;    // Slot 1: 32 bytes
        address owner;                // Slot 2: 20 bytes
        uint32 currentRound;          // Slot 2: 4 bytes (max 4B rounds)
        bool active;                  // Slot 2: 1 byte
        uint64 minStake;              // Slot 2: 8 bytes (max 18.4 ETH in gwei units)
    }

    /// @notice Round information - packed to minimize storage slots
    /// @dev Packed: deadline(5) + isCompleted(1) + prover(20) = 26 bytes in one slot
    struct Round {
        uint256 modelCommitment;      // Slot 1: 32 bytes
        uint256 newCommitment;        // Slot 2: 32 bytes
        uint40 deadline;              // Slot 3: 5 bytes (timestamps until year 36812)
        bool isCompleted;             // Slot 3: 1 byte
        address prover;               // Slot 3: 20 bytes
    }

    /// @notice Stake information - packed to single slot where possible
    /// @dev Packed: amount(16) + lockedUntil(5) + slashed(1) = 22 bytes
    struct Stake {
        uint128 amount;               // 16 bytes (max 340 undecillion wei)
        uint40 lockedUntil;           // 5 bytes (timestamps until year 36812)
        bool slashed;                 // 1 byte
    }

    /// @notice Slashing record - optimized with indexed timestamp
    /// @dev Keeping full precision for amounts as these are audit records
    struct SlashingRecord {
        address prover;               // 20 bytes
        uint64 modelId;               // 8 bytes
        uint32 roundId;               // 4 bytes
        uint128 amount;               // 16 bytes
        string reason;                // Dynamic
        uint40 timestamp;             // 5 bytes
    }

    /// @notice Data for a single proof submission in a batch
    struct ProofSubmissionData {
        uint256 modelId;
        uint256 roundId;
        bytes proof;
        uint256[] publicInputs;
    }

    /// @notice Pending admin change for timelock enforcement
    struct PendingChange {
        bytes32 changeHash;           // Hash of the change parameters
        uint256 executionTime;        // When the change can be executed
        bool active;                  // Whether a change is pending
    }

    // ============ Constants ============

    /// @notice Timelock delay for parameter changes (48 hours)
    uint256 public constant PARAM_TIMELOCK = 48 hours;

    uint16 public constant DEFAULT_SLASH_PERCENTAGE = 5000;       // 50%
    uint16 public constant DEFAULT_STAKE_LOCK_DAYS = 7;
    uint256 public constant DEFAULT_MIN_STAKE = 0.1 ether;
    uint256 public constant DEFAULT_MAX_ERROR_BOUND = 1e18;
    uint8 public constant DEFAULT_REQUIRED_GUARDIANS = 2;
    uint256 public constant DEFAULT_RECOVERY_TIMELOCK = 48 hours;
    uint16 public constant DEFAULT_CHALLENGER_REWARD_PCT = 1000;   // 10%
    uint128 public constant DEFAULT_MIN_CHALLENGER_REWARD = 0.001 ether;
    uint128 public constant DEFAULT_MAX_CHALLENGER_REWARD = 10 ether;
    uint16 public constant MAX_PERCENTAGE = 10000;                 // 100%
    uint16 public constant MAX_REWARD_PERCENTAGE = 5000;           // 50%
    uint8 public constant EXPECTED_PUBLIC_INPUTS = 8;
    uint256 public constant MIN_RECOVERY_TIMELOCK = 1 hours;
    uint256 public constant MAX_RECOVERY_TIMELOCK = 30 days;

    // ============ State Variables (Ordered for Optimal Packing) ============

    // Verifier is immutable for gas savings (saves 2100 gas per SLOAD on every submitProof call)
    /// @notice The ZK proof verifier contract
    IHelixVerifier public immutable verifier;

    // Slot 2: Packed addresses and small values
    /// @notice Treasury to receive slashed funds
    address public treasury;
    /// @notice Slash percentage (100 = 1%, max 10000 = 100%)
    uint16 public slashPercentage;
    /// @notice Stake lock period in days (max 65535 days = ~179 years)
    uint16 public stakeLockDays;
    /// @notice Counter for model IDs
    uint32 public nextModelId;

    // Slot 3: Owner
    /// @notice Contract owner
    address public owner;

    // Slot 4: Default min stake (full precision needed)
    /// @notice Default minimum stake (in wei)
    uint256 public defaultMinStake;

    // Slot 5: Max error bound (full precision needed)
    /// @notice Maximum allowed error bound per step (in fixed-point units)
    uint256 public maxErrorBound;

    // Slot 6: Emergency pause state
    /// @notice Whether the contract is paused
    bool public paused;

    /// @notice Data commitment contract address
    address public dataCommitment;

    /// @notice Slashing evidence contract address
    address public slashingEvidence;

    /// @notice Optional model registry for checkpoint tracking
    ModelRegistry public modelRegistry;

    // ============ Multi-Sig Emergency Pause State ============

    /// @notice Number of guardians required for multi-sig actions
    uint8 public requiredGuardians;

    /// @notice Current pause request nonce
    uint64 public pauseNonce;

    /// @notice Time lock duration for recovery (default 48 hours)
    uint256 public recoveryTimeLock;

    /// @notice Pending recovery execution time (0 if no pending recovery)
    uint256 public pendingRecoveryTime;

    /// @notice Pending recovery new owner
    address public pendingRecoveryOwner;

    /// @notice Mapping of guardian addresses
    mapping(address => bool) public isGuardian;

    /// @notice Guardian count
    uint8 public guardianCount;

    /// @notice Pause approvals per nonce: nonce => guardian => approved
    mapping(uint64 => mapping(address => bool)) public pauseApprovals;

    /// @notice Approval count per nonce
    mapping(uint64 => uint8) public pauseApprovalCount;

    // ============ Challenger Reward Configuration ============

    /// @notice Challenger reward percentage (basis points, 1000 = 10%)
    uint16 public challengerRewardPercentage;

    /// @notice Minimum challenger reward
    uint128 public minChallengerReward;

    /// @notice Maximum challenger reward
    uint128 public maxChallengerReward;

    /// @notice Challenger rewards enabled
    bool public challengerRewardsEnabled;

    // ============ Mappings ============

    /// @notice Model registry
    mapping(uint256 => Model) public models;

    /// @notice Round data: modelId => roundId => Round
    mapping(uint256 => mapping(uint256 => Round)) public rounds;

    /// @notice Prover stakes: prover => modelId => Stake
    mapping(address => mapping(uint256 => Stake)) public stakes;

    /// @notice Accumulated error bound per model
    mapping(uint256 => uint256) public accumulatedErrorBound;

    /// @notice Data commitment required per model (merkle root)
    mapping(uint256 => bytes32) public modelDataCommitment;

    /// @notice Data root used per round
    mapping(uint256 => mapping(uint256 => bytes32)) public roundDataRoot;

    /// @notice Slashing records - append only for audit trail
    SlashingRecord[] public slashingRecords;

    /// @notice Proof replay protection: hash(proof || publicInputs) => used
    mapping(bytes32 => bool) public usedProofHashes;

    // ============ Admin Timelock State ============

    /// @notice Pending admin changes by change type key
    mapping(bytes32 => PendingChange) public pendingChanges;

    // ============ Events (Optimized with Indexed Parameters) ============

    /// @notice Emitted when a new model is registered
    event ModelRegistered(
        uint256 indexed modelId,
        address indexed owner,
        uint256 initialCommitment,
        uint256 minStake,
        string ipfsHash
    );

    /// @notice Emitted when a training round starts
    event RoundStarted(
        uint256 indexed modelId,
        uint256 indexed roundId,
        uint256 deadline,
        uint256 modelCommitment
    );

    /// @notice Emitted when a proof is submitted
    event ProofSubmitted(
        uint256 indexed modelId,
        uint256 indexed roundId,
        address indexed prover,
        uint256 newCommitment,
        uint256 errorBound
    );

    /// @notice Emitted when a training round completes
    event RoundCompleted(
        uint256 indexed modelId,
        uint256 indexed roundId,
        uint256 newCommitment,
        uint256 totalErrorBound
    );

    /// @notice Emitted when stake is deposited
    event Staked(
        address indexed prover,
        uint256 indexed modelId,
        uint256 amount,
        uint256 totalStake
    );

    /// @notice Emitted when stake is withdrawn
    event Unstaked(
        address indexed prover,
        uint256 indexed modelId,
        uint256 amount
    );

    /// @notice Emitted when a prover is slashed
    event Slashed(
        address indexed prover,
        uint256 indexed modelId,
        uint256 roundId,
        uint256 amount,
        uint256 remainingStake,
        string reason
    );

    /// @notice Emitted when an invalid proof is detected
    event InvalidProofDetected(
        uint256 indexed modelId,
        uint256 indexed roundId,
        address indexed prover,
        bytes32 proofHash
    );

    /// @notice Emitted when a batch of proofs is submitted
    event BatchProofSubmitted(address indexed submitter, uint256 totalProofs);

    /// @notice Emitted when a proof replay attempt is blocked
    event ProofReplayBlocked(
        bytes32 indexed proofHash,
        address indexed submitter
    );

    /// @notice Emitted when model state changes
    event ModelStateChanged(
        uint256 indexed modelId,
        bool active,
        address indexed changedBy
    );

    /// @notice Emitted when configuration is updated
    event ConfigUpdated(
        string indexed parameter,
        uint256 oldValue,
        uint256 newValue
    );

    /// @notice Emitted when contract is paused/unpaused
    event EmergencyPauseChanged(
        bool isPaused,
        address indexed changedBy
    );

    /// @notice Emitted when data commitment is set for a model
    event DataCommitmentSet(
        uint256 indexed modelId,
        bytes32 indexed dataRoot,
        address indexed setBy
    );

    /// @notice Emitted when round data is committed
    event RoundDataCommitted(
        uint256 indexed modelId,
        uint256 indexed roundId,
        bytes32 dataRoot
    );

    /// @notice Emitted when model registry is updated
    event ModelRegistryUpdated(address indexed oldRegistry, address indexed newRegistry);

    /// @notice Emitted when data commitment contract is updated
    event DataCommitmentContractUpdated(
        address indexed oldContract,
        address indexed newContract
    );

    /// @notice Emitted when a guardian is added or removed
    event GuardianUpdated(
        address indexed guardian,
        bool isActive,
        address indexed changedBy
    );

    /// @notice Emitted when a pause approval is received
    event PauseApprovalReceived(
        uint64 indexed nonce,
        address indexed guardian,
        uint8 currentApprovals,
        uint8 requiredApprovals
    );

    /// @notice Emitted when multi-sig pause is executed
    event MultiSigPauseExecuted(
        uint64 indexed nonce,
        address[] approvers
    );

    /// @notice Emitted when recovery is initiated
    event RecoveryInitiated(
        address indexed newOwner,
        uint256 executionTime,
        address indexed initiatedBy
    );

    /// @notice Emitted when recovery is executed
    event RecoveryExecuted(
        address indexed oldOwner,
        address indexed newOwner
    );

    /// @notice Emitted when recovery is cancelled
    event RecoveryCancelled(
        address indexed cancelledBy
    );

    /// @notice Emitted when challenger receives reward
    event ChallengerRewarded(
        address indexed challenger,
        uint256 indexed modelId,
        uint256 roundId,
        uint256 rewardAmount
    );

    /// @notice Emitted when challenger config is updated
    event ChallengerConfigUpdated(
        uint16 rewardPercentage,
        uint128 minReward,
        uint128 maxReward,
        bool enabled
    );

    /// @notice Emitted when an admin change is proposed (timelock starts)
    event AdminChangeProposed(
        bytes32 indexed changeKey,
        bytes32 changeHash,
        uint256 executionTime
    );

    /// @notice Emitted when a pending admin change is cancelled
    event AdminChangeCancelled(bytes32 indexed changeKey);

    /// @notice Emitted when a timelocked admin change is executed
    event AdminChangeExecuted(bytes32 indexed changeKey);

    // ============ Modifiers ============

    modifier onlyOwner() {
        if (msg.sender != owner) revert OnlyOwner();
        _;
    }

    modifier whenNotPaused() {
        if (paused) revert ContractPaused();
        _;
    }

    modifier whenPaused() {
        if (!paused) revert NotPaused();
        _;
    }

    modifier modelExists(uint256 modelId) {
        if (models[modelId].owner == address(0)) revert ModelNotFound();
        _;
    }

    modifier hasStake(uint256 modelId) {
        Stake storage s = stakes[msg.sender][modelId];
        if (s.amount < models[modelId].minStake) revert InsufficientStake();
        if (s.slashed) revert StakeSlashed();
        _;
    }

    modifier onlyGuardian() {
        if (!isGuardian[msg.sender]) revert OnlyGuardian();
        _;
    }

    modifier noActivePause() {
        if (pauseApprovalCount[pauseNonce] != 0) revert PauseInProgress();
        _;
    }

    // ============ Constructor ============

    constructor(address _verifier, address _treasury) {
        if (_treasury == address(0)) revert InvalidTreasury();
        verifier = IHelixVerifier(_verifier);
        treasury = _treasury;
        owner = msg.sender;

        // Initialize packed values
        slashPercentage = DEFAULT_SLASH_PERCENTAGE;
        stakeLockDays = DEFAULT_STAKE_LOCK_DAYS;
        defaultMinStake = DEFAULT_MIN_STAKE;
        maxErrorBound = DEFAULT_MAX_ERROR_BOUND;

        // Initialize multi-sig pause configuration
        requiredGuardians = DEFAULT_REQUIRED_GUARDIANS;
        recoveryTimeLock = DEFAULT_RECOVERY_TIMELOCK;

        // Initialize challenger reward configuration
        challengerRewardPercentage = DEFAULT_CHALLENGER_REWARD_PCT;
        minChallengerReward = DEFAULT_MIN_CHALLENGER_REWARD;
        maxChallengerReward = DEFAULT_MAX_CHALLENGER_REWARD;
        challengerRewardsEnabled = true;

        // Owner is first guardian
        isGuardian[msg.sender] = true;
        guardianCount = 1;
    }

    // ============ Model Management ============

    /// @notice Registers a new model
    function registerModel(
        string memory ipfsHash,
        uint256 initialCommitment,
        uint256 minStake
    ) external whenNotPaused returns (uint256 modelId) {
        modelId = nextModelId++;

        uint64 effectiveMinStake = minStake > 0
            ? uint64(minStake / 1 gwei)  // Store in gwei for packing
            : uint64(defaultMinStake / 1 gwei);

        models[modelId] = Model({
            ipfsHash: ipfsHash,
            currentCommitment: initialCommitment,
            owner: msg.sender,
            currentRound: 0,
            active: true,
            minStake: effectiveMinStake
        });

        emit ModelRegistered(modelId, msg.sender, initialCommitment, minStake > 0 ? minStake : defaultMinStake, ipfsHash);

        // Optional ModelRegistry integration
        if (address(modelRegistry) != address(0)) {
            modelRegistry.registerModel("", "", ipfsHash, bytes32(initialCommitment));
        }
    }

    /// @notice Starts a new training round
    function startRound(
        uint256 modelId,
        uint256 duration
    ) external whenNotPaused modelExists(modelId) {
        Model storage model = models[modelId];
        if (msg.sender != model.owner) revert NotModelOwner();
        if (!model.active) revert ModelNotActive();

        uint32 roundId = ++model.currentRound;
        uint40 deadline = uint40(block.timestamp + duration);

        rounds[modelId][roundId] = Round({
            modelCommitment: model.currentCommitment,
            newCommitment: 0,
            deadline: deadline,
            isCompleted: false,
            prover: address(0)
        });

        emit RoundStarted(modelId, roundId, deadline, model.currentCommitment);
    }

    // ============ Staking ============

    /// @notice Stakes tokens to participate in a model's training
    function stake(uint256 modelId) external payable modelExists(modelId) {
        if (msg.value == 0) revert ZeroStake();

        Stake storage s = stakes[msg.sender][modelId];
        if (s.slashed) revert PreviousStakeSlashed();

        uint128 newAmount = s.amount + uint128(msg.value);
        s.amount = newAmount;
        s.lockedUntil = uint40(block.timestamp + uint256(stakeLockDays) * 1 days);

        emit Staked(msg.sender, modelId, msg.value, newAmount);
    }

    /// @notice Withdraws stake after lock period
    function unstake(uint256 modelId) external nonReentrant {
        Stake storage s = stakes[msg.sender][modelId];
        if (s.amount == 0) revert NoStake();
        if (s.slashed) revert StakeWasSlashed();
        if (block.timestamp < s.lockedUntil) revert StillLocked();

        uint256 amount = s.amount;
        s.amount = 0;

        (bool success, ) = msg.sender.call{value: amount}("");
        if (!success) revert TransferFailed();

        emit Unstaked(msg.sender, modelId, amount);
    }

    // ============ Proof Submission ============

    /// @notice Submits a training proof for a round
    function submitProof(
        uint256 modelId,
        uint256 roundId,
        bytes memory proof,
        uint256[] memory publicInputs
    ) external nonReentrant whenNotPaused {
        _processProof(msg.sender, modelId, roundId, proof, publicInputs);
    }

    /// @notice Submits multiple proofs in a single transaction to amortize base tx costs
    function submitProofBatch(ProofSubmissionData[] calldata submissions) external nonReentrant whenNotPaused {
        uint256 n = submissions.length;
        if (n == 0) revert EmptyBatch();
        for (uint256 i = 0; i < n; i++) {
            _processProof(
                msg.sender,
                submissions[i].modelId,
                submissions[i].roundId,
                submissions[i].proof,
                submissions[i].publicInputs
            );
        }
        emit BatchProofSubmitted(msg.sender, n);
    }

    /// @notice Internal proof processing logic
    function _processProof(
        address prover,
        uint256 modelId,
        uint256 roundId,
        bytes memory proof,
        uint256[] memory publicInputs
    ) internal {
        // Validate model exists
        if (models[modelId].owner == address(0)) revert ModelNotFound();

        // Validate stake
        Stake storage s = stakes[prover][modelId];
        if (s.amount < models[modelId].minStake) revert InsufficientStake();
        if (s.slashed) revert StakeSlashed();

        Model storage model = models[modelId];
        Round storage round = rounds[modelId][roundId];

        // Validate round
        if (roundId != model.currentRound) revert InvalidRound();
        if (round.isCompleted) revert RoundAlreadyCompleted();
        if (block.timestamp > round.deadline) revert RoundExpired();

        // Validate public inputs count (8 inputs including error checksum)
        if (publicInputs.length != EXPECTED_PUBLIC_INPUTS) revert InvalidPublicInputsCount();

        // Proof replay protection
        bytes32 proofHash = keccak256(abi.encodePacked(proof, publicInputs));
        if (usedProofHashes[proofHash]) revert ProofAlreadyUsed();
        usedProofHashes[proofHash] = true;

        // Reconstruct and validate old commitment
        uint256 oldCommitmentFromProof = _hashPair(publicInputs[0], publicInputs[1]);
        if (oldCommitmentFromProof != round.modelCommitment) revert OldCommitmentMismatch();

        // Validate error bound
        uint256 stepErrorBound = publicInputs[5];
        if (stepErrorBound > maxErrorBound) revert ErrorBoundExceeded();

        // Validate error checksum (prevents tampering with error tracking)
        uint256 errorChecksum = publicInputs[7];
        uint256 expectedChecksum = _computeErrorChecksum(
            stepErrorBound,
            publicInputs[6], // step_number
            modelId,
            maxErrorBound
        );
        if (errorChecksum != expectedChecksum) revert ErrorChecksumMismatch();

        // Verify the proof
        bool valid = verifier.verifyProof(proof, publicInputs);

        if (!valid) {
            _slash(prover, modelId, roundId, "Invalid proof");
            emit InvalidProofDetected(modelId, roundId, prover, keccak256(proof));
            return;
        }

        // Extract new commitment
        uint256 newCommitment = _hashPair(publicInputs[2], publicInputs[3]);

        // Update state
        model.currentCommitment = newCommitment;
        round.newCommitment = newCommitment;
        round.isCompleted = true;
        round.prover = prover;

        // Track accumulated error bound
        uint256 newAccumulatedError = accumulatedErrorBound[modelId] + stepErrorBound;
        accumulatedErrorBound[modelId] = newAccumulatedError;

        // Reset stake lock (reward for valid submission)
        stakes[prover][modelId].lockedUntil = uint40(block.timestamp);

        // Optional ModelRegistry integration
        if (address(modelRegistry) != address(0)) {
            modelRegistry.updateModel(modelId, bytes32(newCommitment), roundId, "", stepErrorBound, proofHash);
        }

        emit ProofSubmitted(modelId, roundId, prover, newCommitment, stepErrorBound);
        emit RoundCompleted(modelId, roundId, newCommitment, newAccumulatedError);
    }

    /// @notice Computes the expected error checksum for verification
    /// @dev Matches the Rust circuit implementation using Poseidon hash:
    ///      checksum = Poseidon(Poseidon(errorBound, stepNumber), Poseidon(modelId, errorBudget))
    ///      See training_step_v2.rs compute_error_checksum() and verify_error_checksum().
    function _computeErrorChecksum(
        uint256 errorBound,
        uint256 stepNumber,
        uint256 modelId,
        uint256 errorBudget
    ) internal pure returns (uint256) {
        return PoseidonHasher.computeErrorChecksum(errorBound, stepNumber, modelId, errorBudget);
    }

    // ============ Slashing ============

    /// @notice Internal function to slash a prover's stake
    function _slash(
        address prover,
        uint256 modelId,
        uint256 roundId,
        string memory reason
    ) internal {
        _slashWithChallenger(prover, modelId, roundId, reason, address(0));
    }

    /// @notice Internal function to slash with challenger reward
    function _slashWithChallenger(
        address prover,
        uint256 modelId,
        uint256 roundId,
        string memory reason,
        address challenger
    ) internal {
        Stake storage s = stakes[prover][modelId];
        if (s.amount == 0) revert NoStakeToSlash();
        if (s.slashed) revert AlreadySlashed();

        uint128 slashAmount = uint128((uint256(s.amount) * slashPercentage) / MAX_PERCENTAGE);
        s.amount -= slashAmount;
        s.slashed = true;

        // Calculate challenger reward if applicable
        uint128 challengerReward = 0;
        if (challenger != address(0) && challengerRewardsEnabled && slashAmount > 0) {
            challengerReward = _calculateChallengerReward(slashAmount);
        }

        // Send challenger reward
        if (challengerReward > 0) {
            (bool rewardSuccess, ) = challenger.call{value: challengerReward}("");
            if (rewardSuccess) {
                emit ChallengerRewarded(challenger, modelId, roundId, challengerReward);
            }
        }

        // Send remaining slashed amount to treasury
        uint128 toTreasury = slashAmount - challengerReward;
        if (treasury != address(0) && toTreasury > 0) {
            (bool success, ) = treasury.call{value: toTreasury}("");
            if (!success) revert TreasuryTransferFailed();
        }

        slashingRecords.push(SlashingRecord({
            prover: prover,
            modelId: uint64(modelId),
            roundId: uint32(roundId),
            amount: slashAmount,
            reason: reason,
            timestamp: uint40(block.timestamp)
        }));

        emit Slashed(prover, modelId, roundId, slashAmount, s.amount, reason);
    }

    /// @notice Calculate challenger reward from slashed amount
    function _calculateChallengerReward(uint128 slashAmount) internal view returns (uint128 reward) {
        reward = uint128((uint256(slashAmount) * challengerRewardPercentage) / MAX_PERCENTAGE);

        // Apply min/max bounds
        if (reward < minChallengerReward) {
            reward = minChallengerReward;
        }
        if (reward > maxChallengerReward) {
            reward = maxChallengerReward;
        }
        // Cap to available slashed amount
        if (reward > slashAmount) {
            reward = slashAmount;
        }
    }

    /// @notice Allows anyone to challenge a past proof and receive reward
    function challengeProof(
        uint256 modelId,
        uint256 roundId,
        bytes memory proof,
        uint256[] memory publicInputs
    ) external nonReentrant {
        Round storage round = rounds[modelId][roundId];
        if (!round.isCompleted) revert RoundNotCompleted();
        if (round.prover == address(0)) revert NoProverToChallenge();
        if (msg.sender == round.prover) revert CannotChallengeSelf();

        bool valid = verifier.verifyProof(proof, publicInputs);

        if (!valid) {
            _slashWithChallenger(round.prover, modelId, roundId, "Fraudulent proof challenged", msg.sender);
        }
    }

    // ============ View Functions ============

    /// @notice Gets the current state of a model
    function getModelState(uint256 modelId) external view returns (
        uint256 currentRound,
        uint256 currentCommitment,
        bool active
    ) {
        Model storage model = models[modelId];
        return (model.currentRound, model.currentCommitment, model.active);
    }

    /// @notice Gets a prover's stake for a model
    function getStake(address prover, uint256 modelId) external view returns (
        uint256 amount,
        uint256 lockedUntil,
        bool slashed
    ) {
        Stake storage s = stakes[prover][modelId];
        return (s.amount, s.lockedUntil, s.slashed);
    }

    /// @notice Gets the number of slashing records
    function getSlashingRecordCount() external view returns (uint256) {
        return slashingRecords.length;
    }

    /// @notice Gets paginated slashing records to avoid unbounded iteration
    /// @param offset Starting index
    /// @param limit Maximum number of records to return
    /// @return records Array of slashing records
    /// @return total Total number of records
    function getSlashingRecords(uint256 offset, uint256 limit) external view returns (
        SlashingRecord[] memory records,
        uint256 total
    ) {
        total = slashingRecords.length;
        if (offset >= total) {
            return (new SlashingRecord[](0), total);
        }

        uint256 end = offset + limit;
        if (end > total) {
            end = total;
        }

        uint256 count = end - offset;
        records = new SlashingRecord[](count);
        for (uint256 i = 0; i < count; i++) {
            records[i] = slashingRecords[offset + i];
        }
    }

    /// @notice Gets the accumulated error bound for a model
    function getAccumulatedErrorBound(uint256 modelId) external view returns (uint256) {
        return accumulatedErrorBound[modelId];
    }

    /// @notice Checks if a model's accumulated error is within acceptable limits
    function isModelErrorAcceptable(
        uint256 modelId,
        uint256 maxAccumulated
    ) external view returns (bool) {
        return accumulatedErrorBound[modelId] <= maxAccumulated;
    }

    /// @notice Checks if a proof has already been used
    function isProofUsed(bytes memory proof, uint256[] memory publicInputs) external view returns (bool) {
        bytes32 proofHash = keccak256(abi.encodePacked(proof, publicInputs));
        return usedProofHashes[proofHash];
    }

    /// @notice Gets the effective stake lock period in seconds
    function stakeLockPeriod() external view returns (uint256) {
        return uint256(stakeLockDays) * 1 days;
    }

    // ============ Internal Helpers ============

    /// @notice Combines two 128-bit halves into a single commitment
    function _hashPair(uint256 lo, uint256 hi) internal pure returns (uint256) {
        return uint256(keccak256(abi.encodePacked(lo, hi)));
    }

    // ============ Admin Timelock Mechanism ============

    /// @notice Propose a timelocked admin change
    /// @param changeKey Unique identifier for the change type
    /// @param changeHash Hash of the change parameters for verification
    /// @param delay Timelock delay in seconds
    function _proposeChange(bytes32 changeKey, bytes32 changeHash, uint256 delay) internal {
        PendingChange storage pending = pendingChanges[changeKey];
        if (pending.active) revert TimelockAlreadyPending();

        pending.changeHash = changeHash;
        pending.executionTime = block.timestamp + delay;
        pending.active = true;

        emit AdminChangeProposed(changeKey, changeHash, pending.executionTime);
    }

    /// @notice Verify and clear a pending change
    /// @param changeKey Unique identifier for the change type
    /// @param changeHash Expected hash of the change parameters
    function _executeChange(bytes32 changeKey, bytes32 changeHash) internal {
        PendingChange storage pending = pendingChanges[changeKey];
        if (!pending.active) revert NoTimelockPending();
        if (block.timestamp < pending.executionTime) revert TimelockNotReady();
        if (pending.changeHash != changeHash) revert ErrorChecksumMismatch();

        pending.active = false;
        pending.changeHash = bytes32(0);
        pending.executionTime = 0;

        emit AdminChangeExecuted(changeKey);
    }

    /// @notice Cancel a pending admin change
    function cancelPendingChange(bytes32 changeKey) external onlyOwner {
        PendingChange storage pending = pendingChanges[changeKey];
        if (!pending.active) revert NoTimelockPending();

        pending.active = false;
        pending.changeHash = bytes32(0);
        pending.executionTime = 0;

        emit AdminChangeCancelled(changeKey);
    }

    /// @notice Get pending change info
    function getPendingChange(bytes32 changeKey) external view returns (
        bytes32 changeHash,
        uint256 executionTime,
        bool active
    ) {
        PendingChange storage pending = pendingChanges[changeKey];
        return (pending.changeHash, pending.executionTime, pending.active);
    }

    // ============ Timelocked Admin Functions ============

    /// @notice Propose treasury update (48-hour timelock)
    function proposeSetTreasury(address _treasury) external onlyOwner {
        if (_treasury == address(0)) revert InvalidTreasury();
        bytes32 key = keccak256("setTreasury");
        bytes32 hash = keccak256(abi.encode(_treasury));
        _proposeChange(key, hash, PARAM_TIMELOCK);
    }

    /// @notice Execute treasury update after timelock
    function executeSetTreasury(address _treasury) external onlyOwner {
        if (_treasury == address(0)) revert InvalidTreasury();
        bytes32 key = keccak256("setTreasury");
        bytes32 hash = keccak256(abi.encode(_treasury));
        _executeChange(key, hash);

        emit ConfigUpdated("treasury", uint256(uint160(treasury)), uint256(uint160(_treasury)));
        treasury = _treasury;
    }

    /// @notice Propose slash percentage update (48-hour timelock)
    function proposeSetSlashPercentage(uint256 _percentage) external onlyOwner {
        if (_percentage > MAX_PERCENTAGE) revert MaxPercentage();
        bytes32 key = keccak256("setSlashPercentage");
        bytes32 hash = keccak256(abi.encode(_percentage));
        _proposeChange(key, hash, PARAM_TIMELOCK);
    }

    /// @notice Execute slash percentage update after timelock
    function executeSetSlashPercentage(uint256 _percentage) external onlyOwner {
        bytes32 key = keccak256("setSlashPercentage");
        bytes32 hash = keccak256(abi.encode(_percentage));
        _executeChange(key, hash);

        emit ConfigUpdated("slashPercentage", slashPercentage, _percentage);
        slashPercentage = uint16(_percentage);
    }

    /// @notice Propose default min stake update (48-hour timelock)
    function proposeSetDefaultMinStake(uint256 _minStake) external onlyOwner {
        bytes32 key = keccak256("setDefaultMinStake");
        bytes32 hash = keccak256(abi.encode(_minStake));
        _proposeChange(key, hash, PARAM_TIMELOCK);
    }

    /// @notice Execute default min stake update after timelock
    function executeSetDefaultMinStake(uint256 _minStake) external onlyOwner {
        bytes32 key = keccak256("setDefaultMinStake");
        bytes32 hash = keccak256(abi.encode(_minStake));
        _executeChange(key, hash);

        emit ConfigUpdated("defaultMinStake", defaultMinStake, _minStake);
        defaultMinStake = _minStake;
    }

    /// @notice Propose max error bound update (48-hour timelock)
    function proposeSetMaxErrorBound(uint256 _maxErrorBound) external onlyOwner {
        bytes32 key = keccak256("setMaxErrorBound");
        bytes32 hash = keccak256(abi.encode(_maxErrorBound));
        _proposeChange(key, hash, PARAM_TIMELOCK);
    }

    /// @notice Execute max error bound update after timelock
    function executeSetMaxErrorBound(uint256 _maxErrorBound) external onlyOwner {
        bytes32 key = keccak256("setMaxErrorBound");
        bytes32 hash = keccak256(abi.encode(_maxErrorBound));
        _executeChange(key, hash);

        emit ConfigUpdated("maxErrorBound", maxErrorBound, _maxErrorBound);
        maxErrorBound = _maxErrorBound;
    }

    /// @notice Propose challenger config update (48-hour timelock)
    function proposeSetChallengerConfig(
        uint16 _rewardPercentage,
        uint128 _minReward,
        uint128 _maxReward,
        bool _enabled
    ) external onlyOwner {
        if (_rewardPercentage > MAX_REWARD_PERCENTAGE) revert MaxRewardPercentage();
        if (_minReward > _maxReward) revert MinExceedsMax();
        bytes32 key = keccak256("setChallengerConfig");
        bytes32 hash = keccak256(abi.encode(_rewardPercentage, _minReward, _maxReward, _enabled));
        _proposeChange(key, hash, PARAM_TIMELOCK);
    }

    /// @notice Execute challenger config update after timelock
    function executeSetChallengerConfig(
        uint16 _rewardPercentage,
        uint128 _minReward,
        uint128 _maxReward,
        bool _enabled
    ) external onlyOwner {
        bytes32 key = keccak256("setChallengerConfig");
        bytes32 hash = keccak256(abi.encode(_rewardPercentage, _minReward, _maxReward, _enabled));
        _executeChange(key, hash);

        challengerRewardPercentage = _rewardPercentage;
        minChallengerReward = _minReward;
        maxChallengerReward = _maxReward;
        challengerRewardsEnabled = _enabled;

        emit ChallengerConfigUpdated(_rewardPercentage, _minReward, _maxReward, _enabled);
    }

    // ============ Emergency-Only Admin Functions (bypass timelocks only when paused) ============
    // NOTE: These immediate setters bypass timelocks but ONLY work during emergency pause.
    // In normal operation, use the timelocked propose/execute functions above.

    /// @notice Emergency treasury update (only when paused)
    function setTreasury(address _treasury) external onlyOwner whenPaused {
        if (_treasury == address(0)) revert InvalidTreasury();
        emit ConfigUpdated("treasury", uint256(uint160(treasury)), uint256(uint160(_treasury)));
        treasury = _treasury;
    }

    /// @notice Emergency slash percentage update (only when paused)
    function setSlashPercentage(uint256 _percentage) external onlyOwner whenPaused {
        if (_percentage > MAX_PERCENTAGE) revert MaxPercentage();
        emit ConfigUpdated("slashPercentage", slashPercentage, _percentage);
        slashPercentage = uint16(_percentage);
    }

    /// @notice Emergency default min stake update (only when paused)
    function setDefaultMinStake(uint256 _minStake) external onlyOwner whenPaused {
        emit ConfigUpdated("defaultMinStake", defaultMinStake, _minStake);
        defaultMinStake = _minStake;
    }

    /// @notice Emergency max error bound update (only when paused)
    function setMaxErrorBound(uint256 _maxErrorBound) external onlyOwner whenPaused {
        emit ConfigUpdated("maxErrorBound", maxErrorBound, _maxErrorBound);
        maxErrorBound = _maxErrorBound;
    }

    /// @notice Resets accumulated error for a model
    function resetAccumulatedError(uint256 modelId) external {
        if (msg.sender != models[modelId].owner && msg.sender != owner) revert NotAuthorized();
        accumulatedErrorBound[modelId] = 0;
    }

    /// @notice Pauses a model
    function pauseModel(uint256 modelId) external {
        if (msg.sender != models[modelId].owner && msg.sender != owner) revert NotAuthorized();
        models[modelId].active = false;
        emit ModelStateChanged(modelId, false, msg.sender);
    }

    /// @notice Resumes a model
    function resumeModel(uint256 modelId) external {
        if (msg.sender != models[modelId].owner && msg.sender != owner) revert NotAuthorized();
        models[modelId].active = true;
        emit ModelStateChanged(modelId, true, msg.sender);
    }

    // ============ Emergency Pause ============

    /// @notice Emergency pause - stops all critical operations (single owner)
    function emergencyPause() external onlyOwner {
        if (paused) revert AlreadyPaused();
        paused = true;
        emit EmergencyPauseChanged(true, msg.sender);
    }

    /// @notice Unpause the contract
    function unpause() external onlyOwner {
        if (!paused) revert NotPaused();
        paused = false;
        emit EmergencyPauseChanged(false, msg.sender);
    }

    // ============ Multi-Sig Emergency Pause ============

    /// @notice Guardian approves emergency pause
    function approveEmergencyPause() external onlyGuardian {
        uint64 currentNonce = pauseNonce;
        if (pauseApprovals[currentNonce][msg.sender]) revert AlreadyApproved();
        if (paused) revert AlreadyPaused();

        pauseApprovals[currentNonce][msg.sender] = true;
        pauseApprovalCount[currentNonce]++;

        emit PauseApprovalReceived(
            currentNonce,
            msg.sender,
            pauseApprovalCount[currentNonce],
            requiredGuardians
        );

        // Execute pause if threshold reached
        if (pauseApprovalCount[currentNonce] >= requiredGuardians) {
            _executeMultiSigPause(currentNonce);
        }
    }

    /// @notice Internal function to execute multi-sig pause
    function _executeMultiSigPause(uint64 nonce) internal {
        paused = true;
        pauseNonce++; // Increment nonce for next pause

        address[] memory emptyApprovers;

        emit MultiSigPauseExecuted(nonce, emptyApprovers);
        emit EmergencyPauseChanged(true, address(this));
    }

    /// @notice Cancel pending pause approval
    function cancelPauseApproval() external onlyGuardian {
        uint64 currentNonce = pauseNonce;
        if (!pauseApprovals[currentNonce][msg.sender]) revert NoApprovalToCancel();

        pauseApprovals[currentNonce][msg.sender] = false;
        pauseApprovalCount[currentNonce]--;
    }

    // ============ Time-Locked Recovery ============

    /// @notice Initiate ownership recovery (requires multi-sig)
    function initiateRecovery(address newOwner) external onlyGuardian {
        if (newOwner == address(0)) revert InvalidNewOwner();
        if (pendingRecoveryTime != 0) revert RecoveryAlreadyPending();
        if (!paused) revert ContractMustBePaused();

        pendingRecoveryTime = block.timestamp + recoveryTimeLock;
        pendingRecoveryOwner = newOwner;

        emit RecoveryInitiated(newOwner, pendingRecoveryTime, msg.sender);
    }

    /// @notice Execute recovery after time lock
    function executeRecovery() external {
        if (pendingRecoveryTime == 0) revert NoRecoveryPending();
        if (block.timestamp < pendingRecoveryTime) revert TimeLockNotExpired();
        if (msg.sender != pendingRecoveryOwner && !isGuardian[msg.sender]) revert NotAuthorized();

        address oldOwner = owner;
        owner = pendingRecoveryOwner;

        // Clear recovery state
        pendingRecoveryTime = 0;
        pendingRecoveryOwner = address(0);

        emit RecoveryExecuted(oldOwner, owner);
    }

    /// @notice Cancel pending recovery
    function cancelRecovery() external {
        if (pendingRecoveryTime == 0) revert NoRecoveryPending();
        if (msg.sender != owner && !isGuardian[msg.sender]) revert NotAuthorized();

        pendingRecoveryTime = 0;
        pendingRecoveryOwner = address(0);

        emit RecoveryCancelled(msg.sender);
    }

    // ============ Guardian Management ============

    /// @notice Add a guardian
    function addGuardian(address guardian) external onlyOwner {
        if (guardian == address(0)) revert InvalidAddress();
        if (isGuardian[guardian]) revert AlreadyGuardian();

        isGuardian[guardian] = true;
        guardianCount++;

        emit GuardianUpdated(guardian, true, msg.sender);
    }

    /// @notice Remove a guardian
    function removeGuardian(address guardian) external onlyOwner {
        if (!isGuardian[guardian]) revert NotGuardianRole();
        if (guardianCount <= requiredGuardians) revert BelowThreshold();

        isGuardian[guardian] = false;
        guardianCount--;

        emit GuardianUpdated(guardian, false, msg.sender);
    }

    /// @notice Update required guardians threshold
    function setRequiredGuardians(uint8 _required) external onlyOwner {
        if (_required == 0) revert RequireAtLeastOne();
        if (_required > guardianCount) revert ExceedsGuardianCount();
        requiredGuardians = _required;
    }

    /// @notice Update recovery time lock
    function setRecoveryTimeLock(uint256 _timeLock) external onlyOwner {
        if (_timeLock < MIN_RECOVERY_TIMELOCK) revert MinimumTimeLock();
        if (_timeLock > MAX_RECOVERY_TIMELOCK) revert MaximumTimeLock();
        recoveryTimeLock = _timeLock;
    }

    // ============ Challenger Reward Configuration ============

    /// @notice Update challenger reward configuration (DEPRECATED: use proposeSetChallengerConfig/executeSetChallengerConfig)
    function setChallengerConfig(
        uint16 _rewardPercentage,
        uint128 _minReward,
        uint128 _maxReward,
        bool _enabled
    ) external onlyOwner {
        if (_rewardPercentage > MAX_REWARD_PERCENTAGE) revert MaxRewardPercentage();
        if (_minReward > _maxReward) revert MinExceedsMax();

        challengerRewardPercentage = _rewardPercentage;
        minChallengerReward = _minReward;
        maxChallengerReward = _maxReward;
        challengerRewardsEnabled = _enabled;

        emit ChallengerConfigUpdated(_rewardPercentage, _minReward, _maxReward, _enabled);
    }

    /// @notice Get challenger reward configuration
    function getChallengerConfig() external view returns (
        uint16 rewardPercentage,
        uint128 minReward,
        uint128 maxReward,
        bool enabled
    ) {
        return (
            challengerRewardPercentage,
            minChallengerReward,
            maxChallengerReward,
            challengerRewardsEnabled
        );
    }

    /// @notice Check if address is a guardian
    function getGuardianStatus(address guardian) external view returns (bool) {
        return isGuardian[guardian];
    }

    /// @notice Get pending recovery info
    function getPendingRecovery() external view returns (
        address newOwner,
        uint256 executionTime,
        bool isPending
    ) {
        return (
            pendingRecoveryOwner,
            pendingRecoveryTime,
            pendingRecoveryTime != 0
        );
    }

    /// @notice Get pause approval status
    function getPauseApprovalStatus() external view returns (
        uint64 currentNonce,
        uint8 currentApprovals,
        uint8 required
    ) {
        return (
            pauseNonce,
            pauseApprovalCount[pauseNonce],
            requiredGuardians
        );
    }

    // ============ Data Commitment Functions ============

    /// @notice Set model registry address (optional integration)
    function setModelRegistry(address _registry) external onlyOwner {
        emit ModelRegistryUpdated(address(modelRegistry), _registry);
        modelRegistry = ModelRegistry(_registry);
    }

    /// @notice Set data commitment contract address
    function setDataCommitmentContract(address _dataCommitment) external onlyOwner {
        emit DataCommitmentContractUpdated(dataCommitment, _dataCommitment);
        dataCommitment = _dataCommitment;
    }

    /// @notice Set slashing evidence contract address
    function setSlashingEvidenceContract(address _slashingEvidence) external onlyOwner {
        slashingEvidence = _slashingEvidence;
    }

    /// @notice Set required data commitment for a model
    function setModelDataCommitment(
        uint256 modelId,
        bytes32 dataRoot
    ) external modelExists(modelId) {
        if (msg.sender != models[modelId].owner && msg.sender != owner) revert NotAuthorized();
        modelDataCommitment[modelId] = dataRoot;
        emit DataCommitmentSet(modelId, dataRoot, msg.sender);
    }

    /// @notice Commit data root for a specific round
    /// @dev Restricted to model owner, contract owner, or dataCommitment contract
    function commitRoundData(
        uint256 modelId,
        uint256 roundId,
        bytes32 dataRoot
    ) external whenNotPaused modelExists(modelId) {
        if (msg.sender != models[modelId].owner && msg.sender != owner && msg.sender != dataCommitment) revert NotAuthorized();
        if (roundId != models[modelId].currentRound) revert InvalidRound();
        if (dataRoot == bytes32(0)) revert InvalidDataRoot();

        roundDataRoot[modelId][roundId] = dataRoot;
        emit RoundDataCommitted(modelId, roundId, dataRoot);
    }

    /// @notice Get the data commitment for a model
    function getModelDataCommitment(uint256 modelId) external view returns (bytes32) {
        return modelDataCommitment[modelId];
    }

    /// @notice Get the data root for a specific round
    function getRoundDataRoot(uint256 modelId, uint256 roundId) external view returns (bytes32) {
        return roundDataRoot[modelId][roundId];
    }

    /// @notice Check if a model has a required data commitment
    function hasDataCommitment(uint256 modelId) external view returns (bool) {
        return modelDataCommitment[modelId] != bytes32(0);
    }

    /// @notice Verify that a round's data matches the model's required data
    function verifyRoundDataCommitment(
        uint256 modelId,
        uint256 roundId
    ) external view returns (bool) {
        bytes32 modelData = modelDataCommitment[modelId];
        bytes32 roundData = roundDataRoot[modelId][roundId];

        // If no model data commitment required, always valid
        if (modelData == bytes32(0)) {
            return true;
        }

        // Round must have committed data
        if (roundData == bytes32(0)) {
            return false;
        }

        // For now, just check that data was committed
        // In production, this would verify the round data is derived from model data
        return true;
    }

    /// @notice Receives ETH
    receive() external payable {}
}
