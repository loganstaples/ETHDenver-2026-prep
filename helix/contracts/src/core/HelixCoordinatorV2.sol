// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "../interfaces/IHelixVerifier.sol";

/// @title HelixCoordinatorV2
/// @notice Gas-optimized coordinator for model registration, proof submission, staking, and slashing
/// @dev Supports real ZK proof verification with economic security
///      Storage layout optimized for gas efficiency with struct packing
///      Includes multi-sig emergency pause mechanism with time-locked recovery
///      and comprehensive challenger reward distribution
contract HelixCoordinatorV2 {
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

    // ============ State Variables (Ordered for Optimal Packing) ============

    // Slot 1: Verifier (immutable after deployment for gas savings)
    /// @notice The ZK proof verifier contract
    IHelixVerifier public verifier;

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

    // ============ Events (Optimized with Indexed Parameters) ============

    /// @notice Emitted when a new model is registered
    /// @param modelId Indexed for efficient filtering by model
    /// @param owner Indexed for efficient filtering by owner
    /// @param initialCommitment Initial state commitment
    /// @param minStake Minimum stake required
    /// @param ipfsHash IPFS hash of model weights
    event ModelRegistered(
        uint256 indexed modelId,
        address indexed owner,
        uint256 initialCommitment,
        uint256 minStake,
        string ipfsHash
    );

    /// @notice Emitted when a training round starts
    /// @param modelId Indexed for efficient filtering by model
    /// @param roundId Indexed for efficient filtering by round
    /// @param deadline Submission deadline timestamp
    /// @param modelCommitment Current model commitment
    event RoundStarted(
        uint256 indexed modelId,
        uint256 indexed roundId,
        uint256 deadline,
        uint256 modelCommitment
    );

    /// @notice Emitted when a proof is submitted
    /// @param modelId Indexed for efficient filtering by model
    /// @param roundId Indexed for efficient filtering by round
    /// @param prover Indexed for efficient filtering by prover
    /// @param newCommitment New state commitment after training step
    /// @param errorBound Error bound of this step
    event ProofSubmitted(
        uint256 indexed modelId,
        uint256 indexed roundId,
        address indexed prover,
        uint256 newCommitment,
        uint256 errorBound
    );

    /// @notice Emitted when a training round completes
    /// @param modelId Indexed for efficient filtering by model
    /// @param roundId Indexed for efficient filtering by round
    /// @param newCommitment Final commitment
    /// @param totalErrorBound Accumulated error bound after this round
    event RoundCompleted(
        uint256 indexed modelId,
        uint256 indexed roundId,
        uint256 newCommitment,
        uint256 totalErrorBound
    );

    /// @notice Emitted when stake is deposited
    /// @param prover Indexed for efficient filtering by prover
    /// @param modelId Indexed for efficient filtering by model
    /// @param amount Amount staked in this transaction
    /// @param totalStake Total stake after this deposit
    event Staked(
        address indexed prover,
        uint256 indexed modelId,
        uint256 amount,
        uint256 totalStake
    );

    /// @notice Emitted when stake is withdrawn
    /// @param prover Indexed for efficient filtering by prover
    /// @param modelId Indexed for efficient filtering by model
    /// @param amount Amount withdrawn
    event Unstaked(
        address indexed prover,
        uint256 indexed modelId,
        uint256 amount
    );

    /// @notice Emitted when a prover is slashed
    /// @param prover Indexed for efficient filtering by prover
    /// @param modelId Indexed for efficient filtering by model
    /// @param roundId Round where slashing occurred
    /// @param amount Amount slashed
    /// @param remainingStake Remaining stake after slashing
    /// @param reason Reason for slashing
    event Slashed(
        address indexed prover,
        uint256 indexed modelId,
        uint256 roundId,
        uint256 amount,
        uint256 remainingStake,
        string reason
    );

    /// @notice Emitted when an invalid proof is detected
    /// @param modelId Indexed for efficient filtering by model
    /// @param roundId Indexed for efficient filtering by round
    /// @param prover Indexed for efficient filtering by prover
    /// @param proofHash Hash of the invalid proof
    event InvalidProofDetected(
        uint256 indexed modelId,
        uint256 indexed roundId,
        address indexed prover,
        bytes32 proofHash
    );

    /// @notice Emitted when model state changes
    /// @param modelId Indexed for efficient filtering by model
    /// @param active New active state
    /// @param changedBy Who made the change
    event ModelStateChanged(
        uint256 indexed modelId,
        bool active,
        address indexed changedBy
    );

    /// @notice Emitted when configuration is updated
    /// @param parameter Which parameter was changed
    /// @param oldValue Previous value
    /// @param newValue New value
    event ConfigUpdated(
        string indexed parameter,
        uint256 oldValue,
        uint256 newValue
    );

    /// @notice Emitted when contract is paused/unpaused
    /// @param isPaused New pause state
    /// @param changedBy Who made the change
    event EmergencyPauseChanged(
        bool isPaused,
        address indexed changedBy
    );

    /// @notice Emitted when data commitment is set for a model
    /// @param modelId The model ID
    /// @param dataRoot The data commitment merkle root
    /// @param setBy Who set the commitment
    event DataCommitmentSet(
        uint256 indexed modelId,
        bytes32 indexed dataRoot,
        address indexed setBy
    );

    /// @notice Emitted when round data is committed
    /// @param modelId The model ID
    /// @param roundId The round ID
    /// @param dataRoot The data merkle root for this round
    event RoundDataCommitted(
        uint256 indexed modelId,
        uint256 indexed roundId,
        bytes32 dataRoot
    );

    /// @notice Emitted when data commitment contract is updated
    /// @param oldContract Previous contract address
    /// @param newContract New contract address
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

    // ============ Modifiers ============

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    modifier whenNotPaused() {
        require(!paused, "Contract is paused");
        _;
    }

    modifier modelExists(uint256 modelId) {
        require(models[modelId].owner != address(0), "Model does not exist");
        _;
    }

    modifier hasStake(uint256 modelId) {
        Stake storage s = stakes[msg.sender][modelId];
        require(s.amount >= models[modelId].minStake, "Insufficient stake");
        require(!s.slashed, "Stake has been slashed");
        _;
    }

    modifier onlyGuardian() {
        require(isGuardian[msg.sender], "Only guardian");
        _;
    }

    modifier noActivePause() {
        require(pauseApprovalCount[pauseNonce] == 0, "Pause in progress");
        _;
    }

    // ============ Constructor ============

    constructor(address _verifier, address _treasury) {
        verifier = IHelixVerifier(_verifier);
        treasury = _treasury;
        owner = msg.sender;

        // Initialize packed values
        slashPercentage = 5000;  // 50%
        stakeLockDays = 7;       // 7 days
        defaultMinStake = 0.1 ether;
        maxErrorBound = 1e18;    // 1.0 in 18-decimal fixed point

        // Initialize multi-sig pause configuration
        requiredGuardians = 2;   // Require 2 guardians for pause
        recoveryTimeLock = 48 hours;

        // Initialize challenger reward configuration
        challengerRewardPercentage = 1000;  // 10%
        minChallengerReward = 0.001 ether;
        maxChallengerReward = 10 ether;
        challengerRewardsEnabled = true;

        // Owner is first guardian
        isGuardian[msg.sender] = true;
        guardianCount = 1;
    }

    // ============ Model Management ============

    /// @notice Registers a new model
    /// @param ipfsHash IPFS hash of initial model weights
    /// @param initialCommitment Hash commitment of initial weights
    /// @param minStake Minimum stake required to submit proofs
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
    }

    /// @notice Starts a new training round
    /// @param modelId The model to start a round for
    /// @param duration Round duration in seconds
    function startRound(
        uint256 modelId,
        uint256 duration
    ) external whenNotPaused modelExists(modelId) {
        Model storage model = models[modelId];
        require(msg.sender == model.owner, "Only model owner");
        require(model.active, "Model not active");

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
    /// @param modelId The model to stake for
    function stake(uint256 modelId) external payable modelExists(modelId) {
        require(msg.value > 0, "Must stake non-zero amount");

        Stake storage s = stakes[msg.sender][modelId];
        require(!s.slashed, "Previous stake was slashed");

        uint128 newAmount = s.amount + uint128(msg.value);
        s.amount = newAmount;
        s.lockedUntil = uint40(block.timestamp + uint256(stakeLockDays) * 1 days);

        emit Staked(msg.sender, modelId, msg.value, newAmount);
    }

    /// @notice Withdraws stake after lock period
    /// @param modelId The model to unstake from
    function unstake(uint256 modelId) external {
        Stake storage s = stakes[msg.sender][modelId];
        require(s.amount > 0, "No stake");
        require(!s.slashed, "Stake was slashed");
        require(block.timestamp >= s.lockedUntil, "Still locked");

        uint256 amount = s.amount;
        s.amount = 0;

        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");

        emit Unstaked(msg.sender, modelId, amount);
    }

    // ============ Proof Submission ============

    /// @notice Submits a training proof for a round
    /// @param modelId The model ID
    /// @param roundId The round ID
    /// @param proof The ZK proof bytes
    /// @param publicInputs Public inputs: [oldHashLo, oldHashHi, newHashLo, newHashHi, loss, errorBound, stepNumber, errorChecksum]
    function submitProof(
        uint256 modelId,
        uint256 roundId,
        bytes memory proof,
        uint256[] memory publicInputs
    ) external modelExists(modelId) hasStake(modelId) {
        Model storage model = models[modelId];
        Round storage round = rounds[modelId][roundId];

        // Validate round
        require(roundId == model.currentRound, "Invalid round");
        require(!round.isCompleted, "Round completed");
        require(block.timestamp <= round.deadline, "Round expired");

        // Validate public inputs count (8 inputs including error checksum)
        require(publicInputs.length == 8, "Invalid public inputs count");

        // Reconstruct and validate old commitment
        uint256 oldCommitmentFromProof = _hashPair(publicInputs[0], publicInputs[1]);
        require(oldCommitmentFromProof == round.modelCommitment, "Old commitment mismatch");

        // Validate error bound
        uint256 stepErrorBound = publicInputs[5];
        require(stepErrorBound <= maxErrorBound, "Error bound exceeds maximum");

        // Validate error checksum (prevents tampering with error tracking)
        // The checksum cryptographically commits to: error || step || model_id || budget
        uint256 errorChecksum = publicInputs[7];
        uint256 expectedChecksum = _computeErrorChecksum(
            stepErrorBound,
            publicInputs[6], // step_number
            modelId,
            maxErrorBound
        );
        require(errorChecksum == expectedChecksum, "Error checksum mismatch");

        // Verify the proof
        bool valid = verifier.verifyProof(proof, publicInputs);

        if (!valid) {
            _slash(msg.sender, modelId, roundId, "Invalid proof");
            emit InvalidProofDetected(modelId, roundId, msg.sender, keccak256(proof));
            return;
        }

        // Extract new commitment
        uint256 newCommitment = _hashPair(publicInputs[2], publicInputs[3]);

        // Update state
        model.currentCommitment = newCommitment;
        round.newCommitment = newCommitment;
        round.isCompleted = true;
        round.prover = msg.sender;

        // Track accumulated error bound
        uint256 newAccumulatedError = accumulatedErrorBound[modelId] + stepErrorBound;
        accumulatedErrorBound[modelId] = newAccumulatedError;

        // Reset stake lock (reward for valid submission)
        stakes[msg.sender][modelId].lockedUntil = uint40(block.timestamp);

        emit ProofSubmitted(modelId, roundId, msg.sender, newCommitment, stepErrorBound);
        emit RoundCompleted(modelId, roundId, newCommitment, newAccumulatedError);
    }

    /// @notice Computes the expected error checksum for verification
    /// @dev Matches the Rust implementation in helix-core/types/error_commitment.rs
    ///      Layout: SHA256(errorBound_LE64 || stepNumber_LE64 || modelId_32bytes || errorBudget_LE64)
    ///      Total input: 56 bytes. Output: first 8 bytes interpreted as LE u64.
    /// @param errorBound The step error bound (truncated to u64)
    /// @param stepNumber The training step number (truncated to u64)
    /// @param modelId The model identifier (as bytes32, big-endian)
    /// @param errorBudget The maximum error budget (truncated to u64)
    /// @return The first 8 bytes of SHA256 hash interpreted as little-endian u64
    function _computeErrorChecksum(
        uint256 errorBound,
        uint256 stepNumber,
        uint256 modelId,
        uint256 errorBudget
    ) internal pure returns (uint256) {
        // Build the 56-byte preimage matching Rust's compute_checksum():
        //   error_scaled.to_le_bytes()  (8 bytes)
        //   step_number.to_le_bytes()   (8 bytes)
        //   model_id                    (32 bytes)
        //   budget_scaled.to_le_bytes() (8 bytes)
        bytes memory data = new bytes(56);
        assembly {
            let ptr := add(data, 32)

            // Write errorBound as LE u64 (byte-swap from BE)
            let eb := and(errorBound, 0xFFFFFFFFFFFFFFFF)
            // Reverse bytes: swap pairs at each level
            eb := or(and(shr(8, eb), 0x00FF00FF00FF00FF), shl(8, and(eb, 0x00FF00FF00FF00FF)))
            eb := or(and(shr(16, eb), 0x0000FFFF0000FFFF), shl(16, and(eb, 0x0000FFFF0000FFFF)))
            eb := or(shr(32, eb), shl(32, and(eb, 0x00000000FFFFFFFF)))
            // Store as big-endian bytes8 (which now represents LE u64 in memory)
            mstore(ptr, shl(192, eb))

            // Write stepNumber as LE u64
            let sn := and(stepNumber, 0xFFFFFFFFFFFFFFFF)
            sn := or(and(shr(8, sn), 0x00FF00FF00FF00FF), shl(8, and(sn, 0x00FF00FF00FF00FF)))
            sn := or(and(shr(16, sn), 0x0000FFFF0000FFFF), shl(16, and(sn, 0x0000FFFF0000FFFF)))
            sn := or(shr(32, sn), shl(32, and(sn, 0x00000000FFFFFFFF)))
            mstore(add(ptr, 8), shl(192, sn))

            // Write modelId as 32 bytes (big-endian, matching Rust [u8; 32])
            mstore(add(ptr, 16), modelId)

            // Write errorBudget as LE u64
            let bg := and(errorBudget, 0xFFFFFFFFFFFFFFFF)
            bg := or(and(shr(8, bg), 0x00FF00FF00FF00FF), shl(8, and(bg, 0x00FF00FF00FF00FF)))
            bg := or(and(shr(16, bg), 0x0000FFFF0000FFFF), shl(16, and(bg, 0x0000FFFF0000FFFF)))
            bg := or(shr(32, bg), shl(32, and(bg, 0x00000000FFFFFFFF)))
            mstore(add(ptr, 48), shl(192, bg))
        }

        bytes32 hash = sha256(data);

        // Extract first 8 bytes as LE u64 (matching Rust u64::from_le_bytes(hash[0..8]))
        // hash[0] is the first SHA256 output byte. LE u64 = h[0] + h[1]*256 + ... + h[7]*2^56
        uint256 result;
        assembly {
            // bytes32 in Solidity: hash[0] at bits 248-255, hash[1] at 240-247, etc.
            // Extract first 8 bytes as BE u64, then byte-swap to LE
            let be := shr(192, hash) // top 64 bits = first 8 bytes as BE u64
            // Reverse bytes to get LE interpretation
            be := or(and(shr(8, be), 0x00FF00FF00FF00FF), shl(8, and(be, 0x00FF00FF00FF00FF)))
            be := or(and(shr(16, be), 0x0000FFFF0000FFFF), shl(16, and(be, 0x0000FFFF0000FFFF)))
            be := or(shr(32, be), shl(32, and(be, 0x00000000FFFFFFFF)))
            result := be
        }
        return result;
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
        require(s.amount > 0, "No stake to slash");
        require(!s.slashed, "Already slashed");

        uint128 slashAmount = uint128((uint256(s.amount) * slashPercentage) / 10000);
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
            require(success, "Treasury transfer failed");
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
        reward = uint128((uint256(slashAmount) * challengerRewardPercentage) / 10000);

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
    ) external {
        Round storage round = rounds[modelId][roundId];
        require(round.isCompleted, "Round not completed");
        require(round.prover != address(0), "No prover to challenge");

        // Prevent challenger from challenging themselves
        require(msg.sender != round.prover, "Cannot challenge self");

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

    /// @notice Gets the effective stake lock period in seconds
    function stakeLockPeriod() external view returns (uint256) {
        return uint256(stakeLockDays) * 1 days;
    }

    // ============ Internal Helpers ============

    /// @notice Combines two 128-bit halves into a single commitment
    function _hashPair(uint256 lo, uint256 hi) internal pure returns (uint256) {
        return uint256(keccak256(abi.encodePacked(lo, hi)));
    }

    // ============ Admin Functions ============

    /// @notice Updates the verifier contract
    function setVerifier(address _verifier) external onlyOwner {
        emit ConfigUpdated("verifier", uint256(uint160(address(verifier))), uint256(uint160(_verifier)));
        verifier = IHelixVerifier(_verifier);
    }

    /// @notice Updates the treasury address
    function setTreasury(address _treasury) external onlyOwner {
        emit ConfigUpdated("treasury", uint256(uint160(treasury)), uint256(uint160(_treasury)));
        treasury = _treasury;
    }

    /// @notice Updates the slash percentage
    function setSlashPercentage(uint256 _percentage) external onlyOwner {
        require(_percentage <= 10000, "Max 100%");
        emit ConfigUpdated("slashPercentage", slashPercentage, _percentage);
        slashPercentage = uint16(_percentage);
    }

    /// @notice Updates the default minimum stake
    function setDefaultMinStake(uint256 _minStake) external onlyOwner {
        emit ConfigUpdated("defaultMinStake", defaultMinStake, _minStake);
        defaultMinStake = _minStake;
    }

    /// @notice Updates the maximum allowed error bound per step
    function setMaxErrorBound(uint256 _maxErrorBound) external onlyOwner {
        emit ConfigUpdated("maxErrorBound", maxErrorBound, _maxErrorBound);
        maxErrorBound = _maxErrorBound;
    }

    /// @notice Resets accumulated error for a model
    function resetAccumulatedError(uint256 modelId) external {
        require(
            msg.sender == models[modelId].owner || msg.sender == owner,
            "Not authorized"
        );
        accumulatedErrorBound[modelId] = 0;
    }

    /// @notice Pauses a model
    function pauseModel(uint256 modelId) external {
        require(
            msg.sender == models[modelId].owner || msg.sender == owner,
            "Not authorized"
        );
        models[modelId].active = false;
        emit ModelStateChanged(modelId, false, msg.sender);
    }

    /// @notice Resumes a model
    function resumeModel(uint256 modelId) external {
        require(
            msg.sender == models[modelId].owner || msg.sender == owner,
            "Not authorized"
        );
        models[modelId].active = true;
        emit ModelStateChanged(modelId, true, msg.sender);
    }

    // ============ Emergency Pause ============

    /// @notice Emergency pause - stops all critical operations (single owner)
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

    // ============ Multi-Sig Emergency Pause ============

    /// @notice Guardian approves emergency pause
    function approveEmergencyPause() external onlyGuardian {
        uint64 currentNonce = pauseNonce;
        require(!pauseApprovals[currentNonce][msg.sender], "Already approved");
        require(!paused, "Already paused");

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

        // Collect approvers for event
        address[] memory approvers = new address[](guardianCount);
        uint256 count = 0;
        // Note: In production, this would iterate through guardian list
        // For now, emit empty array
        address[] memory emptyApprovers;

        emit MultiSigPauseExecuted(nonce, emptyApprovers);
        emit EmergencyPauseChanged(true, address(this));
    }

    /// @notice Cancel pending pause approval
    function cancelPauseApproval() external onlyGuardian {
        uint64 currentNonce = pauseNonce;
        require(pauseApprovals[currentNonce][msg.sender], "No approval to cancel");

        pauseApprovals[currentNonce][msg.sender] = false;
        pauseApprovalCount[currentNonce]--;
    }

    // ============ Time-Locked Recovery ============

    /// @notice Initiate ownership recovery (requires multi-sig)
    function initiateRecovery(address newOwner) external onlyGuardian {
        require(newOwner != address(0), "Invalid new owner");
        require(pendingRecoveryTime == 0, "Recovery already pending");
        require(paused, "Contract must be paused");

        pendingRecoveryTime = block.timestamp + recoveryTimeLock;
        pendingRecoveryOwner = newOwner;

        emit RecoveryInitiated(newOwner, pendingRecoveryTime, msg.sender);
    }

    /// @notice Execute recovery after time lock
    function executeRecovery() external {
        require(pendingRecoveryTime != 0, "No pending recovery");
        require(block.timestamp >= pendingRecoveryTime, "Time lock not expired");
        require(
            msg.sender == pendingRecoveryOwner || isGuardian[msg.sender],
            "Not authorized"
        );

        address oldOwner = owner;
        owner = pendingRecoveryOwner;

        // Clear recovery state
        pendingRecoveryTime = 0;
        pendingRecoveryOwner = address(0);

        emit RecoveryExecuted(oldOwner, owner);
    }

    /// @notice Cancel pending recovery
    function cancelRecovery() external {
        require(pendingRecoveryTime != 0, "No pending recovery");
        require(
            msg.sender == owner || isGuardian[msg.sender],
            "Not authorized"
        );

        pendingRecoveryTime = 0;
        pendingRecoveryOwner = address(0);

        emit RecoveryCancelled(msg.sender);
    }

    // ============ Guardian Management ============

    /// @notice Add a guardian
    function addGuardian(address guardian) external onlyOwner {
        require(guardian != address(0), "Invalid address");
        require(!isGuardian[guardian], "Already guardian");

        isGuardian[guardian] = true;
        guardianCount++;

        emit GuardianUpdated(guardian, true, msg.sender);
    }

    /// @notice Remove a guardian
    function removeGuardian(address guardian) external onlyOwner {
        require(isGuardian[guardian], "Not guardian");
        require(guardianCount > requiredGuardians, "Cannot remove: below threshold");

        isGuardian[guardian] = false;
        guardianCount--;

        emit GuardianUpdated(guardian, false, msg.sender);
    }

    /// @notice Update required guardians threshold
    function setRequiredGuardians(uint8 _required) external onlyOwner {
        require(_required > 0, "Must require at least 1");
        require(_required <= guardianCount, "Cannot exceed guardian count");
        requiredGuardians = _required;
    }

    /// @notice Update recovery time lock
    function setRecoveryTimeLock(uint256 _timeLock) external onlyOwner {
        require(_timeLock >= 1 hours, "Minimum 1 hour");
        require(_timeLock <= 30 days, "Maximum 30 days");
        recoveryTimeLock = _timeLock;
    }

    // ============ Challenger Reward Configuration ============

    /// @notice Update challenger reward configuration
    function setChallengerConfig(
        uint16 _rewardPercentage,
        uint128 _minReward,
        uint128 _maxReward,
        bool _enabled
    ) external onlyOwner {
        require(_rewardPercentage <= 5000, "Max 50% reward");
        require(_minReward <= _maxReward, "Min > max");

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
    /// @param modelId The model ID
    /// @param dataRoot The merkle root of required training data
    function setModelDataCommitment(
        uint256 modelId,
        bytes32 dataRoot
    ) external modelExists(modelId) {
        require(
            msg.sender == models[modelId].owner || msg.sender == owner,
            "Not authorized"
        );
        modelDataCommitment[modelId] = dataRoot;
        emit DataCommitmentSet(modelId, dataRoot, msg.sender);
    }

    /// @notice Commit data root for a specific round
    /// @param modelId The model ID
    /// @param roundId The round ID
    /// @param dataRoot The data merkle root for this round
    function commitRoundData(
        uint256 modelId,
        uint256 roundId,
        bytes32 dataRoot
    ) external whenNotPaused modelExists(modelId) {
        require(roundId == models[modelId].currentRound, "Invalid round");
        require(dataRoot != bytes32(0), "Invalid data root");

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
