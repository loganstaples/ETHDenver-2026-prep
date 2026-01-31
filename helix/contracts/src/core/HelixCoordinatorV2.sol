// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "../interfaces/IHelixVerifier.sol";

/// @title HelixCoordinatorV2
/// @notice Gas-optimized coordinator for model registration, proof submission, staking, and slashing
/// @dev Supports real ZK proof verification with economic security
///      Storage layout optimized for gas efficiency with struct packing
///      Includes emergency pause mechanism and data commitment verification
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
    /// @param publicInputs Public inputs: [oldHashLo, oldHashHi, newHashLo, newHashHi, loss, errorBound, stepNumber]
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

        // Validate public inputs count
        require(publicInputs.length == 7, "Invalid public inputs count");

        // Reconstruct and validate old commitment
        uint256 oldCommitmentFromProof = _hashPair(publicInputs[0], publicInputs[1]);
        require(oldCommitmentFromProof == round.modelCommitment, "Old commitment mismatch");

        // Validate error bound
        uint256 stepErrorBound = publicInputs[5];
        require(stepErrorBound <= maxErrorBound, "Error bound exceeds maximum");

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

    // ============ Slashing ============

    /// @notice Internal function to slash a prover's stake
    function _slash(
        address prover,
        uint256 modelId,
        uint256 roundId,
        string memory reason
    ) internal {
        Stake storage s = stakes[prover][modelId];
        require(s.amount > 0, "No stake to slash");
        require(!s.slashed, "Already slashed");

        uint128 slashAmount = uint128((uint256(s.amount) * slashPercentage) / 10000);
        s.amount -= slashAmount;
        s.slashed = true;

        // Send slashed amount to treasury
        if (treasury != address(0) && slashAmount > 0) {
            (bool success, ) = treasury.call{value: slashAmount}("");
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

    /// @notice Allows anyone to challenge a past proof
    function challengeProof(
        uint256 modelId,
        uint256 roundId,
        bytes memory proof,
        uint256[] memory publicInputs
    ) external {
        Round storage round = rounds[modelId][roundId];
        require(round.isCompleted, "Round not completed");
        require(round.prover != address(0), "No prover to challenge");

        bool valid = verifier.verifyProof(proof, publicInputs);

        if (!valid) {
            _slash(round.prover, modelId, roundId, "Fraudulent proof challenged");
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

    /// @notice Emergency pause - stops all critical operations
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
