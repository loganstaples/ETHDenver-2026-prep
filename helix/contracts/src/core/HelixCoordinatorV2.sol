// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "../interfaces/IHelixVerifier.sol";

/// @title HelixCoordinatorV2
/// @notice Manages model registration, proof submission, staking, and slashing
/// @dev Supports real ZK proof verification with economic security
contract HelixCoordinatorV2 {
    // ============ Structs ============

    struct Model {
        string ipfsHash;              // IPFS hash of model weights
        uint256 currentCommitment;    // Current state commitment (hash of weights)
        uint256 currentRound;         // Current training round
        address owner;                // Model owner
        uint256 minStake;             // Minimum stake required to participate
        bool active;                  // Whether the model is accepting submissions
    }

    struct Round {
        uint256 modelCommitment;      // Expected old commitment for this round
        uint256 newCommitment;        // New commitment after round completes
        bool isCompleted;             // Whether round is completed
        uint256 deadline;             // Deadline for submissions
        address prover;               // Who submitted the winning proof
    }

    struct Stake {
        uint256 amount;               // Staked amount
        uint256 lockedUntil;          // Lock period end timestamp
        bool slashed;                 // Whether stake has been slashed
    }

    struct SlashingRecord {
        address prover;               // Who was slashed
        uint256 modelId;              // Which model
        uint256 roundId;              // Which round
        uint256 amount;               // Amount slashed
        string reason;                // Why they were slashed
        uint256 timestamp;            // When it happened
    }

    // ============ State Variables ============

    /// @notice The ZK proof verifier contract
    IHelixVerifier public verifier;

    /// @notice Model registry
    mapping(uint256 => Model) public models;

    /// @notice Round data: modelId => roundId => Round
    mapping(uint256 => mapping(uint256 => Round)) public rounds;

    /// @notice Prover stakes: prover => modelId => Stake
    mapping(address => mapping(uint256 => Stake)) public stakes;

    /// @notice Slashing records
    SlashingRecord[] public slashingRecords;

    /// @notice Counter for model IDs
    uint256 public nextModelId;

    /// @notice Default minimum stake (in wei)
    uint256 public defaultMinStake = 0.1 ether;

    /// @notice Stake lock period (in seconds)
    uint256 public stakeLockPeriod = 7 days;

    /// @notice Slash percentage (100 = 1%)
    uint256 public slashPercentage = 5000; // 50%

    /// @notice Treasury to receive slashed funds
    address public treasury;

    /// @notice Contract owner
    address public owner;

    // ============ Events ============

    event ModelRegistered(
        uint256 indexed modelId,
        address indexed owner,
        uint256 initialCommitment,
        string ipfsHash
    );

    event RoundStarted(
        uint256 indexed modelId,
        uint256 indexed roundId,
        uint256 deadline
    );

    event ProofSubmitted(
        uint256 indexed modelId,
        uint256 indexed roundId,
        address indexed prover,
        uint256 newCommitment
    );

    event RoundCompleted(
        uint256 indexed modelId,
        uint256 indexed roundId,
        uint256 newCommitment
    );

    event Staked(
        address indexed prover,
        uint256 indexed modelId,
        uint256 amount
    );

    event Unstaked(
        address indexed prover,
        uint256 indexed modelId,
        uint256 amount
    );

    event Slashed(
        address indexed prover,
        uint256 indexed modelId,
        uint256 amount,
        string reason
    );

    event InvalidProofDetected(
        uint256 indexed modelId,
        uint256 indexed roundId,
        address indexed prover,
        bytes32 proofHash
    );

    // ============ Modifiers ============

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    modifier modelExists(uint256 modelId) {
        require(models[modelId].owner != address(0), "Model does not exist");
        _;
    }

    modifier hasStake(uint256 modelId) {
        require(
            stakes[msg.sender][modelId].amount >= models[modelId].minStake,
            "Insufficient stake"
        );
        require(
            !stakes[msg.sender][modelId].slashed,
            "Stake has been slashed"
        );
        _;
    }

    // ============ Constructor ============

    constructor(address _verifier, address _treasury) {
        verifier = IHelixVerifier(_verifier);
        treasury = _treasury;
        owner = msg.sender;
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
    ) external returns (uint256 modelId) {
        modelId = nextModelId++;

        models[modelId] = Model({
            ipfsHash: ipfsHash,
            currentCommitment: initialCommitment,
            currentRound: 0,
            owner: msg.sender,
            minStake: minStake > 0 ? minStake : defaultMinStake,
            active: true
        });

        emit ModelRegistered(modelId, msg.sender, initialCommitment, ipfsHash);
    }

    /// @notice Starts a new training round
    /// @param modelId The model to start a round for
    /// @param duration Round duration in seconds
    function startRound(
        uint256 modelId,
        uint256 duration
    ) external modelExists(modelId) {
        Model storage model = models[modelId];
        require(msg.sender == model.owner, "Only model owner");
        require(model.active, "Model not active");

        uint256 roundId = ++model.currentRound;
        uint256 deadline = block.timestamp + duration;

        rounds[modelId][roundId] = Round({
            modelCommitment: model.currentCommitment,
            newCommitment: 0,
            isCompleted: false,
            deadline: deadline,
            prover: address(0)
        });

        emit RoundStarted(modelId, roundId, deadline);
    }

    // ============ Staking ============

    /// @notice Stakes tokens to participate in a model's training
    /// @param modelId The model to stake for
    function stake(uint256 modelId) external payable modelExists(modelId) {
        require(msg.value > 0, "Must stake non-zero amount");

        Stake storage s = stakes[msg.sender][modelId];
        require(!s.slashed, "Previous stake was slashed");

        s.amount += msg.value;
        s.lockedUntil = block.timestamp + stakeLockPeriod;

        emit Staked(msg.sender, modelId, msg.value);
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

        // Validate public inputs (7 inputs for MLTrainingStepCircuit)
        require(publicInputs.length == 7, "Invalid public inputs count");

        // Reconstruct old commitment from public inputs [0] and [1]
        uint256 oldCommitmentFromProof = _hashPair(publicInputs[0], publicInputs[1]);
        require(
            oldCommitmentFromProof == round.modelCommitment,
            "Old commitment mismatch"
        );

        // Verify the proof
        bool valid = verifier.verifyProof(proof, publicInputs);

        if (!valid) {
            // Slash the prover for submitting invalid proof
            // NOTE: We do NOT revert here - that would roll back the slashing!
            // Instead, we slash and return early. The prover loses their stake
            // and must re-stake to participate again.
            _slash(msg.sender, modelId, "Invalid proof");
            emit InvalidProofDetected(
                modelId,
                roundId,
                msg.sender,
                keccak256(proof)
            );
            return; // Do not revert - let the slashing persist
        }

        // Extract new commitment from public inputs [2] and [3]
        uint256 newCommitment = _hashPair(publicInputs[2], publicInputs[3]);

        // Update state
        model.currentCommitment = newCommitment;
        round.newCommitment = newCommitment;
        round.isCompleted = true;
        round.prover = msg.sender;

        // Reset stake lock (reward for valid submission)
        stakes[msg.sender][modelId].lockedUntil = block.timestamp;

        emit ProofSubmitted(modelId, roundId, msg.sender, newCommitment);
        emit RoundCompleted(modelId, roundId, newCommitment);
    }

    // ============ Slashing ============

    /// @notice Internal function to slash a prover's stake
    function _slash(
        address prover,
        uint256 modelId,
        string memory reason
    ) internal {
        Stake storage s = stakes[prover][modelId];
        require(s.amount > 0, "No stake to slash");
        require(!s.slashed, "Already slashed");

        uint256 slashAmount = (s.amount * slashPercentage) / 10000;
        s.amount -= slashAmount;
        s.slashed = true;

        // Send slashed amount to treasury
        if (treasury != address(0) && slashAmount > 0) {
            (bool success, ) = treasury.call{value: slashAmount}("");
            require(success, "Treasury transfer failed");
        }

        slashingRecords.push(SlashingRecord({
            prover: prover,
            modelId: modelId,
            roundId: models[modelId].currentRound,
            amount: slashAmount,
            reason: reason,
            timestamp: block.timestamp
        }));

        emit Slashed(prover, modelId, slashAmount, reason);
    }

    /// @notice Allows anyone to challenge a past proof (if fraud is detected later)
    /// @dev This could be called if someone finds a proof was actually invalid
    function challengeProof(
        uint256 modelId,
        uint256 roundId,
        bytes memory proof,
        uint256[] memory publicInputs
    ) external {
        Round storage round = rounds[modelId][roundId];
        require(round.isCompleted, "Round not completed");
        require(round.prover != address(0), "No prover to challenge");

        // Re-verify the proof
        bool valid = verifier.verifyProof(proof, publicInputs);

        if (!valid) {
            // The original prover submitted fraud!
            _slash(round.prover, modelId, "Fraudulent proof challenged");
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

    // ============ Internal Helpers ============

    /// @notice Combines two 128-bit halves into a single commitment
    function _hashPair(uint256 lo, uint256 hi) internal pure returns (uint256) {
        return uint256(keccak256(abi.encodePacked(lo, hi)));
    }

    // ============ Admin Functions ============

    /// @notice Updates the verifier contract
    function setVerifier(address _verifier) external onlyOwner {
        verifier = IHelixVerifier(_verifier);
    }

    /// @notice Updates the treasury address
    function setTreasury(address _treasury) external onlyOwner {
        treasury = _treasury;
    }

    /// @notice Updates the slash percentage
    function setSlashPercentage(uint256 _percentage) external onlyOwner {
        require(_percentage <= 10000, "Max 100%");
        slashPercentage = _percentage;
    }

    /// @notice Updates the default minimum stake
    function setDefaultMinStake(uint256 _minStake) external onlyOwner {
        defaultMinStake = _minStake;
    }

    /// @notice Pauses a model
    function pauseModel(uint256 modelId) external {
        require(
            msg.sender == models[modelId].owner || msg.sender == owner,
            "Not authorized"
        );
        models[modelId].active = false;
    }

    /// @notice Resumes a model
    function resumeModel(uint256 modelId) external {
        require(
            msg.sender == models[modelId].owner || msg.sender == owner,
            "Not authorized"
        );
        models[modelId].active = true;
    }

    /// @notice Receives ETH
    receive() external payable {}
}
