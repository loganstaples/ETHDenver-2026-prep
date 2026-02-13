// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// @title TrainingRound
/// @notice Manages individual training rounds for a model
/// @dev Handles participant registration, gradient submission, and round lifecycle
contract TrainingRound {
    /// @notice Round states
    enum RoundState {
        Created,      // Round created, accepting registrations
        Active,       // Training in progress, accepting gradients
        Aggregating,  // Aggregating gradients
        Completed,    // Round completed successfully
        Failed        // Round failed (timeout or other issue)
    }
    
    /// @notice Round information
    struct Round {
        uint256 modelId;
        uint256 roundId;
        bytes32 startCommitment;      // Model state at round start
        bytes32 endCommitment;        // Model state after round
        RoundState state;
        uint256 startTime;
        uint256 endTime;
        uint256 deadline;             // Submission deadline
        uint256 minParticipants;
        uint256 maxParticipants;
        uint256 registeredCount;
        uint256 submittedCount;
        uint256 errorBound;
        bytes32 datasetHash;          // SHA-256 of training dataset for auditability
    }
    
    /// @notice Participant information
    struct Participant {
        address addr;
        bool registered;
        bool submitted;
        bytes32 gradientCommitment;
        uint256 errorBound;
        uint256 submittedAt;
    }
    
    /// @notice Round counter per model
    mapping(uint256 => uint256) public roundCount;
    
    /// @notice Mapping of modelId -> roundId -> Round
    mapping(uint256 => mapping(uint256 => Round)) public rounds;
    
    /// @notice Mapping of modelId -> roundId -> participant address -> Participant
    mapping(uint256 => mapping(uint256 => mapping(address => Participant))) public participants;
    
    /// @notice Mapping of modelId -> roundId -> participant addresses array
    mapping(uint256 => mapping(uint256 => address[])) public participantList;
    
    /// @notice Coordinator contract
    address public coordinator;
    
    /// @notice Owner
    address public owner;
    
    /// @notice Default round duration (1 hour)
    uint256 public defaultRoundDuration = 1 hours;
    
    /// @notice Events
    event RoundCreated(
        uint256 indexed modelId,
        uint256 indexed roundId,
        bytes32 startCommitment,
        uint256 deadline
    );
    event ParticipantRegistered(
        uint256 indexed modelId,
        uint256 indexed roundId,
        address indexed participant
    );
    event GradientSubmitted(
        uint256 indexed modelId,
        uint256 indexed roundId,
        address indexed participant,
        bytes32 gradientCommitment
    );
    event RoundActivated(uint256 indexed modelId, uint256 indexed roundId);
    event RoundAggregating(uint256 indexed modelId, uint256 indexed roundId);
    event RoundCompleted(
        uint256 indexed modelId,
        uint256 indexed roundId,
        bytes32 endCommitment,
        uint256 participantCount
    );
    event RoundFailed(uint256 indexed modelId, uint256 indexed roundId, string reason);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }
    
    modifier onlyCoordinator() {
        require(msg.sender == coordinator || msg.sender == owner, "Only coordinator");
        _;
    }
    
    constructor() {
        owner = msg.sender;
    }
    
    /// @notice Create a new training round (without dataset hash)
    function createRound(
        uint256 modelId,
        bytes32 startCommitment,
        uint256 minParticipants,
        uint256 maxParticipants,
        uint256 duration
    ) external onlyCoordinator returns (uint256 roundId) {
        return _createRound(modelId, startCommitment, minParticipants, maxParticipants, duration, bytes32(0));
    }

    /// @notice Create a new training round with dataset hash
    /// @param modelId Model ID
    /// @param startCommitment Current model state commitment
    /// @param minParticipants Minimum participants required
    /// @param maxParticipants Maximum participants allowed
    /// @param duration Round duration in seconds
    /// @param datasetHash SHA-256 hash of the training dataset (for auditability)
    /// @return roundId ID of the created round
    function createRound(
        uint256 modelId,
        bytes32 startCommitment,
        uint256 minParticipants,
        uint256 maxParticipants,
        uint256 duration,
        bytes32 datasetHash
    ) external onlyCoordinator returns (uint256 roundId) {
        return _createRound(modelId, startCommitment, minParticipants, maxParticipants, duration, datasetHash);
    }

    /// @dev Internal implementation for createRound
    function _createRound(
        uint256 modelId,
        bytes32 startCommitment,
        uint256 minParticipants,
        uint256 maxParticipants,
        uint256 duration,
        bytes32 datasetHash
    ) internal returns (uint256 roundId) {
        roundId = ++roundCount[modelId];

        uint256 deadline = block.timestamp + (duration > 0 ? duration : defaultRoundDuration);

        rounds[modelId][roundId] = Round({
            modelId: modelId,
            roundId: roundId,
            startCommitment: startCommitment,
            endCommitment: bytes32(0),
            state: RoundState.Created,
            startTime: block.timestamp,
            endTime: 0,
            deadline: deadline,
            minParticipants: minParticipants,
            maxParticipants: maxParticipants,
            registeredCount: 0,
            submittedCount: 0,
            errorBound: 0,
            datasetHash: datasetHash
        });

        emit RoundCreated(modelId, roundId, startCommitment, deadline);
    }

    /// @notice Register for a training round
    /// @param modelId Model ID
    /// @param roundId Round ID
    function registerForRound(uint256 modelId, uint256 roundId) external {
        Round storage round = rounds[modelId][roundId];
        
        require(round.state == RoundState.Created || round.state == RoundState.Active, "Round not accepting registrations");
        require(block.timestamp < round.deadline, "Round deadline passed");
        require(round.registeredCount < round.maxParticipants, "Round full");
        require(!participants[modelId][roundId][msg.sender].registered, "Already registered");
        
        participants[modelId][roundId][msg.sender] = Participant({
            addr: msg.sender,
            registered: true,
            submitted: false,
            gradientCommitment: bytes32(0),
            errorBound: 0,
            submittedAt: 0
        });
        
        participantList[modelId][roundId].push(msg.sender);
        round.registeredCount++;
        
        emit ParticipantRegistered(modelId, roundId, msg.sender);
        
        // Activate round if minimum participants reached
        if (round.state == RoundState.Created && round.registeredCount >= round.minParticipants) {
            round.state = RoundState.Active;
            emit RoundActivated(modelId, roundId);
        }
    }
    
    /// @notice Submit gradient for a round
    /// @param modelId Model ID
    /// @param roundId Round ID
    /// @param gradientCommitment Commitment to the gradient
    /// @param errorBound Error bound of the gradient
    function submitGradient(
        uint256 modelId,
        uint256 roundId,
        bytes32 gradientCommitment,
        uint256 errorBound
    ) external {
        Round storage round = rounds[modelId][roundId];
        Participant storage participant = participants[modelId][roundId][msg.sender];
        
        require(round.state == RoundState.Active, "Round not active");
        require(block.timestamp < round.deadline, "Round deadline passed");
        require(participant.registered, "Not registered");
        require(!participant.submitted, "Already submitted");
        
        participant.submitted = true;
        participant.gradientCommitment = gradientCommitment;
        participant.errorBound = errorBound;
        participant.submittedAt = block.timestamp;
        
        round.submittedCount++;
        
        emit GradientSubmitted(modelId, roundId, msg.sender, gradientCommitment);
    }
    
    /// @notice Transition round to aggregating state
    function startAggregation(uint256 modelId, uint256 roundId) external onlyCoordinator {
        Round storage round = rounds[modelId][roundId];
        
        require(
            round.state == RoundState.Active ||
            (round.state == RoundState.Created && block.timestamp >= round.deadline),
            "Cannot start aggregation"
        );
        require(round.submittedCount >= round.minParticipants, "Not enough submissions");
        
        round.state = RoundState.Aggregating;
        
        emit RoundAggregating(modelId, roundId);
    }
    
    /// @notice Complete the round with aggregated result
    /// @param modelId Model ID
    /// @param roundId Round ID
    /// @param endCommitment Final model state commitment
    /// @param totalErrorBound Total error bound after aggregation
    function completeRound(
        uint256 modelId,
        uint256 roundId,
        bytes32 endCommitment,
        uint256 totalErrorBound
    ) external onlyCoordinator {
        Round storage round = rounds[modelId][roundId];
        
        require(round.state == RoundState.Aggregating, "Not aggregating");
        
        round.state = RoundState.Completed;
        round.endCommitment = endCommitment;
        round.endTime = block.timestamp;
        round.errorBound = totalErrorBound;
        
        emit RoundCompleted(modelId, roundId, endCommitment, round.submittedCount);
    }
    
    /// @notice Fail a round
    function failRound(
        uint256 modelId,
        uint256 roundId,
        string calldata reason
    ) external onlyCoordinator {
        Round storage round = rounds[modelId][roundId];
        
        require(round.state != RoundState.Completed && round.state != RoundState.Failed, "Round already finalized");
        
        round.state = RoundState.Failed;
        round.endTime = block.timestamp;
        
        emit RoundFailed(modelId, roundId, reason);
    }
    
    /// @notice Get round info
    function getRound(
        uint256 modelId,
        uint256 roundId
    ) external view returns (Round memory) {
        return rounds[modelId][roundId];
    }
    
    /// @notice Get participant list for a round
    function getParticipants(
        uint256 modelId,
        uint256 roundId
    ) external view returns (address[] memory) {
        return participantList[modelId][roundId];
    }
    
    /// @notice Get participant info
    function getParticipant(
        uint256 modelId,
        uint256 roundId,
        address addr
    ) external view returns (Participant memory) {
        return participants[modelId][roundId][addr];
    }
    
    /// @notice Check if round deadline has passed
    function isDeadlinePassed(
        uint256 modelId,
        uint256 roundId
    ) external view returns (bool) {
        return block.timestamp >= rounds[modelId][roundId].deadline;
    }
    
    /// @notice Set coordinator
    function setCoordinator(address _coordinator) external onlyOwner {
        require(_coordinator != address(0), "Invalid address");
        coordinator = _coordinator;
    }
    
    /// @notice Set default round duration
    function setDefaultRoundDuration(uint256 duration) external onlyOwner {
        require(duration > 0, "Invalid duration");
        defaultRoundDuration = duration;
    }
    
    /// @notice Transfer ownership
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid address");
        owner = newOwner;
    }
}
