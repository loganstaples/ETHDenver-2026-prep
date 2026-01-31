// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// @title SlashingEvidence
/// @notice Comprehensive slashing evidence verification for HELIX protocol
/// @dev Production-grade evidence recording, verification, dispute resolution,
///      gradual slashing, challenger rewards, and appeals mechanism
contract SlashingEvidence {
    // ============ Enums ============

    /// @notice Types of violations that can trigger slashing
    /// @dev Each type has specific error codes for detailed tracking
    enum ViolationType {
        InvalidProof,           // ZK proof verification failed
        CommitmentMismatch,     // Old/new commitment doesn't match
        ErrorBoundExceeded,     // Error bound exceeds maximum
        DataIntegrityFailure,   // Training data commitment mismatch
        DoubleSubmission,       // Same prover submitted twice
        TimeoutViolation,       // Submitted after deadline
        MaliciousGradient,      // Detected gradient poisoning
        ProtocolViolation,      // Generic protocol rule violation
        ReplayAttack,           // Attempted proof replay
        SybilBehavior,          // Suspected Sybil attack pattern
        CollaborativeFraud      // Multiple workers coordinating fraud
    }

    /// @notice Specific error codes for detailed violation tracking
    /// @dev 16-bit codes: upper 8 bits = category, lower 8 bits = specific error
    enum ErrorCode {
        // InvalidProof errors (0x01XX)
        PROOF_MALFORMED,              // Proof bytes invalid format
        PROOF_TOO_SHORT,              // Proof length < minimum
        PROOF_VERIFICATION_FAILED,    // Cryptographic verification failed
        PROOF_FIELD_ELEMENT_INVALID,  // Field element exceeds scalar field
        PROOF_PAIRING_FAILED,         // Elliptic curve pairing failed

        // CommitmentMismatch errors (0x02XX)
        OLD_COMMITMENT_WRONG,         // Old commitment doesn't match model state
        NEW_COMMITMENT_INVALID,       // New commitment computation incorrect
        COMMITMENT_HASH_COLLISION,    // Hash collision detected

        // ErrorBound errors (0x03XX)
        ERROR_BOUND_EXCEEDED,         // Single step error too high
        ACCUMULATED_ERROR_EXCEEDED,   // Total accumulated error too high
        ERROR_BOUND_MANIPULATION,     // Suspected error bound tampering

        // DataIntegrity errors (0x04XX)
        DATA_ROOT_MISMATCH,           // Data merkle root doesn't match
        DATA_PROOF_INVALID,           // Merkle proof verification failed
        DATA_CORRUPTION_DETECTED,     // Data integrity check failed

        // Timing errors (0x05XX)
        SUBMISSION_TOO_LATE,          // After round deadline
        SUBMISSION_TOO_EARLY,         // Before round started
        ROUND_ALREADY_COMPLETED,      // Round was already finalized

        // Protocol errors (0x06XX)
        DOUBLE_SUBMISSION_DETECTED,   // Same prover, same round
        REPLAY_DETECTED,              // Same proof reused
        INVALID_PUBLIC_INPUTS,        // Wrong number/format of public inputs
        STAKE_INSUFFICIENT,           // Below minimum stake

        // Gradient errors (0x07XX)
        GRADIENT_OUTLIER,             // Gradient significantly different from median
        GRADIENT_POISONING_DETECTED,  // Deliberate poisoning pattern
        GRADIENT_MAGNITUDE_INVALID,   // Gradient norm out of bounds

        // Unknown
        UNKNOWN_ERROR                 // Catch-all for unclassified errors
    }

    /// @notice Status of an evidence submission
    enum EvidenceStatus {
        Pending,       // Evidence submitted, awaiting verification
        Verified,      // Evidence verified as valid
        Rejected,      // Evidence rejected as invalid
        Disputed,      // Evidence is being disputed
        Resolved,      // Dispute resolved
        Appealed,      // Appeal filed (second review)
        AppealResolved // Appeal resolved
    }

    /// @notice Severity level for gradual slashing
    enum SeverityLevel {
        Warning,       // First offense - warning only
        Minor,         // Minor violation - 10% slash
        Moderate,      // Moderate violation - 25% slash
        Major,         // Major violation - 50% slash
        Critical,      // Critical violation - 75% slash
        Terminal       // Terminal violation - 100% slash + permanent ban
    }

    // ============ Structs (Gas Optimized) ============

    /// @notice Complete evidence record for a slashing event
    /// @dev Packed for storage efficiency where possible
    struct Evidence {
        // Slot 1: Core identifiers
        address prover;               // 20 bytes - who was slashed
        uint64 modelId;               // 8 bytes
        uint32 roundId;               // 4 bytes

        // Slot 2: Status and type info
        ViolationType violationType;  // 1 byte
        EvidenceStatus status;        // 1 byte
        ErrorCode errorCode;          // 1 byte
        SeverityLevel severity;       // 1 byte
        uint40 timestamp;             // 5 bytes
        uint40 disputeDeadline;       // 5 bytes
        uint40 appealDeadline;        // 5 bytes
        bool challengerRewarded;      // 1 byte

        // Slot 3: Addresses
        address submitter;            // 20 bytes - who submitted evidence
        address challenger;           // 20 bytes - who challenged (if any)

        // Slot 4: Financial
        uint128 slashedAmount;        // 16 bytes
        uint128 remainingStake;       // 16 bytes

        // Slot 5: Challenger reward
        uint128 challengerReward;     // 16 bytes
        uint128 protocolFee;          // 16 bytes

        // Dynamic data
        bytes32 proofHash;            // Hash of the proof that failed
        bytes32 evidenceHash;         // Hash of all evidence data
        string reason;                // Detailed reason string
    }

    /// @notice Cryptographic evidence data
    struct CryptoEvidence {
        bytes invalidProof;           // The proof that was invalid
        uint256[] publicInputs;       // The public inputs used
        bytes32 expectedCommitment;   // What commitment was expected
        bytes32 actualCommitment;     // What commitment was provided
        uint256 claimedErrorBound;    // What error bound was claimed
        uint256 actualErrorBound;     // What error bound was computed
        bytes32 dataCommitment;       // Data commitment if applicable
        bytes32 gradientCommitment;   // Gradient commitment if applicable
        bytes additionalData;         // Any additional verification data
    }

    /// @notice Warning record for gradual slashing
    struct WarningRecord {
        uint256 warningCount;         // Number of warnings
        uint40 firstWarningTime;      // When first warning was issued
        uint40 lastWarningTime;       // When last warning was issued
        uint128 totalSlashedAmount;   // Total amount slashed across all incidents
        SeverityLevel maxSeverity;    // Maximum severity encountered
        bool permanentlyBanned;       // Whether permanently banned
    }

    /// @notice Dispute information
    struct Dispute {
        uint256 evidenceId;           // Which evidence is disputed
        address disputer;             // Who filed the dispute
        uint40 filedAt;               // When dispute was filed
        uint40 resolvedAt;            // When dispute was resolved
        bool inFavorOfProver;         // Resolution outcome
        bytes32 disputeEvidenceHash;  // Hash of dispute evidence
        string disputeReason;         // Reason for dispute
        uint256 disputeStake;         // Stake deposited for dispute
    }

    /// @notice Appeal information
    struct Appeal {
        uint256 evidenceId;           // Which evidence is appealed
        uint256 disputeId;            // Related dispute if any
        address appellant;            // Who filed the appeal
        uint40 filedAt;               // When appeal was filed
        uint40 resolvedAt;            // When appeal was resolved
        bool successful;              // Whether appeal succeeded
        bytes32 appealEvidenceHash;   // Hash of appeal evidence
        string appealReason;          // Reason for appeal
        uint256 appealStake;          // Stake deposited for appeal
        address[] arbitrators;        // Addresses that participated in ruling
    }

    /// @notice Challenger reward configuration
    struct ChallengerConfig {
        uint16 rewardPercentage;      // Percentage of slashed amount (basis points)
        uint16 protocolFeePercentage; // Protocol fee (basis points)
        uint128 minimumReward;        // Minimum reward amount
        uint128 maximumReward;        // Maximum reward cap
        bool enabled;                 // Whether rewards are enabled
    }

    // ============ State Variables ============

    /// @notice Contract owner
    address public owner;

    /// @notice Coordinator contract
    address public coordinator;

    /// @notice Treasury for protocol fees
    address public treasury;

    /// @notice Counter for evidence IDs
    uint256 public nextEvidenceId;

    /// @notice Counter for dispute IDs
    uint256 public nextDisputeId;

    /// @notice Counter for appeal IDs
    uint256 public nextAppealId;

    /// @notice Dispute period in seconds
    uint256 public disputePeriod;

    /// @notice Appeal period in seconds
    uint256 public appealPeriod;

    /// @notice Minimum stake required to file a dispute
    uint256 public disputeStakeRequired;

    /// @notice Minimum stake required to file an appeal
    uint256 public appealStakeRequired;

    /// @notice Challenger reward configuration
    ChallengerConfig public challengerConfig;

    /// @notice Evidence records
    mapping(uint256 => Evidence) public evidenceRecords;

    /// @notice Crypto evidence for each evidence ID
    mapping(uint256 => CryptoEvidence) internal _cryptoEvidence;

    /// @notice Warning records per prover
    mapping(address => WarningRecord) public warningRecords;

    /// @notice Dispute records
    mapping(uint256 => Dispute) public disputes;

    /// @notice Appeal records
    mapping(uint256 => Appeal) public appeals;

    /// @notice Mapping of prover to their evidence IDs
    mapping(address => uint256[]) internal _proverEvidence;

    /// @notice Mapping of model+round to evidence IDs
    mapping(uint256 => mapping(uint256 => uint256[])) internal _roundEvidence;

    /// @notice Mapping of challenger to their evidence IDs
    mapping(address => uint256[]) internal _challengerEvidence;

    /// @notice Verified evidence hashes (prevent duplicates)
    mapping(bytes32 => bool) public verifiedEvidenceHashes;

    /// @notice Severity level slashing percentages (basis points)
    mapping(SeverityLevel => uint16) public severitySlashPercentage;

    /// @notice Warning thresholds for severity escalation
    mapping(SeverityLevel => uint256) public warningEscalationThreshold;

    /// @notice Arbitrators who can resolve appeals
    mapping(address => bool) public isArbitrator;

    /// @notice Arbitrator count
    uint256 public arbitratorCount;

    // ============ Events ============

    event EvidenceSubmitted(
        uint256 indexed evidenceId,
        address indexed prover,
        uint64 indexed modelId,
        ViolationType violationType,
        ErrorCode errorCode,
        SeverityLevel severity,
        uint128 slashedAmount
    );

    event EvidenceVerified(
        uint256 indexed evidenceId,
        address indexed verifier,
        EvidenceStatus status
    );

    event DisputeFiled(
        uint256 indexed disputeId,
        uint256 indexed evidenceId,
        address indexed disputer,
        uint256 stakeDeposited,
        string reason
    );

    event DisputeResolved(
        uint256 indexed disputeId,
        uint256 indexed evidenceId,
        bool inFavorOfProver,
        address resolver
    );

    event AppealFiled(
        uint256 indexed appealId,
        uint256 indexed evidenceId,
        address indexed appellant,
        uint256 stakeDeposited,
        string reason
    );

    event AppealResolved(
        uint256 indexed appealId,
        uint256 indexed evidenceId,
        bool successful,
        address[] arbitrators
    );

    event ChallengerRewarded(
        uint256 indexed evidenceId,
        address indexed challenger,
        uint128 rewardAmount,
        uint128 protocolFee
    );

    event WarningIssued(
        address indexed prover,
        uint256 indexed evidenceId,
        uint256 warningCount,
        SeverityLevel nextSeverity
    );

    event ProverBanned(
        address indexed prover,
        uint256 totalIncidents,
        uint128 totalSlashed,
        string reason
    );

    event CryptoEvidenceStored(
        uint256 indexed evidenceId,
        bytes32 proofHash,
        bytes32 expectedCommitment,
        bytes32 actualCommitment
    );

    event SeverityEscalated(
        address indexed prover,
        SeverityLevel oldSeverity,
        SeverityLevel newSeverity,
        uint256 warningCount
    );

    event ArbitratorAdded(address indexed arbitrator);
    event ArbitratorRemoved(address indexed arbitrator);
    event ChallengerConfigUpdated(ChallengerConfig config);

    // ============ Errors ============

    error EvidenceNotFound(uint256 evidenceId);
    error DisputeNotFound(uint256 disputeId);
    error AppealNotFound(uint256 appealId);
    error NotAuthorized(address caller);
    error DisputePeriodEnded(uint256 evidenceId);
    error AppealPeriodEnded(uint256 evidenceId);
    error InsufficientDisputeStake(uint256 required, uint256 provided);
    error InsufficientAppealStake(uint256 required, uint256 provided);
    error DuplicateEvidence(bytes32 hash);
    error InvalidStatus(EvidenceStatus current, EvidenceStatus required);
    error ProverPermanentlyBanned(address prover);
    error NotArbitrator(address caller);
    error AlreadyResolved();
    error InvalidSeverity();

    // ============ Modifiers ============

    modifier onlyOwner() {
        if (msg.sender != owner) revert NotAuthorized(msg.sender);
        _;
    }

    modifier onlyCoordinator() {
        if (msg.sender != coordinator) revert NotAuthorized(msg.sender);
        _;
    }

    modifier onlyArbitrator() {
        if (!isArbitrator[msg.sender] && msg.sender != owner) revert NotArbitrator(msg.sender);
        _;
    }

    modifier evidenceExists(uint256 evidenceId) {
        if (evidenceRecords[evidenceId].timestamp == 0) revert EvidenceNotFound(evidenceId);
        _;
    }

    // ============ Constructor ============

    constructor(address _coordinator) {
        owner = msg.sender;
        coordinator = _coordinator;
        treasury = msg.sender;

        // Default periods
        disputePeriod = 7 days;
        appealPeriod = 14 days;
        disputeStakeRequired = 0.01 ether;
        appealStakeRequired = 0.05 ether;

        // Default severity percentages (basis points)
        severitySlashPercentage[SeverityLevel.Warning] = 0;        // 0% - just warning
        severitySlashPercentage[SeverityLevel.Minor] = 1000;       // 10%
        severitySlashPercentage[SeverityLevel.Moderate] = 2500;    // 25%
        severitySlashPercentage[SeverityLevel.Major] = 5000;       // 50%
        severitySlashPercentage[SeverityLevel.Critical] = 7500;    // 75%
        severitySlashPercentage[SeverityLevel.Terminal] = 10000;   // 100%

        // Warning escalation thresholds
        warningEscalationThreshold[SeverityLevel.Warning] = 1;
        warningEscalationThreshold[SeverityLevel.Minor] = 2;
        warningEscalationThreshold[SeverityLevel.Moderate] = 3;
        warningEscalationThreshold[SeverityLevel.Major] = 4;
        warningEscalationThreshold[SeverityLevel.Critical] = 5;
        warningEscalationThreshold[SeverityLevel.Terminal] = 6;

        // Default challenger config
        challengerConfig = ChallengerConfig({
            rewardPercentage: 1000,     // 10% of slashed amount
            protocolFeePercentage: 500, // 5% protocol fee
            minimumReward: 0.001 ether,
            maximumReward: 10 ether,
            enabled: true
        });
    }

    // ============ Evidence Submission Functions ============

    /// @notice Submit slashing evidence with full details (called by coordinator)
    /// @param prover The address being slashed
    /// @param modelId The model ID
    /// @param roundId The round ID
    /// @param violationType Type of violation
    /// @param errorCode Specific error code
    /// @param slashedAmount Amount being slashed
    /// @param remainingStake Remaining stake after slashing
    /// @param proofHash Hash of the relevant proof
    /// @param challenger Address of the challenger (address(0) if system detected)
    /// @param reason Detailed reason for slashing
    /// @return evidenceId The ID of the created evidence record
    function submitEvidence(
        address prover,
        uint64 modelId,
        uint32 roundId,
        ViolationType violationType,
        ErrorCode errorCode,
        uint128 slashedAmount,
        uint128 remainingStake,
        bytes32 proofHash,
        address challenger,
        string calldata reason
    ) external onlyCoordinator returns (uint256 evidenceId) {
        // Check if prover is permanently banned
        if (warningRecords[prover].permanentlyBanned) {
            revert ProverPermanentlyBanned(prover);
        }

        // Compute evidence hash for deduplication
        bytes32 evidenceHash = keccak256(abi.encodePacked(
            prover, modelId, roundId, violationType, errorCode, proofHash, block.timestamp
        ));

        if (verifiedEvidenceHashes[evidenceHash]) {
            revert DuplicateEvidence(evidenceHash);
        }

        // Determine severity based on violation type and warning history
        SeverityLevel severity = _calculateSeverity(prover, violationType);

        // Calculate challenger reward
        (uint128 challengerReward, uint128 protocolFee) = _calculateChallengerReward(slashedAmount);

        evidenceId = nextEvidenceId++;

        evidenceRecords[evidenceId] = Evidence({
            prover: prover,
            modelId: modelId,
            roundId: roundId,
            violationType: violationType,
            status: EvidenceStatus.Pending,
            errorCode: errorCode,
            severity: severity,
            timestamp: uint40(block.timestamp),
            disputeDeadline: uint40(block.timestamp + disputePeriod),
            appealDeadline: uint40(block.timestamp + appealPeriod),
            challengerRewarded: false,
            submitter: msg.sender,
            challenger: challenger,
            slashedAmount: slashedAmount,
            remainingStake: remainingStake,
            challengerReward: challengerReward,
            protocolFee: protocolFee,
            proofHash: proofHash,
            evidenceHash: evidenceHash,
            reason: reason
        });

        verifiedEvidenceHashes[evidenceHash] = true;
        _proverEvidence[prover].push(evidenceId);
        _roundEvidence[modelId][roundId].push(evidenceId);

        if (challenger != address(0)) {
            _challengerEvidence[challenger].push(evidenceId);
        }

        // Update warning records
        _updateWarningRecord(prover, severity, slashedAmount);

        emit EvidenceSubmitted(
            evidenceId, prover, modelId, violationType, errorCode, severity, slashedAmount
        );

        // Handle warning vs actual slashing
        if (severity == SeverityLevel.Warning) {
            emit WarningIssued(
                prover,
                evidenceId,
                warningRecords[prover].warningCount,
                _getNextSeverity(prover)
            );
        }
    }

    /// @notice Submit slashing evidence with automatic severity calculation (simplified)
    function submitEvidenceSimple(
        address prover,
        uint64 modelId,
        uint32 roundId,
        ViolationType violationType,
        uint128 slashedAmount,
        uint128 remainingStake,
        bytes32 proofHash,
        string calldata reason
    ) external onlyCoordinator returns (uint256 evidenceId) {
        ErrorCode errorCode = _mapViolationToErrorCode(violationType);
        return this.submitEvidence(
            prover,
            modelId,
            roundId,
            violationType,
            errorCode,
            slashedAmount,
            remainingStake,
            proofHash,
            address(0),
            reason
        );
    }

    /// @notice Store cryptographic evidence for an evidence record
    function storeCryptoEvidence(
        uint256 evidenceId,
        bytes calldata invalidProof,
        uint256[] calldata publicInputs,
        bytes32 expectedCommitment,
        bytes32 actualCommitment,
        uint256 claimedErrorBound,
        uint256 actualErrorBound,
        bytes32 dataCommitment,
        bytes32 gradientCommitment,
        bytes calldata additionalData
    ) external onlyCoordinator evidenceExists(evidenceId) {
        _cryptoEvidence[evidenceId] = CryptoEvidence({
            invalidProof: invalidProof,
            publicInputs: publicInputs,
            expectedCommitment: expectedCommitment,
            actualCommitment: actualCommitment,
            claimedErrorBound: claimedErrorBound,
            actualErrorBound: actualErrorBound,
            dataCommitment: dataCommitment,
            gradientCommitment: gradientCommitment,
            additionalData: additionalData
        });

        emit CryptoEvidenceStored(
            evidenceId,
            keccak256(invalidProof),
            expectedCommitment,
            actualCommitment
        );
    }

    // ============ Challenger Reward Functions ============

    /// @notice Distribute challenger reward for verified evidence
    /// @param evidenceId The evidence ID
    function distributesChallengerReward(uint256 evidenceId) external evidenceExists(evidenceId) {
        Evidence storage evidence = evidenceRecords[evidenceId];

        require(
            evidence.status == EvidenceStatus.Verified ||
            evidence.status == EvidenceStatus.Resolved ||
            evidence.status == EvidenceStatus.AppealResolved,
            "Evidence not verified"
        );
        require(!evidence.challengerRewarded, "Already rewarded");
        require(evidence.challenger != address(0), "No challenger");
        require(challengerConfig.enabled, "Rewards disabled");

        evidence.challengerRewarded = true;

        // Transfer reward to challenger
        if (evidence.challengerReward > 0) {
            (bool success, ) = evidence.challenger.call{value: evidence.challengerReward}("");
            require(success, "Reward transfer failed");
        }

        // Transfer protocol fee to treasury
        if (evidence.protocolFee > 0 && treasury != address(0)) {
            (bool success, ) = treasury.call{value: evidence.protocolFee}("");
            require(success, "Fee transfer failed");
        }

        emit ChallengerRewarded(
            evidenceId,
            evidence.challenger,
            evidence.challengerReward,
            evidence.protocolFee
        );
    }

    /// @notice Batch distribute challenger rewards
    function batchDistributeRewards(uint256[] calldata evidenceIds) external {
        for (uint256 i = 0; i < evidenceIds.length; i++) {
            try this.distributesChallengerReward(evidenceIds[i]) {} catch {}
        }
    }

    // ============ Verification Functions ============

    /// @notice Verify and finalize evidence (after dispute period)
    function verifyEvidence(uint256 evidenceId) external onlyOwner evidenceExists(evidenceId) {
        Evidence storage evidence = evidenceRecords[evidenceId];

        if (evidence.status != EvidenceStatus.Pending) {
            revert InvalidStatus(evidence.status, EvidenceStatus.Pending);
        }
        require(block.timestamp >= evidence.disputeDeadline, "Dispute period active");

        evidence.status = EvidenceStatus.Verified;

        emit EvidenceVerified(evidenceId, msg.sender, EvidenceStatus.Verified);
    }

    /// @notice Reject evidence
    function rejectEvidence(uint256 evidenceId) external onlyOwner evidenceExists(evidenceId) {
        Evidence storage evidence = evidenceRecords[evidenceId];

        if (evidence.status != EvidenceStatus.Pending && evidence.status != EvidenceStatus.Disputed) {
            revert InvalidStatus(evidence.status, EvidenceStatus.Pending);
        }

        evidence.status = EvidenceStatus.Rejected;

        emit EvidenceVerified(evidenceId, msg.sender, EvidenceStatus.Rejected);
    }

    /// @notice Batch verify multiple evidence records
    function batchVerifyEvidence(uint256[] calldata evidenceIds) external onlyOwner {
        for (uint256 i = 0; i < evidenceIds.length; i++) {
            Evidence storage evidence = evidenceRecords[evidenceIds[i]];
            if (evidence.status == EvidenceStatus.Pending &&
                block.timestamp >= evidence.disputeDeadline) {
                evidence.status = EvidenceStatus.Verified;
                emit EvidenceVerified(evidenceIds[i], msg.sender, EvidenceStatus.Verified);
            }
        }
    }

    // ============ Dispute Functions ============

    /// @notice File a dispute against slashing evidence
    function fileDispute(
        uint256 evidenceId,
        bytes32 disputeEvidenceHash,
        string calldata reason
    ) external payable evidenceExists(evidenceId) returns (uint256 disputeId) {
        Evidence storage evidence = evidenceRecords[evidenceId];

        if (evidence.status != EvidenceStatus.Pending) {
            revert InvalidStatus(evidence.status, EvidenceStatus.Pending);
        }
        if (block.timestamp >= evidence.disputeDeadline) {
            revert DisputePeriodEnded(evidenceId);
        }
        if (msg.sender != evidence.prover && msg.sender != owner) {
            revert NotAuthorized(msg.sender);
        }
        if (msg.value < disputeStakeRequired) {
            revert InsufficientDisputeStake(disputeStakeRequired, msg.value);
        }

        disputeId = nextDisputeId++;

        disputes[disputeId] = Dispute({
            evidenceId: evidenceId,
            disputer: msg.sender,
            filedAt: uint40(block.timestamp),
            resolvedAt: 0,
            inFavorOfProver: false,
            disputeEvidenceHash: disputeEvidenceHash,
            disputeReason: reason,
            disputeStake: msg.value
        });

        evidence.status = EvidenceStatus.Disputed;

        emit DisputeFiled(disputeId, evidenceId, msg.sender, msg.value, reason);
    }

    /// @notice Resolve a dispute
    function resolveDispute(
        uint256 disputeId,
        bool inFavorOfProver
    ) external onlyOwner {
        Dispute storage dispute = disputes[disputeId];
        if (dispute.filedAt == 0) revert DisputeNotFound(disputeId);
        if (dispute.resolvedAt != 0) revert AlreadyResolved();

        dispute.resolvedAt = uint40(block.timestamp);
        dispute.inFavorOfProver = inFavorOfProver;

        Evidence storage evidence = evidenceRecords[dispute.evidenceId];

        if (inFavorOfProver) {
            evidence.status = EvidenceStatus.Rejected;
            // Return dispute stake to disputer
            (bool success, ) = dispute.disputer.call{value: dispute.disputeStake}("");
            require(success, "Stake return failed");
        } else {
            evidence.status = EvidenceStatus.Resolved;
            // Dispute stake goes to treasury
            if (treasury != address(0)) {
                (bool success, ) = treasury.call{value: dispute.disputeStake}("");
                require(success, "Fee transfer failed");
            }
        }

        emit DisputeResolved(disputeId, dispute.evidenceId, inFavorOfProver, msg.sender);
    }

    // ============ Appeal Functions ============

    /// @notice File an appeal against a dispute resolution
    function fileAppeal(
        uint256 evidenceId,
        uint256 disputeId,
        bytes32 appealEvidenceHash,
        string calldata reason
    ) external payable evidenceExists(evidenceId) returns (uint256 appealId) {
        Evidence storage evidence = evidenceRecords[evidenceId];

        // Can only appeal resolved disputes or rejected evidence
        if (evidence.status != EvidenceStatus.Resolved &&
            evidence.status != EvidenceStatus.Rejected &&
            evidence.status != EvidenceStatus.Verified) {
            revert InvalidStatus(evidence.status, EvidenceStatus.Resolved);
        }
        if (block.timestamp >= evidence.appealDeadline) {
            revert AppealPeriodEnded(evidenceId);
        }
        // Only affected parties can appeal
        if (msg.sender != evidence.prover &&
            msg.sender != evidence.challenger &&
            msg.sender != owner) {
            revert NotAuthorized(msg.sender);
        }
        if (msg.value < appealStakeRequired) {
            revert InsufficientAppealStake(appealStakeRequired, msg.value);
        }

        appealId = nextAppealId++;

        Appeal storage appeal = appeals[appealId];
        appeal.evidenceId = evidenceId;
        appeal.disputeId = disputeId;
        appeal.appellant = msg.sender;
        appeal.filedAt = uint40(block.timestamp);
        appeal.appealEvidenceHash = appealEvidenceHash;
        appeal.appealReason = reason;
        appeal.appealStake = msg.value;

        evidence.status = EvidenceStatus.Appealed;

        emit AppealFiled(appealId, evidenceId, msg.sender, msg.value, reason);
    }

    /// @notice Resolve an appeal (requires arbitrators)
    function resolveAppeal(
        uint256 appealId,
        bool successful
    ) external onlyArbitrator {
        Appeal storage appeal = appeals[appealId];
        if (appeal.filedAt == 0) revert AppealNotFound(appealId);
        if (appeal.resolvedAt != 0) revert AlreadyResolved();

        appeal.resolvedAt = uint40(block.timestamp);
        appeal.successful = successful;
        appeal.arbitrators.push(msg.sender);

        Evidence storage evidence = evidenceRecords[appeal.evidenceId];

        if (successful) {
            // Appeal successful - reverse the original decision
            if (evidence.status == EvidenceStatus.Appealed) {
                // If prover was slashed, mark as rejected
                evidence.status = EvidenceStatus.Rejected;
            }
            // Return appeal stake
            (bool success, ) = appeal.appellant.call{value: appeal.appealStake}("");
            require(success, "Stake return failed");
        } else {
            // Appeal failed - confirm original decision
            evidence.status = EvidenceStatus.AppealResolved;
            // Appeal stake goes to treasury
            if (treasury != address(0)) {
                (bool success, ) = treasury.call{value: appeal.appealStake}("");
                require(success, "Fee transfer failed");
            }
        }

        emit AppealResolved(appealId, appeal.evidenceId, successful, appeal.arbitrators);
    }

    // ============ Gradual Slashing Internal Functions ============

    /// @notice Calculate severity based on violation type and history
    function _calculateSeverity(
        address prover,
        ViolationType violationType
    ) internal view returns (SeverityLevel) {
        WarningRecord storage record = warningRecords[prover];

        // First offense for minor violations starts as warning
        if (record.warningCount == 0 && _isMinorViolation(violationType)) {
            return SeverityLevel.Warning;
        }

        // Escalate based on warning count
        if (record.warningCount >= warningEscalationThreshold[SeverityLevel.Terminal]) {
            return SeverityLevel.Terminal;
        }
        if (record.warningCount >= warningEscalationThreshold[SeverityLevel.Critical]) {
            return SeverityLevel.Critical;
        }
        if (record.warningCount >= warningEscalationThreshold[SeverityLevel.Major]) {
            return SeverityLevel.Major;
        }
        if (record.warningCount >= warningEscalationThreshold[SeverityLevel.Moderate]) {
            return SeverityLevel.Moderate;
        }
        if (record.warningCount >= warningEscalationThreshold[SeverityLevel.Minor]) {
            return SeverityLevel.Minor;
        }

        // Critical violations skip warning
        if (_isCriticalViolation(violationType)) {
            return SeverityLevel.Major;
        }

        return SeverityLevel.Warning;
    }

    /// @notice Check if violation type is minor
    function _isMinorViolation(ViolationType violationType) internal pure returns (bool) {
        return violationType == ViolationType.TimeoutViolation ||
               violationType == ViolationType.ErrorBoundExceeded;
    }

    /// @notice Check if violation type is critical
    function _isCriticalViolation(ViolationType violationType) internal pure returns (bool) {
        return violationType == ViolationType.MaliciousGradient ||
               violationType == ViolationType.CollaborativeFraud ||
               violationType == ViolationType.ReplayAttack;
    }

    /// @notice Update warning record for a prover
    function _updateWarningRecord(
        address prover,
        SeverityLevel severity,
        uint128 slashedAmount
    ) internal {
        WarningRecord storage record = warningRecords[prover];

        record.warningCount++;
        record.lastWarningTime = uint40(block.timestamp);
        record.totalSlashedAmount += slashedAmount;

        if (record.firstWarningTime == 0) {
            record.firstWarningTime = uint40(block.timestamp);
        }

        if (uint8(severity) > uint8(record.maxSeverity)) {
            SeverityLevel oldSeverity = record.maxSeverity;
            record.maxSeverity = severity;
            emit SeverityEscalated(prover, oldSeverity, severity, record.warningCount);
        }

        // Check for permanent ban
        if (severity == SeverityLevel.Terminal ||
            record.warningCount >= warningEscalationThreshold[SeverityLevel.Terminal]) {
            record.permanentlyBanned = true;
            emit ProverBanned(
                prover,
                record.warningCount,
                record.totalSlashedAmount,
                "Terminal severity or maximum warnings exceeded"
            );
        }
    }

    /// @notice Get next severity level for a prover
    function _getNextSeverity(address prover) internal view returns (SeverityLevel) {
        WarningRecord storage record = warningRecords[prover];
        uint256 nextWarningCount = record.warningCount + 1;

        if (nextWarningCount >= warningEscalationThreshold[SeverityLevel.Terminal]) {
            return SeverityLevel.Terminal;
        }
        if (nextWarningCount >= warningEscalationThreshold[SeverityLevel.Critical]) {
            return SeverityLevel.Critical;
        }
        if (nextWarningCount >= warningEscalationThreshold[SeverityLevel.Major]) {
            return SeverityLevel.Major;
        }
        if (nextWarningCount >= warningEscalationThreshold[SeverityLevel.Moderate]) {
            return SeverityLevel.Moderate;
        }
        return SeverityLevel.Minor;
    }

    /// @notice Calculate challenger reward
    function _calculateChallengerReward(
        uint128 slashedAmount
    ) internal view returns (uint128 reward, uint128 fee) {
        if (!challengerConfig.enabled || slashedAmount == 0) {
            return (0, 0);
        }

        uint256 rawReward = (uint256(slashedAmount) * challengerConfig.rewardPercentage) / 10000;
        uint256 rawFee = (uint256(slashedAmount) * challengerConfig.protocolFeePercentage) / 10000;

        // Apply min/max bounds
        if (rawReward < challengerConfig.minimumReward) {
            rawReward = challengerConfig.minimumReward;
        }
        if (rawReward > challengerConfig.maximumReward) {
            rawReward = challengerConfig.maximumReward;
        }

        reward = uint128(rawReward);
        fee = uint128(rawFee);
    }

    /// @notice Map violation type to error code
    function _mapViolationToErrorCode(ViolationType violationType) internal pure returns (ErrorCode) {
        if (violationType == ViolationType.InvalidProof) {
            return ErrorCode.PROOF_VERIFICATION_FAILED;
        }
        if (violationType == ViolationType.CommitmentMismatch) {
            return ErrorCode.OLD_COMMITMENT_WRONG;
        }
        if (violationType == ViolationType.ErrorBoundExceeded) {
            return ErrorCode.ERROR_BOUND_EXCEEDED;
        }
        if (violationType == ViolationType.DataIntegrityFailure) {
            return ErrorCode.DATA_ROOT_MISMATCH;
        }
        if (violationType == ViolationType.DoubleSubmission) {
            return ErrorCode.DOUBLE_SUBMISSION_DETECTED;
        }
        if (violationType == ViolationType.TimeoutViolation) {
            return ErrorCode.SUBMISSION_TOO_LATE;
        }
        if (violationType == ViolationType.MaliciousGradient) {
            return ErrorCode.GRADIENT_POISONING_DETECTED;
        }
        if (violationType == ViolationType.ReplayAttack) {
            return ErrorCode.REPLAY_DETECTED;
        }
        return ErrorCode.UNKNOWN_ERROR;
    }

    // ============ View Functions ============

    /// @notice Get evidence record
    function getEvidence(uint256 evidenceId) external view returns (Evidence memory) {
        return evidenceRecords[evidenceId];
    }

    /// @notice Get crypto evidence
    function getCryptoEvidence(uint256 evidenceId) external view returns (CryptoEvidence memory) {
        return _cryptoEvidence[evidenceId];
    }

    /// @notice Get warning record for a prover
    function getWarningRecord(address prover) external view returns (WarningRecord memory) {
        return warningRecords[prover];
    }

    /// @notice Get all evidence for a prover
    function getProverEvidence(address prover) external view returns (uint256[] memory) {
        return _proverEvidence[prover];
    }

    /// @notice Get evidence for a specific round
    function getRoundEvidence(uint256 modelId, uint256 roundId) external view returns (uint256[] memory) {
        return _roundEvidence[modelId][roundId];
    }

    /// @notice Get all evidence for a challenger
    function getChallengerEvidence(address challenger) external view returns (uint256[] memory) {
        return _challengerEvidence[challenger];
    }

    /// @notice Get dispute details
    function getDispute(uint256 disputeId) external view returns (Dispute memory) {
        return disputes[disputeId];
    }

    /// @notice Get appeal details
    function getAppeal(uint256 appealId) external view returns (Appeal memory) {
        return appeals[appealId];
    }

    /// @notice Check if evidence is disputed
    function isDisputed(uint256 evidenceId) external view returns (bool) {
        return evidenceRecords[evidenceId].status == EvidenceStatus.Disputed;
    }

    /// @notice Check if dispute period is active
    function isDisputePeriodActive(uint256 evidenceId) external view returns (bool) {
        return block.timestamp < evidenceRecords[evidenceId].disputeDeadline;
    }

    /// @notice Check if appeal period is active
    function isAppealPeriodActive(uint256 evidenceId) external view returns (bool) {
        return block.timestamp < evidenceRecords[evidenceId].appealDeadline;
    }

    /// @notice Get evidence count for a prover
    function getProverEvidenceCount(address prover) external view returns (uint256) {
        return _proverEvidence[prover].length;
    }

    /// @notice Check if prover is banned
    function isProverBanned(address prover) external view returns (bool) {
        return warningRecords[prover].permanentlyBanned;
    }

    /// @notice Get slashing percentage for severity level
    function getSlashPercentage(SeverityLevel severity) external view returns (uint16) {
        return severitySlashPercentage[severity];
    }

    /// @notice Calculate expected slash amount for severity
    function calculateSlashAmount(
        uint256 stakeAmount,
        SeverityLevel severity
    ) external view returns (uint256) {
        return (stakeAmount * severitySlashPercentage[severity]) / 10000;
    }

    // ============ Admin Functions ============

    /// @notice Update dispute period
    function setDisputePeriod(uint256 _disputePeriod) external onlyOwner {
        require(_disputePeriod >= 1 days, "Period too short");
        require(_disputePeriod <= 30 days, "Period too long");
        disputePeriod = _disputePeriod;
    }

    /// @notice Update appeal period
    function setAppealPeriod(uint256 _appealPeriod) external onlyOwner {
        require(_appealPeriod >= 3 days, "Period too short");
        require(_appealPeriod <= 60 days, "Period too long");
        appealPeriod = _appealPeriod;
    }

    /// @notice Update dispute stake requirement
    function setDisputeStakeRequired(uint256 _stake) external onlyOwner {
        disputeStakeRequired = _stake;
    }

    /// @notice Update appeal stake requirement
    function setAppealStakeRequired(uint256 _stake) external onlyOwner {
        appealStakeRequired = _stake;
    }

    /// @notice Update severity slash percentage
    function setSeveritySlashPercentage(
        SeverityLevel severity,
        uint16 percentage
    ) external onlyOwner {
        require(percentage <= 10000, "Max 100%");
        severitySlashPercentage[severity] = percentage;
    }

    /// @notice Update warning escalation threshold
    function setWarningEscalationThreshold(
        SeverityLevel severity,
        uint256 threshold
    ) external onlyOwner {
        warningEscalationThreshold[severity] = threshold;
    }

    /// @notice Update challenger configuration
    function setChallengerConfig(ChallengerConfig calldata config) external onlyOwner {
        require(config.rewardPercentage <= 5000, "Max 50% reward");
        require(config.protocolFeePercentage <= 2000, "Max 20% fee");
        challengerConfig = config;
        emit ChallengerConfigUpdated(config);
    }

    /// @notice Add arbitrator
    function addArbitrator(address arbitrator) external onlyOwner {
        require(arbitrator != address(0), "Invalid address");
        require(!isArbitrator[arbitrator], "Already arbitrator");
        isArbitrator[arbitrator] = true;
        arbitratorCount++;
        emit ArbitratorAdded(arbitrator);
    }

    /// @notice Remove arbitrator
    function removeArbitrator(address arbitrator) external onlyOwner {
        require(isArbitrator[arbitrator], "Not arbitrator");
        isArbitrator[arbitrator] = false;
        arbitratorCount--;
        emit ArbitratorRemoved(arbitrator);
    }

    /// @notice Update coordinator
    function setCoordinator(address _coordinator) external onlyOwner {
        coordinator = _coordinator;
    }

    /// @notice Update treasury
    function setTreasury(address _treasury) external onlyOwner {
        treasury = _treasury;
    }

    /// @notice Transfer ownership
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid address");
        owner = newOwner;
    }

    /// @notice Clear permanent ban (emergency function)
    function clearBan(address prover) external onlyOwner {
        warningRecords[prover].permanentlyBanned = false;
    }

    /// @notice Reset warning record (emergency function)
    function resetWarningRecord(address prover) external onlyOwner {
        delete warningRecords[prover];
    }

    /// @notice Receive ETH for rewards
    receive() external payable {}
}
