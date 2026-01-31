// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// @title SlashingEvidence
/// @notice Detailed slashing evidence verification for HELIX protocol
/// @dev Provides comprehensive evidence recording, verification, and dispute resolution
///      for slashing events with cryptographic proof support
contract SlashingEvidence {
    // ============ Enums ============

    /// @notice Types of violations that can trigger slashing
    enum ViolationType {
        InvalidProof,           // ZK proof verification failed
        CommitmentMismatch,     // Old/new commitment doesn't match
        ErrorBoundExceeded,     // Error bound exceeds maximum
        DataIntegrityFailure,   // Training data commitment mismatch
        DoubleSubmission,       // Same prover submitted twice
        TimeoutViolation,       // Submitted after deadline
        MaliciousGradient,      // Detected gradient poisoning
        ProtocolViolation       // Generic protocol rule violation
    }

    /// @notice Status of an evidence submission
    enum EvidenceStatus {
        Pending,       // Evidence submitted, awaiting verification
        Verified,      // Evidence verified as valid
        Rejected,      // Evidence rejected as invalid
        Disputed,      // Evidence is being disputed
        Resolved       // Dispute resolved
    }

    // ============ Structs (Gas Optimized) ============

    /// @notice Complete evidence record for a slashing event
    /// @dev Packed for storage efficiency where possible
    struct Evidence {
        // Slot 1: Core identifiers
        address prover;               // 20 bytes - who was slashed
        uint64 modelId;               // 8 bytes
        uint32 roundId;               // 4 bytes

        // Slot 2: Status and type
        ViolationType violationType;  // 1 byte
        EvidenceStatus status;        // 1 byte
        uint40 timestamp;             // 5 bytes
        uint40 disputeDeadline;       // 5 bytes
        address submitter;            // 20 bytes

        // Slot 3: Financial
        uint128 slashedAmount;        // 16 bytes
        uint128 remainingStake;       // 16 bytes

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
        uint256 errorBound;           // Error bound if applicable
        bytes32 dataCommitment;       // Data commitment if applicable
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
    }

    // ============ State Variables ============

    /// @notice Contract owner
    address public owner;

    /// @notice Coordinator contract
    address public coordinator;

    /// @notice Counter for evidence IDs
    uint256 public nextEvidenceId;

    /// @notice Counter for dispute IDs
    uint256 public nextDisputeId;

    /// @notice Dispute period in seconds
    uint256 public disputePeriod;

    /// @notice Evidence records
    mapping(uint256 => Evidence) public evidenceRecords;

    /// @notice Crypto evidence for each evidence ID
    mapping(uint256 => CryptoEvidence) internal _cryptoEvidence;

    /// @notice Dispute records
    mapping(uint256 => Dispute) public disputes;

    /// @notice Mapping of prover to their evidence IDs
    mapping(address => uint256[]) internal _proverEvidence;

    /// @notice Mapping of model+round to evidence IDs
    mapping(uint256 => mapping(uint256 => uint256[])) internal _roundEvidence;

    /// @notice Verified evidence hashes (prevent duplicates)
    mapping(bytes32 => bool) public verifiedEvidenceHashes;

    // ============ Events ============

    event EvidenceSubmitted(
        uint256 indexed evidenceId,
        address indexed prover,
        uint64 indexed modelId,
        ViolationType violationType,
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
        string reason
    );

    event DisputeResolved(
        uint256 indexed disputeId,
        uint256 indexed evidenceId,
        bool inFavorOfProver,
        address resolver
    );

    event CryptoEvidenceStored(
        uint256 indexed evidenceId,
        bytes32 proofHash,
        bytes32 expectedCommitment,
        bytes32 actualCommitment
    );

    // ============ Modifiers ============

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    modifier onlyCoordinator() {
        require(msg.sender == coordinator, "Only coordinator");
        _;
    }

    modifier evidenceExists(uint256 evidenceId) {
        require(evidenceRecords[evidenceId].timestamp > 0, "Evidence does not exist");
        _;
    }

    // ============ Constructor ============

    constructor(address _coordinator) {
        owner = msg.sender;
        coordinator = _coordinator;
        disputePeriod = 7 days; // Default 7 day dispute period
    }

    // ============ Evidence Submission Functions ============

    /// @notice Submit slashing evidence (called by coordinator)
    /// @param prover The address being slashed
    /// @param modelId The model ID
    /// @param roundId The round ID
    /// @param violationType Type of violation
    /// @param slashedAmount Amount being slashed
    /// @param remainingStake Remaining stake after slashing
    /// @param proofHash Hash of the relevant proof
    /// @param reason Detailed reason for slashing
    /// @return evidenceId The ID of the created evidence record
    function submitEvidence(
        address prover,
        uint64 modelId,
        uint32 roundId,
        ViolationType violationType,
        uint128 slashedAmount,
        uint128 remainingStake,
        bytes32 proofHash,
        string calldata reason
    ) external onlyCoordinator returns (uint256 evidenceId) {
        // Compute evidence hash for deduplication
        bytes32 evidenceHash = keccak256(abi.encodePacked(
            prover, modelId, roundId, violationType, proofHash, block.timestamp
        ));

        require(!verifiedEvidenceHashes[evidenceHash], "Duplicate evidence");

        evidenceId = nextEvidenceId++;

        evidenceRecords[evidenceId] = Evidence({
            prover: prover,
            modelId: modelId,
            roundId: roundId,
            violationType: violationType,
            status: EvidenceStatus.Pending,
            timestamp: uint40(block.timestamp),
            disputeDeadline: uint40(block.timestamp + disputePeriod),
            submitter: msg.sender,
            slashedAmount: slashedAmount,
            remainingStake: remainingStake,
            proofHash: proofHash,
            evidenceHash: evidenceHash,
            reason: reason
        });

        verifiedEvidenceHashes[evidenceHash] = true;
        _proverEvidence[prover].push(evidenceId);
        _roundEvidence[modelId][roundId].push(evidenceId);

        emit EvidenceSubmitted(evidenceId, prover, modelId, violationType, slashedAmount);
    }

    /// @notice Store cryptographic evidence for an evidence record
    /// @param evidenceId The evidence ID
    /// @param invalidProof The proof that failed verification
    /// @param publicInputs The public inputs used
    /// @param expectedCommitment Expected commitment value
    /// @param actualCommitment Actual commitment provided
    /// @param errorBound Error bound if applicable
    /// @param dataCommitment Data commitment if applicable
    function storeCryptoEvidence(
        uint256 evidenceId,
        bytes calldata invalidProof,
        uint256[] calldata publicInputs,
        bytes32 expectedCommitment,
        bytes32 actualCommitment,
        uint256 errorBound,
        bytes32 dataCommitment
    ) external onlyCoordinator evidenceExists(evidenceId) {
        _cryptoEvidence[evidenceId] = CryptoEvidence({
            invalidProof: invalidProof,
            publicInputs: publicInputs,
            expectedCommitment: expectedCommitment,
            actualCommitment: actualCommitment,
            errorBound: errorBound,
            dataCommitment: dataCommitment
        });

        emit CryptoEvidenceStored(
            evidenceId,
            keccak256(invalidProof),
            expectedCommitment,
            actualCommitment
        );
    }

    // ============ Verification Functions ============

    /// @notice Verify and finalize evidence (after dispute period)
    /// @param evidenceId The evidence to verify
    function verifyEvidence(uint256 evidenceId) external onlyOwner evidenceExists(evidenceId) {
        Evidence storage evidence = evidenceRecords[evidenceId];

        require(evidence.status == EvidenceStatus.Pending, "Not pending");
        require(block.timestamp >= evidence.disputeDeadline, "Dispute period active");

        evidence.status = EvidenceStatus.Verified;

        emit EvidenceVerified(evidenceId, msg.sender, EvidenceStatus.Verified);
    }

    /// @notice Reject evidence
    /// @param evidenceId The evidence to reject
    function rejectEvidence(uint256 evidenceId) external onlyOwner evidenceExists(evidenceId) {
        Evidence storage evidence = evidenceRecords[evidenceId];

        require(evidence.status == EvidenceStatus.Pending || evidence.status == EvidenceStatus.Disputed, "Cannot reject");

        evidence.status = EvidenceStatus.Rejected;

        emit EvidenceVerified(evidenceId, msg.sender, EvidenceStatus.Rejected);
    }

    // ============ Dispute Functions ============

    /// @notice File a dispute against slashing evidence
    /// @param evidenceId The evidence to dispute
    /// @param disputeEvidenceHash Hash of counter-evidence
    /// @param reason Reason for dispute
    /// @return disputeId The created dispute ID
    function fileDispute(
        uint256 evidenceId,
        bytes32 disputeEvidenceHash,
        string calldata reason
    ) external evidenceExists(evidenceId) returns (uint256 disputeId) {
        Evidence storage evidence = evidenceRecords[evidenceId];

        require(evidence.status == EvidenceStatus.Pending, "Cannot dispute");
        require(block.timestamp < evidence.disputeDeadline, "Dispute period ended");
        require(msg.sender == evidence.prover || msg.sender == owner, "Not authorized");

        disputeId = nextDisputeId++;

        disputes[disputeId] = Dispute({
            evidenceId: evidenceId,
            disputer: msg.sender,
            filedAt: uint40(block.timestamp),
            resolvedAt: 0,
            inFavorOfProver: false,
            disputeEvidenceHash: disputeEvidenceHash,
            disputeReason: reason
        });

        evidence.status = EvidenceStatus.Disputed;

        emit DisputeFiled(disputeId, evidenceId, msg.sender, reason);
    }

    /// @notice Resolve a dispute
    /// @param disputeId The dispute to resolve
    /// @param inFavorOfProver Whether the resolution favors the prover
    function resolveDispute(
        uint256 disputeId,
        bool inFavorOfProver
    ) external onlyOwner {
        Dispute storage dispute = disputes[disputeId];
        require(dispute.filedAt > 0, "Dispute does not exist");
        require(dispute.resolvedAt == 0, "Already resolved");

        dispute.resolvedAt = uint40(block.timestamp);
        dispute.inFavorOfProver = inFavorOfProver;

        Evidence storage evidence = evidenceRecords[dispute.evidenceId];

        if (inFavorOfProver) {
            evidence.status = EvidenceStatus.Rejected;
        } else {
            evidence.status = EvidenceStatus.Verified;
        }

        emit DisputeResolved(disputeId, dispute.evidenceId, inFavorOfProver, msg.sender);
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

    /// @notice Get all evidence for a prover
    function getProverEvidence(address prover) external view returns (uint256[] memory) {
        return _proverEvidence[prover];
    }

    /// @notice Get evidence for a specific round
    function getRoundEvidence(uint256 modelId, uint256 roundId) external view returns (uint256[] memory) {
        return _roundEvidence[modelId][roundId];
    }

    /// @notice Get dispute details
    function getDispute(uint256 disputeId) external view returns (Dispute memory) {
        return disputes[disputeId];
    }

    /// @notice Check if evidence is disputed
    function isDisputed(uint256 evidenceId) external view returns (bool) {
        return evidenceRecords[evidenceId].status == EvidenceStatus.Disputed;
    }

    /// @notice Check if dispute period is active
    function isDisputePeriodActive(uint256 evidenceId) external view returns (bool) {
        return block.timestamp < evidenceRecords[evidenceId].disputeDeadline;
    }

    /// @notice Get evidence count for a prover
    function getProverEvidenceCount(address prover) external view returns (uint256) {
        return _proverEvidence[prover].length;
    }

    // ============ Admin Functions ============

    /// @notice Update dispute period
    function setDisputePeriod(uint256 _disputePeriod) external onlyOwner {
        require(_disputePeriod >= 1 days, "Period too short");
        require(_disputePeriod <= 30 days, "Period too long");
        disputePeriod = _disputePeriod;
    }

    /// @notice Update coordinator
    function setCoordinator(address _coordinator) external onlyOwner {
        coordinator = _coordinator;
    }

    /// @notice Transfer ownership
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid address");
        owner = newOwner;
    }
}
