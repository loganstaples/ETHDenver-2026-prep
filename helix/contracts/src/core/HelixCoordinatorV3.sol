// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "../interfaces/IHelixVerifier.sol";
import "../verification/PoseidonHasher.sol";
import "../token/Staking.sol";
import "../token/Rewards.sol";
import "../core/ModelRegistry.sol";

/// @title HelixCoordinatorV3
/// @notice Production coordinator integrating token staking, rewards, model registry, and security hardening
/// @dev Upgrades from V2:
///   - Token-based staking via Staking.sol (replaces raw ETH)
///   - Reward distribution via Rewards.sol on valid proof submission
///   - ModelRegistry integration for checkpoint history
///   - Proof replay protection via proof hash tracking
///   - ReentrancyGuard on all state-changing external functions
///   - Treasury zero-address validation
///   - Reentrancy-safe challenger rewards (state updates before external calls)
contract HelixCoordinatorV3 is ReentrancyGuard {
    using SafeERC20 for IERC20;

    // ============ Structs ============

    /// @notice Internal model tracking (gas-efficient access to round/commitment state)
    struct Model {
        string ipfsHash;
        uint256 currentCommitment;
        address owner;
        uint32 currentRound;
        bool active;
    }

    /// @notice Round information
    struct Round {
        uint256 modelCommitment;
        uint256 newCommitment;
        uint40 deadline;
        bool isCompleted;
        address prover;
    }

    /// @notice Slashing record for audit trail
    struct SlashingRecord {
        address prover;
        uint64 modelId;
        uint32 roundId;
        uint128 amount;
        string reason;
        uint40 timestamp;
    }

    /// @notice Data for a single proof submission in a batch
    struct ProofSubmissionData {
        uint256 modelId;
        uint256 roundId;
        bytes proof;
        uint256[] publicInputs;
    }

    /// @notice Per-round participant proof data
    struct RoundParticipant {
        uint256 newCommitmentLo;
        uint256 newCommitmentHi;
        uint256 loss;
        uint256 errorBound;
        uint40 submittedAt;
        bytes32 proofHash;
    }

    /// @notice Extended round data for multi-participant rounds
    struct RoundExt {
        uint32 minParticipants;
        uint32 validProofs;
        uint40 disputeDeadline;
        uint40 startedAt;
        bool finalized;
        address bestProver;
        uint256 bestLoss;
        uint256 bestNewCommitment;
    }

    /// @notice Training job — model owner deposits tokens to pay workers
    struct TrainingJob {
        uint256 modelId;
        address jobOwner;
        uint256 totalDeposit;
        uint256 totalRounds;
        uint256 completedRounds;
        uint256 feePerRound;
        uint256 remainingBalance;
        bool active;
    }

    // ============ Constants ============

    uint256 public constant DEFAULT_MAX_ERROR_BOUND = 1e18;
    uint8 public constant DEFAULT_REQUIRED_GUARDIANS = 2;
    uint256 public constant DEFAULT_RECOVERY_TIMELOCK = 48 hours;
    uint16 public constant DEFAULT_CHALLENGER_REWARD_PCT = 1000;   // 10%
    uint16 public constant MAX_REWARD_PERCENTAGE = 5000;           // 50%
    uint8 public constant EXPECTED_PUBLIC_INPUTS = 8;
    uint256 public constant MIN_RECOVERY_TIMELOCK = 1 hours;
    uint256 public constant MAX_RECOVERY_TIMELOCK = 30 days;
    uint256 public constant DISPUTE_PERIOD = 1 hours;
    uint32 public constant DEFAULT_MIN_PARTICIPANTS = 1;

    // ============ External Contracts ============

    /// @notice ZK proof verifier (Halo2Verifier)
    IHelixVerifier public immutable verifier;

    /// @notice Token-based staking contract
    Staking public stakingContract;

    /// @notice Reward distribution contract
    Rewards public rewardsContract;

    /// @notice Model checkpoint registry
    ModelRegistry public modelRegistry;

    // ============ State Variables ============

    /// @notice Contract owner
    address public owner;

    /// @notice Treasury for slashed funds
    address public treasury;

    /// @notice Counter for model IDs
    uint32 public nextModelId;

    /// @notice Maximum allowed error bound per step (in fixed-point units)
    uint256 public maxErrorBound;

    /// @notice Whether the contract is paused
    bool public paused;

    // ============ Multi-Sig Emergency Pause State ============

    uint8 public requiredGuardians;
    uint64 public pauseNonce;
    uint256 public recoveryTimeLock;
    uint256 public pendingRecoveryTime;
    address public pendingRecoveryOwner;
    mapping(address => bool) public isGuardian;
    uint8 public guardianCount;
    mapping(uint64 => mapping(address => bool)) public pauseApprovals;
    mapping(uint64 => uint8) public pauseApprovalCount;

    // ============ Challenger Reward Configuration ============

    uint16 public challengerRewardPercentage;
    bool public challengerRewardsEnabled;

    // ============ Mappings ============

    /// @notice Model registry (internal tracking)
    mapping(uint256 => Model) public models;

    /// @notice Round data: modelId => roundId => Round
    mapping(uint256 => mapping(uint256 => Round)) public rounds;

    /// @notice Accumulated error bound per model
    mapping(uint256 => uint256) public accumulatedErrorBound;

    /// @notice Proof replay protection: hash(proof || publicInputs) => used
    mapping(bytes32 => bool) public usedProofHashes;

    /// @notice Slashing records for audit trail
    SlashingRecord[] public slashingRecords;

    /// @notice Data commitment per model (merkle root)
    mapping(uint256 => bytes32) public modelDataCommitment;

    /// @notice Data root per round
    mapping(uint256 => mapping(uint256 => bytes32)) public roundDataRoot;

    /// @notice Extended round data for multi-participant rounds
    mapping(uint256 => mapping(uint256 => RoundExt)) public roundsExt;

    /// @notice Per-round participant proof data
    mapping(uint256 => mapping(uint256 => mapping(address => RoundParticipant))) public roundParticipants;

    /// @notice List of participants per round
    mapping(uint256 => mapping(uint256 => address[])) internal roundParticipantList;

    /// @notice Counter for training job IDs
    uint256 public nextJobId;

    /// @notice Training jobs: jobId => TrainingJob
    mapping(uint256 => TrainingJob) public trainingJobs;

    /// @notice Active job per model: modelId => jobId (0 = no job)
    mapping(uint256 => uint256) public activeModelJob;

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
        uint256 deadline,
        uint256 modelCommitment
    );
    event ProofSubmitted(
        uint256 indexed modelId,
        uint256 indexed roundId,
        address indexed prover,
        uint256 newCommitment,
        uint256 errorBound
    );
    event RoundCompleted(
        uint256 indexed modelId,
        uint256 indexed roundId,
        uint256 newCommitment,
        uint256 totalErrorBound
    );
    event InvalidProofDetected(
        uint256 indexed modelId,
        uint256 indexed roundId,
        address indexed prover,
        bytes32 proofHash
    );
    event BatchProofSubmitted(address indexed submitter, uint256 totalProofs);
    event ProofReplayBlocked(
        bytes32 indexed proofHash,
        address indexed submitter
    );
    event ModelStateChanged(
        uint256 indexed modelId,
        bool active,
        address indexed changedBy
    );
    event ConfigUpdated(
        string indexed parameter,
        uint256 oldValue,
        uint256 newValue
    );
    event EmergencyPauseChanged(bool isPaused, address indexed changedBy);
    event GuardianUpdated(address indexed guardian, bool isActive, address indexed changedBy);
    event PauseApprovalReceived(uint64 indexed nonce, address indexed guardian, uint8 currentApprovals, uint8 requiredApprovals);
    event MultiSigPauseExecuted(uint64 indexed nonce);
    event RecoveryInitiated(address indexed newOwner, uint256 executionTime, address indexed initiatedBy);
    event RecoveryExecuted(address indexed oldOwner, address indexed newOwner);
    event RecoveryCancelled(address indexed cancelledBy);
    event ChallengerConfigUpdated(uint16 rewardPercentage, bool enabled);
    event DataCommitmentSet(uint256 indexed modelId, bytes32 indexed dataRoot, address indexed setBy);
    event RoundDataCommitted(uint256 indexed modelId, uint256 indexed roundId, bytes32 dataRoot);
    event RoundExpired(uint256 indexed modelId, uint256 indexed roundId, uint32 validProofs, uint32 required);
    event RoundFinalized(uint256 indexed modelId, uint256 indexed roundId, address indexed bestProver, uint256 bestLoss);
    event TrainingJobCreated(uint256 indexed modelId, uint256 indexed jobId, uint256 deposit, uint256 rounds);
    event TrainingJobFunded(uint256 indexed modelId, uint256 indexed roundId, uint256 feeAmount);
    event TrainingJobRefunded(uint256 indexed modelId, uint256 indexed roundId, uint256 refundAmount);
    event TrainingJobCancelled(uint256 indexed jobId, uint256 refundAmount);

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

    modifier onlyGuardian() {
        require(isGuardian[msg.sender], "Only guardian");
        _;
    }

    // ============ Constructor ============

    /// @param _verifier The Halo2Verifier address
    /// @param _staking The Staking contract address
    /// @param _rewards The Rewards contract address
    /// @param _registry The ModelRegistry address
    /// @param _treasury Treasury address for slashed funds
    constructor(
        address _verifier,
        address _staking,
        address _rewards,
        address _registry,
        address _treasury
    ) {
        require(_verifier != address(0), "Invalid verifier");
        require(_staking != address(0), "Invalid staking");
        require(_rewards != address(0), "Invalid rewards");
        require(_registry != address(0), "Invalid registry");
        require(_treasury != address(0), "Invalid treasury");

        verifier = IHelixVerifier(_verifier);
        stakingContract = Staking(_staking);
        rewardsContract = Rewards(_rewards);
        modelRegistry = ModelRegistry(_registry);
        treasury = _treasury;
        owner = msg.sender;

        maxErrorBound = DEFAULT_MAX_ERROR_BOUND;

        // Multi-sig pause
        requiredGuardians = DEFAULT_REQUIRED_GUARDIANS;
        recoveryTimeLock = DEFAULT_RECOVERY_TIMELOCK;

        // Challenger rewards
        challengerRewardPercentage = DEFAULT_CHALLENGER_REWARD_PCT;
        challengerRewardsEnabled = true;

        // Owner is first guardian
        isGuardian[msg.sender] = true;
        guardianCount = 1;
    }

    // ============ Model Management ============

    /// @notice Registers a new model and creates initial checkpoint in ModelRegistry
    /// @param name Model name
    /// @param description Model description
    /// @param ipfsHash IPFS hash of initial model weights
    /// @param initialCommitment Hash commitment of initial weights
    function registerModel(
        string memory name,
        string memory description,
        string memory ipfsHash,
        uint256 initialCommitment
    ) external whenNotPaused nonReentrant returns (uint256 modelId) {
        modelId = nextModelId++;

        models[modelId] = Model({
            ipfsHash: ipfsHash,
            currentCommitment: initialCommitment,
            owner: msg.sender,
            currentRound: 0,
            active: true
        });

        // Register in ModelRegistry for checkpoint history
        modelRegistry.registerModel(name, description, ipfsHash, bytes32(initialCommitment));

        emit ModelRegistered(modelId, msg.sender, initialCommitment, ipfsHash);
    }

    /// @notice Starts a new training round (backward compatible, single-participant auto-finalize)
    function startRound(
        uint256 modelId,
        uint256 duration
    ) external whenNotPaused nonReentrant modelExists(modelId) {
        _startRound(modelId, duration, DEFAULT_MIN_PARTICIPANTS);
    }

    /// @notice Starts a new training round requiring multiple participants
    function startRoundWithThreshold(
        uint256 modelId,
        uint256 duration,
        uint32 minParticipants
    ) external whenNotPaused nonReentrant modelExists(modelId) {
        require(minParticipants > 0, "Min participants must be > 0");
        _startRound(modelId, duration, minParticipants);
    }

    function _startRound(uint256 modelId, uint256 duration, uint32 minParticipants) internal {
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

        roundsExt[modelId][roundId] = RoundExt({
            minParticipants: minParticipants,
            validProofs: 0,
            disputeDeadline: uint40(block.timestamp + duration + DISPUTE_PERIOD),
            startedAt: uint40(block.timestamp),
            finalized: false,
            bestProver: address(0),
            bestLoss: type(uint256).max,
            bestNewCommitment: 0
        });

        emit RoundStarted(modelId, roundId, deadline, model.currentCommitment);
    }

    // ============ Proof Submission ============

    /// @notice Submits a training proof for a round
    function submitProof(
        uint256 modelId,
        uint256 roundId,
        bytes memory proof,
        uint256[] memory publicInputs
    ) external whenNotPaused nonReentrant {
        _processProof(msg.sender, modelId, roundId, proof, publicInputs);
    }

    /// @notice Submits multiple proofs in a single transaction to amortize base tx costs
    function submitProofBatch(ProofSubmissionData[] calldata submissions) external whenNotPaused nonReentrant {
        uint256 n = submissions.length;
        require(n > 0, "Empty batch");
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
        require(models[modelId].owner != address(0), "Model does not exist");

        // Check staker has sufficient tokens via Staking.sol
        require(stakingContract.canParticipate(prover), "Insufficient stake or not active");

        Model storage model = models[modelId];
        Round storage round = rounds[modelId][roundId];
        RoundExt storage ext = roundsExt[modelId][roundId];

        // Validate round
        require(roundId == model.currentRound, "Invalid round");
        require(!round.isCompleted, "Round completed");
        require(block.timestamp <= round.deadline, "Round expired");

        // Validate public inputs count
        require(publicInputs.length == EXPECTED_PUBLIC_INPUTS, "Invalid public inputs count");

        // Proof replay protection
        bytes32 proofHash = keccak256(abi.encodePacked(proof, publicInputs));
        require(!usedProofHashes[proofHash], "Proof already used");
        usedProofHashes[proofHash] = true;

        // Reconstruct and validate old commitment
        uint256 oldCommitmentFromProof = _hashPair(publicInputs[0], publicInputs[1]);
        require(oldCommitmentFromProof == round.modelCommitment, "Old commitment mismatch");

        // Validate error bound
        uint256 stepErrorBound = publicInputs[5];
        require(stepErrorBound <= maxErrorBound, "Error bound exceeds maximum");

        // Validate error checksum
        uint256 errorChecksum = publicInputs[7];
        uint256 expectedChecksum = _computeErrorChecksum(
            stepErrorBound,
            publicInputs[6],
            modelId,
            maxErrorBound
        );
        require(errorChecksum == expectedChecksum, "Error checksum mismatch");

        // Verify the proof
        bool valid = verifier.verifyProof(proof, publicInputs);

        if (!valid) {
            // Slash via Staking.sol - invalid proofs are Major severity (50% slash)
            stakingContract.slashWithSeverity(
                prover,
                Staking.SeverityLevel.Major,
                Staking.ViolationType.InvalidProof,
                address(0),
                "Invalid ZK proof submitted"
            );

            emit InvalidProofDetected(modelId, roundId, prover, proofHash);

            // Record slashing
            slashingRecords.push(SlashingRecord({
                prover: prover,
                modelId: uint64(modelId),
                roundId: uint32(roundId),
                amount: 0, // Actual amount determined by Staking.sol severity
                reason: "Invalid proof",
                timestamp: uint40(block.timestamp)
            }));
            return;
        }

        // Record participant data
        uint256 loss = publicInputs[4];
        roundParticipants[modelId][roundId][prover] = RoundParticipant({
            newCommitmentLo: publicInputs[2],
            newCommitmentHi: publicInputs[3],
            loss: loss,
            errorBound: stepErrorBound,
            submittedAt: uint40(block.timestamp),
            proofHash: proofHash
        });
        roundParticipantList[modelId][roundId].push(prover);
        ext.validProofs++;

        // Track best (lowest) loss
        if (loss < ext.bestLoss) {
            ext.bestLoss = loss;
            ext.bestProver = prover;
            ext.bestNewCommitment = _hashPair(publicInputs[2], publicInputs[3]);
        }

        uint256 newCommitment = _hashPair(publicInputs[2], publicInputs[3]);
        emit ProofSubmitted(modelId, roundId, prover, newCommitment, stepErrorBound);

        // For single-participant rounds (backward compat), auto-finalize
        if (ext.minParticipants <= 1) {
            _finalizeRound(modelId, roundId);
        }
    }

    // ============ Round Finalization ============

    /// @notice Finalize a multi-participant round after dispute period
    function finalizeRound(uint256 modelId, uint256 roundId) external whenNotPaused nonReentrant {
        RoundExt storage ext = roundsExt[modelId][roundId];
        Round storage round = rounds[modelId][roundId];

        require(!round.isCompleted, "Round already completed");
        require(!ext.finalized, "Round already finalized");
        require(ext.minParticipants > 1, "Use submitProof for single-participant rounds");
        require(block.timestamp > ext.disputeDeadline, "Dispute period not ended");
        require(ext.validProofs >= ext.minParticipants, "Insufficient participants");

        _finalizeRound(modelId, roundId);
        emit RoundFinalized(modelId, roundId, ext.bestProver, ext.bestLoss);
    }

    /// @notice Expire a round that didn't meet participant threshold
    function expireRound(uint256 modelId, uint256 roundId) external whenNotPaused nonReentrant {
        Round storage round = rounds[modelId][roundId];
        RoundExt storage ext = roundsExt[modelId][roundId];

        require(!round.isCompleted, "Round already completed");
        require(!ext.finalized, "Round already finalized");
        require(block.timestamp > round.deadline, "Submission still open");
        require(ext.validProofs < ext.minParticipants, "Threshold met, use finalizeRound");

        round.isCompleted = true;
        ext.finalized = true;

        _refundRoundFees(modelId, roundId);

        emit RoundExpired(modelId, roundId, ext.validProofs, ext.minParticipants);
    }

    /// @dev Internal finalization - updates model commitment, registry, rewards
    function _finalizeRound(uint256 modelId, uint256 roundId) internal {
        Model storage model = models[modelId];
        Round storage round = rounds[modelId][roundId];
        RoundExt storage ext = roundsExt[modelId][roundId];

        address bestProver = ext.bestProver;
        uint256 newCommitment = ext.bestNewCommitment;
        RoundParticipant storage best = roundParticipants[modelId][roundId][bestProver];

        model.currentCommitment = newCommitment;
        round.newCommitment = newCommitment;
        round.isCompleted = true;
        round.prover = bestProver;
        ext.finalized = true;

        uint256 stepErrorBound = best.errorBound;
        uint256 newAccumulatedError = accumulatedErrorBound[modelId] + stepErrorBound;
        accumulatedErrorBound[modelId] = newAccumulatedError;

        modelRegistry.updateModel(modelId, bytes32(newCommitment), roundId, "", stepErrorBound, best.proofHash);

        // Register all round participants for rewards
        address[] storage participants = roundParticipantList[modelId][roundId];
        for (uint256 i = 0; i < participants.length; i++) {
            rewardsContract.registerParticipant(modelId, roundId, participants[i]);
        }
        try rewardsContract.allocateRoundRewards(modelId, roundId) {} catch {}

        _distributeRoundFees(modelId, roundId);

        emit RoundCompleted(modelId, roundId, newCommitment, newAccumulatedError);
    }

    // ============ Training Jobs ============

    /// @notice Create a training job with token deposit to pay workers
    function createTrainingJob(
        uint256 modelId,
        uint256 rounds,
        uint256 depositAmount
    ) external whenNotPaused nonReentrant modelExists(modelId) returns (uint256 jobId) {
        require(msg.sender == models[modelId].owner, "Only model owner");
        require(rounds > 0, "Must fund at least 1 round");
        require(depositAmount > 0, "Deposit must be positive");
        require(activeModelJob[modelId] == 0, "Model already has active job");

        IERC20 feeToken = stakingContract.helixToken();
        feeToken.safeTransferFrom(msg.sender, address(this), depositAmount);

        jobId = ++nextJobId;
        trainingJobs[jobId] = TrainingJob({
            modelId: modelId,
            jobOwner: msg.sender,
            totalDeposit: depositAmount,
            totalRounds: rounds,
            completedRounds: 0,
            feePerRound: depositAmount / rounds,
            remainingBalance: depositAmount,
            active: true
        });
        activeModelJob[modelId] = jobId;

        emit TrainingJobCreated(modelId, jobId, depositAmount, rounds);
    }

    /// @notice Cancel a training job and refund remaining balance
    function cancelTrainingJob(uint256 jobId) external nonReentrant {
        TrainingJob storage job = trainingJobs[jobId];
        require(job.active, "Job not active");
        require(msg.sender == job.jobOwner, "Only job owner");

        uint256 refund = job.remainingBalance;
        job.active = false;
        job.remainingBalance = 0;
        activeModelJob[job.modelId] = 0;

        if (refund > 0) {
            IERC20 feeToken = stakingContract.helixToken();
            feeToken.safeTransfer(msg.sender, refund);
        }

        emit TrainingJobCancelled(jobId, refund);
    }

    /// @dev Distribute training job fees for a completed round
    function _distributeRoundFees(uint256 modelId, uint256 roundId) internal {
        uint256 jobId = activeModelJob[modelId];
        if (jobId == 0) return;

        TrainingJob storage job = trainingJobs[jobId];
        if (!job.active || job.remainingBalance == 0) return;

        uint256 fee = job.feePerRound;
        if (fee > job.remainingBalance) {
            fee = job.remainingBalance;
        }

        job.remainingBalance -= fee;
        job.completedRounds++;

        // Distribute fee equally among round participants
        address[] storage participants = roundParticipantList[modelId][roundId];
        uint256 count = participants.length;
        if (count == 0 || fee == 0) return;

        IERC20 feeToken = stakingContract.helixToken();
        uint256 perWorker = fee / count;
        for (uint256 i = 0; i < count; i++) {
            if (perWorker > 0) {
                feeToken.safeTransfer(participants[i], perWorker);
            }
        }

        // Deactivate job if fully used
        if (job.completedRounds >= job.totalRounds || job.remainingBalance == 0) {
            job.active = false;
            activeModelJob[modelId] = 0;
        }

        emit TrainingJobFunded(modelId, roundId, fee);
    }

    /// @dev Called on round expiry — fees stay in pool for next round
    function _refundRoundFees(uint256 modelId, uint256 roundId) internal {
        // No-op: fees remain available for subsequent rounds
        emit TrainingJobRefunded(modelId, roundId, 0);
    }

    // ============ Challenge ============

    /// @notice Challenge a past proof - challenger receives reward from Staking.sol if proof is invalid
    function challengeProof(
        uint256 modelId,
        uint256 roundId,
        bytes memory proof,
        uint256[] memory publicInputs
    ) external nonReentrant {
        Round storage round = rounds[modelId][roundId];
        require(round.isCompleted, "Round not completed");
        require(round.prover != address(0), "No prover to challenge");
        require(msg.sender != round.prover, "Cannot challenge self");

        bool valid = verifier.verifyProof(proof, publicInputs);

        if (!valid) {
            // Slash via Staking.sol - challenged proofs are Major severity with challenger reward
            stakingContract.slashWithSeverity(
                round.prover,
                Staking.SeverityLevel.Major,
                Staking.ViolationType.InvalidProof,
                msg.sender,
                "Fraudulent proof challenged"
            );

            slashingRecords.push(SlashingRecord({
                prover: round.prover,
                modelId: uint64(modelId),
                roundId: uint32(roundId),
                amount: 0,
                reason: "Fraudulent proof challenged",
                timestamp: uint40(block.timestamp)
            }));
        }
    }

    // ============ Error Checksum ============

    /// @notice Computes the expected error checksum for verification
    /// @dev Matches the Rust circuit implementation using Poseidon hash:
    ///      checksum = Poseidon(Poseidon(errorBound, stepNumber), Poseidon(modelId, errorBudget))
    function _computeErrorChecksum(
        uint256 errorBound,
        uint256 stepNumber,
        uint256 modelId,
        uint256 errorBudget
    ) internal pure returns (uint256) {
        return PoseidonHasher.computeErrorChecksum(errorBound, stepNumber, modelId, errorBudget);
    }

    // ============ Internal Helpers ============

    function _hashPair(uint256 lo, uint256 hi) internal pure returns (uint256) {
        return uint256(keccak256(abi.encodePacked(lo, hi)));
    }

    // ============ View Functions ============

    function getModelState(uint256 modelId) external view returns (
        uint256 currentRound,
        uint256 currentCommitment,
        bool active
    ) {
        Model storage model = models[modelId];
        return (model.currentRound, model.currentCommitment, model.active);
    }

    function getAccumulatedErrorBound(uint256 modelId) external view returns (uint256) {
        return accumulatedErrorBound[modelId];
    }

    function isModelErrorAcceptable(uint256 modelId, uint256 maxAccumulated) external view returns (bool) {
        return accumulatedErrorBound[modelId] <= maxAccumulated;
    }

    function getSlashingRecordCount() external view returns (uint256) {
        return slashingRecords.length;
    }

    function isProofUsed(bytes32 proofHash) external view returns (bool) {
        return usedProofHashes[proofHash];
    }

    function getRoundParticipants(uint256 modelId, uint256 roundId) external view returns (address[] memory) {
        return roundParticipantList[modelId][roundId];
    }

    function getRoundExt(uint256 modelId, uint256 roundId) external view returns (
        uint32 minParticipants, uint32 validProofs, uint40 disputeDeadline,
        uint40 startedAt, bool finalized, address bestProver, uint256 bestLoss
    ) {
        RoundExt storage ext = roundsExt[modelId][roundId];
        return (ext.minParticipants, ext.validProofs, ext.disputeDeadline,
                ext.startedAt, ext.finalized, ext.bestProver, ext.bestLoss);
    }

    // ============ Admin Functions ============

    function setTreasury(address _treasury) external onlyOwner {
        require(_treasury != address(0), "Invalid treasury");
        emit ConfigUpdated("treasury", uint256(uint160(treasury)), uint256(uint160(_treasury)));
        treasury = _treasury;
    }

    function setMaxErrorBound(uint256 _maxErrorBound) external onlyOwner {
        emit ConfigUpdated("maxErrorBound", maxErrorBound, _maxErrorBound);
        maxErrorBound = _maxErrorBound;
    }

    function setStakingContract(address _staking) external onlyOwner {
        require(_staking != address(0), "Invalid staking");
        stakingContract = Staking(_staking);
    }

    function setRewardsContract(address _rewards) external onlyOwner {
        require(_rewards != address(0), "Invalid rewards");
        rewardsContract = Rewards(_rewards);
    }

    function setModelRegistry(address _registry) external onlyOwner {
        require(_registry != address(0), "Invalid registry");
        modelRegistry = ModelRegistry(_registry);
    }

    function resetAccumulatedError(uint256 modelId) external {
        require(
            msg.sender == models[modelId].owner || msg.sender == owner,
            "Not authorized"
        );
        accumulatedErrorBound[modelId] = 0;
    }

    function pauseModel(uint256 modelId) external {
        require(
            msg.sender == models[modelId].owner || msg.sender == owner,
            "Not authorized"
        );
        models[modelId].active = false;
        emit ModelStateChanged(modelId, false, msg.sender);
    }

    function resumeModel(uint256 modelId) external {
        require(
            msg.sender == models[modelId].owner || msg.sender == owner,
            "Not authorized"
        );
        models[modelId].active = true;
        emit ModelStateChanged(modelId, true, msg.sender);
    }

    // ============ Emergency Pause ============

    function emergencyPause() external onlyOwner {
        require(!paused, "Already paused");
        paused = true;
        emit EmergencyPauseChanged(true, msg.sender);
    }

    function unpause() external onlyOwner {
        require(paused, "Not paused");
        paused = false;
        emit EmergencyPauseChanged(false, msg.sender);
    }

    // ============ Multi-Sig Emergency Pause ============

    function approveEmergencyPause() external onlyGuardian {
        uint64 currentNonce = pauseNonce;
        require(!pauseApprovals[currentNonce][msg.sender], "Already approved");
        require(!paused, "Already paused");

        pauseApprovals[currentNonce][msg.sender] = true;
        pauseApprovalCount[currentNonce]++;

        emit PauseApprovalReceived(currentNonce, msg.sender, pauseApprovalCount[currentNonce], requiredGuardians);

        if (pauseApprovalCount[currentNonce] >= requiredGuardians) {
            paused = true;
            pauseNonce++;
            emit MultiSigPauseExecuted(currentNonce);
            emit EmergencyPauseChanged(true, address(this));
        }
    }

    function cancelPauseApproval() external onlyGuardian {
        uint64 currentNonce = pauseNonce;
        require(pauseApprovals[currentNonce][msg.sender], "No approval to cancel");
        pauseApprovals[currentNonce][msg.sender] = false;
        pauseApprovalCount[currentNonce]--;
    }

    // ============ Time-Locked Recovery ============

    function initiateRecovery(address newOwner) external onlyGuardian {
        require(newOwner != address(0), "Invalid new owner");
        require(pendingRecoveryTime == 0, "Recovery already pending");
        require(paused, "Contract must be paused");

        pendingRecoveryTime = block.timestamp + recoveryTimeLock;
        pendingRecoveryOwner = newOwner;
        emit RecoveryInitiated(newOwner, pendingRecoveryTime, msg.sender);
    }

    function executeRecovery() external {
        require(pendingRecoveryTime != 0, "No pending recovery");
        require(block.timestamp >= pendingRecoveryTime, "Time lock not expired");
        require(msg.sender == pendingRecoveryOwner || isGuardian[msg.sender], "Not authorized");

        address oldOwner = owner;
        owner = pendingRecoveryOwner;
        pendingRecoveryTime = 0;
        pendingRecoveryOwner = address(0);
        emit RecoveryExecuted(oldOwner, owner);
    }

    function cancelRecovery() external {
        require(pendingRecoveryTime != 0, "No pending recovery");
        require(msg.sender == owner || isGuardian[msg.sender], "Not authorized");
        pendingRecoveryTime = 0;
        pendingRecoveryOwner = address(0);
        emit RecoveryCancelled(msg.sender);
    }

    // ============ Guardian Management ============

    function addGuardian(address guardian) external onlyOwner {
        require(guardian != address(0), "Invalid address");
        require(!isGuardian[guardian], "Already guardian");
        isGuardian[guardian] = true;
        guardianCount++;
        emit GuardianUpdated(guardian, true, msg.sender);
    }

    function removeGuardian(address guardian) external onlyOwner {
        require(isGuardian[guardian], "Not guardian");
        require(guardianCount > requiredGuardians, "Cannot remove: below threshold");
        isGuardian[guardian] = false;
        guardianCount--;
        emit GuardianUpdated(guardian, false, msg.sender);
    }

    function setRequiredGuardians(uint8 _required) external onlyOwner {
        require(_required > 0, "Must require at least 1");
        require(_required <= guardianCount, "Cannot exceed guardian count");
        requiredGuardians = _required;
    }

    function setRecoveryTimeLock(uint256 _timeLock) external onlyOwner {
        require(_timeLock >= MIN_RECOVERY_TIMELOCK, "Minimum 1 hour");
        require(_timeLock <= MAX_RECOVERY_TIMELOCK, "Maximum 30 days");
        recoveryTimeLock = _timeLock;
    }

    // ============ Challenger Config ============

    function setChallengerConfig(uint16 _rewardPercentage, bool _enabled) external onlyOwner {
        require(_rewardPercentage <= MAX_REWARD_PERCENTAGE, "Max 50% reward");
        challengerRewardPercentage = _rewardPercentage;
        challengerRewardsEnabled = _enabled;
        emit ChallengerConfigUpdated(_rewardPercentage, _enabled);
    }

    function getChallengerConfig() external view returns (uint16 rewardPercentage, bool enabled) {
        return (challengerRewardPercentage, challengerRewardsEnabled);
    }

    // ============ Guardian View Functions ============

    function getGuardianStatus(address guardian) external view returns (bool) {
        return isGuardian[guardian];
    }

    function getPendingRecovery() external view returns (address newOwner, uint256 executionTime, bool isPending) {
        return (pendingRecoveryOwner, pendingRecoveryTime, pendingRecoveryTime != 0);
    }

    function getPauseApprovalStatus() external view returns (uint64 currentNonce, uint8 currentApprovals, uint8 required) {
        return (pauseNonce, pauseApprovalCount[pauseNonce], requiredGuardians);
    }

    // ============ Data Commitment Functions ============

    function setModelDataCommitment(uint256 modelId, bytes32 dataRoot) external modelExists(modelId) {
        require(msg.sender == models[modelId].owner || msg.sender == owner, "Not authorized");
        modelDataCommitment[modelId] = dataRoot;
        emit DataCommitmentSet(modelId, dataRoot, msg.sender);
    }

    function commitRoundData(uint256 modelId, uint256 roundId, bytes32 dataRoot) external whenNotPaused modelExists(modelId) {
        require(msg.sender == models[modelId].owner || msg.sender == owner, "Not authorized");
        require(roundId == models[modelId].currentRound, "Invalid round");
        require(dataRoot != bytes32(0), "Invalid data root");
        roundDataRoot[modelId][roundId] = dataRoot;
        emit RoundDataCommitted(modelId, roundId, dataRoot);
    }

    function getModelDataCommitment(uint256 modelId) external view returns (bytes32) {
        return modelDataCommitment[modelId];
    }

    function getRoundDataRoot(uint256 modelId, uint256 roundId) external view returns (bytes32) {
        return roundDataRoot[modelId][roundId];
    }
}
