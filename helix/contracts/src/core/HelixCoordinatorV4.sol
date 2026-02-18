// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/cryptography/MessageHashUtils.sol";
import "../interfaces/IHelixVerifier.sol";

/// @title HelixCoordinatorV4
/// @notice MPC-primary coordinator for HELIX decentralized ML training
/// @dev Supports multi-party attestation checkpoints, MAC failure reporting with
///      individual slashing, per-job staking, proportional payment distribution,
///      and optional ZK proof verification for checkpoint state transitions.
///
///      Key differences from V2/V3:
///      - Workers submit signed attestation checkpoints, not ZK proofs per step
///      - SPDZ MAC failure reporting replaces proof-based slashing
///      - Staking and payment are per-job (not per-model)
///      - Payment is proportional to steps participated
///      - Optional ZK verification via submitCheckpointWithProof()
contract HelixCoordinatorV4 is ReentrancyGuard {
    using ECDSA for bytes32;
    using MessageHashUtils for bytes32;

    // ============ Structs ============

    /// @notice Training job configuration and state
    struct Job {
        address owner;
        address operator;          // Can call assignPoolWorkers on behalf of owner
        bytes32 architectureHash;
        uint256 checkpointFreq;
        uint256 numRounds;
        uint256 paymentAmount;
        uint256 minStake;
        uint256 currentStep;
        bytes32 latestWeightCommitment;
        uint256 latestLoss;
        bool active;
        bool completed;
        // Risk-based ZK activation fields
        bool zkEnabled;               // User toggle: always require ZK
        uint256 zkCheckpointFreq;     // ZK proof every N checkpoints (0 = end only)
        bool riskZkEnabled;           // User toggle: enable auto-ZK on risk
        uint256 minWorkersForMpc;     // Threshold (default 2)
        bool zkActivatedByRisk;       // Set true when workers drop below threshold
    }

    /// @notice Per-job worker state
    struct WorkerInfo {
        uint256 stakeAmount;
        uint256 joinedAtStep;
        uint256 lastActiveStep;
        bool registered;
        bool slashed;
    }

    /// @notice Checkpoint record
    struct Checkpoint {
        uint256 stepNumber;
        bytes32 weightCommitment;
        uint256 loss;
        uint256 timestamp;
        uint256 signerCount;
    }

    /// @notice MAC failure report record
    struct MACFailureReport {
        uint256 stepNumber;
        address cheater;
        uint256 timestamp;
        uint256 slashedAmount;
        uint256 reporterCount;
    }

    /// @notice Inference result attestation record
    struct InferenceResult {
        uint256 jobId;
        uint256 prediction;
        bytes32 inputHash;
        bytes32 outputHash;
        uint256 timestamp;
        uint256 signerCount;
    }

    /// @notice Global worker pool entry — workers register once, available for any job
    struct PoolWorker {
        string endpoint;            // TCP endpoint (e.g. "192.168.1.5:9001")
        uint256 stakeAmount;        // ETH staked as collateral
        uint256 registeredAt;       // block.timestamp when registered
        bool available;             // true = accepting new jobs
        uint256 activeJobId;        // current job assignment (0 = none)
    }

    // ============ Constants ============

    /// @notice Bounty percentage for MAC failure reporters (basis points)
    uint16 public constant REPORTER_BOUNTY_BPS = 1000; // 10%

    /// @notice Basis points denominator
    uint16 public constant BPS_DENOMINATOR = 10000;

    /// @notice Cooldown period before workers can withdraw stake after job completion
    uint256 public constant STAKE_COOLDOWN = 7 days;

    /// @notice Minimum number of workers for a job
    uint256 public constant MIN_WORKERS = 2;

    // ============ State Variables ============

    /// @notice Contract owner
    address public owner;

    /// @notice Treasury for slashed funds (after bounty)
    address public treasury;

    /// @notice Optional ZK proof verifier for submitCheckpointWithProof
    IHelixVerifier public verifier;

    /// @notice Counter for job IDs
    uint256 public nextJobId;

    /// @notice Job registry: jobId => Job
    mapping(uint256 => Job) public jobs;

    /// @notice Workers per job: jobId => worker address => WorkerInfo
    mapping(uint256 => mapping(address => WorkerInfo)) public workers;

    /// @notice Active worker list per job: jobId => worker addresses
    mapping(uint256 => address[]) internal _activeWorkers;

    /// @notice Worker index in active list (for O(1) removal): jobId => worker => index+1 (0 = not in list)
    mapping(uint256 => mapping(address => uint256)) internal _workerIndex;

    /// @notice Checkpoints per job: jobId => checkpoint index => Checkpoint
    mapping(uint256 => Checkpoint[]) public checkpoints;

    /// @notice MAC failure reports per job: jobId => report index => MACFailureReport
    mapping(uint256 => MACFailureReport[]) public macFailureReports;

    /// @notice Track slashed workers per job per step to prevent double-slash
    /// @dev jobId => stepNumber => cheater => slashed
    mapping(uint256 => mapping(uint256 => mapping(address => bool))) public slashedAt;

    /// @notice Job completion timestamp (for cooldown): jobId => timestamp
    mapping(uint256 => uint256) public jobCompletionTime;

    /// @notice Whether a worker has withdrawn stake for a job: jobId => worker => withdrawn
    mapping(uint256 => mapping(address => bool)) public stakeWithdrawn;

    /// @notice Total staked ETH by each worker address across all jobs
    mapping(address => uint256) public workerTotalStaked;

    /// @notice Number of active jobs a worker is currently participating in
    mapping(address => uint256) public workerActiveJobs;

    /// @notice Stake required per active job slot (rate limiting)
    uint256 public stakePerJobSlot = 0.001 ether;

    /// @notice ZK weight hash chain: jobId => index => value (index 0 = lo, index 1 = hi)
    mapping(uint256 => mapping(uint256 => uint256)) public zkWeightHash;

    /// @notice Counter for inference result IDs
    uint256 public inferenceCount;

    /// @notice Inference results: inferenceId => InferenceResult
    mapping(uint256 => InferenceResult) public inferenceResults;

    // ============ Global Worker Pool State ============

    /// @notice Pool worker registry: address => PoolWorker
    mapping(address => PoolWorker) public poolWorkers;

    /// @notice Ordered list of pool worker addresses (for enumeration)
    address[] internal _poolWorkerList;

    /// @notice Pool worker index: address => index+1 in _poolWorkerList (0 = not registered)
    mapping(address => uint256) internal _poolWorkerIndex;

    /// @notice Minimum stake required to join the worker pool
    uint256 public poolMinStake = 0.001 ether;

    // ============ Events ============

    event JobRegistered(
        uint256 indexed jobId,
        address indexed owner,
        bytes32 architectureHash,
        uint256 checkpointFreq,
        uint256 numRounds,
        uint256 paymentAmount
    );

    event WorkerJoined(
        uint256 indexed jobId,
        address indexed worker,
        uint256 stakeAmount
    );

    event CheckpointSubmitted(
        uint256 indexed jobId,
        uint256 indexed stepNumber,
        bytes32 weightCommitment,
        uint256 loss,
        uint256 signerCount
    );

    event CheckpointWithProofSubmitted(
        uint256 indexed jobId,
        uint256 indexed stepNumber,
        bytes32 weightCommitment,
        uint256 loss,
        bool proofValid
    );

    event WorkerSlashed(
        uint256 indexed jobId,
        address indexed cheater,
        uint256 stepNumber,
        uint256 slashedAmount,
        uint256 bountyAmount,
        uint256 reporterCount
    );

    event TrainingCompleted(
        uint256 indexed jobId,
        bytes32 finalCommitment,
        uint256 totalSteps,
        uint256 activeWorkerCount
    );

    event PaymentDistributed(
        uint256 indexed jobId,
        address indexed worker,
        uint256 amount
    );

    event StakeReturned(
        uint256 indexed jobId,
        address indexed worker,
        uint256 amount
    );

    event TreasuryUpdated(address indexed oldTreasury, address indexed newTreasury);
    event VerifierUpdated(address indexed oldVerifier, address indexed newVerifier);
    event ZkActivatedByRisk(uint256 indexed jobId, uint256 activeWorkerCount);

    event InferenceResultCommitted(
        uint256 indexed inferenceId,
        uint256 indexed jobId,
        uint256 prediction,
        bytes32 inputHash,
        bytes32 outputHash,
        uint256 signerCount
    );

    // Worker pool events
    event WorkerPoolRegistered(address indexed worker, string endpoint, uint256 stakeAmount);
    event WorkerPoolDeregistered(address indexed worker, uint256 stakeReturned);
    event WorkerPoolAssigned(uint256 indexed jobId, address indexed worker, uint256 stakeAmount);

    // ============ Errors ============

    error OnlyOwner();
    error InvalidTreasury();
    error InvalidPayment();
    error InvalidCheckpointFreq();
    error InvalidNumRounds();
    error JobNotFound();
    error JobNotActive();
    error JobAlreadyCompleted();
    error AlreadyRegistered();
    error InsufficientStake();
    error WorkerNotRegistered();
    error WorkerSlashedError();
    error InvalidSignatureCount();
    error InvalidSigner();
    error DuplicateSigner();
    error StepMismatch();
    error InsufficientReporters();
    error CheaterNotRegistered();
    error CheaterAlreadySlashed();
    error JobNotCompleted();
    error CooldownNotExpired();
    error AlreadyWithdrawn();
    error TransferFailed();
    error VerifierNotSet();
    error InvalidProof();
    error ZkHashChainMismatch();
    error InvalidPublicInputsLength();
    error NoWorkersJoined();
    error WorkerAlreadySlashed();
    error RateLimited();
    error ZkRequired();
    error WorkerAlreadyInPool();
    error WorkerNotInPool();
    error WorkerBusy();
    error NotEnoughPoolWorkers();

    // ============ Modifiers ============

    modifier onlyOwner() {
        if (msg.sender != owner) revert OnlyOwner();
        _;
    }

    modifier jobExists(uint256 jobId) {
        if (jobs[jobId].owner == address(0)) revert JobNotFound();
        _;
    }

    modifier jobActive(uint256 jobId) {
        if (!jobs[jobId].active) revert JobNotActive();
        if (jobs[jobId].completed) revert JobAlreadyCompleted();
        _;
    }

    // ============ Constructor ============

    /// @param _treasury Treasury address for slashed funds
    /// @param _verifier Optional ZK verifier address (can be address(0) if ZK not used)
    constructor(address _treasury, address _verifier) {
        if (_treasury == address(0)) revert InvalidTreasury();
        treasury = _treasury;
        owner = msg.sender;
        if (_verifier != address(0)) {
            verifier = IHelixVerifier(_verifier);
        }
    }

    // ============ Job Registration ============

    /// @notice Register a new MPC training job with payment deposit
    /// @param architectureHash Hash of the model architecture (e.g., keccak256("784,32,10"))
    /// @param checkpointFreq How often checkpoints are submitted (every N steps)
    /// @param numRounds Total number of training steps/rounds
    /// @param paymentAmount Total payment for workers (sent as msg.value)
    /// @param zkEnabled If true, always require ZK proofs for checkpoints
    /// @param zkCheckpointFreq ZK proof every N checkpoints (0 = end only, ignored if !zkEnabled)
    /// @param riskZkEnabled If true, auto-activate ZK when workers drop below minWorkersForMpc
    /// @param minWorkersForMpc Minimum worker threshold for MPC-only mode (default 2)
    /// @return jobId The unique job identifier
    function registerTrainingJob(
        bytes32 architectureHash,
        uint256 checkpointFreq,
        uint256 numRounds,
        uint256 paymentAmount,
        bool zkEnabled,
        uint256 zkCheckpointFreq,
        bool riskZkEnabled,
        uint256 minWorkersForMpc,
        address operator
    ) external payable nonReentrant returns (uint256 jobId) {
        if (msg.value != paymentAmount || paymentAmount == 0) revert InvalidPayment();
        if (checkpointFreq == 0) revert InvalidCheckpointFreq();
        if (numRounds == 0) revert InvalidNumRounds();

        jobId = nextJobId++;

        // Default minWorkersForMpc to 2 if not specified
        uint256 effectiveMinWorkers = minWorkersForMpc > 0 ? minWorkersForMpc : 2;

        jobs[jobId] = Job({
            owner: msg.sender,
            operator: operator,
            architectureHash: architectureHash,
            checkpointFreq: checkpointFreq,
            numRounds: numRounds,
            paymentAmount: paymentAmount,
            minStake: 0.001 ether,
            currentStep: 0,
            latestWeightCommitment: bytes32(0),
            latestLoss: 0,
            active: true,
            completed: false,
            zkEnabled: zkEnabled,
            zkCheckpointFreq: zkCheckpointFreq,
            riskZkEnabled: riskZkEnabled,
            minWorkersForMpc: effectiveMinWorkers,
            zkActivatedByRisk: false
        });

        emit JobRegistered(jobId, msg.sender, architectureHash, checkpointFreq, numRounds, paymentAmount);
    }

    // ============ Worker Registration ============

    /// @notice Stake ETH and join a training job as a worker
    /// @param jobId The job to join
    function stakeAndJoin(uint256 jobId) external payable nonReentrant jobExists(jobId) jobActive(jobId) {
        if (workers[jobId][msg.sender].registered) revert AlreadyRegistered();
        if (msg.value < jobs[jobId].minStake) revert InsufficientStake();

        // Rate limiting: worker must have enough total stake to support another active job
        if (workerActiveJobs[msg.sender] >= (workerTotalStaked[msg.sender] + msg.value) / stakePerJobSlot) {
            revert RateLimited();
        }

        workers[jobId][msg.sender] = WorkerInfo({
            stakeAmount: msg.value,
            joinedAtStep: jobs[jobId].currentStep,
            lastActiveStep: jobs[jobId].currentStep,
            registered: true,
            slashed: false
        });

        _activeWorkers[jobId].push(msg.sender);
        _workerIndex[jobId][msg.sender] = _activeWorkers[jobId].length; // 1-indexed

        // Update rate limiting state
        workerTotalStaked[msg.sender] += msg.value;
        workerActiveJobs[msg.sender]++;

        emit WorkerJoined(jobId, msg.sender, msg.value);
    }

    // ============ Checkpoint Submission ============

    /// @notice Submit a multi-party attestation checkpoint signed by all active workers
    /// @param jobId The training job
    /// @param stepNumber The training step this checkpoint covers
    /// @param weightCommitment Pedersen commitment to the current model weights
    /// @param loss Current training loss value
    /// @param signatures ECDSA signatures from all active workers
    function submitCheckpoint(
        uint256 jobId,
        uint256 stepNumber,
        bytes32 weightCommitment,
        uint256 loss,
        bytes[] calldata signatures
    ) external nonReentrant jobExists(jobId) jobActive(jobId) {
        // Enforce ZK when required — but only for checkpoints where ZK is actually needed.
        // In MPC training, intermediate weights remain secret-shared so ZK proofs can only
        // be generated for the final checkpoint (where weights are reconstructed).
        // When zkCheckpointFreq == 0, ZK is required only at the final step (stepNumber >= numRounds).
        // When zkCheckpointFreq > 0, ZK is required every N-th checkpoint.
        if (jobs[jobId].zkEnabled || jobs[jobId].zkActivatedByRisk) {
            bool zkRequiredHere;
            if (jobs[jobId].zkCheckpointFreq == 0) {
                // ZK only at the final checkpoint
                zkRequiredHere = stepNumber >= jobs[jobId].numRounds;
            } else {
                // ZK every N-th checkpoint (based on checkpoint count, not step)
                uint256 checkpointIndex = stepNumber / jobs[jobId].checkpointFreq;
                zkRequiredHere = checkpointIndex % jobs[jobId].zkCheckpointFreq == 0;
            }
            if (zkRequiredHere) revert ZkRequired();
        }

        uint256 activeCount = _activeWorkers[jobId].length;
        if (activeCount == 0) revert NoWorkersJoined();
        if (signatures.length != activeCount) revert InvalidSignatureCount();

        // Build the attestation message
        bytes32 message = _buildCheckpointMessage(jobId, stepNumber, weightCommitment, loss);
        bytes32 ethSignedHash = message.toEthSignedMessageHash();

        // Verify all signatures are from registered active workers
        _verifyAllWorkerSignatures(jobId, ethSignedHash, signatures);

        // Update job state
        Job storage job = jobs[jobId];
        job.currentStep = stepNumber;
        job.latestWeightCommitment = weightCommitment;
        job.latestLoss = loss;

        // Update all active workers' lastActiveStep
        for (uint256 i = 0; i < activeCount; i++) {
            workers[jobId][_activeWorkers[jobId][i]].lastActiveStep = stepNumber;
        }

        // Store checkpoint
        checkpoints[jobId].push(Checkpoint({
            stepNumber: stepNumber,
            weightCommitment: weightCommitment,
            loss: loss,
            timestamp: block.timestamp,
            signerCount: activeCount
        }));

        emit CheckpointSubmitted(jobId, stepNumber, weightCommitment, loss, activeCount);
    }

    // ============ MAC Failure Reporting ============

    /// @notice Report a worker who failed MAC verification, signed by majority of workers
    /// @param jobId The training job
    /// @param stepNumber The step where cheating was detected
    /// @param cheater Address of the cheating worker
    /// @param evidence Encoded evidence data (MAC verification sigma values, etc.)
    /// @param reporterSignatures Signatures from honest workers reporting the failure
    function reportMACFailure(
        uint256 jobId,
        uint256 stepNumber,
        address cheater,
        bytes calldata evidence,
        bytes[] calldata reporterSignatures
    ) external nonReentrant jobExists(jobId) jobActive(jobId) {
        // Validate cheater is a registered worker
        if (!workers[jobId][cheater].registered) revert CheaterNotRegistered();
        if (workers[jobId][cheater].slashed) revert CheaterAlreadySlashed();

        // Prevent double-slash for same step
        if (slashedAt[jobId][stepNumber][cheater]) revert CheaterAlreadySlashed();

        uint256 activeCount = _activeWorkers[jobId].length;
        // Need strict majority of workers (excluding the cheater)
        // For 3 workers with 1 cheater: need 2 signatures (2 out of 3 is majority)
        uint256 majority = (activeCount / 2) + 1;
        if (reporterSignatures.length < majority) revert InsufficientReporters();

        // Build the MAC failure report message
        bytes32 message = _buildMACFailureMessage(jobId, stepNumber, cheater, evidence);
        bytes32 ethSignedHash = message.toEthSignedMessageHash();

        // Verify all reporter signatures are from registered, non-slashed workers
        // and that the cheater did not sign the report
        address[] memory reporters = _verifyReporterSignatures(
            jobId, ethSignedHash, reporterSignatures, cheater
        );

        // Slash the cheater's full stake
        uint256 stakeAmount = workers[jobId][cheater].stakeAmount;
        workers[jobId][cheater].slashed = true;
        workers[jobId][cheater].stakeAmount = 0;
        slashedAt[jobId][stepNumber][cheater] = true;

        // Remove cheater from active worker list
        _removeActiveWorker(jobId, cheater);

        // Distribute: 10% bounty to reporters, rest to treasury
        uint256 bountyTotal = (stakeAmount * REPORTER_BOUNTY_BPS) / BPS_DENOMINATOR;
        uint256 bountyPerReporter = bountyTotal / reporters.length;
        uint256 actualBountyDistributed = bountyPerReporter * reporters.length;
        uint256 toTreasury = stakeAmount - actualBountyDistributed;

        // Pay reporters
        for (uint256 i = 0; i < reporters.length; i++) {
            (bool success, ) = reporters[i].call{value: bountyPerReporter}("");
            if (!success) revert TransferFailed();
        }

        // Pay treasury
        if (toTreasury > 0) {
            (bool success, ) = treasury.call{value: toTreasury}("");
            if (!success) revert TransferFailed();
        }

        // Decrement slashed worker's active job count and total staked
        if (workerActiveJobs[cheater] > 0) {
            workerActiveJobs[cheater]--;
        }
        // workerTotalStaked was already set when they joined; stakeAmount was their contribution
        if (workerTotalStaked[cheater] >= stakeAmount) {
            workerTotalStaked[cheater] -= stakeAmount;
        } else {
            workerTotalStaked[cheater] = 0;
        }

        // Remove slashed worker from pool entirely (they lose their pool registration)
        if (_poolWorkerIndex[cheater] != 0) {
            uint256 poolStake = poolWorkers[cheater].stakeAmount;
            // Remove from pool list (swap-and-pop)
            uint256 pIdx = _poolWorkerIndex[cheater] - 1;
            uint256 pLast = _poolWorkerList.length - 1;
            if (pIdx != pLast) {
                address lastPw = _poolWorkerList[pLast];
                _poolWorkerList[pIdx] = lastPw;
                _poolWorkerIndex[lastPw] = pIdx + 1;
            }
            _poolWorkerList.pop();
            _poolWorkerIndex[cheater] = 0;
            delete poolWorkers[cheater];
            // Pool stake is also forfeited (already held in contract)
            if (poolStake > 0) {
                (bool s, ) = treasury.call{value: poolStake}("");
                if (!s) revert TransferFailed();
            }
        }

        // Check risk threshold: activate ZK if workers drop below minimum
        uint256 activeWorkerCount = _activeWorkers[jobId].length;
        if (jobs[jobId].riskZkEnabled && !jobs[jobId].zkActivatedByRisk && activeWorkerCount < jobs[jobId].minWorkersForMpc) {
            jobs[jobId].zkActivatedByRisk = true;
            emit ZkActivatedByRisk(jobId, activeWorkerCount);
        }

        // Store report
        macFailureReports[jobId].push(MACFailureReport({
            stepNumber: stepNumber,
            cheater: cheater,
            timestamp: block.timestamp,
            slashedAmount: stakeAmount,
            reporterCount: reporters.length
        }));

        emit WorkerSlashed(jobId, cheater, stepNumber, stakeAmount, actualBountyDistributed, reporters.length);
    }

    // ============ Training Completion ============

    /// @notice Complete training with a final checkpoint signed by all active workers
    /// @param jobId The training job
    /// @param finalCommitment Final weight commitment
    /// @param signatures Signatures from all active workers
    function completeTraining(
        uint256 jobId,
        bytes32 finalCommitment,
        bytes[] calldata signatures
    ) external nonReentrant jobExists(jobId) jobActive(jobId) {
        uint256 activeCount = _activeWorkers[jobId].length;
        if (activeCount == 0) revert NoWorkersJoined();
        if (signatures.length != activeCount) revert InvalidSignatureCount();

        Job storage job = jobs[jobId];

        // Build and verify the completion message
        bytes32 message = _buildCompletionMessage(jobId, finalCommitment);
        bytes32 ethSignedHash = message.toEthSignedMessageHash();
        _verifyAllWorkerSignatures(jobId, ethSignedHash, signatures);

        // Mark job as completed
        job.completed = true;
        job.active = false;
        job.latestWeightCommitment = finalCommitment;
        jobCompletionTime[jobId] = block.timestamp;

        // Update all workers' lastActiveStep to final and decrement active job counts
        uint256 finalStep = job.currentStep;
        for (uint256 i = 0; i < activeCount; i++) {
            address w = _activeWorkers[jobId][i];
            workers[jobId][w].lastActiveStep = finalStep;
            // Decrement active job count for rate limiting
            if (workerActiveJobs[w] > 0) {
                workerActiveJobs[w]--;
            }
            // Release pool workers back to available
            if (_poolWorkerIndex[w] != 0) {
                poolWorkers[w].available = true;
                poolWorkers[w].activeJobId = 0;
            }
        }

        // Distribute payment proportionally to participation
        _distributePayment(jobId);

        emit TrainingCompleted(jobId, finalCommitment, finalStep, activeCount);
    }

    // ============ Optional ZK Checkpoint ============

    /// @notice Submit a checkpoint with an optional ZK proof of state transition
    /// @dev Uses the existing Halo2Verifier for proof verification. Only for jobs with ZK enabled.
    /// @param jobId The training job
    /// @param stepNumber The training step this checkpoint covers
    /// @param weightCommitment Pedersen commitment to the current model weights
    /// @param loss Current training loss value
    /// @param proof ZK proof bytes
    /// @param publicInputs Public inputs to the ZK circuit
    function submitCheckpointWithProof(
        uint256 jobId,
        uint256 stepNumber,
        bytes32 weightCommitment,
        uint256 loss,
        bytes calldata proof,
        uint256[] calldata publicInputs
    ) external nonReentrant jobExists(jobId) jobActive(jobId) {
        if (address(verifier) == address(0)) revert VerifierNotSet();

        // Require exactly 6 public inputs (StateTransitionCircuit)
        if (publicInputs.length != 6) revert InvalidPublicInputsLength();

        // Verify the ZK proof
        bool valid = verifier.verifyProof(proof, publicInputs);
        if (!valid) revert InvalidProof();

        // Verify hash chain continuity: if a previous ZK checkpoint exists,
        // the proof's old_hash must match the stored new_hash from last time.
        uint256 prevLo = zkWeightHash[jobId][0];
        uint256 prevHi = zkWeightHash[jobId][1];
        if (prevLo != 0 || prevHi != 0) {
            // Previous ZK checkpoint exists — verify chain
            if (publicInputs[0] != prevLo || publicInputs[1] != prevHi) {
                revert ZkHashChainMismatch();
            }
        }

        // Store the new weight hash for chain continuity
        zkWeightHash[jobId][0] = publicInputs[2]; // new_hash_lo
        zkWeightHash[jobId][1] = publicInputs[3]; // new_hash_hi

        // Update job state
        Job storage job = jobs[jobId];
        job.currentStep = stepNumber;
        job.latestWeightCommitment = weightCommitment;
        job.latestLoss = loss;

        // Store checkpoint
        checkpoints[jobId].push(Checkpoint({
            stepNumber: stepNumber,
            weightCommitment: weightCommitment,
            loss: loss,
            timestamp: block.timestamp,
            signerCount: 0 // ZK proof, not signatures
        }));

        emit CheckpointWithProofSubmitted(jobId, stepNumber, weightCommitment, loss, true);
    }

    // ============ Stake Withdrawal ============

    /// @notice Withdraw stake after job completion and cooldown period
    /// @param jobId The completed job
    function withdrawStake(uint256 jobId) external nonReentrant jobExists(jobId) {
        if (!jobs[jobId].completed) revert JobNotCompleted();
        if (block.timestamp < jobCompletionTime[jobId] + STAKE_COOLDOWN) revert CooldownNotExpired();
        if (stakeWithdrawn[jobId][msg.sender]) revert AlreadyWithdrawn();

        WorkerInfo storage worker = workers[jobId][msg.sender];
        if (!worker.registered) revert WorkerNotRegistered();
        if (worker.slashed) revert WorkerSlashedError();
        if (worker.stakeAmount == 0) revert InsufficientStake();

        uint256 amount = worker.stakeAmount;
        worker.stakeAmount = 0;
        stakeWithdrawn[jobId][msg.sender] = true;

        (bool success, ) = msg.sender.call{value: amount}("");
        if (!success) revert TransferFailed();

        emit StakeReturned(jobId, msg.sender, amount);
    }

    // ============ Internal Functions ============

    /// @dev Build the checkpoint attestation message
    function _buildCheckpointMessage(
        uint256 jobId,
        uint256 stepNumber,
        bytes32 weightCommitment,
        uint256 loss
    ) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked(
            "HELIX_CHECKPOINT",
            jobId,
            stepNumber,
            weightCommitment,
            loss
        ));
    }

    /// @dev Build the MAC failure report message
    function _buildMACFailureMessage(
        uint256 jobId,
        uint256 stepNumber,
        address cheater,
        bytes calldata evidence
    ) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked(
            "HELIX_MAC_FAILURE",
            jobId,
            stepNumber,
            cheater,
            evidence
        ));
    }

    /// @dev Build the training completion message
    function _buildCompletionMessage(
        uint256 jobId,
        bytes32 finalCommitment
    ) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked(
            "HELIX_COMPLETE",
            jobId,
            finalCommitment
        ));
    }

    /// @dev Verify all signatures are from active registered workers, with no duplicates
    function _verifyAllWorkerSignatures(
        uint256 jobId,
        bytes32 ethSignedHash,
        bytes[] calldata signatures
    ) internal view {
        uint256 count = signatures.length;
        // Track seen signers to prevent duplicates
        address[] memory seen = new address[](count);

        for (uint256 i = 0; i < count; i++) {
            address signer = ethSignedHash.recover(signatures[i]);

            // Must be a registered, non-slashed worker for this job
            if (!workers[jobId][signer].registered) revert InvalidSigner();
            if (workers[jobId][signer].slashed) revert InvalidSigner();
            if (_workerIndex[jobId][signer] == 0) revert InvalidSigner();

            // Check for duplicates
            for (uint256 j = 0; j < i; j++) {
                if (seen[j] == signer) revert DuplicateSigner();
            }
            seen[i] = signer;
        }
    }

    /// @dev Verify reporter signatures for MAC failure report
    /// @return reporters Array of validated reporter addresses
    function _verifyReporterSignatures(
        uint256 jobId,
        bytes32 ethSignedHash,
        bytes[] calldata signatures,
        address cheater
    ) internal view returns (address[] memory reporters) {
        uint256 count = signatures.length;
        reporters = new address[](count);

        for (uint256 i = 0; i < count; i++) {
            address signer = ethSignedHash.recover(signatures[i]);

            // Must be a registered, non-slashed, active worker
            if (!workers[jobId][signer].registered) revert InvalidSigner();
            if (workers[jobId][signer].slashed) revert InvalidSigner();
            if (_workerIndex[jobId][signer] == 0) revert InvalidSigner();

            // Cheater cannot sign their own report
            if (signer == cheater) revert InvalidSigner();

            // Check for duplicates
            for (uint256 j = 0; j < i; j++) {
                if (reporters[j] == signer) revert DuplicateSigner();
            }
            reporters[i] = signer;
        }
    }

    /// @dev Remove a worker from the active worker list (O(1) swap-and-pop)
    function _removeActiveWorker(uint256 jobId, address worker) internal {
        uint256 index = _workerIndex[jobId][worker];
        if (index == 0) return; // not in list

        uint256 arrayIndex = index - 1; // convert from 1-indexed
        uint256 lastIndex = _activeWorkers[jobId].length - 1;

        if (arrayIndex != lastIndex) {
            address lastWorker = _activeWorkers[jobId][lastIndex];
            _activeWorkers[jobId][arrayIndex] = lastWorker;
            _workerIndex[jobId][lastWorker] = index; // maintain 1-indexed
        }

        _activeWorkers[jobId].pop();
        _workerIndex[jobId][worker] = 0;
    }

    /// @dev Distribute payment proportionally based on steps participated
    function _distributePayment(uint256 jobId) internal {
        Job storage job = jobs[jobId];
        address[] storage active = _activeWorkers[jobId];
        uint256 activeCount = active.length;
        if (activeCount == 0) return;

        // Calculate total participation weight (steps participated by each active worker)
        uint256 totalWeight = 0;
        uint256[] memory weights = new uint256[](activeCount);

        for (uint256 i = 0; i < activeCount; i++) {
            WorkerInfo storage w = workers[jobId][active[i]];
            uint256 stepsParticipated = w.lastActiveStep - w.joinedAtStep;
            if (stepsParticipated == 0) stepsParticipated = 1; // minimum 1 for participation
            weights[i] = stepsParticipated;
            totalWeight += stepsParticipated;
        }

        // Distribute payment proportionally
        uint256 totalDistributed = 0;
        for (uint256 i = 0; i < activeCount; i++) {
            uint256 payment;
            if (i == activeCount - 1) {
                // Last worker gets remainder to avoid dust
                payment = job.paymentAmount - totalDistributed;
            } else {
                payment = (job.paymentAmount * weights[i]) / totalWeight;
            }
            totalDistributed += payment;

            if (payment > 0) {
                (bool success, ) = active[i].call{value: payment}("");
                if (!success) revert TransferFailed();
                emit PaymentDistributed(jobId, active[i], payment);
            }
        }
    }

    // ============ Global Worker Pool ============

    /// @notice Register in the global worker pool, staking ETH as collateral.
    ///         Any user who submits a training job can auto-assign pool workers.
    /// @param endpoint TCP endpoint the worker is listening on (e.g. "192.168.1.5:9001")
    function registerInPool(string calldata endpoint) external payable nonReentrant {
        if (_poolWorkerIndex[msg.sender] != 0) revert WorkerAlreadyInPool();
        if (msg.value < poolMinStake) revert InsufficientStake();

        poolWorkers[msg.sender] = PoolWorker({
            endpoint: endpoint,
            stakeAmount: msg.value,
            registeredAt: block.timestamp,
            available: true,
            activeJobId: 0
        });

        _poolWorkerList.push(msg.sender);
        _poolWorkerIndex[msg.sender] = _poolWorkerList.length; // 1-indexed

        emit WorkerPoolRegistered(msg.sender, endpoint, msg.value);
    }

    /// @notice Leave the worker pool and withdraw staked ETH.
    ///         Only possible if the worker is not currently assigned to a job.
    function deregisterFromPool() external nonReentrant {
        if (_poolWorkerIndex[msg.sender] == 0) revert WorkerNotInPool();
        if (poolWorkers[msg.sender].activeJobId != 0) revert WorkerBusy();

        uint256 stake = poolWorkers[msg.sender].stakeAmount;

        // Remove from list (swap-and-pop)
        uint256 index = _poolWorkerIndex[msg.sender] - 1;
        uint256 lastIndex = _poolWorkerList.length - 1;
        if (index != lastIndex) {
            address lastWorker = _poolWorkerList[lastIndex];
            _poolWorkerList[index] = lastWorker;
            _poolWorkerIndex[lastWorker] = index + 1;
        }
        _poolWorkerList.pop();
        _poolWorkerIndex[msg.sender] = 0;
        delete poolWorkers[msg.sender];

        // Return stake
        if (stake > 0) {
            (bool success, ) = msg.sender.call{value: stake}("");
            if (!success) revert TransferFailed();
        }

        emit WorkerPoolDeregistered(msg.sender, stake);
    }

    /// @notice Assign available pool workers to a job. Called by the job owner.
    ///         Each assigned worker's pool stake is locked into the job.
    /// @param jobId The job to assign workers to
    /// @param count Number of workers to assign
    function assignPoolWorkers(uint256 jobId, uint256 count) external nonReentrant jobExists(jobId) jobActive(jobId) {
        if (msg.sender != jobs[jobId].owner && msg.sender != jobs[jobId].operator) revert OnlyOwner();

        uint256 assigned = 0;
        uint256 poolLen = _poolWorkerList.length;

        for (uint256 i = 0; i < poolLen && assigned < count; i++) {
            address w = _poolWorkerList[i];
            PoolWorker storage pw = poolWorkers[w];

            if (!pw.available || pw.activeJobId != 0) continue;
            if (pw.stakeAmount < jobs[jobId].minStake) continue;
            if (workers[jobId][w].registered) continue; // already in this job

            // Register this pool worker into the job
            uint256 jobStake = jobs[jobId].minStake;
            workers[jobId][w] = WorkerInfo({
                stakeAmount: jobStake,
                joinedAtStep: jobs[jobId].currentStep,
                lastActiveStep: jobs[jobId].currentStep,
                registered: true,
                slashed: false
            });

            _activeWorkers[jobId].push(w);
            _workerIndex[jobId][w] = _activeWorkers[jobId].length;

            workerTotalStaked[w] += jobStake;
            workerActiveJobs[w]++;

            // Mark pool worker as busy
            pw.available = false;
            pw.activeJobId = jobId;

            assigned++;

            emit WorkerPoolAssigned(jobId, w, jobStake);
            emit WorkerJoined(jobId, w, jobStake);
        }

        if (assigned < count) revert NotEnoughPoolWorkers();
    }

    // ============ View Functions ============

    /// @notice Get the number of active workers for a job
    function getActiveWorkerCount(uint256 jobId) external view returns (uint256) {
        return _activeWorkers[jobId].length;
    }

    /// @notice Get the list of active workers for a job
    function getActiveWorkers(uint256 jobId) external view returns (address[] memory) {
        return _activeWorkers[jobId];
    }

    /// @notice Get worker info for a job
    function getWorkerInfo(uint256 jobId, address worker) external view returns (
        uint256 stakeAmount,
        uint256 joinedAtStep,
        uint256 lastActiveStep,
        bool registered,
        bool slashed
    ) {
        WorkerInfo storage w = workers[jobId][worker];
        return (w.stakeAmount, w.joinedAtStep, w.lastActiveStep, w.registered, w.slashed);
    }

    /// @notice Get the number of checkpoints for a job
    function getCheckpointCount(uint256 jobId) external view returns (uint256) {
        return checkpoints[jobId].length;
    }

    /// @notice Get a checkpoint by index
    function getCheckpoint(uint256 jobId, uint256 index) external view returns (
        uint256 stepNumber,
        bytes32 weightCommitment,
        uint256 loss,
        uint256 timestamp,
        uint256 signerCount
    ) {
        Checkpoint storage cp = checkpoints[jobId][index];
        return (cp.stepNumber, cp.weightCommitment, cp.loss, cp.timestamp, cp.signerCount);
    }

    /// @notice Get the number of MAC failure reports for a job
    function getMACFailureReportCount(uint256 jobId) external view returns (uint256) {
        return macFailureReports[jobId].length;
    }

    /// @notice Get a MAC failure report by index
    function getMACFailureReport(uint256 jobId, uint256 index) external view returns (
        uint256 stepNumber,
        address cheater,
        uint256 timestamp,
        uint256 slashedAmount,
        uint256 reporterCount
    ) {
        MACFailureReport storage report = macFailureReports[jobId][index];
        return (report.stepNumber, report.cheater, report.timestamp, report.slashedAmount, report.reporterCount);
    }

    /// @notice Get job summary
    function getJobSummary(uint256 jobId) external view returns (
        address jobOwner,
        uint256 currentStep,
        uint256 numRounds,
        uint256 paymentAmount,
        uint256 activeWorkerCount,
        bool active,
        bool completed,
        bool zkEnabled,
        bool zkActivatedByRisk
    ) {
        Job storage job = jobs[jobId];
        return (
            job.owner,
            job.currentStep,
            job.numRounds,
            job.paymentAmount,
            _activeWorkers[jobId].length,
            job.active,
            job.completed,
            job.zkEnabled,
            job.zkActivatedByRisk
        );
    }

    /// @notice Check if an address is an active worker for a job
    function isActiveWorker(uint256 jobId, address worker) external view returns (bool) {
        return _workerIndex[jobId][worker] > 0 && !workers[jobId][worker].slashed;
    }

    /// @notice Check if ZK proofs are required for a job (either user-enabled or risk-activated)
    function isZkRequired(uint256 jobId) external view returns (bool) {
        return jobs[jobId].zkEnabled || jobs[jobId].zkActivatedByRisk;
    }

    /// @notice Get the total number of workers in the pool
    function getPoolWorkerCount() external view returns (uint256) {
        return _poolWorkerList.length;
    }

    /// @notice Get the number of available (idle) pool workers
    function getAvailablePoolWorkerCount() external view returns (uint256) {
        uint256 count = 0;
        for (uint256 i = 0; i < _poolWorkerList.length; i++) {
            if (poolWorkers[_poolWorkerList[i]].available) count++;
        }
        return count;
    }

    /// @notice Get all pool worker addresses
    function getPoolWorkers() external view returns (address[] memory) {
        return _poolWorkerList;
    }

    /// @notice Get pool worker info
    function getPoolWorkerInfo(address worker) external view returns (
        string memory endpoint,
        uint256 stakeAmount,
        uint256 registeredAt,
        bool available,
        uint256 activeJobId
    ) {
        PoolWorker storage pw = poolWorkers[worker];
        return (pw.endpoint, pw.stakeAmount, pw.registeredAt, pw.available, pw.activeJobId);
    }

    // ============ Inference Attestation ============

    /// @notice Submit a multi-party attested inference result
    /// @param jobId The training job that produced the model
    /// @param prediction The predicted class (e.g. 0-9 for MNIST)
    /// @param inputHash Hash of the inference input data
    /// @param outputHash Hash of the full output probabilities
    /// @param signatures Worker ECDSA signatures attesting to the result
    function submitInferenceResult(
        uint256 jobId,
        uint256 prediction,
        bytes32 inputHash,
        bytes32 outputHash,
        bytes[] calldata signatures
    ) external nonReentrant {
        // Job must exist and be completed (model trained)
        if (jobs[jobId].owner == address(0)) revert JobNotFound();
        if (!jobs[jobId].completed) revert JobNotCompleted();

        // Must have at least MIN_WORKERS signatures
        if (signatures.length < MIN_WORKERS) revert InvalidSignatureCount();

        // Verify each signature
        bytes32 messageHash = keccak256(abi.encodePacked(
            "HELIX_INFERENCE",
            jobId,
            prediction,
            inputHash,
            outputHash
        ));
        bytes32 ethHash = messageHash.toEthSignedMessageHash();

        address[] memory seen = new address[](signatures.length);
        uint256 validSigners = 0;

        for (uint256 i = 0; i < signatures.length; i++) {
            address signer = ethHash.recover(signatures[i]);

            // Signer must be a registered, non-slashed worker for this job
            if (!workers[jobId][signer].registered || workers[jobId][signer].slashed) {
                revert InvalidSigner();
            }

            // No duplicate signers
            for (uint256 j = 0; j < validSigners; j++) {
                if (seen[j] == signer) revert DuplicateSigner();
            }
            seen[validSigners] = signer;
            validSigners++;
        }

        // Store inference result
        uint256 inferenceId = ++inferenceCount;
        inferenceResults[inferenceId] = InferenceResult({
            jobId: jobId,
            prediction: prediction,
            inputHash: inputHash,
            outputHash: outputHash,
            timestamp: block.timestamp,
            signerCount: validSigners
        });

        emit InferenceResultCommitted(
            inferenceId,
            jobId,
            prediction,
            inputHash,
            outputHash,
            validSigners
        );
    }

    /// @notice Get inference result details
    function getInferenceResult(uint256 inferenceId) external view returns (
        uint256 jobId,
        uint256 prediction,
        bytes32 inputHash,
        bytes32 outputHash,
        uint256 timestamp,
        uint256 signerCount
    ) {
        InferenceResult storage r = inferenceResults[inferenceId];
        return (r.jobId, r.prediction, r.inputHash, r.outputHash, r.timestamp, r.signerCount);
    }

    // ============ Admin Functions ============

    /// @notice Update treasury address
    function setTreasury(address _treasury) external onlyOwner {
        if (_treasury == address(0)) revert InvalidTreasury();
        emit TreasuryUpdated(treasury, _treasury);
        treasury = _treasury;
    }

    /// @notice Update ZK verifier address
    function setVerifier(address _verifier) external onlyOwner {
        emit VerifierUpdated(address(verifier), _verifier);
        verifier = IHelixVerifier(_verifier);
    }

    /// @notice Receives ETH (for job payments)
    receive() external payable {}
}
