// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./IHelixVerifier.sol";

/// @title IHelixCoordinatorV2
/// @notice Interface for external integration with HelixCoordinatorV2
interface IHelixCoordinatorV2 {
    // ============ Structs ============

    struct ProofSubmissionData {
        uint256 modelId;
        uint256 roundId;
        bytes proof;
        uint256[] publicInputs;
    }

    // ============ Events ============

    event ModelRegistered(
        uint256 indexed modelId,
        address indexed owner,
        uint256 initialCommitment,
        uint256 minStake,
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
    event BatchProofSubmitted(address indexed submitter, uint256 totalProofs);
    event Staked(
        address indexed prover,
        uint256 indexed modelId,
        uint256 amount,
        uint256 totalStake
    );
    event Unstaked(
        address indexed prover,
        uint256 indexed modelId,
        uint256 amount
    );
    event Slashed(
        address indexed prover,
        uint256 indexed modelId,
        uint256 roundId,
        uint256 amount,
        uint256 remainingStake,
        string reason
    );

    // ============ Model Management ============

    function registerModel(
        string memory ipfsHash,
        uint256 initialCommitment,
        uint256 minStake,
        uint32 dIn,
        uint32 dHidden,
        uint32 dOut,
        uint32 numLayers,
        uint8 activationType
    ) external returns (uint256 modelId);

    function startRound(uint256 modelId, uint256 duration) external;

    function pauseModel(uint256 modelId) external;

    function resumeModel(uint256 modelId) external;

    // ============ Staking ============

    function stake(uint256 modelId) external payable;

    function unstake(uint256 modelId) external;

    // ============ Proof Submission ============

    function submitProof(
        uint256 modelId,
        uint256 roundId,
        bytes memory proof,
        uint256[] memory publicInputs
    ) external;

    function submitProofBatch(ProofSubmissionData[] calldata submissions) external;

    function challengeProof(
        uint256 modelId,
        uint256 roundId,
        bytes memory proof,
        uint256[] memory publicInputs
    ) external;

    // ============ View Functions ============

    function getModelState(uint256 modelId) external view returns (
        uint256 currentRound,
        uint256 currentCommitment,
        bool active
    );

    function getStake(address prover, uint256 modelId) external view returns (
        uint256 amount,
        uint256 lockedUntil,
        bool slashed
    );

    function getSlashingRecordCount() external view returns (uint256);

    function getAccumulatedErrorBound(uint256 modelId) external view returns (uint256);

    function isProofUsed(bytes memory proof, uint256[] memory publicInputs) external view returns (bool);

    function getChallengerConfig() external view returns (
        uint16 rewardPercentage,
        uint128 minReward,
        uint128 maxReward,
        bool enabled
    );

    // ============ State Accessors ============

    function verifier() external view returns (IHelixVerifier);

    function treasury() external view returns (address);

    function owner() external view returns (address);

    function paused() external view returns (bool);

    function nextModelId() external view returns (uint32);

    function maxErrorBound() external view returns (uint256);
}
