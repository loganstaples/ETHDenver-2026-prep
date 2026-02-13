// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// @title ITrainingRound
/// @notice Interface for the training round contract
interface ITrainingRound {
    /// @notice Round states
    enum RoundState {
        Created,
        Active,
        Aggregating,
        Completed,
        Failed
    }

    /// @notice Create a new training round (without dataset hash)
    function createRound(
        uint256 modelId,
        bytes32 startCommitment,
        uint256 minParticipants,
        uint256 maxParticipants,
        uint256 duration
    ) external returns (uint256 roundId);

    /// @notice Create a new training round with dataset hash
    function createRound(
        uint256 modelId,
        bytes32 startCommitment,
        uint256 minParticipants,
        uint256 maxParticipants,
        uint256 duration,
        bytes32 datasetHash
    ) external returns (uint256 roundId);

    /// @notice Register for a round
    function registerForRound(uint256 modelId, uint256 roundId) external;

    /// @notice Submit gradient for a round
    function submitGradient(
        uint256 modelId,
        uint256 roundId,
        bytes32 gradientCommitment,
        uint256 errorBound
    ) external;

    /// @notice Start aggregation phase
    function startAggregation(uint256 modelId, uint256 roundId) external;

    /// @notice Complete a round
    function completeRound(
        uint256 modelId,
        uint256 roundId,
        bytes32 endCommitment,
        uint256 totalErrorBound
    ) external;

    /// @notice Fail a round
    function failRound(
        uint256 modelId,
        uint256 roundId,
        string calldata reason
    ) external;

    /// @notice Get round count for a model
    function roundCount(uint256 modelId) external view returns (uint256);

    /// @notice Check if deadline passed
    function isDeadlinePassed(uint256 modelId, uint256 roundId) external view returns (bool);

    /// @notice Events
    event RoundCreated(uint256 indexed modelId, uint256 indexed roundId, bytes32 startCommitment, uint256 deadline);
    event ParticipantRegistered(uint256 indexed modelId, uint256 indexed roundId, address indexed participant);
    event GradientSubmitted(uint256 indexed modelId, uint256 indexed roundId, address indexed participant, bytes32 gradientCommitment);
    event RoundActivated(uint256 indexed modelId, uint256 indexed roundId);
    event RoundAggregating(uint256 indexed modelId, uint256 indexed roundId);
    event RoundCompleted(uint256 indexed modelId, uint256 indexed roundId, bytes32 endCommitment, uint256 participantCount);
    event RoundFailed(uint256 indexed modelId, uint256 indexed roundId, string reason);
}
