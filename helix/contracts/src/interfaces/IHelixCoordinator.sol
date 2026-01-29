// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// @title IHelixCoordinator
/// @notice Interface for the HELIX training coordinator
interface IHelixCoordinator {
    /// @notice Model information
    struct Model {
        string ipfsHash;
        uint256 currentCommitment;
        uint256 currentRound;
        address owner;
    }

    /// @notice Round information
    struct Round {
        uint256 modelCommitment;
        bool isCompleted;
    }

    /// @notice Register a new model
    function registerModel(string memory ipfsHash, uint256 initialCommitment) external;

    /// @notice Start a new training round
    function startRound(uint256 modelId) external;

    /// @notice Submit a gradient with proof
    function submitGradient(
        uint256 modelId,
        uint256 roundId,
        bytes memory proof,
        uint256[] memory publicInputs
    ) external;

    /// @notice Get model information
    function models(uint256 modelId) external view returns (
        string memory ipfsHash,
        uint256 currentCommitment,
        uint256 currentRound,
        address owner
    );

    /// @notice Get round information
    function rounds(uint256 modelId, uint256 roundId) external view returns (
        uint256 modelCommitment,
        bool isCompleted
    );

    /// @notice Get the next model ID
    function nextModelId() external view returns (uint256);

    /// @notice Get the verifier contract
    function verifier() external view returns (address);

    /// @notice Events
    event ModelRegistered(uint256 indexed modelId, address indexed owner, uint256 initialCommitment);
    event RoundStarted(uint256 indexed modelId, uint256 indexed roundId);
    event GradientSubmitted(uint256 indexed modelId, uint256 indexed roundId, address indexed prover);
    event RoundCompleted(uint256 indexed modelId, uint256 indexed roundId, uint256 newCommitment);
}
