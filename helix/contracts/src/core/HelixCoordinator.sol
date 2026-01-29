// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "../interfaces/IHelixVerifier.sol";

contract HelixCoordinator {
    struct Model {
        string ipfsHash;
        uint256 currentCommitment;
        uint256 currentRound;
        address owner;
    }

    struct Round {
        uint256 modelCommitment; // Expected commitment this round
        bool isCompleted;
    }

    mapping(uint256 => Model) public models;
    mapping(uint256 => mapping(uint256 => Round)) public rounds; // modelId -> roundId -> Round
    uint256 public nextModelId;

    IHelixVerifier public verifier;

    event ModelRegistered(uint256 indexed modelId, address indexed owner, uint256 initialCommitment);
    event RoundStarted(uint256 indexed modelId, uint256 indexed roundId);
    event GradientSubmitted(uint256 indexed modelId, uint256 indexed roundId, address indexed prover);
    event RoundCompleted(uint256 indexed modelId, uint256 indexed roundId, uint256 newCommitment);

    constructor(address _verifier) {
        verifier = IHelixVerifier(_verifier);
    }

    function registerModel(string memory ipfsHash, uint256 initialCommitment) external {
        uint256 id = nextModelId++;
        models[id] = Model({
            ipfsHash: ipfsHash,
            currentCommitment: initialCommitment,
            currentRound: 0,
            owner: msg.sender
        });
        emit ModelRegistered(id, msg.sender, initialCommitment);
    }

    function startRound(uint256 modelId) external {
        Model storage model = models[modelId];
        require(msg.sender == model.owner, "Only owner can start round");
        
        uint256 roundId = ++model.currentRound;
        rounds[modelId][roundId] = Round({
            modelCommitment: model.currentCommitment,
            isCompleted: false
        });

        emit RoundStarted(modelId, roundId);
    }

    /// @notice Submits a proof of a valid state transition (Application of Gradient).
    /// Public Inputs expected: [oldCommitment, newCommitment, gradientCommitment]
    function submitGradient(
        uint256 modelId,
        uint256 roundId,
        bytes memory proof,
        uint256[] memory publicInputs
    ) external {
        Model storage model = models[modelId];
        Round storage round = rounds[modelId][roundId];
        
        require(roundId == model.currentRound, "Invalid round");
        require(!round.isCompleted, "Round already completed");
        require(publicInputs.length >= 2, "Invalid public inputs");
        
        // Verify Old Commitment matches on-chain state
        require(publicInputs[0] == round.modelCommitment, "Commitment mismatch");

        // Verify Proof
        require(verifier.verifyProof(proof, publicInputs), "Invalid proof");

        // Update State (In a real FL system, we aggregate. Here we accept the first valid transition as the update)
        uint256 newCommitment = publicInputs[1];
        model.currentCommitment = newCommitment;
        round.isCompleted = true;

        emit GradientSubmitted(modelId, roundId, msg.sender);
        emit RoundCompleted(modelId, roundId, newCommitment);
    }
}
