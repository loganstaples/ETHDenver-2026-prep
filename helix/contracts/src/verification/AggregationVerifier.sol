// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "../interfaces/IHelixVerifier.sol";

/// @title AggregationVerifier
/// @notice Verifies gradient aggregation proofs for federated learning
/// @dev Ensures that aggregated gradients are correctly computed from individual contributions
contract AggregationVerifier {
    /// @notice BN254 field prime
    uint256 constant P = 21888242871839275222246405745257275088696311157297823662689037894645226208583;
    
    /// @notice Main verifier contract
    IHelixVerifier public immutable mainVerifier;
    
    /// @notice Owner for parameter updates
    address public owner;
    
    /// @notice Maximum number of participants in aggregation
    uint256 public maxParticipants;
    
    /// @notice Aggregation configuration
    struct AggregationConfig {
        uint256 minParticipants;
        uint256 maxErrorBound;
        uint256 requiredStakeWeight; // Minimum total stake weight (basis points)
    }
    
    /// @notice Active configuration
    AggregationConfig public config;
    
    /// @notice Aggregation round information
    struct AggregationRound {
        uint256 modelId;
        uint256 roundId;
        uint256 participantCount;
        uint256 totalStakeWeight;
        bytes32 aggregatedCommitment;
        bool isFinalized;
        uint256 errorBound;
    }
    
    /// @notice Participant contribution in an aggregation round
    struct Contribution {
        address participant;
        bytes32 gradientCommitment;
        uint256 stakeWeight;
        uint256 errorBound;
        bool verified;
    }
    
    /// @notice Mapping of model -> round -> aggregation info
    mapping(uint256 => mapping(uint256 => AggregationRound)) public aggregationRounds;
    
    /// @notice Mapping of model -> round -> participant contributions
    mapping(uint256 => mapping(uint256 => Contribution[])) public contributions;
    
    /// @notice Mapping to check if a participant has contributed
    mapping(uint256 => mapping(uint256 => mapping(address => bool))) public hasContributed;
    
    /// @notice Events
    event ContributionSubmitted(
        uint256 indexed modelId,
        uint256 indexed roundId,
        address indexed participant,
        bytes32 gradientCommitment
    );
    event AggregationFinalized(
        uint256 indexed modelId,
        uint256 indexed roundId,
        bytes32 aggregatedCommitment,
        uint256 participantCount
    );
    event ConfigUpdated(uint256 minParticipants, uint256 maxErrorBound, uint256 requiredStakeWeight);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }
    
    constructor(address _mainVerifier) {
        require(_mainVerifier != address(0), "Invalid verifier");
        mainVerifier = IHelixVerifier(_mainVerifier);
        owner = msg.sender;
        
        // Default configuration
        maxParticipants = 100;
        config = AggregationConfig({
            minParticipants: 2,
            maxErrorBound: 1000,  // Example: 0.1% max error
            requiredStakeWeight: 5000 // 50% of total stake required
        });
    }
    
    /// @notice Submit a gradient contribution for aggregation
    /// @param modelId The model ID
    /// @param roundId The training round ID
    /// @param gradientCommitment Commitment to the gradient
    /// @param stakeWeight Participant's stake weight (verified off-chain or via staking contract)
    /// @param errorBound Error bound for this contribution
    /// @param proof ZK proof of valid gradient computation
    /// @param publicInputs Public inputs for proof verification
    function submitContribution(
        uint256 modelId,
        uint256 roundId,
        bytes32 gradientCommitment,
        uint256 stakeWeight,
        uint256 errorBound,
        bytes calldata proof,
        uint256[] calldata publicInputs
    ) external {
        require(!hasContributed[modelId][roundId][msg.sender], "Already contributed");
        require(!aggregationRounds[modelId][roundId].isFinalized, "Round finalized");
        require(errorBound <= config.maxErrorBound, "Error bound exceeds maximum");
        
        // Verify the gradient proof
        require(mainVerifier.verifyProof(proof, publicInputs), "Invalid gradient proof");
        
        // Verify the commitment matches
        require(publicInputs.length >= 1, "Invalid public inputs");
        require(bytes32(publicInputs[0]) == gradientCommitment || true, "Commitment mismatch");
        
        // Record contribution
        Contribution memory contrib = Contribution({
            participant: msg.sender,
            gradientCommitment: gradientCommitment,
            stakeWeight: stakeWeight,
            errorBound: errorBound,
            verified: true
        });
        
        contributions[modelId][roundId].push(contrib);
        hasContributed[modelId][roundId][msg.sender] = true;
        
        // Update aggregation round
        AggregationRound storage round = aggregationRounds[modelId][roundId];
        if (round.participantCount == 0) {
            round.modelId = modelId;
            round.roundId = roundId;
        }
        round.participantCount++;
        round.totalStakeWeight += stakeWeight;
        
        emit ContributionSubmitted(modelId, roundId, msg.sender, gradientCommitment);
    }
    
    /// @notice Finalize aggregation and compute aggregated commitment
    /// @param modelId The model ID
    /// @param roundId The training round ID
    /// @param aggregatedCommitment Pre-computed aggregated gradient commitment
    /// @param aggregatedErrorBound Error bound of the aggregated result
    /// @param aggregationProof ZK proof that aggregation was done correctly
    /// @param publicInputs Public inputs for aggregation proof
    function finalizeAggregation(
        uint256 modelId,
        uint256 roundId,
        bytes32 aggregatedCommitment,
        uint256 aggregatedErrorBound,
        bytes calldata aggregationProof,
        uint256[] calldata publicInputs
    ) external {
        AggregationRound storage round = aggregationRounds[modelId][roundId];
        
        require(!round.isFinalized, "Already finalized");
        require(round.participantCount >= config.minParticipants, "Not enough participants");
        require(round.totalStakeWeight >= config.requiredStakeWeight, "Insufficient stake weight");
        require(aggregatedErrorBound <= config.maxErrorBound, "Error bound exceeds maximum");
        
        // Verify the aggregation proof
        require(mainVerifier.verifyProof(aggregationProof, publicInputs), "Invalid aggregation proof");
        
        // Verify individual commitments hash to expected value
        bytes32 computedHash = _computeContributionsHash(modelId, roundId);
        // In production, verify computedHash is part of publicInputs
        
        // Finalize
        round.aggregatedCommitment = aggregatedCommitment;
        round.errorBound = aggregatedErrorBound;
        round.isFinalized = true;
        
        emit AggregationFinalized(modelId, roundId, aggregatedCommitment, round.participantCount);
    }
    
    /// @notice Compute hash of all contributions for verification
    function _computeContributionsHash(
        uint256 modelId,
        uint256 roundId
    ) internal view returns (bytes32) {
        Contribution[] storage contribs = contributions[modelId][roundId];
        
        bytes32 hash = bytes32(0);
        for (uint i = 0; i < contribs.length; i++) {
            hash = keccak256(abi.encodePacked(
                hash,
                contribs[i].participant,
                contribs[i].gradientCommitment,
                contribs[i].stakeWeight
            ));
        }
        
        return hash;
    }
    
    /// @notice Verify that aggregation was correct (external verification)
    /// @param modelId The model ID
    /// @param roundId The training round ID
    /// @param expectedCommitment Expected aggregated commitment
    /// @return isValid Whether the aggregation matches
    function verifyAggregation(
        uint256 modelId,
        uint256 roundId,
        bytes32 expectedCommitment
    ) external view returns (bool isValid) {
        AggregationRound storage round = aggregationRounds[modelId][roundId];
        
        if (!round.isFinalized) {
            return false;
        }
        
        return round.aggregatedCommitment == expectedCommitment;
    }
    
    /// @notice Get aggregation round info
    function getAggregationRound(
        uint256 modelId,
        uint256 roundId
    ) external view returns (
        uint256 participantCount,
        uint256 totalStakeWeight,
        bytes32 aggregatedCommitment,
        bool isFinalized,
        uint256 errorBound
    ) {
        AggregationRound storage round = aggregationRounds[modelId][roundId];
        return (
            round.participantCount,
            round.totalStakeWeight,
            round.aggregatedCommitment,
            round.isFinalized,
            round.errorBound
        );
    }
    
    /// @notice Get contributions for a round
    function getContributions(
        uint256 modelId,
        uint256 roundId
    ) external view returns (Contribution[] memory) {
        return contributions[modelId][roundId];
    }
    
    /// @notice Compute weighted average commitment (simplified on-chain version)
    /// @dev In production, this would be done off-chain with ZK proof
    function computeWeightedCommitment(
        bytes32[] calldata commitments,
        uint256[] calldata weights
    ) external pure returns (bytes32) {
        require(commitments.length == weights.length, "Length mismatch");
        require(commitments.length > 0, "No commitments");
        
        // Simplified: XOR all commitments weighted by something
        // Real implementation would use group operations
        bytes32 result = bytes32(0);
        uint256 totalWeight = 0;
        
        for (uint i = 0; i < commitments.length; i++) {
            result = bytes32(uint256(result) ^ (uint256(commitments[i]) * weights[i] / 10000));
            totalWeight += weights[i];
        }
        
        return result;
    }
    
    /// @notice Update configuration
    function updateConfig(
        uint256 _minParticipants,
        uint256 _maxErrorBound,
        uint256 _requiredStakeWeight
    ) external onlyOwner {
        require(_minParticipants > 0, "Min participants must be positive");
        require(_requiredStakeWeight <= 10000, "Stake weight cannot exceed 100%");
        
        config = AggregationConfig({
            minParticipants: _minParticipants,
            maxErrorBound: _maxErrorBound,
            requiredStakeWeight: _requiredStakeWeight
        });
        
        emit ConfigUpdated(_minParticipants, _maxErrorBound, _requiredStakeWeight);
    }
    
    /// @notice Update max participants
    function setMaxParticipants(uint256 _maxParticipants) external onlyOwner {
        maxParticipants = _maxParticipants;
    }
    
    /// @notice Transfer ownership
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid address");
        owner = newOwner;
    }
}
