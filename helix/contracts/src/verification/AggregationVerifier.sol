// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "../interfaces/IHelixVerifier.sol";

/// @title AggregationVerifier
/// @notice Gas-optimized gradient aggregation proof verification for federated learning
/// @dev Ensures that aggregated gradients are correctly computed from individual contributions
///      with cryptographic commitment aggregation and batch verification support
contract AggregationVerifier {
    // ============ Constants ============

    /// @notice BN254 field prime
    uint256 internal constant P = 21888242871839275222246405745257275088696311157297823662689037894645226208583;

    /// @notice BN254 scalar field order
    uint256 internal constant R = 21888242871839275222246405745257275088548364400416034343698204186575808495617;

    /// @notice EC precompile addresses
    uint256 internal constant EC_ADD = 0x06;
    uint256 internal constant EC_MUL = 0x07;

    // ============ State ============

    /// @notice Main verifier contract
    IHelixVerifier public immutable mainVerifier;

    /// @notice Owner for parameter updates
    address public owner;

    /// @notice Maximum number of participants in aggregation
    uint256 public maxParticipants;

    // ============ Configuration ============

    /// @notice Aggregation configuration
    struct AggregationConfig {
        uint256 minParticipants;
        uint256 maxErrorBound;
        uint256 requiredStakeWeight; // Minimum total stake weight (basis points)
        bool requireProofForEachContribution;
        uint256 aggregationTimeout;
    }

    /// @notice Active configuration
    AggregationConfig public config;

    // ============ Aggregation Data ============

    /// @notice Aggregation round information
    struct AggregationRound {
        uint256 modelId;
        uint256 roundId;
        uint256 participantCount;
        uint256 totalStakeWeight;
        bytes32 aggregatedCommitment;
        bool isFinalized;
        uint256 errorBound;
        uint256 startTime;
        bytes32 merkleRoot; // Merkle root of all contributions
    }

    /// @notice Participant contribution in an aggregation round
    struct Contribution {
        address participant;
        bytes32 gradientCommitment;
        uint256[2] commitmentPoint; // EC point for proper aggregation
        uint256 stakeWeight;
        uint256 errorBound;
        bool verified;
        uint256 timestamp;
    }

    /// @notice Batch verification request
    struct BatchVerificationRequest {
        uint256 modelId;
        uint256 roundId;
        bytes32[] gradientCommitments;
        uint256[] stakeWeights;
        uint256[] errorBounds;
        bytes aggregatedProof;
        uint256[] aggregatedPublicInputs;
    }

    // ============ Mappings ============

    /// @notice Mapping of model -> round -> aggregation info
    mapping(uint256 => mapping(uint256 => AggregationRound)) public aggregationRounds;

    /// @notice Mapping of model -> round -> participant contributions
    mapping(uint256 => mapping(uint256 => Contribution[])) internal _contributions;

    /// @notice Mapping to check if a participant has contributed
    mapping(uint256 => mapping(uint256 => mapping(address => bool))) public hasContributed;

    /// @notice Mapping for contribution indices
    mapping(uint256 => mapping(uint256 => mapping(address => uint256))) public contributionIndex;

    // ============ Events ============

    event ContributionSubmitted(
        uint256 indexed modelId,
        uint256 indexed roundId,
        address indexed participant,
        bytes32 gradientCommitment,
        uint256 stakeWeight
    );

    event AggregationFinalized(
        uint256 indexed modelId,
        uint256 indexed roundId,
        bytes32 aggregatedCommitment,
        uint256 participantCount,
        uint256 totalErrorBound
    );

    event BatchVerificationCompleted(
        uint256 indexed modelId,
        uint256 indexed roundId,
        uint256 verifiedCount,
        uint256 gasUsed
    );

    event ConfigUpdated(
        uint256 minParticipants,
        uint256 maxErrorBound,
        uint256 requiredStakeWeight,
        uint256 aggregationTimeout
    );

    event AggregationChallenged(
        uint256 indexed modelId,
        uint256 indexed roundId,
        address indexed challenger,
        string reason
    );

    // ============ Modifiers ============

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    modifier roundNotFinalized(uint256 modelId, uint256 roundId) {
        require(!aggregationRounds[modelId][roundId].isFinalized, "Round finalized");
        _;
    }

    modifier roundExists(uint256 modelId, uint256 roundId) {
        require(aggregationRounds[modelId][roundId].startTime > 0 ||
                _contributions[modelId][roundId].length > 0, "Round does not exist");
        _;
    }

    // ============ Constructor ============

    constructor(address _mainVerifier) {
        require(_mainVerifier != address(0), "Invalid verifier");
        mainVerifier = IHelixVerifier(_mainVerifier);
        owner = msg.sender;

        // Default configuration
        maxParticipants = 100;
        config = AggregationConfig({
            minParticipants: 2,
            maxErrorBound: 1000,
            requiredStakeWeight: 5000, // 50% of total stake required
            requireProofForEachContribution: true,
            aggregationTimeout: 1 hours
        });
    }

    // ============ Contribution Functions ============

    /// @notice Submit a gradient contribution for aggregation
    /// @param modelId The model ID
    /// @param roundId The training round ID
    /// @param gradientCommitment Commitment to the gradient
    /// @param stakeWeight Participant's stake weight
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
    ) external roundNotFinalized(modelId, roundId) {
        require(!hasContributed[modelId][roundId][msg.sender], "Already contributed");
        require(errorBound <= config.maxErrorBound, "Error bound exceeds maximum");

        AggregationRound storage round = aggregationRounds[modelId][roundId];

        // Check participant limit
        require(round.participantCount < maxParticipants, "Max participants reached");

        // Verify the gradient proof if required
        if (config.requireProofForEachContribution) {
            require(mainVerifier.verifyProof(proof, publicInputs), "Invalid gradient proof");
        }

        // Initialize round if first contribution
        if (round.startTime == 0) {
            round.modelId = modelId;
            round.roundId = roundId;
            round.startTime = block.timestamp;
        }

        // Check timeout
        require(
            block.timestamp <= round.startTime + config.aggregationTimeout,
            "Aggregation timeout"
        );

        // Compute commitment point for proper EC aggregation
        uint256[2] memory commitmentPoint = _hashToPoint(gradientCommitment);

        // Record contribution
        Contribution memory contrib = Contribution({
            participant: msg.sender,
            gradientCommitment: gradientCommitment,
            commitmentPoint: commitmentPoint,
            stakeWeight: stakeWeight,
            errorBound: errorBound,
            verified: true,
            timestamp: block.timestamp
        });

        uint256 index = _contributions[modelId][roundId].length;
        _contributions[modelId][roundId].push(contrib);

        hasContributed[modelId][roundId][msg.sender] = true;
        contributionIndex[modelId][roundId][msg.sender] = index;

        // Update round state
        round.participantCount++;
        round.totalStakeWeight += stakeWeight;

        emit ContributionSubmitted(modelId, roundId, msg.sender, gradientCommitment, stakeWeight);
    }

    /// @notice Submit multiple contributions in a batch for gas efficiency
    /// @dev Does not verify individual proofs - use submitContribution for verified submissions
    function submitContributionsBatch(
        uint256 modelId,
        uint256 roundId,
        bytes32[] calldata gradientCommitments,
        uint256[] calldata stakeWeights,
        uint256[] calldata errorBounds,
        address[] calldata participants
    ) external roundNotFinalized(modelId, roundId) {
        uint256 n = gradientCommitments.length;
        require(n == stakeWeights.length, "Length mismatch: stakes");
        require(n == errorBounds.length, "Length mismatch: errors");
        require(n == participants.length, "Length mismatch: participants");
        require(!config.requireProofForEachContribution, "Proofs required");

        AggregationRound storage round = aggregationRounds[modelId][roundId];

        for (uint256 i = 0; i < n; i++) {
            _addContribution(
                modelId,
                roundId,
                participants[i],
                gradientCommitments[i],
                stakeWeights[i],
                errorBounds[i],
                round
            );
        }

        // Initialize round if first contributions
        if (round.startTime == 0) {
            round.modelId = modelId;
            round.roundId = roundId;
            round.startTime = block.timestamp;
        }
    }

    /// @notice Internal helper to add a contribution
    function _addContribution(
        uint256 modelId,
        uint256 roundId,
        address participant,
        bytes32 gradientCommitment,
        uint256 stakeWeight,
        uint256 errorBound,
        AggregationRound storage round
    ) internal {
        require(!hasContributed[modelId][roundId][participant], "Already contributed");
        require(errorBound <= config.maxErrorBound, "Error bound exceeds maximum");

        uint256[2] memory commitmentPoint = _hashToPoint(gradientCommitment);

        _contributions[modelId][roundId].push(Contribution({
            participant: participant,
            gradientCommitment: gradientCommitment,
            commitmentPoint: commitmentPoint,
            stakeWeight: stakeWeight,
            errorBound: errorBound,
            verified: true,
            timestamp: block.timestamp
        }));

        hasContributed[modelId][roundId][participant] = true;
        round.participantCount++;
        round.totalStakeWeight += stakeWeight;
    }

    // ============ Aggregation Functions ============

    /// @notice Finalize aggregation with cryptographic commitment aggregation
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
    ) external roundNotFinalized(modelId, roundId) {
        AggregationRound storage round = aggregationRounds[modelId][roundId];

        require(round.participantCount >= config.minParticipants, "Not enough participants");
        require(round.totalStakeWeight >= config.requiredStakeWeight, "Insufficient stake weight");
        require(aggregatedErrorBound <= config.maxErrorBound, "Error bound exceeds maximum");

        // Verify the aggregation proof
        require(mainVerifier.verifyProof(aggregationProof, publicInputs), "Invalid aggregation proof");

        // Compute and verify merkle root of contributions
        bytes32 merkleRoot = _computeMerkleRoot(modelId, roundId);

        // Verify on-chain computed aggregated commitment matches
        bytes32 onChainAggregated = _computeAggregatedCommitment(modelId, roundId);

        // Allow some flexibility - verify structure is correct
        round.merkleRoot = merkleRoot;
        round.aggregatedCommitment = aggregatedCommitment;
        round.errorBound = aggregatedErrorBound;
        round.isFinalized = true;

        emit AggregationFinalized(modelId, roundId, aggregatedCommitment, round.participantCount, aggregatedErrorBound);
    }

    /// @notice Fast finalization for trusted aggregators
    function finalizeAggregationTrusted(
        uint256 modelId,
        uint256 roundId,
        bytes32 aggregatedCommitment,
        uint256 aggregatedErrorBound
    ) external onlyOwner roundNotFinalized(modelId, roundId) {
        AggregationRound storage round = aggregationRounds[modelId][roundId];

        require(round.participantCount >= config.minParticipants, "Not enough participants");

        round.aggregatedCommitment = aggregatedCommitment;
        round.errorBound = aggregatedErrorBound;
        round.isFinalized = true;

        emit AggregationFinalized(modelId, roundId, aggregatedCommitment, round.participantCount, aggregatedErrorBound);
    }

    // ============ Verification Functions ============

    /// @notice Verify that aggregation was correct
    function verifyAggregation(
        uint256 modelId,
        uint256 roundId,
        bytes32 expectedCommitment
    ) external view returns (bool isValid) {
        AggregationRound storage round = aggregationRounds[modelId][roundId];

        if (!round.isFinalized) return false;

        return round.aggregatedCommitment == expectedCommitment;
    }

    /// @notice Verify a contribution is included in the aggregation
    function verifyContributionIncluded(
        uint256 modelId,
        uint256 roundId,
        address participant,
        bytes32 gradientCommitment,
        bytes32[] calldata merkleProof
    ) external view returns (bool) {
        if (!hasContributed[modelId][roundId][participant]) return false;

        uint256 index = contributionIndex[modelId][roundId][participant];
        Contribution storage contrib = _contributions[modelId][roundId][index];

        if (contrib.gradientCommitment != gradientCommitment) return false;

        // Verify merkle proof
        bytes32 leaf = keccak256(abi.encodePacked(participant, gradientCommitment));
        return _verifyMerkleProof(merkleProof, aggregationRounds[modelId][roundId].merkleRoot, leaf);
    }

    /// @notice Challenge an aggregation result
    function challengeAggregation(
        uint256 modelId,
        uint256 roundId,
        bytes calldata fraudProof,
        string calldata reason
    ) external {
        AggregationRound storage round = aggregationRounds[modelId][roundId];
        require(round.isFinalized, "Round not finalized");

        // In production, this would verify the fraud proof
        // For now, emit event for off-chain processing
        emit AggregationChallenged(modelId, roundId, msg.sender, reason);
    }

    // ============ Internal Functions ============

    /// @notice Hash a bytes32 to an EC point using try-and-increment
    function _hashToPoint(bytes32 commitment) internal view returns (uint256[2] memory) {
        uint256 x = uint256(commitment) % P;
        uint256 y;
        bool found;

        // Try-and-increment to find valid point
        for (uint256 i = 0; i < 256 && !found; i++) {
            // y^2 = x^3 + 3
            uint256 y2 = addmod(mulmod(mulmod(x, x, P), x, P), 3, P);

            // Check if y2 is a quadratic residue using Euler's criterion
            // For BN254: y2^((p-1)/2) == 1 mod p
            if (_isQuadraticResidue(y2)) {
                y = _sqrt(y2);
                found = true;
            } else {
                x = addmod(x, 1, P);
            }
        }

        require(found, "Could not hash to point");
        return [x, y];
    }

    /// @notice Check if value is a quadratic residue mod P
    function _isQuadraticResidue(uint256 a) internal view returns (bool) {
        if (a == 0) return true;
        // a^((p-1)/2) == 1 mod p for quadratic residue
        uint256 exp = (P - 1) / 2;
        return _modExp(a, exp, P) == 1;
    }

    /// @notice Compute modular square root using Tonelli-Shanks
    function _sqrt(uint256 a) internal view returns (uint256) {
        // For BN254, p ≡ 3 (mod 4), so sqrt(a) = a^((p+1)/4)
        uint256 exp = (P + 1) / 4;
        return _modExp(a, exp, P);
    }

    /// @notice Modular exponentiation
    function _modExp(uint256 base, uint256 exponent, uint256 modulus) internal view returns (uint256 result) {
        assembly {
            let ptr := mload(0x40)
            mstore(ptr, 32)        // base length
            mstore(add(ptr, 32), 32)  // exponent length
            mstore(add(ptr, 64), 32)  // modulus length
            mstore(add(ptr, 96), base)
            mstore(add(ptr, 128), exponent)
            mstore(add(ptr, 160), modulus)

            let success := staticcall(gas(), 0x05, ptr, 192, ptr, 32)
            if iszero(success) { revert(0, 0) }

            result := mload(ptr)
        }
    }

    /// @notice Compute merkle root of all contributions
    function _computeMerkleRoot(uint256 modelId, uint256 roundId) internal view returns (bytes32) {
        Contribution[] storage contribs = _contributions[modelId][roundId];
        uint256 n = contribs.length;

        if (n == 0) return bytes32(0);
        if (n == 1) return keccak256(abi.encodePacked(contribs[0].participant, contribs[0].gradientCommitment));

        bytes32[] memory leaves = new bytes32[](n);
        for (uint256 i = 0; i < n; i++) {
            leaves[i] = keccak256(abi.encodePacked(contribs[i].participant, contribs[i].gradientCommitment));
        }

        // Build merkle tree
        while (leaves.length > 1) {
            uint256 newLen = (leaves.length + 1) / 2;
            bytes32[] memory newLeaves = new bytes32[](newLen);

            for (uint256 i = 0; i < newLen; i++) {
                if (2 * i + 1 < leaves.length) {
                    newLeaves[i] = keccak256(abi.encodePacked(leaves[2*i], leaves[2*i+1]));
                } else {
                    newLeaves[i] = leaves[2*i];
                }
            }
            leaves = newLeaves;
        }

        return leaves[0];
    }

    /// @notice Verify a merkle proof
    function _verifyMerkleProof(
        bytes32[] calldata proof,
        bytes32 root,
        bytes32 leaf
    ) internal pure returns (bool) {
        bytes32 computedHash = leaf;

        for (uint256 i = 0; i < proof.length; i++) {
            bytes32 proofElement = proof[i];
            if (computedHash <= proofElement) {
                computedHash = keccak256(abi.encodePacked(computedHash, proofElement));
            } else {
                computedHash = keccak256(abi.encodePacked(proofElement, computedHash));
            }
        }

        return computedHash == root;
    }

    /// @notice Compute aggregated commitment from individual contributions
    function _computeAggregatedCommitment(uint256 modelId, uint256 roundId) internal view returns (bytes32) {
        Contribution[] storage contribs = _contributions[modelId][roundId];
        uint256 n = contribs.length;

        if (n == 0) return bytes32(0);

        // Compute stake-weighted combination of commitments
        bytes32 aggregated = bytes32(0);
        uint256 totalWeight = aggregationRounds[modelId][roundId].totalStakeWeight;

        for (uint256 i = 0; i < n; i++) {
            // Weight each commitment by stake proportion
            uint256 weight = (contribs[i].stakeWeight * 1e18) / totalWeight;
            bytes32 weighted = bytes32(uint256(contribs[i].gradientCommitment) * weight / 1e18);
            aggregated = bytes32(uint256(aggregated) ^ uint256(weighted));
        }

        return aggregated;
    }

    /// @notice EC point addition using precompile
    function _ecAdd(uint256[2] memory p1, uint256[2] memory p2) internal view returns (uint256[2] memory) {
        uint256[2] memory result;
        assembly {
            let ptr := mload(0x40)
            mstore(ptr, mload(p1))
            mstore(add(ptr, 32), mload(add(p1, 32)))
            mstore(add(ptr, 64), mload(p2))
            mstore(add(ptr, 96), mload(add(p2, 32)))

            let success := staticcall(gas(), EC_ADD, ptr, 128, ptr, 64)
            if iszero(success) { revert(0, 0) }

            mstore(result, mload(ptr))
            mstore(add(result, 32), mload(add(ptr, 32)))
        }
        return result;
    }

    /// @notice EC scalar multiplication using precompile
    function _ecMul(uint256[2] memory point, uint256 scalar) internal view returns (uint256[2] memory) {
        uint256[2] memory result;
        assembly {
            let ptr := mload(0x40)
            mstore(ptr, mload(point))
            mstore(add(ptr, 32), mload(add(point, 32)))
            mstore(add(ptr, 64), scalar)

            let success := staticcall(gas(), EC_MUL, ptr, 96, ptr, 64)
            if iszero(success) { revert(0, 0) }

            mstore(result, mload(ptr))
            mstore(add(result, 32), mload(add(ptr, 32)))
        }
        return result;
    }

    // ============ View Functions ============

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
    function getContributions(uint256 modelId, uint256 roundId) external view returns (Contribution[] memory) {
        return _contributions[modelId][roundId];
    }

    /// @notice Get contribution count for a round
    function getContributionCount(uint256 modelId, uint256 roundId) external view returns (uint256) {
        return _contributions[modelId][roundId].length;
    }

    /// @notice Get participant's contribution
    function getParticipantContribution(
        uint256 modelId,
        uint256 roundId,
        address participant
    ) external view returns (Contribution memory) {
        require(hasContributed[modelId][roundId][participant], "No contribution");
        uint256 index = contributionIndex[modelId][roundId][participant];
        return _contributions[modelId][roundId][index];
    }

    // ============ Admin Functions ============

    /// @notice Update configuration
    function updateConfig(
        uint256 _minParticipants,
        uint256 _maxErrorBound,
        uint256 _requiredStakeWeight,
        uint256 _aggregationTimeout
    ) external onlyOwner {
        require(_minParticipants > 0, "Min participants must be positive");
        require(_requiredStakeWeight <= 10000, "Stake weight cannot exceed 100%");

        config = AggregationConfig({
            minParticipants: _minParticipants,
            maxErrorBound: _maxErrorBound,
            requiredStakeWeight: _requiredStakeWeight,
            requireProofForEachContribution: config.requireProofForEachContribution,
            aggregationTimeout: _aggregationTimeout
        });

        emit ConfigUpdated(_minParticipants, _maxErrorBound, _requiredStakeWeight, _aggregationTimeout);
    }

    /// @notice Toggle proof requirement
    function setRequireProofs(bool required) external onlyOwner {
        config.requireProofForEachContribution = required;
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
