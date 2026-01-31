// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// @title DataCommitment
/// @notice Manages dataset commitments for verifiable training data integrity
/// @dev Stores Merkle roots and metadata for training datasets used in HELIX protocol
///      Enables cryptographic verification that specific data was used during training
contract DataCommitment {
    // ============ Structs (Gas Optimized) ============

    /// @notice Dataset commitment information
    /// @dev Packed for storage efficiency: merkleRoot(32) + size(8) + timestamp(5) + verified(1) = 46 bytes
    struct DatasetCommitment {
        bytes32 merkleRoot;           // Merkle root of dataset
        uint64 datasetSize;           // Number of samples in dataset
        uint40 timestamp;             // When commitment was created
        bool verified;                // Whether commitment has been verified
        address owner;                // Owner who created the commitment
        string ipfsHash;              // IPFS hash of dataset metadata
    }

    /// @notice Batch commitment for training data batches
    /// @dev Used during training to commit to specific data batches
    struct BatchCommitment {
        bytes32 batchRoot;            // Merkle root of batch data
        uint32 batchIndex;            // Index of this batch
        uint32 batchSize;             // Number of samples in batch
        uint40 timestamp;             // When batch was committed
        bytes32 datasetCommitmentId;  // Parent dataset commitment
    }

    /// @notice Data proof structure for verifying sample inclusion
    struct DataInclusionProof {
        bytes32[] merkleProof;        // Merkle proof path
        uint256 leafIndex;            // Index of the leaf in the tree
        bytes32 leafHash;             // Hash of the data sample
    }

    // ============ State Variables ============

    /// @notice Contract owner
    address public owner;

    /// @notice Coordinator contract authorized to verify commitments
    address public coordinator;

    /// @notice Counter for commitment IDs
    uint256 public nextCommitmentId;

    /// @notice Mapping of commitment ID to dataset commitment
    mapping(uint256 => DatasetCommitment) public datasetCommitments;

    /// @notice Mapping of model ID to required dataset commitment
    mapping(uint256 => uint256) public modelDatasetRequirements;

    /// @notice Mapping of model ID to round ID to batch commitments
    mapping(uint256 => mapping(uint256 => BatchCommitment[])) internal _batchCommitments;

    /// @notice Mapping to track verified data usage per model per round
    mapping(uint256 => mapping(uint256 => bytes32)) public roundDataRoots;

    /// @notice Mapping of commitment hash to commitment ID for lookup
    mapping(bytes32 => uint256) public commitmentIdByRoot;

    // ============ Events ============

    event DatasetCommitted(
        uint256 indexed commitmentId,
        address indexed owner,
        bytes32 merkleRoot,
        uint64 datasetSize,
        string ipfsHash
    );

    event DatasetVerified(
        uint256 indexed commitmentId,
        address indexed verifier,
        bool verified
    );

    event BatchCommitted(
        uint256 indexed modelId,
        uint256 indexed roundId,
        bytes32 batchRoot,
        uint32 batchIndex,
        uint32 batchSize
    );

    event ModelDatasetLinked(
        uint256 indexed modelId,
        uint256 indexed commitmentId
    );

    event DataProofVerified(
        uint256 indexed modelId,
        uint256 indexed roundId,
        bytes32 leafHash,
        bool valid
    );

    event CoordinatorUpdated(
        address indexed oldCoordinator,
        address indexed newCoordinator
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

    modifier commitmentExists(uint256 commitmentId) {
        require(datasetCommitments[commitmentId].merkleRoot != bytes32(0), "Commitment does not exist");
        _;
    }

    // ============ Constructor ============

    constructor(address _coordinator) {
        owner = msg.sender;
        coordinator = _coordinator;
    }

    // ============ Dataset Commitment Functions ============

    /// @notice Commit to a dataset for use in training
    /// @param merkleRoot Merkle root of the dataset
    /// @param datasetSize Number of samples in the dataset
    /// @param ipfsHash IPFS hash of dataset metadata
    /// @return commitmentId The ID of the created commitment
    function commitDataset(
        bytes32 merkleRoot,
        uint64 datasetSize,
        string calldata ipfsHash
    ) external returns (uint256 commitmentId) {
        require(merkleRoot != bytes32(0), "Invalid merkle root");
        require(datasetSize > 0, "Invalid dataset size");

        commitmentId = nextCommitmentId++;

        datasetCommitments[commitmentId] = DatasetCommitment({
            merkleRoot: merkleRoot,
            datasetSize: datasetSize,
            timestamp: uint40(block.timestamp),
            verified: false,
            owner: msg.sender,
            ipfsHash: ipfsHash
        });

        commitmentIdByRoot[merkleRoot] = commitmentId;

        emit DatasetCommitted(commitmentId, msg.sender, merkleRoot, datasetSize, ipfsHash);
    }

    /// @notice Verify a dataset commitment (sets verified flag)
    /// @param commitmentId The commitment to verify
    function verifyDataset(uint256 commitmentId) external onlyOwner commitmentExists(commitmentId) {
        datasetCommitments[commitmentId].verified = true;
        emit DatasetVerified(commitmentId, msg.sender, true);
    }

    /// @notice Link a model to a required dataset
    /// @param modelId The model ID
    /// @param commitmentId The required dataset commitment ID
    function linkModelToDataset(
        uint256 modelId,
        uint256 commitmentId
    ) external onlyCoordinator commitmentExists(commitmentId) {
        modelDatasetRequirements[modelId] = commitmentId;
        emit ModelDatasetLinked(modelId, commitmentId);
    }

    // ============ Batch Commitment Functions ============

    /// @notice Commit to a batch of training data for a specific round
    /// @param modelId The model ID
    /// @param roundId The training round ID
    /// @param batchRoot Merkle root of the batch
    /// @param batchIndex Index of this batch in the round
    /// @param batchSize Number of samples in the batch
    function commitBatch(
        uint256 modelId,
        uint256 roundId,
        bytes32 batchRoot,
        uint32 batchIndex,
        uint32 batchSize
    ) external {
        require(batchRoot != bytes32(0), "Invalid batch root");

        // Verify batch is from the linked dataset (optional check)
        uint256 datasetCommitmentId = modelDatasetRequirements[modelId];

        _batchCommitments[modelId][roundId].push(BatchCommitment({
            batchRoot: batchRoot,
            batchIndex: batchIndex,
            batchSize: batchSize,
            timestamp: uint40(block.timestamp),
            datasetCommitmentId: bytes32(datasetCommitmentId)
        }));

        emit BatchCommitted(modelId, roundId, batchRoot, batchIndex, batchSize);
    }

    /// @notice Set the aggregated data root for a round (called after all batches committed)
    /// @param modelId The model ID
    /// @param roundId The round ID
    /// @param dataRoot The aggregated Merkle root of all batches
    function setRoundDataRoot(
        uint256 modelId,
        uint256 roundId,
        bytes32 dataRoot
    ) external onlyCoordinator {
        roundDataRoots[modelId][roundId] = dataRoot;
    }

    // ============ Verification Functions ============

    /// @notice Verify that a data sample was included in the training dataset
    /// @param commitmentId The dataset commitment ID
    /// @param proof The inclusion proof
    /// @return valid Whether the proof is valid
    function verifyDataInclusion(
        uint256 commitmentId,
        DataInclusionProof calldata proof
    ) external view commitmentExists(commitmentId) returns (bool valid) {
        bytes32 root = datasetCommitments[commitmentId].merkleRoot;
        return _verifyMerkleProof(proof.merkleProof, root, proof.leafHash, proof.leafIndex);
    }

    /// @notice Verify that a batch was used in a specific training round
    /// @param modelId The model ID
    /// @param roundId The round ID
    /// @param batchRoot The batch Merkle root to verify
    /// @return valid Whether the batch was committed for this round
    function verifyBatchUsed(
        uint256 modelId,
        uint256 roundId,
        bytes32 batchRoot
    ) external view returns (bool valid) {
        BatchCommitment[] storage batches = _batchCommitments[modelId][roundId];

        for (uint256 i = 0; i < batches.length; i++) {
            if (batches[i].batchRoot == batchRoot) {
                return true;
            }
        }
        return false;
    }

    /// @notice Verify a Merkle proof for data inclusion
    /// @param proof The Merkle proof array
    /// @param root The expected Merkle root
    /// @param leaf The leaf hash to verify
    /// @param index The leaf index
    /// @return valid Whether the proof is valid
    function _verifyMerkleProof(
        bytes32[] calldata proof,
        bytes32 root,
        bytes32 leaf,
        uint256 index
    ) internal pure returns (bool valid) {
        bytes32 computedHash = leaf;

        for (uint256 i = 0; i < proof.length; i++) {
            bytes32 proofElement = proof[i];

            if (index % 2 == 0) {
                computedHash = keccak256(abi.encodePacked(computedHash, proofElement));
            } else {
                computedHash = keccak256(abi.encodePacked(proofElement, computedHash));
            }

            index = index / 2;
        }

        return computedHash == root;
    }

    /// @notice Compute Merkle root from leaves (for on-chain verification)
    /// @param leaves Array of leaf hashes
    /// @return root The computed Merkle root
    function computeMerkleRoot(bytes32[] calldata leaves) external pure returns (bytes32 root) {
        require(leaves.length > 0, "Empty leaves array");

        if (leaves.length == 1) {
            return leaves[0];
        }

        // Build tree bottom-up
        bytes32[] memory currentLevel = new bytes32[](leaves.length);
        for (uint256 i = 0; i < leaves.length; i++) {
            currentLevel[i] = leaves[i];
        }

        while (currentLevel.length > 1) {
            uint256 newLen = (currentLevel.length + 1) / 2;
            bytes32[] memory nextLevel = new bytes32[](newLen);

            for (uint256 i = 0; i < newLen; i++) {
                if (2 * i + 1 < currentLevel.length) {
                    nextLevel[i] = keccak256(abi.encodePacked(currentLevel[2*i], currentLevel[2*i+1]));
                } else {
                    nextLevel[i] = currentLevel[2*i];
                }
            }

            currentLevel = nextLevel;
        }

        return currentLevel[0];
    }

    // ============ View Functions ============

    /// @notice Get dataset commitment details
    function getDatasetCommitment(uint256 commitmentId) external view returns (
        bytes32 merkleRoot,
        uint64 datasetSize,
        uint40 timestamp,
        bool verified,
        address commitmentOwner,
        string memory ipfsHash
    ) {
        DatasetCommitment storage c = datasetCommitments[commitmentId];
        return (c.merkleRoot, c.datasetSize, c.timestamp, c.verified, c.owner, c.ipfsHash);
    }

    /// @notice Get batch commitments for a round
    function getBatchCommitments(
        uint256 modelId,
        uint256 roundId
    ) external view returns (BatchCommitment[] memory) {
        return _batchCommitments[modelId][roundId];
    }

    /// @notice Get batch count for a round
    function getBatchCount(uint256 modelId, uint256 roundId) external view returns (uint256) {
        return _batchCommitments[modelId][roundId].length;
    }

    /// @notice Check if a model has a linked dataset
    function hasLinkedDataset(uint256 modelId) external view returns (bool) {
        return modelDatasetRequirements[modelId] != 0 ||
               datasetCommitments[0].merkleRoot != bytes32(0); // Handle case where ID 0 is used
    }

    /// @notice Get the required dataset commitment for a model
    function getModelDatasetRequirement(uint256 modelId) external view returns (uint256) {
        return modelDatasetRequirements[modelId];
    }

    // ============ Admin Functions ============

    /// @notice Update the coordinator address
    function setCoordinator(address _coordinator) external onlyOwner {
        emit CoordinatorUpdated(coordinator, _coordinator);
        coordinator = _coordinator;
    }

    /// @notice Transfer ownership
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid address");
        owner = newOwner;
    }
}
