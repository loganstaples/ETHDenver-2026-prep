// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// @title DataVerifier
/// @notice Efficient Merkle proof verification for training data integrity
/// @dev Provides gas-optimized verification of data commitments and proofs
///      Used to verify that specific training data was used during model training
contract DataVerifier {
    // ============ Constants ============

    /// @notice Maximum Merkle tree depth (supports 2^32 leaves)
    uint256 public constant MAX_TREE_DEPTH = 32;

    /// @notice Precomputed empty tree hashes for efficient verification
    bytes32[32] public emptyTreeHashes;

    // ============ State ============

    /// @notice Contract owner
    address public owner;

    /// @notice Verified root registry (root -> timestamp)
    mapping(bytes32 => uint256) public verifiedRoots;

    /// @notice Proof verification records
    mapping(bytes32 => ProofRecord) public proofRecords;

    /// @notice Counter for verified proofs
    uint256 public totalVerifiedProofs;

    // ============ Structs ============

    /// @notice Record of a verified proof
    struct ProofRecord {
        bytes32 root;
        bytes32 leafHash;
        uint256 leafIndex;
        address verifier;
        uint40 timestamp;
        bool valid;
    }

    /// @notice Batch verification request
    struct BatchProofRequest {
        bytes32 root;
        bytes32 leafHash;
        uint256 leafIndex;
        bytes32[] proof;
    }

    /// @notice Sparse Merkle proof for efficient storage
    struct SparseMerkleProof {
        bytes32[] siblings;
        uint256 bitmap;  // Bitmap indicating which siblings are empty
    }

    // ============ Events ============

    event RootVerified(
        bytes32 indexed root,
        address indexed verifier,
        uint256 timestamp
    );

    event ProofVerified(
        bytes32 indexed proofHash,
        bytes32 indexed root,
        bytes32 leafHash,
        bool valid
    );

    event BatchVerificationCompleted(
        bytes32 indexed batchId,
        uint256 totalProofs,
        uint256 validProofs,
        uint256 gasUsed
    );

    // ============ Modifiers ============

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    // ============ Constructor ============

    constructor() {
        owner = msg.sender;
        _initializeEmptyTreeHashes();
    }

    /// @notice Initialize empty tree hashes for sparse Merkle trees
    function _initializeEmptyTreeHashes() internal {
        emptyTreeHashes[0] = bytes32(0);
        for (uint256 i = 1; i < MAX_TREE_DEPTH; i++) {
            emptyTreeHashes[i] = keccak256(abi.encodePacked(emptyTreeHashes[i-1], emptyTreeHashes[i-1]));
        }
    }

    // ============ Core Verification Functions ============

    /// @notice Verify a standard Merkle proof
    /// @param root The Merkle root
    /// @param leafHash The hash of the leaf to verify
    /// @param leafIndex The index of the leaf
    /// @param proof The Merkle proof
    /// @return valid Whether the proof is valid
    function verifyProof(
        bytes32 root,
        bytes32 leafHash,
        uint256 leafIndex,
        bytes32[] calldata proof
    ) external view returns (bool valid) {
        return _verifyProof(root, leafHash, leafIndex, proof);
    }

    /// @notice Verify and record a Merkle proof
    /// @param root The Merkle root
    /// @param leafHash The hash of the leaf to verify
    /// @param leafIndex The index of the leaf
    /// @param proof The Merkle proof
    /// @return valid Whether the proof is valid
    function verifyAndRecord(
        bytes32 root,
        bytes32 leafHash,
        uint256 leafIndex,
        bytes32[] calldata proof
    ) external returns (bool valid) {
        valid = _verifyProof(root, leafHash, leafIndex, proof);

        bytes32 proofHash = keccak256(abi.encodePacked(root, leafHash, leafIndex));

        proofRecords[proofHash] = ProofRecord({
            root: root,
            leafHash: leafHash,
            leafIndex: leafIndex,
            verifier: msg.sender,
            timestamp: uint40(block.timestamp),
            valid: valid
        });

        if (valid) {
            totalVerifiedProofs++;
        }

        emit ProofVerified(proofHash, root, leafHash, valid);
    }

    /// @notice Internal proof verification
    function _verifyProof(
        bytes32 root,
        bytes32 leafHash,
        uint256 leafIndex,
        bytes32[] calldata proof
    ) internal pure returns (bool) {
        require(proof.length <= MAX_TREE_DEPTH, "Proof too long");

        bytes32 computedHash = leafHash;

        for (uint256 i = 0; i < proof.length; i++) {
            bytes32 proofElement = proof[i];

            if ((leafIndex >> i) & 1 == 0) {
                computedHash = _hashPair(computedHash, proofElement);
            } else {
                computedHash = _hashPair(proofElement, computedHash);
            }
        }

        return computedHash == root;
    }

    /// @notice Hash two elements together
    function _hashPair(bytes32 a, bytes32 b) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked(a, b));
    }

    // ============ Batch Verification ============

    /// @notice Verify multiple proofs in a single call (gas optimized)
    /// @param requests Array of batch proof requests
    /// @return validCount Number of valid proofs
    /// @return results Array of validation results
    function batchVerify(
        BatchProofRequest[] calldata requests
    ) external view returns (uint256 validCount, bool[] memory results) {
        results = new bool[](requests.length);

        for (uint256 i = 0; i < requests.length; i++) {
            results[i] = _verifyProof(
                requests[i].root,
                requests[i].leafHash,
                requests[i].leafIndex,
                requests[i].proof
            );
            if (results[i]) {
                validCount++;
            }
        }
    }

    /// @notice Verify multiple proofs sharing the same root (more efficient)
    /// @param root The shared Merkle root
    /// @param leafHashes Array of leaf hashes
    /// @param leafIndices Array of leaf indices
    /// @param proofs Array of proofs
    /// @return validCount Number of valid proofs
    function batchVerifySameRoot(
        bytes32 root,
        bytes32[] calldata leafHashes,
        uint256[] calldata leafIndices,
        bytes32[][] calldata proofs
    ) external view returns (uint256 validCount) {
        require(leafHashes.length == leafIndices.length, "Length mismatch: indices");
        require(leafHashes.length == proofs.length, "Length mismatch: proofs");

        for (uint256 i = 0; i < leafHashes.length; i++) {
            if (_verifyProof(root, leafHashes[i], leafIndices[i], proofs[i])) {
                validCount++;
            }
        }
    }

    // ============ Sparse Merkle Proof Verification ============

    /// @notice Verify a sparse Merkle proof (for trees with many empty leaves)
    /// @param root The Merkle root
    /// @param leafHash The hash of the leaf
    /// @param leafIndex The index of the leaf
    /// @param sparseProof The sparse proof with bitmap
    /// @return valid Whether the proof is valid
    function verifySparseProof(
        bytes32 root,
        bytes32 leafHash,
        uint256 leafIndex,
        SparseMerkleProof calldata sparseProof
    ) external view returns (bool valid) {
        bytes32 computedHash = leafHash;
        uint256 siblingIndex = 0;

        for (uint256 i = 0; i < MAX_TREE_DEPTH; i++) {
            bytes32 sibling;

            if ((sparseProof.bitmap >> i) & 1 == 1) {
                // This sibling is empty, use precomputed hash
                sibling = emptyTreeHashes[i];
            } else {
                // Use the provided sibling
                require(siblingIndex < sparseProof.siblings.length, "Insufficient siblings");
                sibling = sparseProof.siblings[siblingIndex++];
            }

            if ((leafIndex >> i) & 1 == 0) {
                computedHash = _hashPair(computedHash, sibling);
            } else {
                computedHash = _hashPair(sibling, computedHash);
            }
        }

        return computedHash == root;
    }

    // ============ Root Management ============

    /// @notice Register a verified root
    /// @param root The Merkle root to register
    function registerVerifiedRoot(bytes32 root) external onlyOwner {
        require(root != bytes32(0), "Invalid root");
        verifiedRoots[root] = block.timestamp;
        emit RootVerified(root, msg.sender, block.timestamp);
    }

    /// @notice Check if a root is verified
    /// @param root The root to check
    /// @return isVerified Whether the root is verified
    /// @return verifiedAt When the root was verified
    function isRootVerified(bytes32 root) external view returns (bool isVerified, uint256 verifiedAt) {
        verifiedAt = verifiedRoots[root];
        isVerified = verifiedAt > 0;
    }

    // ============ Utility Functions ============

    /// @notice Compute the Merkle root from an array of leaves
    /// @param leaves The leaf hashes
    /// @return root The computed Merkle root
    function computeRoot(bytes32[] calldata leaves) external pure returns (bytes32 root) {
        require(leaves.length > 0, "Empty leaves");

        if (leaves.length == 1) {
            return leaves[0];
        }

        // Pad to power of 2
        uint256 n = 1;
        while (n < leaves.length) {
            n *= 2;
        }

        bytes32[] memory nodes = new bytes32[](n);
        for (uint256 i = 0; i < leaves.length; i++) {
            nodes[i] = leaves[i];
        }
        for (uint256 i = leaves.length; i < n; i++) {
            nodes[i] = bytes32(0);
        }

        // Build tree
        while (n > 1) {
            for (uint256 i = 0; i < n / 2; i++) {
                nodes[i] = _hashPair(nodes[2*i], nodes[2*i + 1]);
            }
            n /= 2;
        }

        return nodes[0];
    }

    /// @notice Compute leaf hash from raw data
    /// @param data The raw data
    /// @return leafHash The computed leaf hash
    function computeLeafHash(bytes calldata data) external pure returns (bytes32 leafHash) {
        return keccak256(data);
    }

    /// @notice Compute leaf hash with domain separation
    /// @param domain Domain separator
    /// @param data The data
    /// @return leafHash The computed leaf hash
    function computeLeafHashWithDomain(
        bytes32 domain,
        bytes calldata data
    ) external pure returns (bytes32 leafHash) {
        return keccak256(abi.encodePacked(domain, keccak256(data)));
    }

    /// @notice Get proof record
    function getProofRecord(bytes32 proofHash) external view returns (ProofRecord memory) {
        return proofRecords[proofHash];
    }

    /// @notice Get empty tree hash at depth
    function getEmptyTreeHash(uint256 depth) external view returns (bytes32) {
        require(depth < MAX_TREE_DEPTH, "Depth too large");
        return emptyTreeHashes[depth];
    }

    // ============ Admin Functions ============

    /// @notice Transfer ownership
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid address");
        owner = newOwner;
    }
}
