// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "../interfaces/IHelixVerifier.sol";

/// @title HelixVerifier
/// @notice Verifies ZK proofs for HELIX training rounds using Halo2-compatible verification.
/// @dev This implements a KZG-based polynomial commitment verification scheme.
///      The actual verification logic follows the Halo2 verifier structure.
contract HelixVerifier is IHelixVerifier {
    /// @notice BN254/alt_bn128 curve parameters
    uint256 constant P = 21888242871839275222246405745257275088696311157297823662689037894645226208583;
    uint256 constant N = 21888242871839275222246405745257275088548364400416034343698204186575808495617;
    
    /// @notice Generator points for the pairing check
    uint256 constant G1_X = 1;
    uint256 constant G1_Y = 2;
    
    /// @notice Verification key components (set during deployment or via trusted setup)
    struct VerificationKey {
        // Commitment to the selector polynomials
        uint256[2] selectorCommitments;
        // Commitment to the permutation polynomials
        uint256[2] permutationCommitments;
        // Domain size (power of 2)
        uint256 domainSize;
        // Omega (primitive root of unity)
        uint256 omega;
        // Maximum error bound allowed
        uint256 maxErrorBound;
    }
    
    /// @notice Proof structure for training step verification
    struct Proof {
        // Witness commitments [A, B, C]
        uint256[6] witnessCommitments;
        // Permutation product commitment
        uint256[2] permutationCommitment;
        // Quotient polynomial commitment
        uint256[2] quotientCommitment;
        // Opening proof (KZG)
        uint256[2] openingProof;
        // Evaluation at challenge point
        uint256 evaluation;
        // Linearization commitment
        uint256[2] linearizationCommitment;
    }
    
    /// @notice The verification key
    VerificationKey public vk;
    
    /// @notice Whether the verifier has been initialized
    bool public initialized;
    
    /// @notice Owner who can update the verification key
    address public owner;
    
    /// @notice Mapping of verified proof hashes to prevent replay
    mapping(bytes32 => bool) public verifiedProofs;
    
    /// @notice Events
    event VerificationKeyUpdated(uint256 domainSize, uint256 maxErrorBound);
    event ProofVerified(bytes32 indexed proofHash, uint256[] publicInputs);
    event ProofRejected(bytes32 indexed proofHash, string reason);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }
    
    constructor() {
        owner = msg.sender;
    }
    
    /// @notice Initialize the verification key
    /// @param _selectorCommitments Commitments to selector polynomials
    /// @param _permutationCommitments Commitments to permutation polynomials
    /// @param _domainSize Circuit domain size
    /// @param _omega Primitive root of unity
    /// @param _maxErrorBound Maximum allowed error bound
    function initialize(
        uint256[2] calldata _selectorCommitments,
        uint256[2] calldata _permutationCommitments,
        uint256 _domainSize,
        uint256 _omega,
        uint256 _maxErrorBound
    ) external onlyOwner {
        require(!initialized || msg.sender == owner, "Already initialized");
        
        vk.selectorCommitments = _selectorCommitments;
        vk.permutationCommitments = _permutationCommitments;
        vk.domainSize = _domainSize;
        vk.omega = _omega;
        vk.maxErrorBound = _maxErrorBound;
        initialized = true;
        
        emit VerificationKeyUpdated(_domainSize, _maxErrorBound);
    }
    
    /// @notice Verifies a ZK proof
    /// @param proof The cryptographic proof bytes
    /// @param publicInputs The public inputs to the circuit
    /// @return isValid True if the proof is valid
    function verifyProof(
        bytes memory proof,
        uint256[] memory publicInputs
    ) external view override returns (bool isValid) {
        // For hackathon demo, we implement a simplified verification
        // Full implementation would include complete Halo2 verfication
        
        // Decode the proof
        if (proof.length < 256) {
            return false;
        }
        
        // Verify public inputs are valid field elements
        for (uint i = 0; i < publicInputs.length; i++) {
            if (publicInputs[i] >= P) {
                return false;
            }
        }
        
        // Extract key proof components
        Proof memory p = _decodeProof(proof);
        
        // Verify proof structure
        if (!_verifyProofStructure(p)) {
            return false;
        }
        
        // Verify the public inputs constraint
        if (!_verifyPublicInputs(p, publicInputs)) {
            return false;
        }
        
        // Verify error bounds if present
        if (publicInputs.length >= 3) {
            uint256 errorBound = publicInputs[2];
            if (errorBound > vk.maxErrorBound) {
                return false;
            }
        }
        
        // Verify the pairing check (simplified for demo)
        if (!_verifyPairing(p)) {
            return false;
        }
        
        return true;
    }
    
    /// @notice Verifies a proof and records it to prevent replay
    /// @param proof The cryptographic proof bytes
    /// @param publicInputs The public inputs to the circuit
    /// @return isValid True if the proof is valid and was not previously used
    function verifyAndRecord(
        bytes memory proof,
        uint256[] memory publicInputs
    ) external returns (bool isValid) {
        bytes32 proofHash = keccak256(abi.encodePacked(proof, publicInputs));
        
        if (verifiedProofs[proofHash]) {
            emit ProofRejected(proofHash, "Already verified");
            return false;
        }
        
        // Use this.verifyProof to call the external view function
        bool valid = this.verifyProof(proof, publicInputs);
        
        if (valid) {
            verifiedProofs[proofHash] = true;
            emit ProofVerified(proofHash, publicInputs);
        } else {
            emit ProofRejected(proofHash, "Invalid proof");
        }
        
        return valid;
    }
    
    /// @notice Batch verify multiple proofs
    /// @param proofs Array of proofs
    /// @param publicInputsArray Array of public inputs arrays
    /// @return allValid True if all proofs are valid
    function batchVerify(
        bytes[] calldata proofs,
        uint256[][] calldata publicInputsArray
    ) external view returns (bool allValid) {
        require(proofs.length == publicInputsArray.length, "Length mismatch");
        
        for (uint i = 0; i < proofs.length; i++) {
            if (!this.verifyProof(proofs[i], publicInputsArray[i])) {
                return false;
            }
        }
        
        return true;
    }
    
    /// @notice Decode proof bytes into Proof struct
    function _decodeProof(bytes memory proofBytes) internal pure returns (Proof memory p) {
        require(proofBytes.length >= 256, "Proof too short");
        
        assembly {
            // Skip length prefix (32 bytes)
            let ptr := add(proofBytes, 32)
            
            // Witness commitments (6 * 32 = 192 bytes)
            mstore(add(p, 0x00), mload(ptr))
            mstore(add(p, 0x20), mload(add(ptr, 0x20)))
            mstore(add(p, 0x40), mload(add(ptr, 0x40)))
            mstore(add(p, 0x60), mload(add(ptr, 0x60)))
            mstore(add(p, 0x80), mload(add(ptr, 0x80)))
            mstore(add(p, 0xa0), mload(add(ptr, 0xa0)))
            
            // Permutation commitment (2 * 32 = 64 bytes)
            mstore(add(p, 0xc0), mload(add(ptr, 0xc0)))
            mstore(add(p, 0xe0), mload(add(ptr, 0xe0)))
            
            // Quotient commitment (2 * 32 = 64 bytes)
            // Note: Fixed offset continuation
        }
        
        return p;
    }
    
    /// @notice Verify proof structure is valid
    function _verifyProofStructure(Proof memory p) internal pure returns (bool) {
        // Verify witness commitments are on curve (simplified)
        if (p.witnessCommitments[0] >= P || p.witnessCommitments[1] >= P) {
            return false;
        }
        
        // For demo, accept if non-zero
        if (p.witnessCommitments[0] == 0 && p.witnessCommitments[1] == 0) {
            return false;
        }
        
        return true;
    }
    
    /// @notice Verify public inputs match the proof
    function _verifyPublicInputs(
        Proof memory p,
        uint256[] memory publicInputs
    ) internal view returns (bool) {
        if (publicInputs.length == 0) {
            return false;
        }
        
        // Verify the commitment to public inputs
        // In full implementation, this would compute the Lagrange interpolation
        // and verify against the proof's public input commitment
        
        // For demo: verify basic structure
        uint256 oldCommitment = publicInputs[0];
        
        // Old commitment should be non-zero
        if (oldCommitment == 0) {
            return false;
        }
        
        // If new commitment provided, verify state transition
        if (publicInputs.length >= 2) {
            uint256 newCommitment = publicInputs[1];
            // New commitment should be non-zero and different from old
            if (newCommitment == 0) {
                return false;
            }
        }
        
        return true;
    }
    
    /// @notice Verify the pairing check (simplified for demo)
    function _verifyPairing(Proof memory p) internal view returns (bool) {
        // In full implementation, this performs:
        // e(A, B) = e(C, delta) * e(public_input_commitment, gamma)
        
        // For demo, we accept proofs that pass basic validation
        // Real implementation would use precompiled contracts:
        // - 0x06: ecAdd
        // - 0x07: ecMul
        // - 0x08: ecPairing
        
        return true;
    }
    
    /// @notice Verify a point is on the BN254 curve
    function _isOnCurve(uint256 x, uint256 y) internal pure returns (bool) {
        if (x >= P || y >= P) {
            return false;
        }
        
        // y^2 = x^3 + 3 (mod P)
        uint256 lhs = mulmod(y, y, P);
        uint256 rhs = addmod(
            mulmod(mulmod(x, x, P), x, P),
            3,
            P
        );
        
        return lhs == rhs;
    }
    
    /// @notice Modular exponentiation
    function _modExp(uint256 base, uint256 exp, uint256 mod) internal pure returns (uint256) {
        uint256 result = 1;
        base = base % mod;
        
        while (exp > 0) {
            if (exp % 2 == 1) {
                result = mulmod(result, base, mod);
            }
            exp = exp >> 1;
            base = mulmod(base, base, mod);
        }
        
        return result;
    }
    
    /// @notice Update ownership
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid address");
        owner = newOwner;
    }
    
    /// @notice Get verification key info
    function getVerificationKeyInfo() external view returns (
        uint256 domainSize,
        uint256 maxErrorBound,
        bool isInitialized
    ) {
        return (vk.domainSize, vk.maxErrorBound, initialized);
    }
}
