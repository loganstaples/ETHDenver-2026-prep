// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// @title BoundsChecker
/// @notice Verifies error bounds claims for approximate computation proofs
/// @dev Ensures that claimed error bounds are valid and within acceptable limits
contract BoundsChecker {
    /// @notice Maximum allowable error bound (in basis points, 10000 = 100%)
    uint256 public maxAllowedError;
    
    /// @notice Owner for parameter updates
    address public owner;
    
    /// @notice Error bound registry for verified computations
    struct BoundRecord {
        uint256 computationId;
        bytes32 inputCommitment;
        bytes32 outputCommitment;
        uint256 claimedError;
        uint256 verifiedAt;
        address verifier;
    }
    
    /// @notice Operation types with their error propagation rules
    enum OperationType {
        Add,          // err = err_a + err_b
        Subtract,     // err = err_a + err_b
        Multiply,     // err = |a|*err_b + |b|*err_a + err_a*err_b
        Divide,       // err = (|a|*err_b + |b|*err_a) / (|b|^2 - err_b^2)
        MatMul,       // err = n * max(|a|*err_b + |b|*err_a + err_a*err_b)
        ReLU,         // err = err_input (preserves)
        Softmax,      // err = 2 * sum(err_input) * max(output)
        LayerNorm     // err = complex formula based on variance
    }
    
    /// @notice Error propagation coefficients for each operation type
    struct ErrorCoefficients {
        uint256 linearCoeff;      // Coefficient for linear term
        uint256 quadraticCoeff;   // Coefficient for quadratic term
        uint256 constantTerm;     // Constant error floor
    }
    
    /// @notice Mapping of operation type to error coefficients
    mapping(OperationType => ErrorCoefficients) public errorCoefficients;
    
    /// @notice Verified bound records
    mapping(bytes32 => BoundRecord) public boundRecords;
    
    /// @notice Model-specific error configurations
    struct ModelErrorConfig {
        uint256 maxLayerError;        // Max error per layer
        uint256 maxTotalError;        // Max cumulative error
        uint256 maxGradientError;     // Max gradient error
        uint256 errorDecayFactor;     // How errors decay between layers (basis points)
        bool enabled;
    }
    
    /// @notice Mapping of model ID to error configuration
    mapping(uint256 => ModelErrorConfig) public modelConfigs;
    
    /// @notice Events
    event BoundVerified(
        bytes32 indexed recordId,
        uint256 computationId,
        uint256 claimedError,
        uint256 maxAllowed
    );
    event BoundRejected(
        bytes32 indexed recordId,
        uint256 claimedError,
        uint256 maxAllowed,
        string reason
    );
    event ModelConfigured(uint256 indexed modelId, ModelErrorConfig config);
    event MaxAllowedErrorUpdated(uint256 oldValue, uint256 newValue);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }
    
    constructor(uint256 _maxAllowedError) {
        require(_maxAllowedError <= 10000, "Max error too high");
        maxAllowedError = _maxAllowedError;
        owner = msg.sender;
        
        // Set default error coefficients
        _setDefaultCoefficients();
    }
    
    /// @notice Set default error propagation coefficients
    function _setDefaultCoefficients() internal {
        // Addition: error adds linearly
        errorCoefficients[OperationType.Add] = ErrorCoefficients({
            linearCoeff: 10000,    // 1.0
            quadraticCoeff: 0,
            constantTerm: 0
        });
        
        // Multiplication: error has quadratic component
        errorCoefficients[OperationType.Multiply] = ErrorCoefficients({
            linearCoeff: 10000,
            quadraticCoeff: 10000,
            constantTerm: 0
        });
        
        // MatMul: scales with dimension
        errorCoefficients[OperationType.MatMul] = ErrorCoefficients({
            linearCoeff: 10000,
            quadraticCoeff: 10000,
            constantTerm: 1         // Minimum 1 unit error
        });
        
        // ReLU: preserves error for positive values
        errorCoefficients[OperationType.ReLU] = ErrorCoefficients({
            linearCoeff: 10000,
            quadraticCoeff: 0,
            constantTerm: 0
        });
        
        // Softmax: amplifies errors
        errorCoefficients[OperationType.Softmax] = ErrorCoefficients({
            linearCoeff: 20000,    // 2.0 multiplier
            quadraticCoeff: 0,
            constantTerm: 1
        });
    }
    
    /// @notice Verify that a claimed error bound is valid
    /// @param computationId Unique ID for the computation
    /// @param inputCommitment Commitment to input values
    /// @param outputCommitment Commitment to output values
    /// @param claimedError The claimed error bound
    /// @param operationType Type of operation performed
    /// @param inputErrorA Error bound of first input
    /// @param inputErrorB Error bound of second input (if applicable)
    /// @return isValid Whether the claimed error bound is acceptable
    function verifyBound(
        uint256 computationId,
        bytes32 inputCommitment,
        bytes32 outputCommitment,
        uint256 claimedError,
        OperationType operationType,
        uint256 inputErrorA,
        uint256 inputErrorB
    ) external returns (bool isValid) {
        // Calculate expected minimum error based on operation type
        uint256 expectedMinError = calculateExpectedError(
            operationType,
            inputErrorA,
            inputErrorB,
            1e18,  // Assume unit magnitude for simplicity
            1e18
        );
        
        // Claimed error must be at least the expected propagated error
        if (claimedError < expectedMinError) {
            emit BoundRejected(
                _computeRecordId(computationId, inputCommitment),
                claimedError,
                expectedMinError,
                "Error bound too optimistic"
            );
            return false;
        }
        
        // Claimed error must not exceed max allowed
        if (claimedError > maxAllowedError) {
            emit BoundRejected(
                _computeRecordId(computationId, inputCommitment),
                claimedError,
                maxAllowedError,
                "Error bound exceeds maximum"
            );
            return false;
        }
        
        // Record the verified bound
        bytes32 recordId = _computeRecordId(computationId, inputCommitment);
        boundRecords[recordId] = BoundRecord({
            computationId: computationId,
            inputCommitment: inputCommitment,
            outputCommitment: outputCommitment,
            claimedError: claimedError,
            verifiedAt: block.timestamp,
            verifier: msg.sender
        });
        
        emit BoundVerified(recordId, computationId, claimedError, maxAllowedError);
        return true;
    }
    
    /// @notice Calculate expected error after an operation
    /// @param opType Type of operation
    /// @param errorA Error of first input
    /// @param errorB Error of second input
    /// @param magnitudeA Magnitude of first input
    /// @param magnitudeB Magnitude of second input
    /// @return expectedError Expected output error
    function calculateExpectedError(
        OperationType opType,
        uint256 errorA,
        uint256 errorB,
        uint256 magnitudeA,
        uint256 magnitudeB
    ) public view returns (uint256 expectedError) {
        ErrorCoefficients storage coeffs = errorCoefficients[opType];
        
        if (opType == OperationType.Add || opType == OperationType.Subtract) {
            // err_out = err_a + err_b
            expectedError = errorA + errorB;
        } else if (opType == OperationType.Multiply) {
            // err_out = |a|*err_b + |b|*err_a + err_a*err_b
            expectedError = (magnitudeA * errorB) / 1e18 +
                           (magnitudeB * errorA) / 1e18 +
                           (errorA * errorB) / 1e18;
        } else if (opType == OperationType.ReLU) {
            // err_out = err_in (preserved)
            expectedError = errorA;
        } else if (opType == OperationType.MatMul) {
            // err_out = sum of element-wise multiplication errors
            // Simplified: scale by 2 for a typical matrix
            expectedError = 2 * ((magnitudeA * errorB) / 1e18 +
                                (magnitudeB * errorA) / 1e18);
        } else if (opType == OperationType.Softmax) {
            // Softmax amplifies errors
            expectedError = 2 * errorA;
        } else {
            // Default: linear propagation
            expectedError = (errorA * coeffs.linearCoeff) / 10000 + coeffs.constantTerm;
        }
        
        return expectedError;
    }
    
    /// @notice Verify cumulative error for a sequence of operations
    /// @param modelId Model ID for configuration lookup
    /// @param operationErrors Array of individual operation errors
    /// @return isValid Whether cumulative error is within bounds
    function verifyCumulativeError(
        uint256 modelId,
        uint256[] calldata operationErrors
    ) external view returns (bool isValid) {
        ModelErrorConfig storage config = modelConfigs[modelId];
        
        if (!config.enabled) {
            // Use default max error
            uint256 total = 0;
            for (uint i = 0; i < operationErrors.length; i++) {
                total += operationErrors[i];
                if (total > maxAllowedError) {
                    return false;
                }
            }
            return true;
        }
        
        // Use model-specific configuration
        uint256 cumulative = 0;
        for (uint i = 0; i < operationErrors.length; i++) {
            // Check individual layer error
            if (operationErrors[i] > config.maxLayerError) {
                return false;
            }
            
            // Accumulate with decay
            cumulative = (cumulative * config.errorDecayFactor) / 10000 + operationErrors[i];
            
            // Check cumulative
            if (cumulative > config.maxTotalError) {
                return false;
            }
        }
        
        return true;
    }
    
    /// @notice Configure error bounds for a specific model
    function configureModel(
        uint256 modelId,
        uint256 maxLayerError,
        uint256 maxTotalError,
        uint256 maxGradientError,
        uint256 errorDecayFactor
    ) external onlyOwner {
        require(errorDecayFactor <= 10000, "Decay factor too high");
        
        modelConfigs[modelId] = ModelErrorConfig({
            maxLayerError: maxLayerError,
            maxTotalError: maxTotalError,
            maxGradientError: maxGradientError,
            errorDecayFactor: errorDecayFactor,
            enabled: true
        });
        
        emit ModelConfigured(modelId, modelConfigs[modelId]);
    }
    
    /// @notice Check if a bound record exists and is valid
    function getBoundRecord(bytes32 recordId) external view returns (
        bool exists,
        uint256 claimedError,
        uint256 verifiedAt,
        address verifier
    ) {
        BoundRecord storage record = boundRecords[recordId];
        exists = record.verifiedAt > 0;
        claimedError = record.claimedError;
        verifiedAt = record.verifiedAt;
        verifier = record.verifier;
    }
    
    /// @notice Compute record ID from inputs
    function _computeRecordId(
        uint256 computationId,
        bytes32 inputCommitment
    ) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked(computationId, inputCommitment));
    }
    
    /// @notice Update error coefficients for an operation type
    function updateErrorCoefficients(
        OperationType opType,
        uint256 linearCoeff,
        uint256 quadraticCoeff,
        uint256 constantTerm
    ) external onlyOwner {
        errorCoefficients[opType] = ErrorCoefficients({
            linearCoeff: linearCoeff,
            quadraticCoeff: quadraticCoeff,
            constantTerm: constantTerm
        });
    }
    
    /// @notice Update maximum allowed error
    function setMaxAllowedError(uint256 newMax) external onlyOwner {
        require(newMax <= 10000, "Max error too high");
        emit MaxAllowedErrorUpdated(maxAllowedError, newMax);
        maxAllowedError = newMax;
    }
    
    /// @notice Transfer ownership
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid address");
        owner = newOwner;
    }
}
