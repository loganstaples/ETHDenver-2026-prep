// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IHelixVerifier {
    /// @notice Verifies a ZK proof.
    /// @param proof The cryptographic proof bytes.
    /// @param publicInputs The public inputs to the circuit.
    /// @return isValid True if the proof is valid, false otherwise.
    function verifyProof(
        bytes memory proof,
        uint256[] memory publicInputs
    ) external view returns (bool isValid);
}
