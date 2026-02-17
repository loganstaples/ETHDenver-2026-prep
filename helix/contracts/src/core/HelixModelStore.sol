// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// @title HelixModelStore
/// @notice Lightweight on-chain registry for user-owned ML models.
///         Stores model version metadata and links to 0G Storage root hashes.
contract HelixModelStore {
    struct ModelEntry {
        string version;      // Semver (e.g. "1.0.0")
        string rootHash;     // 0G Storage root hash (empty if not stored)
        uint96 accuracy;     // Scaled by 1e4 (e.g. 9500 = 95.00%)
        uint40 timestamp;    // Block timestamp of registration
        string sessionId;    // Training session reference
    }

    /// @dev owner address => array of model versions
    mapping(address => ModelEntry[]) private _models;

    event ModelRegistered(
        address indexed owner,
        uint256 indexed index,
        string version,
        string rootHash,
        uint96 accuracy
    );

    /// @notice Register a new model version
    /// @param version   Semantic version string
    /// @param rootHash  0G Storage root hash (pass "" if not stored)
    /// @param accuracy  Accuracy scaled by 1e4 (9500 = 95.00%)
    /// @param sessionId Training session ID reference
    /// @return index    The index of the newly registered entry
    function registerModel(
        string calldata version,
        string calldata rootHash,
        uint96 accuracy,
        string calldata sessionId
    ) external returns (uint256 index) {
        index = _models[msg.sender].length;
        _models[msg.sender].push(ModelEntry({
            version: version,
            rootHash: rootHash,
            accuracy: accuracy,
            timestamp: uint40(block.timestamp),
            sessionId: sessionId
        }));
        emit ModelRegistered(msg.sender, index, version, rootHash, accuracy);
    }

    /// @notice Get all models for a given user
    function getModels(address user) external view returns (ModelEntry[] memory) {
        return _models[user];
    }

    /// @notice Get the number of models a user has registered
    function getModelCount(address user) external view returns (uint256) {
        return _models[user].length;
    }
}
