// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC721/extensions/ERC721Enumerable.sol";

/// @title HelixModelStore
/// @notice ERC-721 model registry. Each model is an NFT with version history
///         and links to encrypted weights on 0G Storage.
contract HelixModelStore is ERC721Enumerable {
    struct Model {
        string slug;           // user-chosen identifier (globally unique)
        string name;           // display name
        string description;
        address creator;       // original creator (immutable, survives transfer)
        uint40 createdAt;
        bool isPublic;         // when true, anyone has access (for paid inference)
        uint16 inferenceFee;   // basis points (500 = 5%), max 5000 (50%)
    }

    struct Version {
        string semver;         // "1.0.0"
        string rootHash;       // 0G Storage root hash (encrypted), "" if not stored
        uint96 accuracy;       // scaled by 1e4 (9500 = 95.00%)
        uint40 timestamp;
        string sessionId;      // training session reference
        bool weightsStored;    // whether weights were uploaded to 0G
    }

    uint256 private _nextTokenId;

    mapping(uint256 => Model) public models;
    mapping(uint256 => Version[]) private _versions;
    mapping(bytes32 => uint256) private _slugToToken;
    mapping(bytes32 => bool) private _slugTaken;

    /// @dev tokenId => (address => hasAccess)
    mapping(uint256 => mapping(address => bool)) public accessGranted;

    // Events
    event ModelCreated(uint256 indexed tokenId, address indexed creator, string slug, string name);
    event VersionAdded(uint256 indexed tokenId, uint256 indexed versionIndex, string semver, string rootHash);
    event ModelPublicityChanged(uint256 indexed tokenId, bool isPublic);
    event InferenceFeeChanged(uint256 indexed tokenId, uint16 feeBps);
    event AccessChanged(uint256 indexed tokenId, address indexed account, bool granted);

    modifier onlyModelOwner(uint256 tokenId) {
        require(ownerOf(tokenId) == msg.sender, "Not model owner");
        _;
    }

    constructor() ERC721("Helix Model", "HMODEL") {}

    /// @notice Create a new model NFT with a unique slug
    /// @param slug   Globally unique identifier for the model
    /// @param name   Display name
    /// @param description Model description
    /// @return tokenId The newly minted token ID
    function createModel(
        string calldata slug,
        string calldata name,
        string calldata description
    ) external returns (uint256 tokenId) {
        require(bytes(slug).length > 0, "Slug required");
        require(bytes(name).length > 0, "Name required");
        bytes32 slugHash = keccak256(abi.encodePacked(slug));
        require(!_slugTaken[slugHash], "Slug already taken");

        tokenId = _nextTokenId++;
        _safeMint(msg.sender, tokenId);

        models[tokenId] = Model({
            slug: slug,
            name: name,
            description: description,
            creator: msg.sender,
            createdAt: uint40(block.timestamp),
            isPublic: false,
            inferenceFee: 0
        });

        _slugTaken[slugHash] = true;
        _slugToToken[slugHash] = tokenId;
        emit ModelCreated(tokenId, msg.sender, slug, name);
    }

    /// @notice Add a new version to an existing model
    /// @param tokenId       The model token ID
    /// @param semver        Semantic version string (e.g. "1.0.0")
    /// @param rootHash      0G Storage root hash (pass "" if not stored)
    /// @param accuracy      Accuracy scaled by 1e4 (9500 = 95.00%)
    /// @param sessionId     Training session ID reference
    /// @param weightsStored Whether weights were uploaded to 0G
    /// @return versionIndex The index of the newly added version
    function addVersion(
        uint256 tokenId,
        string calldata semver,
        string calldata rootHash,
        uint96 accuracy,
        string calldata sessionId,
        bool weightsStored
    ) external onlyModelOwner(tokenId) returns (uint256 versionIndex) {
        versionIndex = _versions[tokenId].length;
        _versions[tokenId].push(Version({
            semver: semver,
            rootHash: rootHash,
            accuracy: accuracy,
            timestamp: uint40(block.timestamp),
            sessionId: sessionId,
            weightsStored: weightsStored
        }));
        emit VersionAdded(tokenId, versionIndex, semver, rootHash);
    }

    /// @notice Set whether the model is publicly accessible
    /// @param tokenId   The model token ID
    /// @param _isPublic True to make publicly accessible
    function setPublic(uint256 tokenId, bool _isPublic) external onlyModelOwner(tokenId) {
        models[tokenId].isPublic = _isPublic;
        emit ModelPublicityChanged(tokenId, _isPublic);
    }

    /// @notice Set the inference fee in basis points
    /// @param tokenId The model token ID
    /// @param feeBps  Fee in basis points (max 5000 = 50%)
    function setInferenceFee(uint256 tokenId, uint16 feeBps) external onlyModelOwner(tokenId) {
        require(feeBps <= 5000, "Fee exceeds 50%");
        models[tokenId].inferenceFee = feeBps;
        emit InferenceFeeChanged(tokenId, feeBps);
    }

    /// @notice Grant access to a specific account
    /// @param tokenId The model token ID
    /// @param account The address to grant access to
    function grantAccess(uint256 tokenId, address account) external onlyModelOwner(tokenId) {
        accessGranted[tokenId][account] = true;
        emit AccessChanged(tokenId, account, true);
    }

    /// @notice Revoke access from a specific account
    /// @param tokenId The model token ID
    /// @param account The address to revoke access from
    function revokeAccess(uint256 tokenId, address account) external onlyModelOwner(tokenId) {
        accessGranted[tokenId][account] = false;
        emit AccessChanged(tokenId, account, false);
    }

    /// @notice Check if an account has access to a model
    /// @param tokenId The model token ID
    /// @param account The address to check
    /// @return True if the account has access
    function hasModelAccess(uint256 tokenId, address account) external view returns (bool) {
        if (ownerOf(tokenId) == account) return true;
        if (models[tokenId].isPublic) return true;
        return accessGranted[tokenId][account];
    }

    /// @notice Get the number of versions for a model
    /// @param tokenId The model token ID
    /// @return The number of versions
    function getVersionCount(uint256 tokenId) external view returns (uint256) {
        return _versions[tokenId].length;
    }

    /// @notice Get a specific version of a model
    /// @param tokenId The model token ID
    /// @param index   The version index
    /// @return The version data
    function getVersion(uint256 tokenId, uint256 index) external view returns (Version memory) {
        require(index < _versions[tokenId].length, "Invalid version index");
        return _versions[tokenId][index];
    }

    /// @notice Get all versions of a model
    /// @param tokenId The model token ID
    /// @return All versions
    function getVersions(uint256 tokenId) external view returns (Version[] memory) {
        return _versions[tokenId];
    }

    /// @notice Look up a model token ID by its slug
    /// @param slug The slug to look up
    /// @return The token ID
    function getModelBySlug(string calldata slug) external view returns (uint256) {
        bytes32 slugHash = keccak256(abi.encodePacked(slug));
        require(_slugTaken[slugHash], "Slug not found");
        return _slugToToken[slugHash];
    }
}
