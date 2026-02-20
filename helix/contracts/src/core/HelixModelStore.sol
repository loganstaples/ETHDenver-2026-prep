// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC721/extensions/ERC721Enumerable.sol";
import "@openzeppelin/contracts/utils/Base64.sol";
import "@openzeppelin/contracts/utils/Strings.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/// @title HelixModelStore
/// @notice ERC-721 model registry. Each model is an NFT with version history
///         and links to encrypted weights on 0G Storage.
contract HelixModelStore is ERC721Enumerable, ReentrancyGuard {
    struct Model {
        string slug;           // user-chosen identifier (globally unique)
        string name;           // display name
        string description;
        string architecture;   // e.g. "784x32x10"
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

    // Marketplace state
    mapping(uint256 => bool) public isForSale;
    mapping(uint256 => uint256) public salePrice;
    mapping(uint256 => uint256) public inferenceFeesAccrued;
    uint256 private _inferenceNonce;

    // Events
    event ModelCreated(uint256 indexed tokenId, address indexed creator, string slug, string name);
    event VersionAdded(uint256 indexed tokenId, uint256 indexed versionIndex, string semver, string rootHash);
    event ModelPublicityChanged(uint256 indexed tokenId, bool isPublic);
    event InferenceFeeChanged(uint256 indexed tokenId, uint16 feeBps);
    event AccessChanged(uint256 indexed tokenId, address indexed account, bool granted);
    event ModelForSaleChanged(uint256 indexed tokenId, bool forSale);
    event SalePriceChanged(uint256 indexed tokenId, uint256 price);
    event ModelSold(uint256 indexed tokenId, address indexed seller, address indexed buyer, uint256 price);
    event InferencePaid(uint256 indexed tokenId, address indexed payer, uint256 nonce, uint256 amount, uint256 ownerShare);

    modifier onlyModelOwner(uint256 tokenId) {
        require(ownerOf(tokenId) == msg.sender, "Not model owner");
        _;
    }

    constructor() ERC721("Helix Model", "HMODEL") {}

    /// @notice Create a new model NFT with a unique slug
    /// @param slug   Globally unique identifier for the model
    /// @param name   Display name
    /// @param description Model description
    /// @param architecture Model architecture string (e.g. "784x32x10"), pass "" if unknown
    /// @return tokenId The newly minted token ID
    function createModel(
        string calldata slug,
        string calldata name,
        string calldata description,
        string calldata architecture
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
            architecture: architecture,
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

    // ─── Metadata ────────────────────────────────────────────────────

    /// @notice Returns on-chain JSON metadata for the NFT (ERC-721 tokenURI).
    function tokenURI(uint256 tokenId) public view override returns (string memory) {
        _requireOwned(tokenId);
        Model storage m = models[tokenId];

        // Get latest version info if available
        string memory latestVersion = "0.0.0";
        string memory accuracy = "N/A";
        string memory storagePointer = "";
        if (_versions[tokenId].length > 0) {
            Version storage v = _versions[tokenId][_versions[tokenId].length - 1];
            latestVersion = v.semver;
            // Zero-pad fractional part: 9505 → "95.05%", 9500 → "95.00%"
            uint256 whole = v.accuracy / 100;
            uint256 frac = v.accuracy % 100;
            accuracy = string.concat(
                Strings.toString(whole), ".",
                frac < 10 ? "0" : "",
                Strings.toString(frac), "%"
            );
            storagePointer = v.rootHash;
        }

        string memory json = string.concat(
            '{"name":"', m.name,
            '","description":"', m.description,
            '","attributes":[',
                '{"trait_type":"Slug","value":"', m.slug, '"},',
                '{"trait_type":"Architecture","value":"', m.architecture, '"},',
                '{"trait_type":"Creator","value":"', Strings.toHexString(m.creator), '"},',
                '{"trait_type":"Version","value":"', latestVersion, '"},',
                '{"trait_type":"Accuracy","value":"', accuracy, '"},',
                '{"trait_type":"StoragePointer","value":"', storagePointer, '"},',
                '{"trait_type":"Public","value":"', m.isPublic ? "Yes" : "No", '"},',
                '{"trait_type":"Versions","display_type":"number","value":', Strings.toString(_versions[tokenId].length), '}',
            ']}'
        );

        return string.concat("data:application/json;base64,", Base64.encode(bytes(json)));
    }

    // ─── Marketplace ──────────────────────────────────────────────────

    /// @notice List or delist a model for sale
    function setForSale(uint256 tokenId, bool _forSale) external onlyModelOwner(tokenId) {
        isForSale[tokenId] = _forSale;
        emit ModelForSaleChanged(tokenId, _forSale);
    }

    /// @notice Set the asking price for a model
    function setSalePrice(uint256 tokenId, uint256 price) external onlyModelOwner(tokenId) {
        salePrice[tokenId] = price;
        emit SalePriceChanged(tokenId, price);
    }

    /// @notice Buy a model NFT that is listed for sale
    function buyModel(uint256 tokenId) external payable nonReentrant {
        require(isForSale[tokenId], "Model not for sale");
        require(salePrice[tokenId] > 0, "Sale price not set");
        require(msg.value >= salePrice[tokenId], "Insufficient payment");
        address seller = ownerOf(tokenId);
        require(msg.sender != seller, "Cannot buy own model");

        // Clear listing
        isForSale[tokenId] = false;
        salePrice[tokenId] = 0;

        // Transfer NFT
        _transfer(seller, msg.sender, tokenId);

        // Pay seller
        (bool sent, ) = payable(seller).call{value: msg.value}("");
        require(sent, "Payment failed");

        emit ModelSold(tokenId, seller, msg.sender, msg.value);
    }

    /// @notice Pay for inference on a public model (non-owner only)
    /// @return nonce Unique nonce for this inference payment
    function payForInference(uint256 tokenId) external payable nonReentrant returns (uint256 nonce) {
        require(models[tokenId].isPublic, "Model is not public");
        require(msg.sender != ownerOf(tokenId), "Owner does not pay for inference");
        require(msg.value > 0, "Payment required");

        uint16 feeBps = models[tokenId].inferenceFee;
        uint256 ownerShare = (msg.value * feeBps) / 10000;
        inferenceFeesAccrued[tokenId] += ownerShare;

        nonce = _inferenceNonce++;
        emit InferencePaid(tokenId, msg.sender, nonce, msg.value, ownerShare);
    }

    /// @notice Withdraw accrued inference fees
    function withdrawInferenceFees(uint256 tokenId) external nonReentrant onlyModelOwner(tokenId) {
        uint256 amount = inferenceFeesAccrued[tokenId];
        require(amount > 0, "No fees to withdraw");
        inferenceFeesAccrued[tokenId] = 0;
        (bool sent, ) = payable(msg.sender).call{value: amount}("");
        require(sent, "Withdrawal failed");
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
