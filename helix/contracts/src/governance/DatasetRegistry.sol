// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// @title DatasetRegistry
/// @notice Registry for approved datasets used in HELIX training
/// @dev Tracks dataset metadata, access permissions, and usage statistics
contract DatasetRegistry {
    /// @notice Dataset information
    struct Dataset {
        string name;
        string description;
        string ipfsHash;           // IPFS hash of dataset or metadata
        string dataType;           // e.g., "text", "image", "tabular"
        uint256 size;              // Approximate size in bytes
        address owner;
        uint256 createdAt;
        bool isApproved;           // Whether DAO has approved this dataset
        bool isActive;
        uint256 usageCount;        // Number of training rounds using this dataset
    }
    
    /// @notice Dataset access permission
    struct AccessPermission {
        bool canRead;
        bool canTrain;
        uint256 grantedAt;
        uint256 expiresAt;
    }
    
    /// @notice Dataset counter
    uint256 public nextDatasetId;
    
    /// @notice Mapping of dataset ID to dataset info
    mapping(uint256 => Dataset) public datasets;
    
    /// @notice Mapping of dataset ID to model ID to usage approval
    mapping(uint256 => mapping(uint256 => bool)) public modelDatasetApproval;
    
    /// @notice Mapping of dataset ID to address to access permission
    mapping(uint256 => mapping(address => AccessPermission)) public accessPermissions;
    
    /// @notice Mapping of owner to their dataset IDs
    mapping(address => uint256[]) public ownerDatasets;
    
    /// @notice Owner for admin functions
    address public owner;
    
    /// @notice DAO contract for approvals
    address public dao;
    
    /// @notice Events
    event DatasetRegistered(
        uint256 indexed datasetId,
        address indexed owner,
        string name,
        string ipfsHash
    );
    event DatasetApproved(uint256 indexed datasetId);
    event DatasetRevoked(uint256 indexed datasetId);
    event AccessGranted(
        uint256 indexed datasetId,
        address indexed grantee,
        bool canRead,
        bool canTrain
    );
    event AccessRevoked(uint256 indexed datasetId, address indexed grantee);
    event ModelDatasetLinked(uint256 indexed datasetId, uint256 indexed modelId);
    event DatasetUsed(uint256 indexed datasetId, uint256 indexed modelId, uint256 roundId);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }
    
    modifier onlyDAO() {
        require(msg.sender == dao || msg.sender == owner, "Only DAO");
        _;
    }
    
    modifier onlyDatasetOwner(uint256 datasetId) {
        require(datasets[datasetId].owner == msg.sender, "Not dataset owner");
        _;
    }
    
    constructor() {
        owner = msg.sender;
    }
    
    /// @notice Register a new dataset
    /// @param name Dataset name
    /// @param description Dataset description
    /// @param ipfsHash IPFS hash of dataset
    /// @param dataType Type of data
    /// @param size Approximate size in bytes
    /// @return datasetId ID of the registered dataset
    function registerDataset(
        string calldata name,
        string calldata description,
        string calldata ipfsHash,
        string calldata dataType,
        uint256 size
    ) external returns (uint256 datasetId) {
        datasetId = nextDatasetId++;
        
        datasets[datasetId] = Dataset({
            name: name,
            description: description,
            ipfsHash: ipfsHash,
            dataType: dataType,
            size: size,
            owner: msg.sender,
            createdAt: block.timestamp,
            isApproved: false,
            isActive: true,
            usageCount: 0
        });
        
        ownerDatasets[msg.sender].push(datasetId);
        
        // Grant owner full access
        accessPermissions[datasetId][msg.sender] = AccessPermission({
            canRead: true,
            canTrain: true,
            grantedAt: block.timestamp,
            expiresAt: type(uint256).max
        });
        
        emit DatasetRegistered(datasetId, msg.sender, name, ipfsHash);
    }
    
    /// @notice Approve a dataset for use in training (DAO action)
    function approveDataset(uint256 datasetId) external onlyDAO {
        require(datasets[datasetId].owner != address(0), "Dataset not found");
        require(!datasets[datasetId].isApproved, "Already approved");
        
        datasets[datasetId].isApproved = true;
        
        emit DatasetApproved(datasetId);
    }
    
    /// @notice Revoke dataset approval (DAO action)
    function revokeApproval(uint256 datasetId) external onlyDAO {
        require(datasets[datasetId].isApproved, "Not approved");
        
        datasets[datasetId].isApproved = false;
        
        emit DatasetRevoked(datasetId);
    }
    
    /// @notice Grant access to a dataset
    /// @param datasetId Dataset ID
    /// @param grantee Address to grant access
    /// @param canRead Whether grantee can read data
    /// @param canTrain Whether grantee can use for training
    /// @param duration Duration of access in seconds (0 for permanent)
    function grantAccess(
        uint256 datasetId,
        address grantee,
        bool canRead,
        bool canTrain,
        uint256 duration
    ) external onlyDatasetOwner(datasetId) {
        require(grantee != address(0), "Invalid address");
        
        uint256 expiresAt = duration > 0 ? block.timestamp + duration : type(uint256).max;
        
        accessPermissions[datasetId][grantee] = AccessPermission({
            canRead: canRead,
            canTrain: canTrain,
            grantedAt: block.timestamp,
            expiresAt: expiresAt
        });
        
        emit AccessGranted(datasetId, grantee, canRead, canTrain);
    }
    
    /// @notice Revoke access to a dataset
    function revokeAccess(
        uint256 datasetId,
        address grantee
    ) external onlyDatasetOwner(datasetId) {
        delete accessPermissions[datasetId][grantee];
        emit AccessRevoked(datasetId, grantee);
    }
    
    /// @notice Link a dataset to a model for training
    function linkToModel(
        uint256 datasetId,
        uint256 modelId
    ) external onlyDatasetOwner(datasetId) {
        require(datasets[datasetId].isApproved, "Dataset not approved");
        
        modelDatasetApproval[datasetId][modelId] = true;
        
        emit ModelDatasetLinked(datasetId, modelId);
    }
    
    /// @notice Record dataset usage in a training round
    function recordUsage(
        uint256 datasetId,
        uint256 modelId,
        uint256 roundId
    ) external {
        require(modelDatasetApproval[datasetId][modelId], "Dataset not linked to model");
        require(hasTrainingAccess(datasetId, msg.sender), "No training access");
        
        datasets[datasetId].usageCount++;
        
        emit DatasetUsed(datasetId, modelId, roundId);
    }
    
    /// @notice Check if an address has read access
    function hasReadAccess(
        uint256 datasetId,
        address addr
    ) public view returns (bool) {
        AccessPermission storage perm = accessPermissions[datasetId][addr];
        return perm.canRead && block.timestamp < perm.expiresAt;
    }
    
    /// @notice Check if an address has training access
    function hasTrainingAccess(
        uint256 datasetId,
        address addr
    ) public view returns (bool) {
        AccessPermission storage perm = accessPermissions[datasetId][addr];
        return perm.canTrain && block.timestamp < perm.expiresAt && datasets[datasetId].isApproved;
    }
    
    /// @notice Get dataset info
    function getDataset(uint256 datasetId) external view returns (Dataset memory) {
        return datasets[datasetId];
    }
    
    /// @notice Get datasets owned by an address
    function getOwnerDatasets(address _owner) external view returns (uint256[] memory) {
        return ownerDatasets[_owner];
    }
    
    /// @notice Check if dataset can be used with a model
    function canUseWithModel(
        uint256 datasetId,
        uint256 modelId
    ) external view returns (bool) {
        return datasets[datasetId].isApproved && 
               datasets[datasetId].isActive && 
               modelDatasetApproval[datasetId][modelId];
    }
    
    /// @notice Deactivate a dataset
    function deactivateDataset(uint256 datasetId) external onlyDatasetOwner(datasetId) {
        datasets[datasetId].isActive = false;
    }
    
    /// @notice Reactivate a dataset
    function reactivateDataset(uint256 datasetId) external onlyDatasetOwner(datasetId) {
        datasets[datasetId].isActive = true;
    }
    
    /// @notice Set DAO address
    function setDAO(address _dao) external onlyOwner {
        require(_dao != address(0), "Invalid address");
        dao = _dao;
    }
    
    /// @notice Transfer ownership
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid address");
        owner = newOwner;
    }
}
