// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// @title ModelRegistry
/// @notice Registry for ML models and their state commitments
/// @dev Tracks model versions, checkpoints, and ownership
contract ModelRegistry {
    /// @notice Model information
    struct Model {
        string name;
        string description;
        string ipfsHash;           // IPFS hash of model architecture
        bytes32 currentCommitment; // Current model state commitment
        address owner;
        uint256 createdAt;
        uint256 lastUpdated;
        uint256 version;
        uint256 totalRounds;
        bool isActive;
    }
    
    /// @notice Model checkpoint information
    struct Checkpoint {
        uint256 roundId;
        bytes32 commitment;
        string ipfsHash;
        uint256 timestamp;
        uint256 errorBound;
        bytes32 proofHash;
    }
    
    /// @notice Model counter
    uint256 public nextModelId;
    
    /// @notice Mapping of model ID to model info
    mapping(uint256 => Model) public models;
    
    /// @notice Mapping of model ID to checkpoints
    mapping(uint256 => Checkpoint[]) public checkpoints;
    
    /// @notice Mapping of commitment hash to model ID
    mapping(bytes32 => uint256) public commitmentToModel;
    
    /// @notice Mapping of owner to their model IDs
    mapping(address => uint256[]) public ownerModels;
    
    /// @notice Owner for admin functions
    address public owner;
    
    /// @notice Coordinator contract (can update models)
    address public coordinator;
    
    /// @notice Events
    event ModelRegistered(
        uint256 indexed modelId,
        address indexed owner,
        string name,
        bytes32 initialCommitment
    );
    event ModelUpdated(
        uint256 indexed modelId,
        bytes32 newCommitment,
        uint256 round,
        uint256 version
    );
    event CheckpointCreated(
        uint256 indexed modelId,
        uint256 indexed checkpointIndex,
        bytes32 commitment
    );
    event ModelDeactivated(uint256 indexed modelId);
    event ModelReactivated(uint256 indexed modelId);
    event OwnershipTransferred(uint256 indexed modelId, address indexed oldOwner, address indexed newOwner);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }
    
    modifier onlyModelOwner(uint256 modelId) {
        require(models[modelId].owner == msg.sender, "Not model owner");
        _;
    }
    
    modifier onlyCoordinator() {
        require(msg.sender == coordinator || msg.sender == owner, "Only coordinator");
        _;
    }
    
    constructor() {
        owner = msg.sender;
    }
    
    /// @notice Register a new model
    /// @param name Model name
    /// @param description Model description
    /// @param ipfsHash IPFS hash of model architecture
    /// @param initialCommitment Initial state commitment
    /// @return modelId ID of the registered model
    function registerModel(
        string calldata name,
        string calldata description,
        string calldata ipfsHash,
        bytes32 initialCommitment
    ) external returns (uint256 modelId) {
        modelId = nextModelId++;
        
        models[modelId] = Model({
            name: name,
            description: description,
            ipfsHash: ipfsHash,
            currentCommitment: initialCommitment,
            owner: msg.sender,
            createdAt: block.timestamp,
            lastUpdated: block.timestamp,
            version: 1,
            totalRounds: 0,
            isActive: true
        });
        
        commitmentToModel[initialCommitment] = modelId;
        ownerModels[msg.sender].push(modelId);
        
        // Create initial checkpoint
        checkpoints[modelId].push(Checkpoint({
            roundId: 0,
            commitment: initialCommitment,
            ipfsHash: ipfsHash,
            timestamp: block.timestamp,
            errorBound: 0,
            proofHash: bytes32(0)
        }));
        
        emit ModelRegistered(modelId, msg.sender, name, initialCommitment);
        emit CheckpointCreated(modelId, 0, initialCommitment);
    }
    
    /// @notice Update model state after a training round
    /// @param modelId Model ID
    /// @param newCommitment New state commitment
    /// @param roundId Training round ID
    /// @param ipfsHash Optional new IPFS hash for checkpoint
    /// @param errorBound Error bound of the new state
    /// @param proofHash Hash of the proof that validated this update
    function updateModel(
        uint256 modelId,
        bytes32 newCommitment,
        uint256 roundId,
        string calldata ipfsHash,
        uint256 errorBound,
        bytes32 proofHash
    ) external onlyCoordinator {
        Model storage model = models[modelId];
        require(model.isActive, "Model not active");
        
        // Update commitment mapping
        delete commitmentToModel[model.currentCommitment];
        commitmentToModel[newCommitment] = modelId;
        
        // Update model
        model.currentCommitment = newCommitment;
        model.lastUpdated = block.timestamp;
        model.version++;
        model.totalRounds++;
        
        // Create checkpoint
        string memory checkpointIpfs;
        if (bytes(ipfsHash).length > 0) {
            checkpointIpfs = ipfsHash;
        } else {
            checkpointIpfs = model.ipfsHash;
        }
        checkpoints[modelId].push(Checkpoint({
            roundId: roundId,
            commitment: newCommitment,
            ipfsHash: checkpointIpfs,
            timestamp: block.timestamp,
            errorBound: errorBound,
            proofHash: proofHash
        }));
        
        emit ModelUpdated(modelId, newCommitment, roundId, model.version);
        emit CheckpointCreated(modelId, checkpoints[modelId].length - 1, newCommitment);
    }
    
    /// @notice Deactivate a model
    function deactivateModel(uint256 modelId) external onlyModelOwner(modelId) {
        models[modelId].isActive = false;
        emit ModelDeactivated(modelId);
    }
    
    /// @notice Reactivate a model
    function reactivateModel(uint256 modelId) external onlyModelOwner(modelId) {
        models[modelId].isActive = true;
        emit ModelReactivated(modelId);
    }
    
    /// @notice Transfer model ownership
    function transferModelOwnership(
        uint256 modelId,
        address newOwner
    ) external onlyModelOwner(modelId) {
        require(newOwner != address(0), "Invalid address");
        
        address oldOwner = models[modelId].owner;
        models[modelId].owner = newOwner;
        
        // Update owner mappings
        _removeFromOwnerModels(oldOwner, modelId);
        ownerModels[newOwner].push(modelId);
        
        emit OwnershipTransferred(modelId, oldOwner, newOwner);
    }
    
    /// @notice Remove model ID from owner's list
    function _removeFromOwnerModels(address _owner, uint256 modelId) internal {
        uint256[] storage modelIds = ownerModels[_owner];
        for (uint i = 0; i < modelIds.length; i++) {
            if (modelIds[i] == modelId) {
                modelIds[i] = modelIds[modelIds.length - 1];
                modelIds.pop();
                break;
            }
        }
    }
    
    /// @notice Get model info
    function getModel(uint256 modelId) external view returns (
        string memory name,
        string memory description,
        bytes32 currentCommitment,
        address modelOwner,
        uint256 version,
        bool isActive
    ) {
        Model storage model = models[modelId];
        return (
            model.name,
            model.description,
            model.currentCommitment,
            model.owner,
            model.version,
            model.isActive
        );
    }
    
    /// @notice Get checkpoint count for a model
    function getCheckpointCount(uint256 modelId) external view returns (uint256) {
        return checkpoints[modelId].length;
    }
    
    /// @notice Get a specific checkpoint
    function getCheckpoint(
        uint256 modelId,
        uint256 index
    ) external view returns (Checkpoint memory) {
        require(index < checkpoints[modelId].length, "Invalid index");
        return checkpoints[modelId][index];
    }
    
    /// @notice Get models owned by an address
    function getOwnerModels(address _owner) external view returns (uint256[] memory) {
        return ownerModels[_owner];
    }
    
    /// @notice Set coordinator address
    function setCoordinator(address _coordinator) external onlyOwner {
        require(_coordinator != address(0), "Invalid address");
        coordinator = _coordinator;
    }
    
    /// @notice Transfer contract ownership
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid address");
        owner = newOwner;
    }
}
