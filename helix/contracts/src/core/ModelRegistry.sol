// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title ModelRegistry
/// @notice Registry for ML models and their state commitments
/// @dev Tracks model versions, checkpoints, ownership, and architecture metadata
contract ModelRegistry {
    /// @notice Model architecture metadata
    struct ModelArchitecture {
        uint32 dIn;              // Input dimension
        uint32 dHidden;          // Hidden dimension
        uint32 dOut;             // Output dimension
        uint32 numLayers;        // Number of layers
        uint8 activationType;    // 0=ReLU, 1=Sigmoid, 2=Tanh, 3=GeLU, 4=LeakyReLU
    }

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
        uint256 parentVersion;  // index of parent checkpoint (0 for initial)
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

    /// @notice Reverse lookup: commitment => version index in checkpoints array (per model)
    mapping(uint256 => mapping(bytes32 => uint256)) public commitmentToVersion;

    /// @notice Model architecture metadata per model ID
    mapping(uint256 => ModelArchitecture) public modelArchitectures;
    
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
    event ModelVersionCreated(uint256 indexed modelId, uint256 version, uint256 parentVersion, bytes32 commitment);
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
    
    /// @notice Register a new model with architecture metadata
    /// @param name Model name
    /// @param description Model description
    /// @param ipfsHash IPFS hash of model weights
    /// @param initialCommitment Initial state commitment
    /// @param arch Model architecture (dimensions, layers, activation)
    /// @return modelId ID of the registered model
    function registerModel(
        string calldata name,
        string calldata description,
        string calldata ipfsHash,
        bytes32 initialCommitment,
        ModelArchitecture calldata arch
    ) external returns (uint256 modelId) {
        require(arch.dIn > 0 && arch.dHidden > 0 && arch.dOut > 0, "Invalid dimensions");
        require(arch.numLayers > 0, "Must have at least 1 layer");
        require(arch.activationType <= 4, "Invalid activation type");

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

        modelArchitectures[modelId] = arch;

        commitmentToModel[initialCommitment] = modelId;
        ownerModels[msg.sender].push(modelId);

        // Create initial checkpoint
        checkpoints[modelId].push(Checkpoint({
            roundId: 0,
            commitment: initialCommitment,
            ipfsHash: ipfsHash,
            timestamp: block.timestamp,
            errorBound: 0,
            proofHash: bytes32(0),
            parentVersion: 0
        }));
        commitmentToVersion[modelId][initialCommitment] = 0;

        emit ModelRegistered(modelId, msg.sender, name, initialCommitment);
        emit CheckpointCreated(modelId, 0, initialCommitment);
        emit ModelVersionCreated(modelId, 0, 0, initialCommitment);
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

        // Capture parent index before push
        uint256 parentIdx = checkpoints[modelId].length - 1;

        checkpoints[modelId].push(Checkpoint({
            roundId: roundId,
            commitment: newCommitment,
            ipfsHash: checkpointIpfs,
            timestamp: block.timestamp,
            errorBound: errorBound,
            proofHash: proofHash,
            parentVersion: parentIdx
        }));

        uint256 newIdx = checkpoints[modelId].length - 1;
        commitmentToVersion[modelId][newCommitment] = newIdx;

        emit ModelUpdated(modelId, newCommitment, roundId, model.version);
        emit CheckpointCreated(modelId, newIdx, newCommitment);
        emit ModelVersionCreated(modelId, newIdx, parentIdx, newCommitment);
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
    
    /// @notice Get model architecture metadata
    function getModelArchitecture(uint256 modelId) external view returns (ModelArchitecture memory) {
        return modelArchitectures[modelId];
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
    
    /// @notice Get models owned by an address (full list - use paginated version for large lists)
    function getOwnerModels(address _owner) external view returns (uint256[] memory) {
        return ownerModels[_owner];
    }

    /// @notice Get paginated models owned by an address
    /// @param _owner Owner address
    /// @param offset Starting index
    /// @param limit Maximum number of model IDs to return
    /// @return modelIds Array of model IDs
    /// @return total Total number of models owned
    function getOwnerModelsPaginated(address _owner, uint256 offset, uint256 limit) external view returns (
        uint256[] memory modelIds,
        uint256 total
    ) {
        uint256[] storage allModels = ownerModels[_owner];
        total = allModels.length;
        if (offset >= total) {
            return (new uint256[](0), total);
        }

        uint256 end = offset + limit;
        if (end > total) {
            end = total;
        }

        uint256 count = end - offset;
        modelIds = new uint256[](count);
        for (uint256 i = 0; i < count; i++) {
            modelIds[i] = allModels[offset + i];
        }
    }

    /// @notice Get paginated checkpoints for a model
    /// @param modelId Model ID
    /// @param offset Starting index
    /// @param limit Maximum number of checkpoints to return
    /// @return result Array of checkpoints
    /// @return total Total number of checkpoints
    function getCheckpointsPaginated(uint256 modelId, uint256 offset, uint256 limit) external view returns (
        Checkpoint[] memory result,
        uint256 total
    ) {
        Checkpoint[] storage allCheckpoints = checkpoints[modelId];
        total = allCheckpoints.length;
        if (offset >= total) {
            return (new Checkpoint[](0), total);
        }

        uint256 end = offset + limit;
        if (end > total) {
            end = total;
        }

        uint256 count = end - offset;
        result = new Checkpoint[](count);
        for (uint256 i = 0; i < count; i++) {
            result[i] = allCheckpoints[offset + i];
        }
    }
    
    /// @notice Get version chain from a starting version going backward
    /// @param modelId Model ID
    /// @param fromVersion Starting version index
    /// @param count Maximum number of versions to return
    /// @return chain Array of checkpoints in reverse order (newest first)
    function getVersionChain(
        uint256 modelId,
        uint256 fromVersion,
        uint256 count
    ) external view returns (Checkpoint[] memory chain) {
        Checkpoint[] storage all = checkpoints[modelId];
        require(fromVersion < all.length, "Invalid version");

        // First pass: count how many we'll actually return
        uint256 actual = 0;
        uint256 ver = fromVersion;
        while (actual < count) {
            actual++;
            if (ver == 0) break;
            ver = all[ver].parentVersion;
        }

        // Second pass: populate the array
        chain = new Checkpoint[](actual);
        ver = fromVersion;
        for (uint256 i = 0; i < actual; i++) {
            chain[i] = all[ver];
            if (ver == 0) break;
            ver = all[ver].parentVersion;
        }
    }

    /// @notice Get the version index for a given commitment
    /// @param modelId Model ID
    /// @param commitment The commitment hash to look up
    /// @return The version index in the checkpoints array
    function getVersionForCommitment(uint256 modelId, bytes32 commitment) external view returns (uint256) {
        return commitmentToVersion[modelId][commitment];
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
