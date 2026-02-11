// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/governance/utils/IVotes.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/// @title TrainingDAO
/// @notice Decentralized governance over training parameters and model updates
/// @dev Uses ERC20Votes snapshot-based voting to prevent flash loan attacks.
///      Voting power is based on getPastVotes() at the block when the proposal was created.
contract TrainingDAO is ReentrancyGuard {
    /// @notice The HELIX token for voting (must implement IVotes / ERC20Votes)
    IERC20 public immutable helixToken;

    /// @notice The voting token interface (same address as helixToken)
    IVotes public immutable votesToken;

    /// @notice Staking contract for voting weight
    address public stakingContract;

    /// @notice Coordinator contract to apply approved changes
    address public coordinator;

    /// @notice Owner (for emergency actions only)
    address public owner;

    /// @notice Proposal states
    enum ProposalState {
        Pending,      // Created but not started
        Active,       // Voting in progress
        Succeeded,    // Passed quorum and majority
        Defeated,     // Failed to pass
        Queued,       // Scheduled for execution
        Executed,     // Executed
        Cancelled     // Cancelled before execution
    }

    /// @notice Proposal types
    enum ProposalType {
        ParameterChange,      // Change training parameters
        ModelUpdate,          // Update model configuration
        RewardDistribution,   // Change reward distribution
        EmergencyAction,      // Emergency protocol action
        Custom                // Custom governance action
    }

    /// @notice Proposal structure
    struct Proposal {
        uint256 id;
        address proposer;
        ProposalType proposalType;
        string description;
        bytes callData;           // Encoded function call
        address target;           // Contract to call
        uint256 value;            // ETH value (usually 0)
        uint256 startTime;        // When voting starts
        uint256 endTime;          // When voting ends
        uint256 forVotes;         // Votes in favor
        uint256 againstVotes;     // Votes against
        uint256 quorumVotes;      // Required votes for quorum
        uint256 snapshotBlock;    // Block number for vote snapshots (flash loan protection)
        bool executed;
        bool cancelled;
        mapping(address => bool) hasVoted;
    }

    /// @notice Training parameter proposal
    struct ParameterProposal {
        uint256 learningRate;     // Learning rate (multiplied by 1e18)
        uint256 batchSize;
        uint256 maxErrorBound;
        uint256 minParticipants;
        uint256 roundDuration;
    }

    /// @notice Governance configuration
    struct GovConfig {
        uint256 votingPeriod;        // Duration of voting (seconds)
        uint256 votingDelay;         // Delay before voting starts
        uint256 executionDelay;      // Delay before execution (timelock)
        uint256 quorumPercentage;    // Percentage of total supply needed (basis points)
        uint256 proposalThreshold;   // Min tokens to create proposal
    }

    /// @notice Active governance configuration
    GovConfig public govConfig;

    /// @notice Proposal counter
    uint256 public proposalCount;

    /// @notice Mapping of proposal ID to proposal
    mapping(uint256 => Proposal) public proposals;

    /// @notice Mapping of proposal ID to parameter proposal
    mapping(uint256 => ParameterProposal) public parameterProposals;

    /// @notice Current training parameters
    ParameterProposal public currentParameters;

    /// @notice Queued proposals ready for execution (proposalId => execution time)
    mapping(uint256 => uint256) public queuedProposals;

    /// @notice Events
    event ProposalCreated(
        uint256 indexed proposalId,
        address indexed proposer,
        ProposalType proposalType,
        string description
    );
    event VoteCast(
        uint256 indexed proposalId,
        address indexed voter,
        bool support,
        uint256 weight
    );
    event ProposalQueued(uint256 indexed proposalId, uint256 executionTime);
    event ProposalExecuted(uint256 indexed proposalId);
    event ProposalCancelled(uint256 indexed proposalId);
    event ParametersUpdated(ParameterProposal newParameters);
    event GovConfigUpdated(GovConfig newConfig);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    constructor(address _helixToken) {
        require(_helixToken != address(0), "Invalid token");
        helixToken = IERC20(_helixToken);
        votesToken = IVotes(_helixToken);
        owner = msg.sender;

        // Default governance config
        govConfig = GovConfig({
            votingPeriod: 3 days,
            votingDelay: 1 days,
            executionDelay: 2 days,
            quorumPercentage: 400,  // 4%
            proposalThreshold: 1000 * 1e18  // 1000 HELIX
        });

        // Default training parameters
        currentParameters = ParameterProposal({
            learningRate: 1e15,      // 0.001
            batchSize: 32,
            maxErrorBound: 1000,     // 0.1%
            minParticipants: 3,
            roundDuration: 1 hours
        });
    }

    /// @notice Create a new proposal
    /// @param proposalType Type of proposal
    /// @param description Description of the proposal
    /// @param target Contract to call (address(0) for internal)
    /// @param callData Encoded function call
    /// @return proposalId ID of the created proposal
    function createProposal(
        ProposalType proposalType,
        string memory description,
        address target,
        bytes memory callData
    ) public returns (uint256 proposalId) {
        require(
            votesToken.getPastVotes(msg.sender, block.number - 1) >= govConfig.proposalThreshold,
            "Below proposal threshold"
        );

        proposalId = ++proposalCount;

        Proposal storage proposal = proposals[proposalId];
        proposal.id = proposalId;
        proposal.proposer = msg.sender;
        proposal.proposalType = proposalType;
        proposal.description = description;
        proposal.callData = callData;
        proposal.target = target;
        proposal.startTime = block.timestamp + govConfig.votingDelay;
        proposal.endTime = proposal.startTime + govConfig.votingPeriod;
        proposal.quorumVotes = (helixToken.totalSupply() * govConfig.quorumPercentage) / 10000;
        // Record snapshot block for flash-loan-resistant voting power lookups
        proposal.snapshotBlock = block.number;

        emit ProposalCreated(proposalId, msg.sender, proposalType, description);
    }

    /// @notice Create a parameter change proposal
    /// @dev Uses internal call to _createProposal to avoid unnecessary external call overhead
    /// @param description Description of changes
    /// @param params New training parameters
    /// @return proposalId ID of the created proposal
    function createParameterProposal(
        string calldata description,
        ParameterProposal calldata params
    ) external returns (uint256 proposalId) {
        // Pre-compute proposalId so calldata is correct from creation (no placeholder needed)
        proposalId = proposalCount + 1;

        createProposal(
            ProposalType.ParameterChange,
            description,
            address(this),
            abi.encodeWithSignature("applyParameters(uint256)", proposalId)
        );

        parameterProposals[proposalId] = params;
    }

    /// @notice Cast a vote on a proposal
    /// @dev Voting power is based on snapshot at proposal creation block (flash loan resistant)
    /// @param proposalId ID of the proposal
    /// @param support Whether to support the proposal
    function castVote(uint256 proposalId, bool support) external {
        Proposal storage proposal = proposals[proposalId];

        require(getProposalState(proposalId) == ProposalState.Active, "Voting not active");
        require(!proposal.hasVoted[msg.sender], "Already voted");

        // Use snapshot-based voting power to prevent flash loan attacks
        uint256 votes = getVotingPower(msg.sender, proposalId);
        require(votes > 0, "No voting power");

        proposal.hasVoted[msg.sender] = true;

        if (support) {
            proposal.forVotes += votes;
        } else {
            proposal.againstVotes += votes;
        }

        emit VoteCast(proposalId, msg.sender, support, votes);
    }

    /// @notice Queue a successful proposal for execution
    /// @param proposalId ID of the proposal
    function queueProposal(uint256 proposalId) external {
        require(getProposalState(proposalId) == ProposalState.Succeeded, "Proposal not succeeded");

        uint256 executionTime = block.timestamp + govConfig.executionDelay;
        queuedProposals[proposalId] = executionTime;

        emit ProposalQueued(proposalId, executionTime);
    }

    /// @notice Execute a queued proposal
    /// @param proposalId ID of the proposal
    function executeProposal(uint256 proposalId) external nonReentrant {
        require(getProposalState(proposalId) == ProposalState.Queued, "Not queued");
        require(block.timestamp >= queuedProposals[proposalId], "Timelock not expired");

        Proposal storage proposal = proposals[proposalId];
        proposal.executed = true;

        // Execute the proposal
        if (proposal.target != address(0)) {
            (bool success, ) = proposal.target.call{value: proposal.value}(proposal.callData);
            require(success, "Execution failed");
        }

        emit ProposalExecuted(proposalId);
    }

    /// @notice Cancel a proposal (only proposer or if under threshold)
    /// @param proposalId ID of the proposal
    function cancelProposal(uint256 proposalId) external {
        Proposal storage proposal = proposals[proposalId];

        require(!proposal.executed, "Already executed");
        require(!proposal.cancelled, "Already cancelled");
        require(
            msg.sender == proposal.proposer ||
            votesToken.getVotes(proposal.proposer) < govConfig.proposalThreshold,
            "Cannot cancel"
        );

        proposal.cancelled = true;

        emit ProposalCancelled(proposalId);
    }

    /// @notice Apply training parameters (called via proposal execution)
    /// @param proposalId ID of the parameter proposal
    function applyParameters(uint256 proposalId) external {
        require(msg.sender == address(this), "Only via proposal");

        ParameterProposal storage params = parameterProposals[proposalId];
        currentParameters = params;

        emit ParametersUpdated(params);
    }

    /// @notice Get proposal state
    /// @param proposalId ID of the proposal
    /// @return Current state of the proposal
    function getProposalState(uint256 proposalId) public view returns (ProposalState) {
        Proposal storage proposal = proposals[proposalId];

        if (proposal.cancelled) {
            return ProposalState.Cancelled;
        }

        if (proposal.executed) {
            return ProposalState.Executed;
        }

        if (queuedProposals[proposalId] > 0) {
            return ProposalState.Queued;
        }

        if (block.timestamp < proposal.startTime) {
            return ProposalState.Pending;
        }

        if (block.timestamp <= proposal.endTime) {
            return ProposalState.Active;
        }

        // Voting ended
        if (proposal.forVotes > proposal.againstVotes &&
            proposal.forVotes + proposal.againstVotes >= proposal.quorumVotes) {
            return ProposalState.Succeeded;
        }

        return ProposalState.Defeated;
    }

    /// @notice Get voting power for an address at a proposal's snapshot block
    /// @dev Uses ERC20Votes getPastVotes for flash-loan-resistant voting
    /// @param account Address to check
    /// @param proposalId Proposal to get voting power for
    /// @return Voting power at the proposal's snapshot block
    function getVotingPower(address account, uint256 proposalId) public view returns (uint256) {
        uint256 snapshotBlock = proposals[proposalId].snapshotBlock;
        require(snapshotBlock > 0, "Invalid proposal");
        return votesToken.getPastVotes(account, snapshotBlock);
    }

    /// @notice Get proposal info
    function getProposalInfo(uint256 proposalId) external view returns (
        address proposer,
        ProposalType proposalType,
        string memory description,
        uint256 startTime,
        uint256 endTime,
        uint256 forVotes,
        uint256 againstVotes,
        ProposalState state
    ) {
        Proposal storage proposal = proposals[proposalId];
        return (
            proposal.proposer,
            proposal.proposalType,
            proposal.description,
            proposal.startTime,
            proposal.endTime,
            proposal.forVotes,
            proposal.againstVotes,
            getProposalState(proposalId)
        );
    }

    /// @notice Get current training parameters
    function getTrainingParameters() external view returns (
        uint256 learningRate,
        uint256 batchSize,
        uint256 maxErrorBound,
        uint256 minParticipants,
        uint256 roundDuration
    ) {
        ParameterProposal storage p = currentParameters;
        return (p.learningRate, p.batchSize, p.maxErrorBound, p.minParticipants, p.roundDuration);
    }

    /// @notice Check if an address has voted on a proposal
    function hasVoted(uint256 proposalId, address account) external view returns (bool) {
        return proposals[proposalId].hasVoted[account];
    }

    /// @notice Get the snapshot block for a proposal
    function getProposalSnapshot(uint256 proposalId) external view returns (uint256) {
        return proposals[proposalId].snapshotBlock;
    }

    /// @notice Update governance configuration (only via proposal or owner initially)
    function updateGovConfig(GovConfig calldata newConfig) external {
        require(msg.sender == address(this) || msg.sender == owner, "Unauthorized");
        govConfig = newConfig;
        emit GovConfigUpdated(newConfig);
    }

    /// @notice Set coordinator contract
    function setCoordinator(address _coordinator) external onlyOwner {
        require(_coordinator != address(0), "Invalid address");
        coordinator = _coordinator;
    }

    /// @notice Set staking contract
    function setStakingContract(address _staking) external onlyOwner {
        require(_staking != address(0), "Invalid address");
        stakingContract = _staking;
    }

    /// @notice Transfer ownership
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid address");
        owner = newOwner;
    }
}
