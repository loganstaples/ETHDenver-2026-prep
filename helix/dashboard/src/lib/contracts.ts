// Contract addresses for different networks
export const CONTRACT_ADDRESSES = {
    // Mainnet
    1: {
        helixCoordinator: '0x0000000000000000000000000000000000000000', // TODO: Deploy
        helixVerifier: '0x0000000000000000000000000000000000000000',
        helixToken: '0x0000000000000000000000000000000000000000',
        helixModelStore: '0x0000000000000000000000000000000000000000',
    },
    // Sepolia
    11155111: {
        helixCoordinator: '0x0000000000000000000000000000000000000000', // TODO: Deploy
        helixVerifier: '0x0000000000000000000000000000000000000000',
        helixToken: '0x0000000000000000000000000000000000000000',
        helixModelStore: '0x0000000000000000000000000000000000000000',
    },
    // Hardhat/Localhost
    31337: {
        helixCoordinator: '0x5FbDB2315678afecb367f032d93F642f64180aa3',
        helixVerifier: '0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512',
        helixToken: '0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0',
        helixModelStore: '0x0000000000000000000000000000000000000000', // Set after deploy
    },
    // ADI Network Testnet
    99999: {
        helixCoordinator: '0x0000000000000000000000000000000000000000', // Set after deploy
        helixVerifier: '0x0000000000000000000000000000000000000000',
        helixToken: '0x0000000000000000000000000000000000000000',
        helixModelStore: '0x0000000000000000000000000000000000000000', // Set after deploy
    },
} as const;

// HelixCoordinatorV2 ABI - Core coordination contract
export const HELIX_COORDINATOR_ABI = [
    // Events
    {
        type: 'event',
        name: 'ModelRegistered',
        inputs: [
            { indexed: true, name: 'modelId', type: 'uint256' },
            { indexed: true, name: 'owner', type: 'address' },
            { indexed: false, name: 'initialCommitment', type: 'uint256' },
            { indexed: false, name: 'ipfsHash', type: 'string' },
        ],
    },
    {
        type: 'event',
        name: 'RoundStarted',
        inputs: [
            { indexed: true, name: 'modelId', type: 'uint256' },
            { indexed: true, name: 'roundId', type: 'uint256' },
            { indexed: false, name: 'deadline', type: 'uint256' },
        ],
    },
    {
        type: 'event',
        name: 'ProofSubmitted',
        inputs: [
            { indexed: true, name: 'modelId', type: 'uint256' },
            { indexed: true, name: 'roundId', type: 'uint256' },
            { indexed: true, name: 'prover', type: 'address' },
            { indexed: false, name: 'newCommitment', type: 'uint256' },
        ],
    },
    {
        type: 'event',
        name: 'RoundCompleted',
        inputs: [
            { indexed: true, name: 'modelId', type: 'uint256' },
            { indexed: true, name: 'roundId', type: 'uint256' },
            { indexed: false, name: 'newCommitment', type: 'uint256' },
        ],
    },
    {
        type: 'event',
        name: 'Staked',
        inputs: [
            { indexed: true, name: 'prover', type: 'address' },
            { indexed: true, name: 'modelId', type: 'uint256' },
            { indexed: false, name: 'amount', type: 'uint256' },
        ],
    },
    {
        type: 'event',
        name: 'Unstaked',
        inputs: [
            { indexed: true, name: 'prover', type: 'address' },
            { indexed: true, name: 'modelId', type: 'uint256' },
            { indexed: false, name: 'amount', type: 'uint256' },
        ],
    },
    {
        type: 'event',
        name: 'Slashed',
        inputs: [
            { indexed: true, name: 'prover', type: 'address' },
            { indexed: true, name: 'modelId', type: 'uint256' },
            { indexed: false, name: 'amount', type: 'uint256' },
            { indexed: false, name: 'reason', type: 'string' },
        ],
    },
    {
        type: 'event',
        name: 'InvalidProofDetected',
        inputs: [
            { indexed: true, name: 'modelId', type: 'uint256' },
            { indexed: true, name: 'roundId', type: 'uint256' },
            { indexed: true, name: 'prover', type: 'address' },
            { indexed: false, name: 'proofHash', type: 'bytes32' },
        ],
    },
    // Read functions
    {
        type: 'function',
        name: 'models',
        inputs: [{ name: 'modelId', type: 'uint256' }],
        outputs: [
            { name: 'ipfsHash', type: 'string' },
            { name: 'currentCommitment', type: 'uint256' },
            { name: 'currentRound', type: 'uint256' },
            { name: 'owner', type: 'address' },
            { name: 'minStake', type: 'uint256' },
            { name: 'active', type: 'bool' },
        ],
        stateMutability: 'view',
    },
    {
        type: 'function',
        name: 'rounds',
        inputs: [
            { name: 'modelId', type: 'uint256' },
            { name: 'roundId', type: 'uint256' },
        ],
        outputs: [
            { name: 'modelCommitment', type: 'uint256' },
            { name: 'newCommitment', type: 'uint256' },
            { name: 'isCompleted', type: 'bool' },
            { name: 'deadline', type: 'uint256' },
            { name: 'prover', type: 'address' },
        ],
        stateMutability: 'view',
    },
    {
        type: 'function',
        name: 'stakes',
        inputs: [
            { name: 'prover', type: 'address' },
            { name: 'modelId', type: 'uint256' },
        ],
        outputs: [
            { name: 'amount', type: 'uint256' },
            { name: 'lockedUntil', type: 'uint256' },
            { name: 'slashed', type: 'bool' },
        ],
        stateMutability: 'view',
    },
    {
        type: 'function',
        name: 'getModelState',
        inputs: [{ name: 'modelId', type: 'uint256' }],
        outputs: [
            { name: 'currentRound', type: 'uint256' },
            { name: 'currentCommitment', type: 'uint256' },
            { name: 'active', type: 'bool' },
        ],
        stateMutability: 'view',
    },
    {
        type: 'function',
        name: 'getStake',
        inputs: [
            { name: 'prover', type: 'address' },
            { name: 'modelId', type: 'uint256' },
        ],
        outputs: [
            { name: 'amount', type: 'uint256' },
            { name: 'lockedUntil', type: 'uint256' },
            { name: 'slashed', type: 'bool' },
        ],
        stateMutability: 'view',
    },
    {
        type: 'function',
        name: 'getSlashingRecordCount',
        inputs: [],
        outputs: [{ name: '', type: 'uint256' }],
        stateMutability: 'view',
    },
    {
        type: 'function',
        name: 'getAccumulatedErrorBound',
        inputs: [{ name: 'modelId', type: 'uint256' }],
        outputs: [{ name: '', type: 'uint256' }],
        stateMutability: 'view',
    },
    {
        type: 'function',
        name: 'isModelErrorAcceptable',
        inputs: [
            { name: 'modelId', type: 'uint256' },
            { name: 'maxAccumulated', type: 'uint256' },
        ],
        outputs: [{ name: '', type: 'bool' }],
        stateMutability: 'view',
    },
    {
        type: 'function',
        name: 'nextModelId',
        inputs: [],
        outputs: [{ name: '', type: 'uint256' }],
        stateMutability: 'view',
    },
    {
        type: 'function',
        name: 'defaultMinStake',
        inputs: [],
        outputs: [{ name: '', type: 'uint256' }],
        stateMutability: 'view',
    },
    {
        type: 'function',
        name: 'stakeLockPeriod',
        inputs: [],
        outputs: [{ name: '', type: 'uint256' }],
        stateMutability: 'view',
    },
    {
        type: 'function',
        name: 'slashPercentage',
        inputs: [],
        outputs: [{ name: '', type: 'uint256' }],
        stateMutability: 'view',
    },
    {
        type: 'function',
        name: 'maxErrorBound',
        inputs: [],
        outputs: [{ name: '', type: 'uint256' }],
        stateMutability: 'view',
    },
    {
        type: 'function',
        name: 'slashingRecords',
        inputs: [{ name: 'index', type: 'uint256' }],
        outputs: [
            { name: 'prover', type: 'address' },
            { name: 'modelId', type: 'uint256' },
            { name: 'roundId', type: 'uint256' },
            { name: 'amount', type: 'uint256' },
            { name: 'reason', type: 'string' },
            { name: 'timestamp', type: 'uint256' },
        ],
        stateMutability: 'view',
    },
    // Write functions
    {
        type: 'function',
        name: 'registerModel',
        inputs: [
            { name: 'ipfsHash', type: 'string' },
            { name: 'initialCommitment', type: 'uint256' },
            { name: 'minStake', type: 'uint256' },
        ],
        outputs: [{ name: 'modelId', type: 'uint256' }],
        stateMutability: 'nonpayable',
    },
    {
        type: 'function',
        name: 'startRound',
        inputs: [
            { name: 'modelId', type: 'uint256' },
            { name: 'duration', type: 'uint256' },
        ],
        outputs: [],
        stateMutability: 'nonpayable',
    },
    {
        type: 'function',
        name: 'stake',
        inputs: [{ name: 'modelId', type: 'uint256' }],
        outputs: [],
        stateMutability: 'payable',
    },
    {
        type: 'function',
        name: 'unstake',
        inputs: [{ name: 'modelId', type: 'uint256' }],
        outputs: [],
        stateMutability: 'nonpayable',
    },
    {
        type: 'function',
        name: 'submitProof',
        inputs: [
            { name: 'modelId', type: 'uint256' },
            { name: 'roundId', type: 'uint256' },
            { name: 'proof', type: 'bytes' },
            { name: 'publicInputs', type: 'uint256[]' },
        ],
        outputs: [],
        stateMutability: 'nonpayable',
    },
    {
        type: 'function',
        name: 'challengeProof',
        inputs: [
            { name: 'modelId', type: 'uint256' },
            { name: 'roundId', type: 'uint256' },
            { name: 'proof', type: 'bytes' },
            { name: 'publicInputs', type: 'uint256[]' },
        ],
        outputs: [],
        stateMutability: 'nonpayable',
    },
] as const;

// HELIX Token ABI (ERC20)
export const HELIX_TOKEN_ABI = [
    {
        type: 'function',
        name: 'balanceOf',
        inputs: [{ name: 'account', type: 'address' }],
        outputs: [{ name: '', type: 'uint256' }],
        stateMutability: 'view',
    },
    {
        type: 'function',
        name: 'approve',
        inputs: [
            { name: 'spender', type: 'address' },
            { name: 'amount', type: 'uint256' },
        ],
        outputs: [{ name: '', type: 'bool' }],
        stateMutability: 'nonpayable',
    },
    {
        type: 'function',
        name: 'transfer',
        inputs: [
            { name: 'to', type: 'address' },
            { name: 'amount', type: 'uint256' },
        ],
        outputs: [{ name: '', type: 'bool' }],
        stateMutability: 'nonpayable',
    },
    {
        type: 'function',
        name: 'allowance',
        inputs: [
            { name: 'owner', type: 'address' },
            { name: 'spender', type: 'address' },
        ],
        outputs: [{ name: '', type: 'uint256' }],
        stateMutability: 'view',
    },
    {
        type: 'event',
        name: 'Transfer',
        inputs: [
            { indexed: true, name: 'from', type: 'address' },
            { indexed: true, name: 'to', type: 'address' },
            { indexed: false, name: 'value', type: 'uint256' },
        ],
    },
    {
        type: 'event',
        name: 'Approval',
        inputs: [
            { indexed: true, name: 'owner', type: 'address' },
            { indexed: true, name: 'spender', type: 'address' },
            { indexed: false, name: 'value', type: 'uint256' },
        ],
    },
] as const;

// HelixModelStore ABI — ERC-721 on-chain model registry
export const HELIX_MODEL_STORE_ABI = [
    // ── Write functions ──────────────────────────────────────────────
    {
        type: 'function',
        name: 'createModel',
        inputs: [
            { name: 'slug', type: 'string' },
            { name: 'name', type: 'string' },
            { name: 'description', type: 'string' },
        ],
        outputs: [{ name: 'tokenId', type: 'uint256' }],
        stateMutability: 'nonpayable',
    },
    {
        type: 'function',
        name: 'addVersion',
        inputs: [
            { name: 'tokenId', type: 'uint256' },
            { name: 'semver', type: 'string' },
            { name: 'rootHash', type: 'string' },
            { name: 'accuracy', type: 'uint96' },
            { name: 'sessionId', type: 'string' },
            { name: 'weightsStored', type: 'bool' },
        ],
        outputs: [{ name: 'versionIndex', type: 'uint256' }],
        stateMutability: 'nonpayable',
    },
    {
        type: 'function',
        name: 'setPublic',
        inputs: [
            { name: 'tokenId', type: 'uint256' },
            { name: 'isPublic', type: 'bool' },
        ],
        outputs: [],
        stateMutability: 'nonpayable',
    },
    {
        type: 'function',
        name: 'setInferenceFee',
        inputs: [
            { name: 'tokenId', type: 'uint256' },
            { name: 'feeBps', type: 'uint16' },
        ],
        outputs: [],
        stateMutability: 'nonpayable',
    },
    {
        type: 'function',
        name: 'grantAccess',
        inputs: [
            { name: 'tokenId', type: 'uint256' },
            { name: 'account', type: 'address' },
        ],
        outputs: [],
        stateMutability: 'nonpayable',
    },
    {
        type: 'function',
        name: 'revokeAccess',
        inputs: [
            { name: 'tokenId', type: 'uint256' },
            { name: 'account', type: 'address' },
        ],
        outputs: [],
        stateMutability: 'nonpayable',
    },
    // ── Read functions ───────────────────────────────────────────────
    {
        type: 'function',
        name: 'models',
        inputs: [{ name: 'tokenId', type: 'uint256' }],
        outputs: [
            { name: 'slug', type: 'string' },
            { name: 'name', type: 'string' },
            { name: 'description', type: 'string' },
            { name: 'creator', type: 'address' },
            { name: 'createdAt', type: 'uint40' },
            { name: 'isPublic', type: 'bool' },
            { name: 'inferenceFee', type: 'uint16' },
        ],
        stateMutability: 'view',
    },
    {
        type: 'function',
        name: 'getVersions',
        inputs: [{ name: 'tokenId', type: 'uint256' }],
        outputs: [
            {
                name: '',
                type: 'tuple[]',
                components: [
                    { name: 'semver', type: 'string' },
                    { name: 'rootHash', type: 'string' },
                    { name: 'accuracy', type: 'uint96' },
                    { name: 'timestamp', type: 'uint40' },
                    { name: 'sessionId', type: 'string' },
                    { name: 'weightsStored', type: 'bool' },
                ],
            },
        ],
        stateMutability: 'view',
    },
    {
        type: 'function',
        name: 'getVersionCount',
        inputs: [{ name: 'tokenId', type: 'uint256' }],
        outputs: [{ name: '', type: 'uint256' }],
        stateMutability: 'view',
    },
    {
        type: 'function',
        name: 'getVersion',
        inputs: [
            { name: 'tokenId', type: 'uint256' },
            { name: 'versionIndex', type: 'uint256' },
        ],
        outputs: [
            {
                name: '',
                type: 'tuple',
                components: [
                    { name: 'semver', type: 'string' },
                    { name: 'rootHash', type: 'string' },
                    { name: 'accuracy', type: 'uint96' },
                    { name: 'timestamp', type: 'uint40' },
                    { name: 'sessionId', type: 'string' },
                    { name: 'weightsStored', type: 'bool' },
                ],
            },
        ],
        stateMutability: 'view',
    },
    {
        type: 'function',
        name: 'getModelBySlug',
        inputs: [{ name: 'slug', type: 'string' }],
        outputs: [{ name: 'tokenId', type: 'uint256' }],
        stateMutability: 'view',
    },
    {
        type: 'function',
        name: 'hasModelAccess',
        inputs: [
            { name: 'tokenId', type: 'uint256' },
            { name: 'account', type: 'address' },
        ],
        outputs: [{ name: '', type: 'bool' }],
        stateMutability: 'view',
    },
    {
        type: 'function',
        name: 'accessGranted',
        inputs: [
            { name: 'tokenId', type: 'uint256' },
            { name: 'account', type: 'address' },
        ],
        outputs: [{ name: '', type: 'bool' }],
        stateMutability: 'view',
    },
    // ── ERC721Enumerable ─────────────────────────────────────────────
    {
        type: 'function',
        name: 'balanceOf',
        inputs: [{ name: 'owner', type: 'address' }],
        outputs: [{ name: '', type: 'uint256' }],
        stateMutability: 'view',
    },
    {
        type: 'function',
        name: 'tokenOfOwnerByIndex',
        inputs: [
            { name: 'owner', type: 'address' },
            { name: 'index', type: 'uint256' },
        ],
        outputs: [{ name: 'tokenId', type: 'uint256' }],
        stateMutability: 'view',
    },
    {
        type: 'function',
        name: 'ownerOf',
        inputs: [{ name: 'tokenId', type: 'uint256' }],
        outputs: [{ name: '', type: 'address' }],
        stateMutability: 'view',
    },
    {
        type: 'function',
        name: 'totalSupply',
        inputs: [],
        outputs: [{ name: '', type: 'uint256' }],
        stateMutability: 'view',
    },
    {
        type: 'function',
        name: 'name',
        inputs: [],
        outputs: [{ name: '', type: 'string' }],
        stateMutability: 'view',
    },
    {
        type: 'function',
        name: 'symbol',
        inputs: [],
        outputs: [{ name: '', type: 'string' }],
        stateMutability: 'view',
    },
    // ── Events ───────────────────────────────────────────────────────
    {
        type: 'event',
        name: 'ModelCreated',
        inputs: [
            { name: 'tokenId', type: 'uint256', indexed: true },
            { name: 'creator', type: 'address', indexed: true },
            { name: 'slug', type: 'string', indexed: false },
            { name: 'name', type: 'string', indexed: false },
        ],
    },
    {
        type: 'event',
        name: 'VersionAdded',
        inputs: [
            { name: 'tokenId', type: 'uint256', indexed: true },
            { name: 'versionIndex', type: 'uint256', indexed: true },
            { name: 'semver', type: 'string', indexed: false },
            { name: 'rootHash', type: 'string', indexed: false },
        ],
    },
    {
        type: 'event',
        name: 'ModelPublicityChanged',
        inputs: [
            { name: 'tokenId', type: 'uint256', indexed: true },
            { name: 'isPublic', type: 'bool', indexed: false },
        ],
    },
    {
        type: 'event',
        name: 'InferenceFeeChanged',
        inputs: [
            { name: 'tokenId', type: 'uint256', indexed: true },
            { name: 'feeBps', type: 'uint16', indexed: false },
        ],
    },
    {
        type: 'event',
        name: 'AccessChanged',
        inputs: [
            { name: 'tokenId', type: 'uint256', indexed: true },
            { name: 'account', type: 'address', indexed: true },
            { name: 'granted', type: 'bool', indexed: false },
        ],
    },
    {
        type: 'event',
        name: 'Transfer',
        inputs: [
            { name: 'from', type: 'address', indexed: true },
            { name: 'to', type: 'address', indexed: true },
            { name: 'tokenId', type: 'uint256', indexed: true },
        ],
    },
    {
        type: 'event',
        name: 'Approval',
        inputs: [
            { name: 'owner', type: 'address', indexed: true },
            { name: 'approved', type: 'address', indexed: true },
            { name: 'tokenId', type: 'uint256', indexed: true },
        ],
    },
] as const;

export const ABIS = {
    helixCoordinator: HELIX_COORDINATOR_ABI,
    helixToken: HELIX_TOKEN_ABI,
    helixModelStore: HELIX_MODEL_STORE_ABI,
};

// Type definitions
export interface Model {
    ipfsHash: string;
    currentCommitment: bigint;
    currentRound: bigint;
    owner: string;
    minStake: bigint;
    active: boolean;
}

export interface Round {
    modelCommitment: bigint;
    newCommitment: bigint;
    isCompleted: boolean;
    deadline: bigint;
    prover: string;
}

export interface Stake {
    amount: bigint;
    lockedUntil: bigint;
    slashed: boolean;
}

export interface SlashingRecord {
    prover: string;
    modelId: bigint;
    roundId: bigint;
    amount: bigint;
    reason: string;
    timestamp: bigint;
}

export interface ProofSubmittedEvent {
    modelId: bigint;
    roundId: bigint;
    prover: string;
    newCommitment: bigint;
    blockNumber: number;
    transactionHash: string;
    timestamp: number;
}

export interface RoundStartedEvent {
    modelId: bigint;
    roundId: bigint;
    deadline: bigint;
    blockNumber: number;
    transactionHash: string;
    timestamp: number;
}

export interface RoundCompletedEvent {
    modelId: bigint;
    roundId: bigint;
    newCommitment: bigint;
    blockNumber: number;
    transactionHash: string;
    timestamp: number;
}

export interface StakedEvent {
    prover: string;
    modelId: bigint;
    amount: bigint;
    blockNumber: number;
    transactionHash: string;
    timestamp: number;
}

export interface SlashedEvent {
    prover: string;
    modelId: bigint;
    amount: bigint;
    reason: string;
    blockNumber: number;
    transactionHash: string;
    timestamp: number;
}

export interface OnChainModel {
    tokenId: number;
    slug: string;
    name: string;
    description: string;
    creator: string;
    createdAt: number;
    isPublic: boolean;
    inferenceFee: number; // basis points
}

export interface OnChainVersion {
    semver: string;
    rootHash: string;
    accuracy: number;     // already divided by 1e4
    timestamp: number;
    sessionId: string;
    weightsStored: boolean;
}

// Helper to get contract address for current chain.
// Supports env-var override: NEXT_PUBLIC_MODEL_STORE_ADDRESS takes precedence.
export function getContractAddress(chainId: number, contract: keyof typeof CONTRACT_ADDRESSES[31337]): string {
    // Allow env-var override for HelixModelStore (set after deploy)
    if (contract === 'helixModelStore') {
        const envAddr = typeof window !== 'undefined'
            ? process.env.NEXT_PUBLIC_MODEL_STORE_ADDRESS
            : process.env.NEXT_PUBLIC_MODEL_STORE_ADDRESS;
        if (envAddr && envAddr !== '0x0000000000000000000000000000000000000000') {
            return envAddr;
        }
    }
    const addresses = CONTRACT_ADDRESSES[chainId as keyof typeof CONTRACT_ADDRESSES];
    if (!addresses) {
        console.warn(`No addresses configured for chain ${chainId}, falling back to localhost`);
        return CONTRACT_ADDRESSES[31337][contract];
    }
    return addresses[contract];
}
