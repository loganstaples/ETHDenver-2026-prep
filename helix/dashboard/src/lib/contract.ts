'use client';

/**
 * HELIX Dashboard Contract Interaction Layer
 * High-level contract interaction utilities with caching, batching, and error handling.
 */

import { type Address, formatEther, parseEther } from 'viem';
import {
    CONTRACT_ADDRESSES,
    HELIX_COORDINATOR_ABI,
    HELIX_TOKEN_ABI,
    getContractAddress,
    type Model,
    type Round,
    type Stake,
    type SlashingRecord,
} from './contracts';

// ============================================================================
// Types
// ============================================================================

export interface ContractConfig {
    chainId: number;
    coordinatorAddress?: Address;
    tokenAddress?: Address;
}

export interface StakingInfo {
    amount: bigint;
    amountFormatted: string;
    lockedUntil: number;
    isLocked: boolean;
    slashed: boolean;
    canUnstake: boolean;
}

export interface ModelInfo extends Model {
    id: bigint;
    errorBound: bigint;
    errorBoundFormatted: number;
    isActive: boolean;
    roundCount: number;
}

export interface RoundInfo extends Round {
    id: bigint;
    modelId: bigint;
    timeRemaining: number;
    isExpired: boolean;
    hasProver: boolean;
}

export interface NetworkState {
    totalModels: number;
    totalSlashings: number;
    defaultMinStake: bigint;
    defaultMinStakeFormatted: string;
    stakeLockPeriod: number;
    slashPercentage: number;
    maxErrorBound: bigint;
}

export interface TransactionResult {
    hash: string;
    success: boolean;
    error?: string;
    gasUsed?: bigint;
    blockNumber?: number;
}

export interface ProofSubmission {
    modelId: bigint;
    roundId: bigint;
    proof: `0x${string}`;
    publicInputs: bigint[];
}

// ============================================================================
// Contract Reader
// ============================================================================

export class ContractReader {
    private chainId: number;
    private coordinatorAddress: Address;
    private tokenAddress: Address;
    private cache: Map<string, { value: unknown; expiry: number }> = new Map();
    private cacheTTL: number;

    constructor(config: ContractConfig, cacheTTL = 5000) {
        this.chainId = config.chainId;
        this.coordinatorAddress = (config.coordinatorAddress ||
            getContractAddress(config.chainId, 'helixCoordinator')) as Address;
        this.tokenAddress = (config.tokenAddress ||
            getContractAddress(config.chainId, 'helixToken')) as Address;
        this.cacheTTL = cacheTTL;
    }

    private getCacheKey(method: string, ...args: unknown[]): string {
        return `${method}:${JSON.stringify(args)}`;
    }

    private getFromCache<T>(key: string): T | null {
        const cached = this.cache.get(key);
        if (cached && cached.expiry > Date.now()) {
            return cached.value as T;
        }
        this.cache.delete(key);
        return null;
    }

    private setCache(key: string, value: unknown): void {
        this.cache.set(key, { value, expiry: Date.now() + this.cacheTTL });
    }

    clearCache(): void {
        this.cache.clear();
    }

    // ========================================================================
    // Read Methods (to be used with publicClient)
    // ========================================================================

    getCoordinatorReadConfig() {
        return {
            address: this.coordinatorAddress,
            abi: HELIX_COORDINATOR_ABI,
        };
    }

    getTokenReadConfig() {
        return {
            address: this.tokenAddress,
            abi: HELIX_TOKEN_ABI,
        };
    }

    // ========================================================================
    // Formatters
    // ========================================================================

    formatStakingInfo(stake: Stake): StakingInfo {
        const now = Math.floor(Date.now() / 1000);
        const lockedUntil = Number(stake.lockedUntil);
        const isLocked = lockedUntil > now;

        return {
            amount: stake.amount,
            amountFormatted: formatEther(stake.amount),
            lockedUntil,
            isLocked,
            slashed: stake.slashed,
            canUnstake: !isLocked && !stake.slashed && stake.amount > BigInt(0),
        };
    }

    formatModelInfo(model: Model, modelId: bigint, errorBound: bigint): ModelInfo {
        return {
            ...model,
            id: modelId,
            errorBound,
            errorBoundFormatted: Number(errorBound) / 1e18,
            isActive: model.active,
            roundCount: Number(model.currentRound),
        };
    }

    formatRoundInfo(round: Round, roundId: bigint, modelId: bigint): RoundInfo {
        const now = Math.floor(Date.now() / 1000);
        const deadline = Number(round.deadline);

        return {
            ...round,
            id: roundId,
            modelId,
            timeRemaining: Math.max(0, deadline - now),
            isExpired: deadline < now,
            hasProver: round.prover !== '0x0000000000000000000000000000000000000000',
        };
    }

    formatNetworkState(
        nextModelId: bigint,
        slashingCount: bigint,
        defaultMinStake: bigint,
        stakeLockPeriod: bigint,
        slashPercentage: bigint,
        maxErrorBound: bigint
    ): NetworkState {
        return {
            totalModels: Number(nextModelId),
            totalSlashings: Number(slashingCount),
            defaultMinStake,
            defaultMinStakeFormatted: formatEther(defaultMinStake),
            stakeLockPeriod: Number(stakeLockPeriod),
            slashPercentage: Number(slashPercentage),
            maxErrorBound,
        };
    }

    // ========================================================================
    // Parse Events
    // ========================================================================

    parseProofSubmittedEvent(log: {
        args: { modelId: bigint; roundId: bigint; prover: string; newCommitment: bigint };
        blockNumber: bigint;
        transactionHash: string;
    }) {
        return {
            modelId: log.args.modelId,
            roundId: log.args.roundId,
            prover: log.args.prover,
            newCommitment: log.args.newCommitment,
            blockNumber: Number(log.blockNumber),
            transactionHash: log.transactionHash,
            timestamp: Date.now(),
        };
    }

    parseSlashedEvent(log: {
        args: { prover: string; modelId: bigint; amount: bigint; reason: string };
        blockNumber: bigint;
        transactionHash: string;
    }) {
        return {
            prover: log.args.prover,
            modelId: log.args.modelId,
            amount: log.args.amount,
            amountFormatted: formatEther(log.args.amount),
            reason: log.args.reason,
            blockNumber: Number(log.blockNumber),
            transactionHash: log.transactionHash,
            timestamp: Date.now(),
        };
    }
}

// ============================================================================
// Contract Writer
// ============================================================================

export class ContractWriter {
    private chainId: number;
    private coordinatorAddress: Address;
    private tokenAddress: Address;

    constructor(config: ContractConfig) {
        this.chainId = config.chainId;
        this.coordinatorAddress = (config.coordinatorAddress ||
            getContractAddress(config.chainId, 'helixCoordinator')) as Address;
        this.tokenAddress = (config.tokenAddress ||
            getContractAddress(config.chainId, 'helixToken')) as Address;
    }

    // ========================================================================
    // Write Configs (to be used with walletClient)
    // ========================================================================

    getStakeConfig(modelId: bigint, amount: string) {
        return {
            address: this.coordinatorAddress,
            abi: HELIX_COORDINATOR_ABI,
            functionName: 'stake' as const,
            args: [modelId] as const,
            value: parseEther(amount),
        };
    }

    getUnstakeConfig(modelId: bigint) {
        return {
            address: this.coordinatorAddress,
            abi: HELIX_COORDINATOR_ABI,
            functionName: 'unstake' as const,
            args: [modelId] as const,
        };
    }

    getRegisterModelConfig(ipfsHash: string, initialCommitment: bigint, minStake: string) {
        return {
            address: this.coordinatorAddress,
            abi: HELIX_COORDINATOR_ABI,
            functionName: 'registerModel' as const,
            args: [ipfsHash, initialCommitment, parseEther(minStake)] as const,
        };
    }

    getStartRoundConfig(modelId: bigint, durationSeconds: number) {
        return {
            address: this.coordinatorAddress,
            abi: HELIX_COORDINATOR_ABI,
            functionName: 'startRound' as const,
            args: [modelId, BigInt(durationSeconds)] as const,
        };
    }

    getSubmitProofConfig(submission: ProofSubmission) {
        return {
            address: this.coordinatorAddress,
            abi: HELIX_COORDINATOR_ABI,
            functionName: 'submitProof' as const,
            args: [
                submission.modelId,
                submission.roundId,
                submission.proof,
                submission.publicInputs,
            ] as const,
        };
    }

    getChallengeProofConfig(
        modelId: bigint,
        roundId: bigint,
        proof: `0x${string}`,
        publicInputs: bigint[]
    ) {
        return {
            address: this.coordinatorAddress,
            abi: HELIX_COORDINATOR_ABI,
            functionName: 'challengeProof' as const,
            args: [modelId, roundId, proof, publicInputs] as const,
        };
    }

    // Token Operations
    getApproveConfig(spender: Address, amount: bigint) {
        return {
            address: this.tokenAddress,
            abi: HELIX_TOKEN_ABI,
            functionName: 'approve' as const,
            args: [spender, amount] as const,
        };
    }

    getTransferConfig(to: Address, amount: bigint) {
        return {
            address: this.tokenAddress,
            abi: HELIX_TOKEN_ABI,
            functionName: 'transfer' as const,
            args: [to, amount] as const,
        };
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

export function formatHashShort(hash: string, chars = 6): string {
    if (!hash || hash.length < chars * 2 + 2) return hash;
    return `${hash.slice(0, chars + 2)}...${hash.slice(-chars)}`;
}

export function formatAddressShort(address: string, chars = 4): string {
    if (!address || address.length < chars * 2 + 2) return address;
    return `${address.slice(0, chars + 2)}...${address.slice(-chars)}`;
}

export function formatBigIntAsEther(value: bigint, decimals = 4): string {
    const formatted = formatEther(value);
    const parts = formatted.split('.');
    if (parts.length === 1) return formatted;
    return `${parts[0]}.${parts[1].slice(0, decimals)}`;
}

export function calculateErrorBoundPercentage(current: bigint, max: bigint): number {
    if (max === BigInt(0)) return 0;
    return Number((current * BigInt(10000)) / max) / 100;
}

export function isValidAddress(address: string): boolean {
    return /^0x[a-fA-F0-9]{40}$/.test(address);
}

export function isValidTransactionHash(hash: string): boolean {
    return /^0x[a-fA-F0-9]{64}$/.test(hash);
}

// ============================================================================
// Singleton Instances
// ============================================================================

let readerInstance: ContractReader | null = null;
let writerInstance: ContractWriter | null = null;

export function getContractReader(chainId: number): ContractReader {
    if (!readerInstance || readerInstance['chainId'] !== chainId) {
        readerInstance = new ContractReader({ chainId });
    }
    return readerInstance;
}

export function getContractWriter(chainId: number): ContractWriter {
    if (!writerInstance || writerInstance['chainId'] !== chainId) {
        writerInstance = new ContractWriter({ chainId });
    }
    return writerInstance;
}

// Re-export types and constants from contracts.ts for convenience
export {
    CONTRACT_ADDRESSES,
    HELIX_COORDINATOR_ABI,
    HELIX_TOKEN_ABI,
    getContractAddress,
    type Model,
    type Round,
    type Stake,
    type SlashingRecord,
};
