'use client';

/**
 * HELIX Dashboard API Client
 * Full API client with WebSocket support for real-time updates
 */

// ============================================================================
// Types
// ============================================================================

export interface ApiConfig {
    baseUrl: string;
    wsUrl: string;
    timeout?: number;
    retryAttempts?: number;
    retryDelay?: number;
}

export interface NodeInfo {
    id: string;
    address: string;
    type: 'compute' | 'aggregator' | 'verifier';
    status: 'online' | 'offline' | 'syncing' | 'proving' | 'training';
    lastSeen: number;
    lastHeartbeat: number;
    metrics: NodeMetrics;
    capabilities: NodeCapabilities;
    stake: {
        amount: bigint;
        lockedUntil: number;
        slashed: boolean;
    };
    performance: NodePerformance;
    location?: {
        region: string;
        latency: number;
    };
}

export interface NodeMetrics {
    cpu: number;
    memory: number;
    gpu?: number;
    gpuMemory?: number;
    networkIn: number;
    networkOut: number;
    diskUsage: number;
    temperature?: number;
}

export interface NodeCapabilities {
    canTrain: boolean;
    canAggregate: boolean;
    canProve: boolean;
    gpuModel?: string;
    gpuMemoryMb: number;
    cudaVersion?: string;
    maxBatchSize: number;
    supportedOperations: string[];
}

export interface NodePerformance {
    proofsSubmitted: number;
    proofsVerified: number;
    proofsFailed: number;
    roundsParticipated: number;
    roundsCompleted: number;
    averageProofTime: number;
    averageVerificationTime: number;
    uptime: number;
    reputation: number;
    totalEarnings: bigint;
}

export interface ProofInfo {
    id: string;
    hash: string;
    modelId: bigint;
    roundId: bigint;
    prover: string;
    type: 'training' | 'aggregation' | 'gradient' | 'computation' | 'verification';
    status: 'generating' | 'pending' | 'verified' | 'failed' | 'challenged';
    createdAt: number;
    verifiedAt?: number;
    size: number;
    generationTime: number;
    verificationTime?: number;
    errorBound: number;
    publicInputsHash: string;
    commitment: bigint;
    gasUsed?: bigint;
    transactionHash?: string;
    blockNumber?: number;
    circuitType: string;
    constraintCount: number;
}

export interface ProofGenerationProgress {
    proofId: string;
    stage: 'witness' | 'setup' | 'proving' | 'verifying' | 'complete' | 'failed';
    progress: number;
    currentStep: string;
    totalSteps: number;
    estimatedTimeRemaining: number;
    memoryUsage: number;
    cpuUsage: number;
}

export interface TrainingSession {
    id: string;
    modelId: bigint;
    modelName: string;
    status: 'initializing' | 'training' | 'paused' | 'completed' | 'failed';
    startedAt: number;
    updatedAt: number;
    completedAt?: number;
    config: TrainingConfig;
    metrics: TrainingMetrics;
    rounds: TrainingRound[];
    workers: string[];
    errorBounds: ErrorBoundEntry[];
}

export interface TrainingConfig {
    totalEpochs: number;
    batchSize: number;
    learningRate: number;
    optimizer: string;
    lossFunction: string;
    maxErrorBound: number;
    mpcThreshold: number;
    proofFrequency: number;
}

export interface TrainingMetrics {
    currentEpoch: number;
    currentBatch: number;
    totalBatches: number;
    loss: number;
    accuracy: number;
    gradientNorm: number;
    learningRate: number;
    throughput: number;
    accumulatedErrorBound: number;
    lossHistory: { epoch: number; loss: number; timestamp: number }[];
    accuracyHistory: { epoch: number; accuracy: number; timestamp: number }[];
    errorBoundHistory: { epoch: number; bound: number; timestamp: number }[];
}

export interface TrainingRound {
    id: bigint;
    status: 'pending' | 'in_progress' | 'aggregating' | 'proving' | 'completed' | 'failed';
    startedAt: number;
    completedAt?: number;
    deadline: number;
    participants: string[];
    prover?: string;
    proofId?: string;
    gradientCommitment?: bigint;
    newModelCommitment?: bigint;
    errorBound: number;
    gasUsed?: bigint;
}

export interface ErrorBoundEntry {
    layer: string;
    operation: string;
    inputBound: number;
    outputBound: number;
    amplification: number;
    timestamp: number;
    riskLevel: 'low' | 'medium' | 'high' | 'critical';
}

export interface AdversarialEvent {
    id: string;
    type: 'invalid_proof' | 'timeout' | 'malicious_gradient' | 'stake_slashed' | 'challenge_submitted' | 'byzantine_behavior';
    severity: 'warning' | 'critical' | 'resolved';
    timestamp: number;
    modelId: bigint;
    roundId?: bigint;
    prover: string;
    description: string;
    details: Record<string, unknown>;
    slashAmount?: bigint;
    transactionHash?: string;
    resolved: boolean;
    resolvedAt?: number;
}

export interface NetworkStats {
    totalNodes: number;
    activeNodes: number;
    totalStaked: bigint;
    totalProofs: number;
    totalRounds: number;
    totalSlashed: bigint;
    averageProofTime: number;
    networkUptime: number;
    throughput: number;
    activeTrainingSessions: number;
}

// ============================================================================
// API Client
// ============================================================================

import { HelixWebSocketClient, type WebSocketMessage } from './websocket';

type MessageHandler<T = unknown> = (message: WebSocketMessage<T>) => void;

export class HelixApiClient {
    private config: Required<ApiConfig>;
    private wsClient: HelixWebSocketClient | null = null;
    private abortControllers: Map<string, AbortController> = new Map();

    constructor(config: Partial<ApiConfig> = {}) {
        this.config = {
            baseUrl: config.baseUrl || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001/api',
            wsUrl: config.wsUrl || process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:3001/ws',
            timeout: config.timeout || 30000,
            retryAttempts: config.retryAttempts || 3,
            retryDelay: config.retryDelay || 1000,
        };
    }

    // ========================================================================
    // WebSocket Methods (delegates to HelixWebSocketClient from websocket.ts)
    // ========================================================================

    async connectWebSocket(): Promise<void> {
        if (!this.wsClient) {
            this.wsClient = new HelixWebSocketClient({ url: this.config.wsUrl });
        }
        await this.wsClient.connect();
    }

    disconnectWebSocket(): void {
        this.wsClient?.disconnect();
        this.wsClient = null;
    }

    subscribeToChannel(channel: string): void {
        this.wsClient?.subscribe(channel);
    }

    unsubscribeFromChannel(channel: string): void {
        this.wsClient?.unsubscribe(channel);
    }

    onWebSocketMessage<T = unknown>(type: string, handler: MessageHandler<T>): () => void {
        if (!this.wsClient) {
            this.wsClient = new HelixWebSocketClient({ url: this.config.wsUrl });
        }
        return this.wsClient.on(type, handler);
    }

    get isWebSocketConnected(): boolean {
        return this.wsClient?.state === 'connected';
    }

    // ========================================================================
    // HTTP Methods
    // ========================================================================

    private async fetch<T>(
        endpoint: string,
        options: RequestInit = {},
        requestId?: string
    ): Promise<T> {
        const url = `${this.config.baseUrl}${endpoint}`;
        const controller = new AbortController();

        if (requestId) {
            this.abortControllers.get(requestId)?.abort();
            this.abortControllers.set(requestId, controller);
        }

        const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

        try {
            const response = await fetch(url, {
                ...options,
                signal: controller.signal,
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers,
                },
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new ApiError(response.status, error.message || 'Request failed', error);
            }

            return response.json();
        } catch (err) {
            clearTimeout(timeoutId);

            if (err instanceof Error && err.name === 'AbortError') {
                throw new ApiError(0, 'Request cancelled');
            }

            throw err;
        } finally {
            if (requestId) {
                this.abortControllers.delete(requestId);
            }
        }
    }

    cancelRequest(requestId: string): void {
        this.abortControllers.get(requestId)?.abort();
        this.abortControllers.delete(requestId);
    }

    // ========================================================================
    // Node Endpoints
    // ========================================================================

    async getNodes(): Promise<NodeInfo[]> {
        return this.fetch<NodeInfo[]>('/nodes');
    }

    async getNode(nodeId: string): Promise<NodeInfo> {
        return this.fetch<NodeInfo>(`/nodes/${nodeId}`);
    }

    async getNodesByType(type: NodeInfo['type']): Promise<NodeInfo[]> {
        return this.fetch<NodeInfo[]>(`/nodes?type=${type}`);
    }

    async getNodeMetrics(nodeId: string): Promise<NodeMetrics> {
        return this.fetch<NodeMetrics>(`/nodes/${nodeId}/metrics`);
    }

    async getNodePerformance(nodeId: string): Promise<NodePerformance> {
        return this.fetch<NodePerformance>(`/nodes/${nodeId}/performance`);
    }

    async getNetworkTopology(): Promise<{
        nodes: NodeInfo[];
        connections: Array<{
            from: string;
            to: string;
            latency: number;
            bandwidth: number;
            status: 'active' | 'degraded' | 'offline';
        }>;
    }> {
        return this.fetch('/network/topology');
    }

    // ========================================================================
    // Proof Endpoints
    // ========================================================================

    async getProofs(params?: {
        modelId?: string;
        roundId?: string;
        prover?: string;
        status?: ProofInfo['status'];
        type?: ProofInfo['type'];
        limit?: number;
        offset?: number;
    }): Promise<{ proofs: ProofInfo[]; total: number }> {
        const queryParams = new URLSearchParams();
        if (params) {
            Object.entries(params).forEach(([key, value]) => {
                if (value !== undefined) queryParams.set(key, String(value));
            });
        }
        return this.fetch(`/proofs?${queryParams}`);
    }

    async getProof(proofId: string): Promise<ProofInfo> {
        return this.fetch<ProofInfo>(`/proofs/${proofId}`);
    }

    async getProofGenerationProgress(proofId: string): Promise<ProofGenerationProgress> {
        return this.fetch<ProofGenerationProgress>(`/proofs/${proofId}/progress`);
    }

    async getProofTimeline(modelId: string, roundId?: string): Promise<{
        events: Array<{
            timestamp: number;
            type: 'started' | 'witness_generated' | 'proof_generated' | 'submitted' | 'verified' | 'failed';
            proofId: string;
            details: Record<string, unknown>;
        }>;
    }> {
        const params = roundId ? `?roundId=${roundId}` : '';
        return this.fetch(`/proofs/timeline/${modelId}${params}`);
    }

    async getProofStats(): Promise<{
        total: number;
        verified: number;
        pending: number;
        failed: number;
        averageGenerationTime: number;
        averageVerificationTime: number;
        totalSize: number;
    }> {
        return this.fetch('/proofs/stats');
    }

    // ========================================================================
    // Training Endpoints
    // ========================================================================

    async getTrainingSessions(params?: {
        modelId?: string;
        status?: TrainingSession['status'];
        limit?: number;
        offset?: number;
    }): Promise<{ sessions: TrainingSession[]; total: number }> {
        const queryParams = new URLSearchParams();
        if (params) {
            Object.entries(params).forEach(([key, value]) => {
                if (value !== undefined) queryParams.set(key, String(value));
            });
        }
        return this.fetch(`/training?${queryParams}`);
    }

    async getTrainingSession(sessionId: string): Promise<TrainingSession> {
        return this.fetch<TrainingSession>(`/training/${sessionId}`);
    }

    async getTrainingMetrics(sessionId: string): Promise<TrainingMetrics> {
        return this.fetch<TrainingMetrics>(`/training/${sessionId}/metrics`);
    }

    async getTrainingRounds(sessionId: string, params?: {
        status?: TrainingRound['status'];
        limit?: number;
        offset?: number;
    }): Promise<{ rounds: TrainingRound[]; total: number }> {
        const queryParams = new URLSearchParams();
        if (params) {
            Object.entries(params).forEach(([key, value]) => {
                if (value !== undefined) queryParams.set(key, String(value));
            });
        }
        return this.fetch(`/training/${sessionId}/rounds?${queryParams}`);
    }

    async getErrorBounds(sessionId: string): Promise<ErrorBoundEntry[]> {
        return this.fetch<ErrorBoundEntry[]>(`/training/${sessionId}/error-bounds`);
    }

    async getLossCurve(sessionId: string): Promise<{
        points: Array<{ epoch: number; batch: number; loss: number; timestamp: number }>;
        smoothed: Array<{ epoch: number; loss: number }>;
    }> {
        return this.fetch(`/training/${sessionId}/loss-curve`);
    }

    // ========================================================================
    // Adversarial & Security Endpoints
    // ========================================================================

    async getAdversarialEvents(params?: {
        modelId?: string;
        type?: AdversarialEvent['type'];
        severity?: AdversarialEvent['severity'];
        resolved?: boolean;
        limit?: number;
        offset?: number;
    }): Promise<{ events: AdversarialEvent[]; total: number }> {
        const queryParams = new URLSearchParams();
        if (params) {
            Object.entries(params).forEach(([key, value]) => {
                if (value !== undefined) queryParams.set(key, String(value));
            });
        }
        return this.fetch(`/adversarial?${queryParams}`);
    }

    async getAdversarialEvent(eventId: string): Promise<AdversarialEvent> {
        return this.fetch<AdversarialEvent>(`/adversarial/${eventId}`);
    }

    async getSlashingHistory(prover?: string): Promise<Array<{
        timestamp: number;
        prover: string;
        modelId: bigint;
        roundId: bigint;
        amount: bigint;
        reason: string;
        transactionHash: string;
    }>> {
        const params = prover ? `?prover=${prover}` : '';
        return this.fetch(`/adversarial/slashing${params}`);
    }

    // ========================================================================
    // Network & Stats Endpoints
    // ========================================================================

    async getNetworkStats(): Promise<NetworkStats> {
        return this.fetch<NetworkStats>('/network/stats');
    }

    async getHealthCheck(): Promise<{
        status: 'healthy' | 'degraded' | 'unhealthy';
        services: Record<string, { status: string; latency: number }>;
        timestamp: number;
    }> {
        return this.fetch('/health');
    }
}

// ============================================================================
// Error Class
// ============================================================================

export class ApiError extends Error {
    constructor(
        public statusCode: number,
        message: string,
        public details?: Record<string, unknown>
    ) {
        super(message);
        this.name = 'ApiError';
    }
}

// ============================================================================
// Singleton Instance & Mock Data Generator
// ============================================================================

let apiClientInstance: HelixApiClient | null = null;

export function getApiClient(config?: Partial<ApiConfig>): HelixApiClient {
    if (!apiClientInstance) {
        apiClientInstance = new HelixApiClient(config);
    }
    return apiClientInstance;
}

// Mock data generator for demo/development
export function generateMockNodeInfo(address: string, index: number): NodeInfo {
    const types: NodeInfo['type'][] = ['compute', 'aggregator', 'verifier'];
    const statuses: NodeInfo['status'][] = ['online', 'online', 'online', 'syncing', 'proving'];
    const regions = ['us-east', 'us-west', 'eu-west', 'ap-south', 'ap-northeast'];

    return {
        id: `node-${address.slice(2, 10)}`,
        address,
        type: types[index % 3],
        status: statuses[Math.floor(Math.random() * statuses.length)],
        lastSeen: Date.now() - Math.random() * 60000,
        lastHeartbeat: Date.now() - Math.random() * 30000,
        metrics: {
            cpu: 20 + Math.random() * 60,
            memory: 30 + Math.random() * 50,
            gpu: Math.random() > 0.3 ? 40 + Math.random() * 50 : undefined,
            gpuMemory: Math.random() > 0.3 ? 30 + Math.random() * 60 : undefined,
            networkIn: Math.random() * 1000,
            networkOut: Math.random() * 500,
            diskUsage: 20 + Math.random() * 40,
            temperature: 40 + Math.random() * 30,
        },
        capabilities: {
            canTrain: Math.random() > 0.2,
            canAggregate: Math.random() > 0.5,
            canProve: Math.random() > 0.3,
            gpuModel: ['RTX 4090', 'RTX 3090', 'A100', 'H100'][Math.floor(Math.random() * 4)],
            gpuMemoryMb: [8192, 16384, 24576, 40960, 81920][Math.floor(Math.random() * 5)],
            cudaVersion: '12.1',
            maxBatchSize: [32, 64, 128, 256][Math.floor(Math.random() * 4)],
            supportedOperations: ['matmul', 'conv2d', 'attention', 'layernorm'],
        },
        stake: {
            amount: BigInt(Math.floor(Math.random() * 10 + 1) * 1e18),
            lockedUntil: Date.now() + Math.random() * 86400000 * 7,
            slashed: Math.random() < 0.1,
        },
        performance: {
            proofsSubmitted: Math.floor(Math.random() * 100),
            proofsVerified: Math.floor(Math.random() * 90),
            proofsFailed: Math.floor(Math.random() * 5),
            roundsParticipated: Math.floor(Math.random() * 50),
            roundsCompleted: Math.floor(Math.random() * 45),
            averageProofTime: 200 + Math.random() * 300,
            averageVerificationTime: 50 + Math.random() * 100,
            uptime: 0.95 + Math.random() * 0.05,
            reputation: 80 + Math.random() * 20,
            totalEarnings: BigInt(Math.floor(Math.random() * 5 * 1e18)),
        },
        location: {
            region: regions[Math.floor(Math.random() * regions.length)],
            latency: 10 + Math.random() * 100,
        },
    };
}

export function generateMockProofInfo(roundId: bigint, index: number): ProofInfo {
    const types: ProofInfo['type'][] = ['training', 'aggregation', 'gradient', 'computation'];
    const statuses: ProofInfo['status'][] = ['verified', 'verified', 'verified', 'pending', 'generating'];

    return {
        id: `proof-${Date.now()}-${index}`,
        hash: `0x${Array(64).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('')}`,
        modelId: BigInt(1),
        roundId,
        prover: `0x${Array(40).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('')}`,
        type: types[index % 4],
        status: statuses[Math.floor(Math.random() * statuses.length)],
        createdAt: Date.now() - Math.random() * 300000,
        verifiedAt: Math.random() > 0.3 ? Date.now() - Math.random() * 60000 : undefined,
        size: 1024 + Math.floor(Math.random() * 4096),
        generationTime: 100 + Math.random() * 400,
        verificationTime: Math.random() > 0.3 ? 20 + Math.random() * 80 : undefined,
        errorBound: Math.random() * 0.001,
        publicInputsHash: `0x${Array(64).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('')}`,
        commitment: BigInt(Math.floor(Math.random() * 1e18)),
        gasUsed: Math.random() > 0.5 ? BigInt(Math.floor(Math.random() * 500000)) : undefined,
        transactionHash: Math.random() > 0.5 ? `0x${Array(64).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('')}` : undefined,
        blockNumber: Math.random() > 0.5 ? Math.floor(Math.random() * 1000000) + 18000000 : undefined,
        circuitType: ['nova_folding', 'groth16', 'plonk'][Math.floor(Math.random() * 3)],
        constraintCount: 10000 + Math.floor(Math.random() * 100000),
    };
}

export function generateMockTrainingMetrics(epoch: number): TrainingMetrics {
    const baseLoss = 2.5 * Math.exp(-0.3 * epoch) + 0.1;
    const baseAccuracy = 1 - Math.exp(-0.2 * epoch) * 0.8;

    return {
        currentEpoch: epoch,
        currentBatch: Math.floor(Math.random() * 100),
        totalBatches: 100,
        loss: baseLoss + (Math.random() - 0.5) * 0.1,
        accuracy: Math.min(0.99, baseAccuracy + (Math.random() - 0.5) * 0.05),
        gradientNorm: 0.1 + Math.random() * 0.5,
        learningRate: 0.001 * Math.pow(0.95, epoch),
        throughput: 50 + Math.random() * 50,
        accumulatedErrorBound: 0.0001 * epoch + Math.random() * 0.00005,
        lossHistory: Array.from({ length: epoch + 1 }, (_, i) => ({
            epoch: i,
            loss: 2.5 * Math.exp(-0.3 * i) + 0.1 + (Math.random() - 0.5) * 0.1,
            timestamp: Date.now() - (epoch - i) * 60000,
        })),
        accuracyHistory: Array.from({ length: epoch + 1 }, (_, i) => ({
            epoch: i,
            accuracy: Math.min(0.99, 1 - Math.exp(-0.2 * i) * 0.8 + (Math.random() - 0.5) * 0.05),
            timestamp: Date.now() - (epoch - i) * 60000,
        })),
        errorBoundHistory: Array.from({ length: epoch + 1 }, (_, i) => ({
            epoch: i,
            bound: 0.0001 * i + Math.random() * 0.00005,
            timestamp: Date.now() - (epoch - i) * 60000,
        })),
    };
}

export function generateMockAdversarialEvent(index: number): AdversarialEvent {
    const types: AdversarialEvent['type'][] = ['invalid_proof', 'timeout', 'malicious_gradient', 'stake_slashed'];
    const severities: AdversarialEvent['severity'][] = ['warning', 'critical', 'resolved'];

    return {
        id: `event-${Date.now()}-${index}`,
        type: types[index % types.length],
        severity: severities[Math.floor(Math.random() * severities.length)],
        timestamp: Date.now() - Math.random() * 3600000,
        modelId: BigInt(1),
        roundId: BigInt(Math.floor(Math.random() * 100)),
        prover: `0x${Array(40).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('')}`,
        description: [
            'Invalid proof detected during verification',
            'Worker timeout during proof generation',
            'Malicious gradient contribution detected',
            'Stake slashed due to protocol violation',
        ][index % 4],
        details: {
            expectedHash: `0x${Array(64).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('')}`,
            actualHash: `0x${Array(64).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('')}`,
        },
        slashAmount: Math.random() > 0.5 ? BigInt(Math.floor(Math.random() * 1e18)) : undefined,
        transactionHash: `0x${Array(64).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('')}`,
        resolved: Math.random() > 0.3,
        resolvedAt: Math.random() > 0.5 ? Date.now() - Math.random() * 1800000 : undefined,
    };
}

// ============================================================================
// Status Page Specific Types and Functions
// ============================================================================

export interface StatusOverview {
    trainingSessions: TrainingSession[];
    workers: NodeInfo[];
    recentProofs: ProofInfo[];
    currentMetrics: TrainingMetrics | null;
    networkStats: NetworkStats;
    errorLogs: StatusErrorLog[];
    contractState: ContractStateSnapshot;
}

export interface StatusErrorLog {
    id: string;
    timestamp: number;
    level: 'error' | 'warning' | 'info';
    message: string;
    source: string;
    details?: Record<string, unknown>;
}

export interface ContractStateSnapshot {
    nextModelId: string;
    defaultMinStake: string;
    stakeLockPeriod: string;
    slashPercentage: string;
    maxErrorBound: string;
    slashingRecordCount: string;
    activeModels: number;
    totalStaked: string;
}

export interface WorkerHealthStatus {
    workerId: string;
    address: string;
    type: NodeInfo['type'];
    status: NodeInfo['status'];
    lastHeartbeat: number;
    uptime: number;
    proofsSubmitted: number;
    proofsVerified: number;
    proofsFailed: number;
    averageProofTime: number;
    stake: {
        amount: string;
        slashed: boolean;
    };
}

export interface RoundState {
    modelId: string;
    currentRound: string;
    roundStatus: 'pending' | 'in_progress' | 'aggregating' | 'proving' | 'completed' | 'failed';
    deadline: number | null;
    participants: string[];
    prover: string | null;
    modelCommitment: string;
    newCommitment: string | null;
    errorBound: number;
    proofSubmitted: boolean;
}

// Add status-specific methods to the API client
export class StatusApiClient extends HelixApiClient {
    // ========================================================================
    // Status Page Endpoints
    // ========================================================================

    /**
     * Get complete status overview for the dashboard
     * Aggregates multiple data sources into a single response
     */
    async getStatusOverview(): Promise<StatusOverview> {
        return this['fetch']<StatusOverview>('/status/overview');
    }

    /**
     * Get worker health status for all connected workers
     */
    async getWorkerHealthStatus(): Promise<WorkerHealthStatus[]> {
        return this['fetch']<WorkerHealthStatus[]>('/status/workers');
    }

    /**
     * Get current round state for a specific model
     */
    async getRoundState(modelId: string | bigint): Promise<RoundState> {
        return this['fetch']<RoundState>(`/status/round/${modelId.toString()}`);
    }

    /**
     * Get error logs with filtering
     */
    async getErrorLogs(params?: {
        level?: StatusErrorLog['level'];
        source?: string;
        since?: number;
        limit?: number;
    }): Promise<{ logs: StatusErrorLog[]; total: number }> {
        const queryParams = new URLSearchParams();
        if (params) {
            Object.entries(params).forEach(([key, value]) => {
                if (value !== undefined) queryParams.set(key, String(value));
            });
        }
        return this['fetch'](`/status/logs?${queryParams}`);
    }

    /**
     * Get contract state snapshot
     */
    async getContractStateSnapshot(): Promise<ContractStateSnapshot> {
        return this['fetch']<ContractStateSnapshot>('/status/contract');
    }

    /**
     * Get recent proof submissions with verification status
     */
    async getRecentProofStatus(params?: {
        modelId?: string;
        limit?: number;
    }): Promise<{
        proofs: Array<ProofInfo & { verificationLatency?: number }>;
        stats: {
            totalSubmitted: number;
            verified: number;
            pending: number;
            failed: number;
            averageVerificationTime: number;
        };
    }> {
        const queryParams = new URLSearchParams();
        if (params) {
            Object.entries(params).forEach(([key, value]) => {
                if (value !== undefined) queryParams.set(key, String(value));
            });
        }
        return this['fetch'](`/status/proofs?${queryParams}`);
    }

    /**
     * Get live training metrics for a session
     */
    async getLiveTrainingMetrics(sessionId: string): Promise<{
        metrics: TrainingMetrics;
        deltaFromLast: {
            loss: number;
            accuracy: number;
            errorBound: number;
        };
        estimatedCompletion: number | null;
    }> {
        return this['fetch'](`/status/training/${sessionId}/live`);
    }
}

// ============================================================================
// Status Mock Data Generators
// ============================================================================

export function generateMockStatusOverview(): StatusOverview {
    const workers = Array.from({ length: 8 }, (_, i) =>
        generateMockNodeInfo(
            `0x${Array(40).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('')}`,
            i
        )
    );

    const proofs = Array.from({ length: 10 }, (_, i) =>
        generateMockProofInfo(BigInt(Math.floor(Math.random() * 50) + 1), i)
    );

    const metrics = generateMockTrainingMetrics(25);

    return {
        trainingSessions: generateMockTrainingSessions(),
        workers,
        recentProofs: proofs,
        currentMetrics: metrics,
        networkStats: generateMockNetworkStats(),
        errorLogs: generateMockErrorLogs(),
        contractState: generateMockContractState(),
    };
}

export function generateMockTrainingSessions(): TrainingSession[] {
    const statuses: TrainingSession['status'][] = ['training', 'initializing', 'completed', 'paused'];
    return Array.from({ length: 3 }, (_, i) => ({
        id: `session-${i + 1}`,
        modelId: BigInt(i + 1),
        modelName: ['GPT-Mini', 'Vision Classifier', 'Sentiment Model'][i] || `Model ${i + 1}`,
        status: statuses[i % statuses.length],
        startedAt: Date.now() - (i + 1) * 3600000,
        updatedAt: Date.now() - Math.random() * 60000,
        completedAt: i === 2 ? Date.now() - 1800000 : undefined,
        config: {
            totalEpochs: 100,
            batchSize: 32,
            learningRate: 0.001,
            optimizer: 'AdamW',
            lossFunction: 'CrossEntropy',
            maxErrorBound: 0.01,
            mpcThreshold: 3,
            proofFrequency: 10,
        },
        metrics: generateMockTrainingMetrics(Math.floor(Math.random() * 50) + 10),
        rounds: [],
        workers: [`0x${Array(40).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('')}`],
        errorBounds: [],
    }));
}

export function generateMockNetworkStats(): NetworkStats {
    return {
        totalNodes: 12,
        activeNodes: 9,
        totalStaked: BigInt(150 * 1e18),
        totalProofs: 1247,
        totalRounds: 89,
        totalSlashed: BigInt(2 * 1e18),
        averageProofTime: 312,
        networkUptime: 0.997,
        throughput: 42.5,
        activeTrainingSessions: 2,
    };
}

export function generateMockErrorLogs(): StatusErrorLog[] {
    const messages = [
        { level: 'error' as const, message: 'Proof verification failed for round 45', source: 'verifier' },
        { level: 'warning' as const, message: 'Worker node-abc123 high latency detected (>500ms)', source: 'network' },
        { level: 'info' as const, message: 'Round 46 aggregation complete', source: 'aggregator' },
        { level: 'warning' as const, message: 'Memory usage above 80% on node-def456', source: 'monitor' },
        { level: 'error' as const, message: 'Invalid gradient detected, triggering slash', source: 'coordinator' },
        { level: 'info' as const, message: 'New worker registered: 0x7a3b...9c2d', source: 'registry' },
        { level: 'warning' as const, message: 'Proof generation time exceeded threshold', source: 'prover' },
        { level: 'info' as const, message: 'Model checkpoint saved to IPFS', source: 'storage' },
    ];

    return messages.map((m, i) => ({
        id: `log-${i}`,
        timestamp: Date.now() - i * 45000,
        ...m,
    }));
}

export function generateMockContractState(): ContractStateSnapshot {
    return {
        nextModelId: '5',
        defaultMinStake: '1000000000000000000', // 1 ETH
        stakeLockPeriod: '604800', // 7 days in seconds
        slashPercentage: '10',
        maxErrorBound: '1000000000000000', // 0.001 in wei-like format
        slashingRecordCount: '3',
        activeModels: 4,
        totalStaked: '150000000000000000000', // 150 ETH
    };
}

export function generateMockWorkerHealth(address: string, index: number): WorkerHealthStatus {
    const types: NodeInfo['type'][] = ['compute', 'aggregator', 'verifier'];
    const statuses: NodeInfo['status'][] = ['online', 'online', 'syncing', 'proving', 'training'];

    return {
        workerId: `worker-${address.slice(2, 10)}`,
        address,
        type: types[index % 3],
        status: statuses[Math.floor(Math.random() * statuses.length)],
        lastHeartbeat: Date.now() - Math.random() * 30000,
        uptime: 0.95 + Math.random() * 0.05,
        proofsSubmitted: Math.floor(Math.random() * 100),
        proofsVerified: Math.floor(Math.random() * 90),
        proofsFailed: Math.floor(Math.random() * 5),
        averageProofTime: 200 + Math.random() * 300,
        stake: {
            amount: (Math.floor(Math.random() * 10 + 1) * 1e18).toString(),
            slashed: Math.random() < 0.1,
        },
    };
}

export function generateMockRoundState(modelId: string): RoundState {
    const statuses: RoundState['roundStatus'][] = ['pending', 'in_progress', 'aggregating', 'proving', 'completed'];
    const status = statuses[Math.floor(Math.random() * statuses.length)];

    return {
        modelId,
        currentRound: Math.floor(Math.random() * 100).toString(),
        roundStatus: status,
        deadline: status !== 'completed' ? Date.now() + Math.random() * 300000 : null,
        participants: Array.from({ length: Math.floor(Math.random() * 5) + 1 }, () =>
            `0x${Array(40).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('')}`
        ),
        prover: status === 'completed' || status === 'proving'
            ? `0x${Array(40).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('')}`
            : null,
        modelCommitment: `0x${Array(64).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('')}`,
        newCommitment: status === 'completed'
            ? `0x${Array(64).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('')}`
            : null,
        errorBound: Math.random() * 0.001,
        proofSubmitted: status === 'completed' || status === 'proving',
    };
}

// ============================================================================
// Status API Client Singleton
// ============================================================================

let statusApiClientInstance: StatusApiClient | null = null;

export function getStatusApiClient(config?: Partial<ApiConfig>): StatusApiClient {
    if (!statusApiClientInstance) {
        statusApiClientInstance = new StatusApiClient(config);
    }
    return statusApiClientInstance;
}

// Default export
export const apiClient = getApiClient();
