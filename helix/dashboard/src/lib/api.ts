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
// MPC Training Types
// ============================================================================

export interface MpcTrainingJobConfig {
    architecture: number[];
    num_workers: number;
    num_steps: number;
    learning_rate: number;
    checkpoint_freq: number;
    mac_interval: number;
    zk_mode: 'off' | 'always' | 'risk';
    zk_checkpoint_freq: number;
    min_workers_for_mpc: number;
    train_size: number;
    test_size: number;
    use_real_mnist: boolean;
    payment_eth: number;
    stake_per_worker_eth: number;
    simulate_cheater: boolean;
    seed: number;
}

export interface MpcTrainingSessionState {
    session_id: string;
    status: 'starting' | 'running' | 'complete' | 'failed';
    current_step: number;
    total_steps: number;
    current_loss: number;
    losses: number[];
    accuracy: number | null;
    checkpoints_submitted: number;
    mac_checks_passed: number;
    cheater_detected: { party_index: number; step: number } | null;
    zk_proofs_generated: number;
    zk_activated_by_risk: boolean;
    phase: number;
    phase_description: string;
    coordinator_address: string;
    job_id: number;
    elapsed_secs: number;
    started_at: number;
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
    // MPC Training Endpoints (backend at /api/training/*)
    // ========================================================================

    async startMpcTraining(config: MpcTrainingJobConfig): Promise<{ session_id: string; status: string }> {
        return this.fetch<{ session_id: string; status: string }>('/training/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });
    }

    async getMpcTrainingSession(sessionId: string): Promise<MpcTrainingSessionState> {
        return this.fetch<MpcTrainingSessionState>(`/training/sessions/${sessionId}`);
    }

    async getMpcSessionLosses(sessionId: string): Promise<{ losses: number[]; current_step: number; total_steps: number }> {
        return this.fetch<{ losses: number[]; current_step: number; total_steps: number }>(`/training/sessions/${sessionId}/losses`);
    }

    async listMpcTrainingSessions(): Promise<MpcTrainingSessionState[]> {
        return this.fetch<MpcTrainingSessionState[]>('/training/sessions');
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

// Default export
export const apiClient = getApiClient();
