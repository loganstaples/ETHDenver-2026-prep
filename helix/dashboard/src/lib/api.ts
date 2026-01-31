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

// WebSocket Message Types
export type WebSocketMessageType =
    | 'node_update'
    | 'proof_update'
    | 'training_update'
    | 'round_update'
    | 'adversarial_event'
    | 'network_stats'
    | 'error_bound_update'
    | 'heartbeat'
    | 'subscribe'
    | 'unsubscribe';

export interface WebSocketMessage<T = unknown> {
    type: WebSocketMessageType;
    timestamp: number;
    data: T;
    channel?: string;
}

// ============================================================================
// WebSocket Manager
// ============================================================================

type MessageHandler<T = unknown> = (message: WebSocketMessage<T>) => void;

export class WebSocketManager {
    private ws: WebSocket | null = null;
    private url: string;
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 10;
    private reconnectDelay = 1000;
    private handlers: Map<WebSocketMessageType, Set<MessageHandler>> = new Map();
    private subscriptions: Set<string> = new Set();
    private heartbeatInterval: ReturnType<typeof setInterval> | null = null;
    private isConnecting = false;
    private messageQueue: WebSocketMessage[] = [];

    constructor(url: string) {
        this.url = url;
    }

    connect(): Promise<void> {
        return new Promise((resolve, reject) => {
            if (this.ws?.readyState === WebSocket.OPEN) {
                resolve();
                return;
            }

            if (this.isConnecting) {
                // Wait for existing connection attempt
                const checkConnection = setInterval(() => {
                    if (this.ws?.readyState === WebSocket.OPEN) {
                        clearInterval(checkConnection);
                        resolve();
                    }
                }, 100);
                return;
            }

            this.isConnecting = true;

            try {
                this.ws = new WebSocket(this.url);

                this.ws.onopen = () => {
                    this.isConnecting = false;
                    this.reconnectAttempts = 0;
                    this.startHeartbeat();

                    // Resubscribe to previous channels
                    this.subscriptions.forEach(channel => {
                        this.send({ type: 'subscribe', timestamp: Date.now(), data: { channel } });
                    });

                    // Flush message queue
                    while (this.messageQueue.length > 0) {
                        const msg = this.messageQueue.shift();
                        if (msg) this.send(msg);
                    }

                    resolve();
                };

                this.ws.onmessage = (event) => {
                    try {
                        const message: WebSocketMessage = JSON.parse(event.data);
                        this.handleMessage(message);
                    } catch (err) {
                        console.error('Failed to parse WebSocket message:', err);
                    }
                };

                this.ws.onclose = () => {
                    this.isConnecting = false;
                    this.stopHeartbeat();
                    this.attemptReconnect();
                };

                this.ws.onerror = (error) => {
                    this.isConnecting = false;
                    console.error('WebSocket error:', error);
                    reject(error);
                };
            } catch (err) {
                this.isConnecting = false;
                reject(err);
            }
        });
    }

    disconnect(): void {
        this.stopHeartbeat();
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.subscriptions.clear();
        this.handlers.clear();
    }

    subscribe(channel: string): void {
        this.subscriptions.add(channel);
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.send({ type: 'subscribe', timestamp: Date.now(), data: { channel } });
        }
    }

    unsubscribe(channel: string): void {
        this.subscriptions.delete(channel);
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.send({ type: 'unsubscribe', timestamp: Date.now(), data: { channel } });
        }
    }

    on<T = unknown>(type: WebSocketMessageType, handler: MessageHandler<T>): () => void {
        if (!this.handlers.has(type)) {
            this.handlers.set(type, new Set());
        }
        this.handlers.get(type)!.add(handler as MessageHandler);

        return () => {
            this.handlers.get(type)?.delete(handler as MessageHandler);
        };
    }

    send(message: WebSocketMessage): void {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        } else {
            this.messageQueue.push(message);
        }
    }

    get isConnected(): boolean {
        return this.ws?.readyState === WebSocket.OPEN;
    }

    private handleMessage(message: WebSocketMessage): void {
        const handlers = this.handlers.get(message.type);
        if (handlers) {
            handlers.forEach(handler => {
                try {
                    handler(message);
                } catch (err) {
                    console.error('Error in WebSocket handler:', err);
                }
            });
        }
    }

    private startHeartbeat(): void {
        this.heartbeatInterval = setInterval(() => {
            this.send({ type: 'heartbeat', timestamp: Date.now(), data: {} });
        }, 30000);
    }

    private stopHeartbeat(): void {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }

    private attemptReconnect(): void {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnect attempts reached');
            return;
        }

        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

        setTimeout(() => {
            this.connect().catch(console.error);
        }, delay);
    }
}

// ============================================================================
// API Client
// ============================================================================

export class HelixApiClient {
    private config: Required<ApiConfig>;
    private wsManager: WebSocketManager | null = null;
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
    // WebSocket Methods
    // ========================================================================

    async connectWebSocket(): Promise<void> {
        if (!this.wsManager) {
            this.wsManager = new WebSocketManager(this.config.wsUrl);
        }
        await this.wsManager.connect();
    }

    disconnectWebSocket(): void {
        this.wsManager?.disconnect();
        this.wsManager = null;
    }

    subscribeToChannel(channel: string): void {
        this.wsManager?.subscribe(channel);
    }

    unsubscribeFromChannel(channel: string): void {
        this.wsManager?.unsubscribe(channel);
    }

    onWebSocketMessage<T = unknown>(type: WebSocketMessageType, handler: MessageHandler<T>): () => void {
        if (!this.wsManager) {
            this.wsManager = new WebSocketManager(this.config.wsUrl);
        }
        return this.wsManager.on(type, handler);
    }

    get isWebSocketConnected(): boolean {
        return this.wsManager?.isConnected || false;
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

// Default export
export const apiClient = getApiClient();
