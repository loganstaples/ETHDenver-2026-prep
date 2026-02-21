'use client';

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { useContractEvents, useContractState } from './useContract';
import {
    type NodeInfo,
    type NodeMetrics,
    getApiClient,
} from '@/lib/api';

// ============================================================================
// Types
// ============================================================================

export interface WorkerNode {
    id: string;
    address: string;
    type: 'compute' | 'aggregator' | 'verifier';
    status: 'active' | 'idle' | 'offline' | 'syncing' | 'proving' | 'training';
    lastSeen: number;
    lastHeartbeat: number;
    stakedAmount: bigint;
    proofsSubmitted: number;
    proofsVerified: number;
    proofsFailed: number;
    roundsParticipated: number;
    roundsCompleted: number;
    reputation: number;
    metrics: {
        cpu: number;
        memory: number;
        gpu?: number;
        gpuMemory?: number;
        networkIn: number;
        networkOut: number;
        temperature?: number;
    };
    capabilities: {
        canTrain: boolean;
        canAggregate: boolean;
        canProve: boolean;
        gpuModel?: string;
        gpuMemoryMb: number;
        maxBatchSize: number;
    };
    location?: {
        region: string;
        latency: number;
    };
    earningsTotal: bigint;
    slashed: boolean;
    // Live fields from backend worker registry
    successRate?: number;
    registeredAt?: number;
    endpoint?: string;
    partyIndex?: number;
}

export interface NetworkConnection {
    from: string;
    to: string;
    latency: number;
    bandwidth: number;
    status: 'active' | 'degraded' | 'offline';
    lastUpdated: number;
}

export interface NetworkTopology {
    nodes: WorkerNode[];
    connections: NetworkConnection[];
    aggregators: string[];
    verifiers: string[];
    computeNodes: string[];
}

export interface NodeFilter {
    type?: WorkerNode['type'];
    status?: WorkerNode['status'];
    minStake?: bigint;
    minReputation?: number;
    hasGpu?: boolean;
    region?: string;
}

export interface UseNodesOptions {
    autoRefresh?: boolean;
    refreshInterval?: number;
    enableWebSocket?: boolean;
    filter?: NodeFilter;
    includeOffline?: boolean;
}

export interface UseNodesReturn {
    nodes: WorkerNode[];
    filteredNodes: WorkerNode[];
    selectedNode: WorkerNode | null;
    selectNode: (nodeId: string | null) => void;
    networkStats: {
        totalNodes: number;
        activeNodes: number;
        computeNodes: number;
        aggregators: number;
        verifiers: number;
        totalStaked: bigint;
        totalProofs: number;
        averageReputation: number;
        networkUptime: number;
    };
    topology: NetworkTopology;
    isLoading: boolean;
    isConnected: boolean;
    error: string | null;
    refetch: () => Promise<void>;
    getNodeById: (nodeId: string) => WorkerNode | undefined;
    getNodesByType: (type: WorkerNode['type']) => WorkerNode[];
    getNodesByStatus: (status: WorkerNode['status']) => WorkerNode[];
    updateFilter: (filter: Partial<NodeFilter>) => void;
}

// ============================================================================
// Hook Implementation
// ============================================================================

export function useNodes(options: UseNodesOptions = {}): UseNodesReturn {
    const {
        autoRefresh = true,
        refreshInterval = 5000,
        enableWebSocket = true,
        filter: initialFilter = {},
        includeOffline = true,
    } = options;

    // State
    const [nodes, setNodes] = useState<WorkerNode[]>([]);
    const [connections, setConnections] = useState<NetworkConnection[]>([]);
    const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
    const [filter, setFilter] = useState<NodeFilter>(initialFilter);
    const [isLoading, setIsLoading] = useState(true);
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Refs
    const wsUnsubscribeRef = useRef<(() => void) | null>(null);
    const refreshIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const lastUpdateRef = useRef<number>(0);
    const initialLoadDone = useRef(false);

    // Contract data
    const { proofEvents, stakedEvents, slashedEvents } = useContractEvents();
    const { slashingRecordCount: _slashingRecordCount } = useContractState();

    // API client
    const apiClient = useMemo(() => getApiClient(), []);

    // ========================================================================
    // Data Fetching
    // ========================================================================

    const buildNodesFromEvents = useCallback(() => {
        const nodeMap = new Map<string, WorkerNode>();

        // Process staked events
        stakedEvents.forEach((event) => {
            const addr = event.prover;
            if (!nodeMap.has(addr)) {
                const types: WorkerNode['type'][] = ['compute', 'aggregator', 'verifier'];
                nodeMap.set(addr, {
                    id: `node-${addr.slice(2, 10)}`,
                    address: addr,
                    type: types[nodeMap.size % 3],
                    status: 'active',
                    lastSeen: event.timestamp,
                    lastHeartbeat: event.timestamp,
                    stakedAmount: event.amount,
                    proofsSubmitted: 0,
                    proofsVerified: 0,
                    proofsFailed: 0,
                    roundsParticipated: 0,
                    roundsCompleted: 0,
                    reputation: 100,
                    metrics: {
                        cpu: 0,
                        memory: 0,
                        gpu: undefined,
                        gpuMemory: undefined,
                        networkIn: 0,
                        networkOut: 0,
                        temperature: undefined,
                    },
                    capabilities: {
                        canTrain: true,
                        canAggregate: nodeMap.size % 3 === 1,
                        canProve: nodeMap.size % 3 === 2,
                        gpuModel: undefined,
                        gpuMemoryMb: 0,
                        maxBatchSize: 64,
                    },
                    location: undefined,
                    earningsTotal: BigInt(0),
                    slashed: false,
                });
            } else {
                const node = nodeMap.get(addr)!;
                node.stakedAmount += event.amount;
                node.lastSeen = Math.max(node.lastSeen, event.timestamp);
            }
        });

        // Process proof events (job completion increases reputation)
        proofEvents.forEach((event) => {
            const addr = event.prover;
            if (nodeMap.has(addr)) {
                const node = nodeMap.get(addr)!;
                node.proofsSubmitted++;
                node.proofsVerified++;
                node.roundsParticipated++;
                node.roundsCompleted++;
                node.lastSeen = Math.max(node.lastSeen, event.timestamp);
                node.lastHeartbeat = event.timestamp;
                node.status = 'active';
            }
        });

        // Process slashed events (slashing decreases reputation significantly)
        slashedEvents.forEach((event) => {
            const addr = event.prover;
            if (nodeMap.has(addr)) {
                const node = nodeMap.get(addr)!;
                node.slashed = true;
                node.proofsFailed++;
            }
        });

        // Compute reputation scores matching the on-chain formula (basis points → 0-100 scale):
        //   Base: 50 + jobBonus (5/job, cap 30) + stakeBonus (scaled, cap 20) - slashPenalty (25/slash)
        nodeMap.forEach((node) => {
            const base = 50;
            const jobBonus = Math.min(node.roundsCompleted * 5, 30);
            // Stake bonus: +1 per 0.001 ETH equivalent, capped at 20
            const stakeEth = Number(node.stakedAmount) / 1e18;
            const stakeBonus = Math.min(Math.floor(stakeEth * 1000) / 10, 20);
            const slashPenalty = node.proofsFailed * 25;
            node.reputation = Math.max(0, Math.min(100, base + jobBonus + stakeBonus - slashPenalty));
        });

        // Update status based on last seen time
        const now = Date.now();
        nodeMap.forEach((node) => {
            const timeSinceLastSeen = now - node.lastSeen;
            if (timeSinceLastSeen > 3600000) { // 1 hour
                node.status = 'offline';
            } else if (timeSinceLastSeen > 300000) { // 5 minutes
                node.status = 'idle';
            } else if (timeSinceLastSeen > 60000) { // 1 minute
                node.status = 'syncing';
            }
        });

        return nodeMap;
    }, [stakedEvents, proofEvents, slashedEvents]);

    const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

    const fetchNodes = useCallback(async () => {
        try {
            if (!initialLoadDone.current) {
                setIsLoading(true);
            }
            setError(null);

            // Build nodes from contract events
            const eventNodes = buildNodesFromEvents();

            if (eventNodes.size === 0) {
                // Fallback: fetch workers from backend API
                try {
                    const controller = new AbortController();
                    const timeout = setTimeout(() => controller.abort(), 3000);
                    const res = await fetch(`${API_BASE}/api/workers`, { signal: controller.signal });
                    clearTimeout(timeout);
                    if (res.ok) {
                        const data = await res.json();
                        // eslint-disable-next-line @typescript-eslint/no-explicit-any
                        const workers: any[] =
                            Array.isArray(data) ? data : (data.workers ?? []);

                        const backendNodes: WorkerNode[] = workers.map((w) => {
                            const addr = w.address || w.addr || w.id || '0x0';
                            const caps = w.capabilities || {};

                            // Derive node type from real capabilities
                            let nodeType: WorkerNode['type'] = 'compute';
                            if (caps.can_aggregate) nodeType = 'aggregator';
                            else if (caps.can_prove) nodeType = 'verifier';

                            // Map backend status to frontend status
                            let status: WorkerNode['status'] = 'active';
                            const rawStatus = (w.status || '').toLowerCase();
                            if (rawStatus === 'offline') status = 'offline';
                            else if (rawStatus === 'idle') status = 'idle';
                            else if (rawStatus === 'busy' || rawStatus === 'training') status = 'training';

                            // Convert reputation_score (0.0-1.0) to 0-100 scale
                            const repScore = typeof w.reputation_score === 'number'
                                ? Math.round(w.reputation_score * 100)
                                : 50;

                            // Convert timestamps: backend uses unix seconds (f64), frontend uses ms
                            const lastHb = typeof w.last_heartbeat === 'number'
                                ? w.last_heartbeat * 1000
                                : Date.now();
                            const registeredAt = typeof w.registered_at === 'number'
                                ? w.registered_at * 1000
                                : Date.now();

                            return {
                                id: w.id || `node-${addr.slice(0, 10)}`,
                                address: addr,
                                type: nodeType,
                                status,
                                lastSeen: lastHb,
                                lastHeartbeat: lastHb,
                                stakedAmount: w.stake_eth ? BigInt(Math.floor(w.stake_eth * 1e18)) : BigInt(0),
                                proofsSubmitted: Number(w.rounds_completed || 0),
                                proofsVerified: Number(w.rounds_completed || 0),
                                proofsFailed: 0,
                                roundsParticipated: Number(w.rounds_participated || 0),
                                roundsCompleted: Number(w.rounds_completed || 0),
                                reputation: repScore,
                                metrics: {
                                    cpu: Number(w.cpu_load || 0),
                                    memory: Number(w.memory_mb || 0),
                                    gpu: undefined,
                                    gpuMemory: undefined,
                                    networkIn: 0,
                                    networkOut: 0,
                                    temperature: undefined,
                                },
                                capabilities: {
                                    canTrain: caps.can_train ?? true,
                                    canAggregate: caps.can_aggregate ?? false,
                                    canProve: caps.can_prove ?? false,
                                    gpuModel: caps.gpu_model,
                                    gpuMemoryMb: Number(caps.gpu_memory_mb || 0),
                                    maxBatchSize: Number(caps.max_batch_size || 64),
                                },
                                location: undefined,
                                earningsTotal: w.earnings_wei ? BigInt(w.earnings_wei) : BigInt(0),
                                slashed: false,
                                // Extra live fields for display
                                successRate: typeof w.success_rate === 'number' ? w.success_rate : 0,
                                registeredAt,
                                endpoint: w.endpoint || '',
                                partyIndex: typeof w.party_index === 'number' ? w.party_index : undefined,
                            } as WorkerNode;
                        });

                        setNodes(backendNodes);

                        // Generate mesh connections
                        const conns: NetworkConnection[] = [];
                        for (let i = 0; i < backendNodes.length; i++) {
                            for (let j = i + 1; j < backendNodes.length; j++) {
                                conns.push({
                                    from: backendNodes[i].id,
                                    to: backendNodes[j].id,
                                    latency: 10 + Math.floor(Math.random() * 40),
                                    bandwidth: 100,
                                    status: 'active',
                                    lastUpdated: Date.now(),
                                });
                            }
                        }
                        setConnections(conns);
                    } else {
                        setNodes([]);
                        setConnections([]);
                    }
                } catch {
                    setNodes([]);
                    setConnections([]);
                }
            } else {
                setNodes(Array.from(eventNodes.values()));
            }

            lastUpdateRef.current = Date.now();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to fetch nodes');
        } finally {
            setIsLoading(false);
            initialLoadDone.current = true;
        }
    }, [buildNodesFromEvents, API_BASE]);

    // ========================================================================
    // WebSocket Setup
    // ========================================================================

    const setupWebSocket = useCallback(async () => {
        if (!enableWebSocket) return;

        try {
            await apiClient.connectWebSocket();
            setIsConnected(true);

            // Subscribe to node updates
            apiClient.subscribeToChannel('nodes');

            // Handle node update messages
            const unsubNodeUpdate = apiClient.onWebSocketMessage<NodeInfo>(
                'node_update',
                (message) => {
                    const nodeData = message.data;
                    setNodes((prevNodes) => {
                        const existingIndex = prevNodes.findIndex(
                            (n) => n.address === nodeData.address
                        );

                        const updatedNode: WorkerNode = {
                            id: nodeData.id,
                            address: nodeData.address,
                            type: nodeData.type,
                            status: nodeData.status as WorkerNode['status'],
                            lastSeen: nodeData.lastSeen,
                            lastHeartbeat: nodeData.lastHeartbeat,
                            stakedAmount: nodeData.stake.amount,
                            proofsSubmitted: nodeData.performance.proofsSubmitted,
                            proofsVerified: nodeData.performance.proofsVerified,
                            proofsFailed: nodeData.performance.proofsFailed,
                            roundsParticipated: nodeData.performance.roundsParticipated,
                            roundsCompleted: nodeData.performance.roundsCompleted,
                            reputation: nodeData.performance.reputation,
                            metrics: {
                                cpu: nodeData.metrics.cpu,
                                memory: nodeData.metrics.memory,
                                gpu: nodeData.metrics.gpu,
                                gpuMemory: nodeData.metrics.gpuMemory,
                                networkIn: nodeData.metrics.networkIn,
                                networkOut: nodeData.metrics.networkOut,
                                temperature: nodeData.metrics.temperature,
                            },
                            capabilities: {
                                canTrain: nodeData.capabilities.canTrain,
                                canAggregate: nodeData.capabilities.canAggregate,
                                canProve: nodeData.capabilities.canProve,
                                gpuModel: nodeData.capabilities.gpuModel,
                                gpuMemoryMb: nodeData.capabilities.gpuMemoryMb,
                                maxBatchSize: nodeData.capabilities.maxBatchSize,
                            },
                            location: nodeData.location,
                            earningsTotal: nodeData.performance.totalEarnings,
                            slashed: nodeData.stake.slashed,
                        };

                        if (existingIndex >= 0) {
                            const newNodes = [...prevNodes];
                            newNodes[existingIndex] = updatedNode;
                            return newNodes;
                        } else {
                            return [...prevNodes, updatedNode];
                        }
                    });
                }
            );

            // Handle worker_joined — add new worker node to the network graph
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const unsubWorkerJoined = apiClient.onWebSocketMessage<any>(
                'worker_joined',
                (message) => {
                    const w = message.data?.worker || (message as any).event?.worker;
                    if (!w?.id) return;

                    setNodes((prev) => {
                        // Don't add if already exists
                        if (prev.some((n) => n.id === w.id)) return prev;

                        const caps = w.capabilities || {};
                        let nodeType: WorkerNode['type'] = 'compute';
                        if (caps.can_aggregate) nodeType = 'aggregator';
                        else if (caps.can_prove) nodeType = 'verifier';

                        const rawStatus = (w.status || 'idle').toLowerCase();
                        let wsStatus: WorkerNode['status'] = 'idle';
                        if (rawStatus === 'busy' || rawStatus === 'training') wsStatus = 'training';
                        else if (rawStatus === 'active') wsStatus = 'active';

                        const newNode: WorkerNode = {
                            id: w.id,
                            address: w.endpoint || w.id,
                            type: nodeType,
                            status: wsStatus,
                            lastSeen: Date.now(),
                            lastHeartbeat: Date.now(),
                            stakedAmount: BigInt(0),
                            proofsSubmitted: 0,
                            proofsVerified: 0,
                            proofsFailed: 0,
                            roundsParticipated: 0,
                            roundsCompleted: 0,
                            reputation: w.reputation_score ? Math.round(w.reputation_score * 100) : 50,
                            metrics: { cpu: w.cpu_load || 0, memory: 0, networkIn: 0, networkOut: 0 },
                            capabilities: {
                                canTrain: caps.can_train ?? true,
                                canAggregate: caps.can_aggregate ?? false,
                                canProve: caps.can_prove ?? false,
                                gpuMemoryMb: 0,
                                maxBatchSize: caps.max_batch_size ?? 64,
                            },
                            earningsTotal: BigInt(0),
                            slashed: false,
                            endpoint: w.endpoint,
                            partyIndex: w.party_index,
                        };
                        return [...prev, newNode];
                    });

                    // Also add mesh connections to existing nodes
                    setConnections((prev) => {
                        const newConns: NetworkConnection[] = [];
                        const existingNodeIds = new Set(prev.flatMap((c) => [c.from, c.to]));
                        existingNodeIds.forEach((existingId) => {
                            newConns.push({
                                from: existingId,
                                to: w.id,
                                latency: 10 + Math.floor(Math.random() * 40),
                                bandwidth: 100,
                                status: 'active',
                                lastUpdated: Date.now(),
                            });
                        });
                        return [...prev, ...newConns];
                    });
                }
            );

            // Handle worker_left — mark worker as offline in the network graph
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const unsubWorkerLeft = apiClient.onWebSocketMessage<any>(
                'worker_left',
                (message) => {
                    const workerId = message.data?.worker_id || (message as any).event?.worker_id;
                    if (!workerId) return;

                    setNodes((prev) =>
                        prev.map((n) =>
                            n.id === workerId
                                ? { ...n, status: 'offline' as const, lastSeen: Date.now() }
                                : n
                        )
                    );

                    // Mark connections involving this worker as offline
                    setConnections((prev) =>
                        prev.map((c) =>
                            c.from === workerId || c.to === workerId
                                ? { ...c, status: 'offline' as const, lastUpdated: Date.now() }
                                : c
                        )
                    );
                }
            );

            // Combine unsubscribe functions
            wsUnsubscribeRef.current = () => {
                unsubNodeUpdate();
                unsubWorkerJoined();
                unsubWorkerLeft();
            };
        } catch (err) {
            console.warn('WebSocket connection failed, falling back to polling:', err);
            setIsConnected(false);
        }
    }, [enableWebSocket, apiClient]);

    // ========================================================================
    // Effects
    // ========================================================================

    // Initial fetch and WebSocket setup
    useEffect(() => {
        fetchNodes();
        setupWebSocket();

        return () => {
            wsUnsubscribeRef.current?.();
            apiClient.disconnectWebSocket();
        };
    }, [fetchNodes, setupWebSocket, apiClient]);

    // Auto refresh — poll backend for updated node list
    useEffect(() => {
        if (!autoRefresh) return;
        const interval = setInterval(() => {
            fetchNodes();
        }, refreshInterval);
        return () => clearInterval(interval);
    }, [autoRefresh, refreshInterval, fetchNodes]);

    // Update nodes when contract events change
    useEffect(() => {
        if (stakedEvents.length > 0 || proofEvents.length > 0 || slashedEvents.length > 0) {
            fetchNodes();
        }
    }, [stakedEvents, proofEvents, slashedEvents, fetchNodes]);

    // ========================================================================
    // Computed Values
    // ========================================================================

    const filteredNodes = useMemo(() => {
        let result = nodes;

        if (!includeOffline) {
            result = result.filter((n) => n.status !== 'offline');
        }

        if (filter.type) {
            result = result.filter((n) => n.type === filter.type);
        }

        if (filter.status) {
            result = result.filter((n) => n.status === filter.status);
        }

        if (filter.minStake !== undefined) {
            result = result.filter((n) => n.stakedAmount >= filter.minStake!);
        }

        if (filter.minReputation !== undefined) {
            result = result.filter((n) => n.reputation >= filter.minReputation!);
        }

        if (filter.hasGpu !== undefined) {
            result = result.filter((n) =>
                filter.hasGpu ? n.metrics.gpu !== undefined : n.metrics.gpu === undefined
            );
        }

        if (filter.region) {
            result = result.filter((n) => n.location?.region === filter.region);
        }

        return result;
    }, [nodes, filter, includeOffline]);

    const selectedNode = useMemo(
        () => nodes.find((n) => n.id === selectedNodeId) || null,
        [nodes, selectedNodeId]
    );

    const networkStats = useMemo(() => {
        const activeNodes = nodes.filter(
            (n) => n.status === 'active' || n.status === 'proving' || n.status === 'training'
        );
        const totalStaked = nodes.reduce((sum, n) => sum + n.stakedAmount, BigInt(0));
        const totalProofs = nodes.reduce((sum, n) => sum + n.proofsSubmitted, 0);
        const avgReputation =
            nodes.length > 0
                ? nodes.reduce((sum, n) => sum + n.reputation, 0) / nodes.length
                : 0;

        const onlineCount = nodes.filter((n) => n.status !== 'offline').length;
        const networkUptime = nodes.length > 0 ? onlineCount / nodes.length : 0;

        return {
            totalNodes: nodes.length,
            activeNodes: activeNodes.length,
            computeNodes: nodes.filter((n) => n.type === 'compute').length,
            aggregators: nodes.filter((n) => n.type === 'aggregator').length,
            verifiers: nodes.filter((n) => n.type === 'verifier').length,
            totalStaked,
            totalProofs,
            averageReputation: avgReputation,
            networkUptime,
        };
    }, [nodes]);

    const topology = useMemo((): NetworkTopology => {
        return {
            nodes,
            connections,
            aggregators: nodes.filter((n) => n.type === 'aggregator').map((n) => n.id),
            verifiers: nodes.filter((n) => n.type === 'verifier').map((n) => n.id),
            computeNodes: nodes.filter((n) => n.type === 'compute').map((n) => n.id),
        };
    }, [nodes, connections]);

    // ========================================================================
    // Actions
    // ========================================================================

    const selectNode = useCallback((nodeId: string | null) => {
        setSelectedNodeId(nodeId);
    }, []);

    const getNodeById = useCallback(
        (nodeId: string) => nodes.find((n) => n.id === nodeId),
        [nodes]
    );

    const getNodesByType = useCallback(
        (type: WorkerNode['type']) => nodes.filter((n) => n.type === type),
        [nodes]
    );

    const getNodesByStatus = useCallback(
        (status: WorkerNode['status']) => nodes.filter((n) => n.status === status),
        [nodes]
    );

    const updateFilter = useCallback((newFilter: Partial<NodeFilter>) => {
        setFilter((prev) => ({ ...prev, ...newFilter }));
    }, []);

    // ========================================================================
    // Return
    // ========================================================================

    return {
        nodes,
        filteredNodes,
        selectedNode,
        selectNode,
        networkStats,
        topology,
        isLoading,
        isConnected,
        error,
        refetch: fetchNodes,
        getNodeById,
        getNodesByType,
        getNodesByStatus,
        updateFilter,
    };
}

// ============================================================================
// Additional Node Hooks
// ============================================================================

/**
 * Hook for tracking a specific node's metrics in real-time
 */
export function useNodeMetrics(nodeId: string, pollingInterval = 2000) {
    const [metrics, setMetrics] = useState<NodeMetrics | null>(null);
    const [history, setHistory] = useState<{ timestamp: number; metrics: NodeMetrics }[]>([]);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        setIsLoading(false);
        // Real metrics come from WebSocket node_update messages
    }, [nodeId]);

    return { metrics, history, isLoading };
}

/**
 * Hook for network health monitoring
 */
export function useNetworkHealth() {
    const { nodes, networkStats, isLoading, error } = useNodes({ autoRefresh: true });

    const healthScore = useMemo(() => {
        if (nodes.length === 0) return 0;

        const factors = [
            networkStats.networkUptime * 30, // 30% weight for uptime
            (networkStats.averageReputation / 100) * 25, // 25% weight for reputation
            Math.min(networkStats.activeNodes / 10, 1) * 20, // 20% weight for active nodes (up to 10)
            Math.min(networkStats.totalProofs / 1000, 1) * 15, // 15% weight for proofs
            (networkStats.aggregators > 0 && networkStats.verifiers > 0 ? 1 : 0) * 10, // 10% weight for role diversity
        ];

        return Math.round(factors.reduce((sum, f) => sum + f, 0));
    }, [nodes, networkStats]);

    const healthStatus = useMemo(() => {
        if (healthScore >= 80) return 'healthy';
        if (healthScore >= 60) return 'good';
        if (healthScore >= 40) return 'degraded';
        if (healthScore >= 20) return 'poor';
        return 'critical';
    }, [healthScore]);

    const issues = useMemo(() => {
        const result: string[] = [];

        if (networkStats.networkUptime < 0.9) {
            result.push('Network uptime below 90%');
        }
        if (networkStats.averageReputation < 80) {
            result.push('Average node reputation is low');
        }
        if (networkStats.aggregators === 0) {
            result.push('No aggregator nodes available');
        }
        if (networkStats.verifiers === 0) {
            result.push('No verifier nodes available');
        }
        if (networkStats.activeNodes < 3) {
            result.push('Insufficient active nodes');
        }

        return result;
    }, [networkStats]);

    return {
        healthScore,
        healthStatus,
        issues,
        networkStats,
        isLoading,
        error,
    };
}

// Default export
export default useNodes;
