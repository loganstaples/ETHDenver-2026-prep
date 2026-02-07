'use client';

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { useContractEvents, useContractState } from './useContract';
import {
    type NodeInfo,
    type NodeMetrics,
    getApiClient,
    generateMockNodeInfo,
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
                const mockData = generateMockNodeInfo(addr, nodeMap.size);
                nodeMap.set(addr, {
                    id: `node-${addr.slice(2, 10)}`,
                    address: addr,
                    type: mockData.type,
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
                        cpu: mockData.metrics.cpu,
                        memory: mockData.metrics.memory,
                        gpu: mockData.metrics.gpu,
                        gpuMemory: mockData.metrics.gpuMemory,
                        networkIn: mockData.metrics.networkIn,
                        networkOut: mockData.metrics.networkOut,
                        temperature: mockData.metrics.temperature,
                    },
                    capabilities: {
                        canTrain: mockData.capabilities.canTrain,
                        canAggregate: mockData.capabilities.canAggregate,
                        canProve: mockData.capabilities.canProve,
                        gpuModel: mockData.capabilities.gpuModel,
                        gpuMemoryMb: mockData.capabilities.gpuMemoryMb,
                        maxBatchSize: mockData.capabilities.maxBatchSize,
                    },
                    location: mockData.location,
                    earningsTotal: BigInt(0),
                    slashed: false,
                });
            } else {
                const node = nodeMap.get(addr)!;
                node.stakedAmount += event.amount;
                node.lastSeen = Math.max(node.lastSeen, event.timestamp);
            }
        });

        // Process proof events
        proofEvents.forEach((event) => {
            const addr = event.prover;
            if (nodeMap.has(addr)) {
                const node = nodeMap.get(addr)!;
                node.proofsSubmitted++;
                node.proofsVerified++;
                node.roundsParticipated++;
                node.lastSeen = Math.max(node.lastSeen, event.timestamp);
                node.lastHeartbeat = event.timestamp;
                node.reputation = Math.min(100, node.reputation + 0.5);
                node.status = 'active';
            }
        });

        // Process slashed events
        slashedEvents.forEach((event) => {
            const addr = event.prover;
            if (nodeMap.has(addr)) {
                const node = nodeMap.get(addr)!;
                node.reputation = Math.max(0, node.reputation - 25);
                node.slashed = true;
                node.proofsFailed++;
            }
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

    const generateDemoNodes = useCallback((): WorkerNode[] => {
        const demoAddresses = [
            '0x742d35Cc6634C0532925a3b844Bc9e7595f01231',
            '0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199',
            '0xdD2FD4581271e230360230F9337D5c0430Bf44C0',
            '0xbDA5747bFD65F08deb54cb465eB87D40e51B197E',
            '0x2546BcD3c84621e976D8185a91A922aE77ECEc30',
            '0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266',
            '0x70997970C51812dc3A010C7d01b50e0d17dc79C8',
            '0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC',
        ];

        return demoAddresses.map((addr, index) => {
            const mockData = generateMockNodeInfo(addr, index);
            const statuses: WorkerNode['status'][] = ['active', 'active', 'active', 'proving', 'training', 'idle', 'syncing', 'active'];

            return {
                id: `node-${addr.slice(2, 10)}`,
                address: addr,
                type: mockData.type,
                status: statuses[index % statuses.length],
                lastSeen: Date.now() - Math.random() * 60000,
                lastHeartbeat: Date.now() - Math.random() * 30000,
                stakedAmount: BigInt(Math.floor((1 + Math.random() * 9) * 1e18)),
                proofsSubmitted: Math.floor(Math.random() * 150),
                proofsVerified: Math.floor(Math.random() * 140),
                proofsFailed: Math.floor(Math.random() * 5),
                roundsParticipated: Math.floor(Math.random() * 80),
                roundsCompleted: Math.floor(Math.random() * 75),
                reputation: 75 + Math.random() * 25,
                metrics: {
                    cpu: mockData.metrics.cpu,
                    memory: mockData.metrics.memory,
                    gpu: mockData.metrics.gpu,
                    gpuMemory: mockData.metrics.gpuMemory,
                    networkIn: mockData.metrics.networkIn,
                    networkOut: mockData.metrics.networkOut,
                    temperature: mockData.metrics.temperature,
                },
                capabilities: {
                    canTrain: mockData.capabilities.canTrain,
                    canAggregate: mockData.capabilities.canAggregate,
                    canProve: mockData.capabilities.canProve,
                    gpuModel: mockData.capabilities.gpuModel,
                    gpuMemoryMb: mockData.capabilities.gpuMemoryMb,
                    maxBatchSize: mockData.capabilities.maxBatchSize,
                },
                location: mockData.location,
                earningsTotal: BigInt(Math.floor(Math.random() * 5 * 1e18)),
                slashed: index === 5, // One slashed node for demo
            };
        });
    }, []);

    const fetchNodes = useCallback(async () => {
        try {
            setIsLoading(true);
            setError(null);

            // Build nodes from contract events
            const eventNodes = buildNodesFromEvents();

            // If we have no event data, use demo data
            if (eventNodes.size === 0) {
                const demoNodes = generateDemoNodes();
                setNodes(demoNodes);

                // Generate demo connections
                const demoConnections: NetworkConnection[] = [];
                const aggregatorNodes = demoNodes.filter(n => n.type === 'aggregator');
                demoNodes.forEach((node, idx) => {
                    if (node.type !== 'aggregator' && aggregatorNodes.length > 0) {
                        const aggregator = aggregatorNodes[idx % aggregatorNodes.length];
                        demoConnections.push({
                            from: node.id,
                            to: aggregator.id,
                            latency: 10 + Math.random() * 50,
                            bandwidth: 100 + Math.random() * 400,
                            status: node.status === 'offline' ? 'offline' : 'active',
                            lastUpdated: Date.now(),
                        });
                    }
                });
                setConnections(demoConnections);
            } else {
                setNodes(Array.from(eventNodes.values()));
            }

            lastUpdateRef.current = Date.now();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to fetch nodes');
        } finally {
            setIsLoading(false);
        }
    }, [buildNodesFromEvents, generateDemoNodes]);

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
            wsUnsubscribeRef.current = apiClient.onWebSocketMessage<NodeInfo>(
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

    // Auto refresh
    useEffect(() => {
        if (!autoRefresh) return;

        refreshIntervalRef.current = setInterval(() => {
            // Simulate real-time metric updates
            setNodes((prevNodes) =>
                prevNodes.map((node) => ({
                    ...node,
                    metrics: {
                        ...node.metrics,
                        cpu: Math.max(5, Math.min(95, node.metrics.cpu + (Math.random() - 0.5) * 10)),
                        memory: Math.max(10, Math.min(95, node.metrics.memory + (Math.random() - 0.5) * 5)),
                        gpu: node.metrics.gpu
                            ? Math.max(10, Math.min(100, node.metrics.gpu + (Math.random() - 0.5) * 15))
                            : undefined,
                        networkIn: Math.max(0, node.metrics.networkIn + (Math.random() - 0.5) * 200),
                        networkOut: Math.max(0, node.metrics.networkOut + (Math.random() - 0.5) * 100),
                    },
                    lastHeartbeat: node.status !== 'offline' ? Date.now() : node.lastHeartbeat,
                }))
            );
        }, refreshInterval);

        return () => {
            if (refreshIntervalRef.current) {
                clearInterval(refreshIntervalRef.current);
            }
        };
    }, [autoRefresh, refreshInterval]);

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
        // Simulate real-time metrics polling
        const interval = setInterval(() => {
            const newMetrics: NodeMetrics = {
                cpu: 20 + Math.random() * 60,
                memory: 30 + Math.random() * 50,
                gpu: Math.random() > 0.3 ? 40 + Math.random() * 50 : undefined,
                gpuMemory: Math.random() > 0.3 ? 30 + Math.random() * 60 : undefined,
                networkIn: Math.random() * 1000,
                networkOut: Math.random() * 500,
                diskUsage: 20 + Math.random() * 40,
                temperature: 40 + Math.random() * 30,
            };

            setMetrics(newMetrics);
            setHistory((prev) => [
                ...prev.slice(-59), // Keep last 60 data points
                { timestamp: Date.now(), metrics: newMetrics },
            ]);
            setIsLoading(false);
        }, pollingInterval);

        return () => clearInterval(interval);
    }, [nodeId, pollingInterval]);

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
