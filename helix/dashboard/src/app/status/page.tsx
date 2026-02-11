'use client';

import { useState, useEffect, useCallback } from 'react';
import { useContractState, useModel, useContractEvents } from '@/hooks/useContract';
import {
    getApiClient,
    generateMockNodeInfo,
    generateMockProofInfo,
    generateMockTrainingMetrics,
    type NodeInfo,
    type ProofInfo,
    type TrainingMetrics,
    type TrainingSession,
    type NetworkStats,
} from '@/lib/api';
import { formatEther } from 'viem';

// ============================================================================
// Types
// ============================================================================

interface StatusData {
    trainingSessions: TrainingSession[];
    workers: NodeInfo[];
    proofs: ProofInfo[];
    metrics: TrainingMetrics | null;
    networkStats: NetworkStats | null;
}

interface ErrorLogEntry {
    id: string;
    timestamp: number;
    level: 'error' | 'warning' | 'info';
    message: string;
    source: string;
}

// ============================================================================
// Polling Configuration
// ============================================================================

const REFRESH_INTERVAL = 5000; // 5 seconds

// ============================================================================
// Mock Data Generators for Demo
// ============================================================================

function generateMockTrainingSessions(): TrainingSession[] {
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

function generateMockNetworkStats(): NetworkStats {
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

function generateMockErrorLogs(): ErrorLogEntry[] {
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

// ============================================================================
// Component: StatusIndicator
// ============================================================================

function StatusIndicator({ status }: { status: 'online' | 'offline' | 'syncing' | 'proving' | 'training' }) {
    const colors = {
        online: 'bg-white',
        offline: 'bg-neutral-600',
        syncing: 'bg-white/60',
        proving: 'bg-white/80',
        training: 'bg-white/80',
    };

    const labels = {
        online: 'Online',
        offline: 'Offline',
        syncing: 'Syncing',
        proving: 'Proving',
        training: 'Training',
    };

    return (
        <span className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${colors[status]}`} />
            <span className="text-xs text-[#888]">{labels[status]}</span>
        </span>
    );
}

// ============================================================================
// Component: RefreshIndicator
// ============================================================================

function RefreshIndicator({ lastRefresh, isRefreshing }: { lastRefresh: number; isRefreshing: boolean }) {
    const secondsAgo = Math.floor((Date.now() - lastRefresh) / 1000);

    return (
        <div className="flex items-center gap-2 text-xs text-[#666]">
            {isRefreshing ? (
                <span className="flex items-center gap-1">
                    <span className="w-3 h-3 border-2 border-[#666] border-t-transparent rounded-full animate-spin" />
                    Refreshing...
                </span>
            ) : (
                <span>Updated {secondsAgo}s ago</span>
            )}
            <span className="text-[#555]">|</span>
            <span>Auto-refresh: {REFRESH_INTERVAL / 1000}s</span>
        </div>
    );
}

// ============================================================================
// Component: SectionCard
// ============================================================================

function SectionCard({ title, children, className = '' }: { title: string; children: React.ReactNode; className?: string }) {
    return (
        <div className={`bg-helix-bg border border-helix-border rounded-md p-4 ${className}`}>
            <h3 className="text-sm font-semibold text-[#aaa] mb-3 uppercase tracking-wide">{title}</h3>
            {children}
        </div>
    );
}

// ============================================================================
// Component: TrainingSessionsSection
// ============================================================================

function TrainingSessionsSection({ sessions }: { sessions: TrainingSession[] }) {
    if (sessions.length === 0) {
        return (
            <SectionCard title="Active Training Sessions">
                <p className="text-[#666] text-sm">No active training sessions</p>
            </SectionCard>
        );
    }

    return (
        <SectionCard title="Active Training Sessions">
            <div className="space-y-3">
                {sessions.map((session) => (
                    <div key={session.id} className="flex items-center justify-between py-2 border-b border-helix-border last:border-0">
                        <div>
                            <div className="font-medium text-white">{session.modelName}</div>
                            <div className="text-xs text-[#666]">
                                Model #{session.modelId.toString()} | Started {new Date(session.startedAt).toLocaleTimeString()}
                            </div>
                        </div>
                        <div className="text-right">
                            <div className={`text-sm font-medium ${
                                session.status === 'training' ? 'text-white' :
                                session.status === 'completed' ? 'text-[#888]' :
                                session.status === 'paused' ? 'text-[#666]' :
                                'text-[#666]'
                            }`}>
                                {session.status.charAt(0).toUpperCase() + session.status.slice(1)}
                            </div>
                            <div className="text-xs text-[#666]">
                                Epoch {session.metrics.currentEpoch}/{session.config.totalEpochs}
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </SectionCard>
    );
}

// ============================================================================
// Component: WorkerListSection
// ============================================================================

function WorkerListSection({ workers }: { workers: NodeInfo[] }) {
    const onlineCount = workers.filter(w => w.status !== 'offline').length;

    return (
        <SectionCard title={`Workers (${onlineCount}/${workers.length} Online)`}>
            <div className="space-y-2 max-h-64 overflow-y-auto">
                {workers.length === 0 ? (
                    <p className="text-[#666] text-sm">No workers connected</p>
                ) : (
                    workers.map((worker) => (
                        <div key={worker.id} className="flex items-center justify-between py-1.5 border-b border-helix-border last:border-0">
                            <div className="flex items-center gap-2">
                                <span className="font-mono text-xs text-[#aaa]">
                                    {worker.address.slice(0, 6)}...{worker.address.slice(-4)}
                                </span>
                                <span className={`text-xs px-1.5 py-0.5 rounded bg-white/5 border border-white/10 text-[#888]`}>
                                    {worker.type}
                                </span>
                            </div>
                            <StatusIndicator status={worker.status} />
                        </div>
                    ))
                )}
            </div>
        </SectionCard>
    );
}

// ============================================================================
// Component: RoundStateSection
// ============================================================================

function RoundStateSection({ modelId }: { modelId: bigint }) {
    const { model, isLoading: modelLoading } = useModel(modelId);

    if (modelLoading) {
        return (
            <SectionCard title="Current Round State">
                <p className="text-[#666] text-sm">Loading...</p>
            </SectionCard>
        );
    }

    if (!model) {
        return (
            <SectionCard title="Current Round State">
                <p className="text-[#666] text-sm">No model data available</p>
            </SectionCard>
        );
    }

    return (
        <SectionCard title="Current Round State">
            <div className="grid grid-cols-2 gap-3">
                <div>
                    <div className="text-xs text-[#666] uppercase">Round Number</div>
                    <div className="text-[15px] font-bold text-white">{model.currentRound.toString()}</div>
                </div>
                <div>
                    <div className="text-xs text-[#666] uppercase">Model Active</div>
                    <div className={`text-[15px] font-bold ${model.active ? 'text-white' : 'text-[#666]'}`}>
                        {model.active ? 'Yes' : 'No'}
                    </div>
                </div>
                <div className="col-span-2">
                    <div className="text-xs text-[#666] uppercase">Model Commitment</div>
                    <div className="font-mono text-xs text-[#aaa] break-all">
                        0x{model.currentCommitment.toString(16).padStart(64, '0')}
                    </div>
                </div>
            </div>
        </SectionCard>
    );
}

// ============================================================================
// Component: ProofSubmissionsSection
// ============================================================================

function ProofSubmissionsSection({ proofs }: { proofs: ProofInfo[] }) {
    const recentProofs = proofs.slice(0, 10);

    return (
        <SectionCard title="Recent Proof Submissions">
            {recentProofs.length === 0 ? (
                <p className="text-[#666] text-sm">No recent proofs</p>
            ) : (
                <div className="space-y-2 max-h-48 overflow-y-auto">
                    {recentProofs.map((proof) => (
                        <div key={proof.id} className="flex items-center justify-between py-1.5 border-b border-helix-border last:border-0">
                            <div>
                                <div className="font-mono text-xs text-[#aaa]">
                                    {proof.hash.slice(0, 10)}...{proof.hash.slice(-6)}
                                </div>
                                <div className="text-xs text-[#666]">
                                    Round {proof.roundId.toString()} | {proof.type}
                                </div>
                            </div>
                            <div className="text-right">
                                <div className={`text-xs font-medium ${
                                    proof.status === 'verified' ? 'text-white' :
                                    proof.status === 'pending' ? 'text-[#888]' :
                                    proof.status === 'failed' ? 'text-[#666]' :
                                    'text-[#666]'
                                }`}>
                                    {proof.status}
                                </div>
                                <div className="text-xs text-[#666]">
                                    {new Date(proof.createdAt).toLocaleTimeString()}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </SectionCard>
    );
}

// ============================================================================
// Component: LossDisplaySection
// ============================================================================

function LossDisplaySection({ metrics }: { metrics: TrainingMetrics | null }) {
    if (!metrics) {
        return (
            <SectionCard title="Training Metrics">
                <p className="text-[#666] text-sm">No metrics available</p>
            </SectionCard>
        );
    }

    return (
        <SectionCard title="Training Metrics">
            <div className="grid grid-cols-2 gap-3">
                <div>
                    <div className="text-xs text-[#666] uppercase">Current Loss</div>
                    <div className="text-xl font-bold text-white">{metrics.loss.toFixed(4)}</div>
                </div>
                <div>
                    <div className="text-xs text-[#666] uppercase">Accuracy</div>
                    <div className="text-xl font-bold text-white">{(metrics.accuracy * 100).toFixed(1)}%</div>
                </div>
                <div>
                    <div className="text-xs text-[#666] uppercase">Learning Rate</div>
                    <div className="text-lg font-medium text-[#aaa]">{metrics.learningRate.toExponential(2)}</div>
                </div>
                <div>
                    <div className="text-xs text-[#666] uppercase">Error Bound</div>
                    <div className="text-lg font-medium text-[#aaa]">{metrics.accumulatedErrorBound.toExponential(2)}</div>
                </div>
                <div>
                    <div className="text-xs text-[#666] uppercase">Gradient Norm</div>
                    <div className="text-lg font-medium text-[#aaa]">{metrics.gradientNorm.toFixed(4)}</div>
                </div>
                <div>
                    <div className="text-xs text-[#666] uppercase">Throughput</div>
                    <div className="text-lg font-medium text-[#aaa]">{metrics.throughput.toFixed(1)} steps/s</div>
                </div>
            </div>
        </SectionCard>
    );
}

// ============================================================================
// Component: ErrorLogSection
// ============================================================================

function ErrorLogSection({ logs }: { logs: ErrorLogEntry[] }) {
    return (
        <SectionCard title="Error Log">
            <div className="space-y-2 max-h-64 overflow-y-auto font-mono text-xs">
                {logs.length === 0 ? (
                    <p className="text-[#666]">No log entries</p>
                ) : (
                    logs.map((log) => (
                        <div key={log.id} className="flex items-start gap-2 py-1 border-b border-helix-border last:border-0">
                            <span className="text-[#555] whitespace-nowrap">
                                {new Date(log.timestamp).toLocaleTimeString()}
                            </span>
                            <span className={`px-1.5 py-0.5 rounded text-[10px] uppercase font-bold ${
                                log.level === 'error' ? 'bg-white/10 text-white' :
                                log.level === 'warning' ? 'bg-white/5 text-[#888]' :
                                'bg-white/[0.03] text-[#666]'
                            }`}>
                                {log.level}
                            </span>
                            <span className="text-[#888]">[{log.source}]</span>
                            <span className="text-[#aaa] flex-1">{log.message}</span>
                        </div>
                    ))
                )}
            </div>
        </SectionCard>
    );
}

// ============================================================================
// Component: ContractStateSection
// ============================================================================

function ContractStateSection() {
    const contractState = useContractState();

    return (
        <SectionCard title="Contract State">
            {contractState.isLoading ? (
                <p className="text-[#666] text-sm">Loading contract state...</p>
            ) : contractState.error ? (
                <p className="text-[#888] text-sm">Error: {contractState.error}</p>
            ) : (
                <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                        <div className="text-xs text-[#666] uppercase">Next Model ID</div>
                        <div className="text-[#aaa]">{contractState.nextModelId?.toString() ?? '-'}</div>
                    </div>
                    <div>
                        <div className="text-xs text-[#666] uppercase">Default Min Stake</div>
                        <div className="text-[#aaa]">
                            {contractState.defaultMinStake ? formatEther(contractState.defaultMinStake) : '-'} ETH
                        </div>
                    </div>
                    <div>
                        <div className="text-xs text-[#666] uppercase">Slash Percentage</div>
                        <div className="text-[#aaa]">{contractState.slashPercentage?.toString() ?? '-'}%</div>
                    </div>
                    <div>
                        <div className="text-xs text-[#666] uppercase">Slashing Records</div>
                        <div className="text-[#aaa]">{contractState.slashingRecordCount?.toString() ?? '-'}</div>
                    </div>
                    <div className="col-span-2">
                        <div className="text-xs text-[#666] uppercase">Max Error Bound</div>
                        <div className="font-mono text-xs text-[#aaa]">
                            {contractState.maxErrorBound?.toString() ?? '-'}
                        </div>
                    </div>
                </div>
            )}
        </SectionCard>
    );
}

// ============================================================================
// Component: NetworkStatsSection
// ============================================================================

function NetworkStatsSection({ stats }: { stats: NetworkStats | null }) {
    if (!stats) {
        return (
            <SectionCard title="Network Statistics">
                <p className="text-[#666] text-sm">Loading network stats...</p>
            </SectionCard>
        );
    }

    return (
        <SectionCard title="Network Statistics">
            <div className="grid grid-cols-3 gap-3 text-center">
                <div>
                    <div className="text-[15px] font-bold text-white">{stats.activeNodes}</div>
                    <div className="text-xs text-[#666]">Active Nodes</div>
                </div>
                <div>
                    <div className="text-[15px] font-bold text-white">{stats.totalProofs}</div>
                    <div className="text-xs text-[#666]">Total Proofs</div>
                </div>
                <div>
                    <div className="text-[15px] font-bold text-white">{stats.totalRounds}</div>
                    <div className="text-xs text-[#666]">Total Rounds</div>
                </div>
                <div>
                    <div className="text-[15px] font-bold text-white">{(stats.networkUptime * 100).toFixed(1)}%</div>
                    <div className="text-xs text-[#666]">Uptime</div>
                </div>
                <div>
                    <div className="text-[15px] font-bold text-white">{stats.averageProofTime}ms</div>
                    <div className="text-xs text-[#666]">Avg Proof Time</div>
                </div>
                <div>
                    <div className="text-[15px] font-bold text-white">{stats.throughput.toFixed(1)}</div>
                    <div className="text-xs text-[#666]">Steps/sec</div>
                </div>
            </div>
        </SectionCard>
    );
}

// ============================================================================
// Main Status Page Component
// ============================================================================

export default function StatusPage() {
    const [data, setData] = useState<StatusData>({
        trainingSessions: [],
        workers: [],
        proofs: [],
        metrics: null,
        networkStats: null,
    });
    const [errorLogs, setErrorLogs] = useState<ErrorLogEntry[]>([]);
    const [lastRefresh, setLastRefresh] = useState(Date.now());
    const [isRefreshing, setIsRefreshing] = useState(false);
    const [useMockData, setUseMockData] = useState(true);

    // Contract events for real-time updates
    const contractEvents = useContractEvents();

    // Fetch data function
    const fetchData = useCallback(async () => {
        setIsRefreshing(true);

        try {
            if (useMockData) {
                // Use mock data for demo
                const mockWorkers = Array.from({ length: 8 }, (_, i) =>
                    generateMockNodeInfo(`0x${Array(40).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('')}`, i)
                );
                const mockProofs = Array.from({ length: 15 }, (_, i) =>
                    generateMockProofInfo(BigInt(Math.floor(Math.random() * 50) + 1), i)
                );
                const mockMetrics = generateMockTrainingMetrics(25);
                const mockSessions = generateMockTrainingSessions();
                const mockStats = generateMockNetworkStats();
                const mockLogs = generateMockErrorLogs();

                setData({
                    trainingSessions: mockSessions,
                    workers: mockWorkers,
                    proofs: mockProofs,
                    metrics: mockMetrics,
                    networkStats: mockStats,
                });
                setErrorLogs(mockLogs);
            } else {
                // Fetch from real API
                const apiClient = getApiClient();

                const [sessionsRes, nodesRes, proofsRes, statsRes] = await Promise.allSettled([
                    apiClient.getTrainingSessions(),
                    apiClient.getNodes(),
                    apiClient.getProofs({ limit: 15 }),
                    apiClient.getNetworkStats(),
                ]);

                setData({
                    trainingSessions: sessionsRes.status === 'fulfilled' ? sessionsRes.value.sessions : [],
                    workers: nodesRes.status === 'fulfilled' ? nodesRes.value : [],
                    proofs: proofsRes.status === 'fulfilled' ? proofsRes.value.proofs : [],
                    metrics: data.metrics, // Keep existing metrics
                    networkStats: statsRes.status === 'fulfilled' ? statsRes.value : null,
                });
            }
        } catch (error) {
            console.error('Failed to fetch status data:', error);
            // Add error to logs
            setErrorLogs(prev => [{
                id: `error-${Date.now()}`,
                timestamp: Date.now(),
                level: 'error' as const,
                message: `Failed to fetch status data: ${error instanceof Error ? error.message : 'Unknown error'}`,
                source: 'dashboard',
            }, ...prev].slice(0, 50));
        } finally {
            setIsRefreshing(false);
            setLastRefresh(Date.now());
        }
    }, [useMockData, data.metrics]);

    // Initial fetch and polling
    useEffect(() => {
        fetchData();

        const interval = setInterval(fetchData, REFRESH_INTERVAL);
        return () => clearInterval(interval);
    }, [fetchData]);

    // Add contract events to logs
    useEffect(() => {
        if (contractEvents.proofEvents.length > 0) {
            const latestProof = contractEvents.proofEvents[contractEvents.proofEvents.length - 1];
            setErrorLogs(prev => [{
                id: `proof-${latestProof.transactionHash}`,
                timestamp: latestProof.timestamp,
                level: 'info' as const,
                message: `Proof submitted for model ${latestProof.modelId} round ${latestProof.roundId}`,
                source: 'contract',
            }, ...prev].slice(0, 50));
        }
    }, [contractEvents.proofEvents]);

    useEffect(() => {
        if (contractEvents.slashedEvents.length > 0) {
            const latestSlash = contractEvents.slashedEvents[contractEvents.slashedEvents.length - 1];
            setErrorLogs(prev => [{
                id: `slash-${latestSlash.transactionHash}`,
                timestamp: latestSlash.timestamp,
                level: 'error' as const,
                message: `Worker ${latestSlash.prover.slice(0, 10)}... slashed: ${latestSlash.reason}`,
                source: 'contract',
            }, ...prev].slice(0, 50));
        }
    }, [contractEvents.slashedEvents]);

    return (
        <div className="min-h-screen bg-neutral-950 text-white p-4">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <div>
                    <h1 className="text-[15px] font-bold">HELIX Status Monitor</h1>
                    <p className="text-sm text-[#666]">Real-time training and network status</p>
                </div>
                <div className="flex items-center gap-3">
                    <RefreshIndicator lastRefresh={lastRefresh} isRefreshing={isRefreshing} />
                    <button
                        onClick={fetchData}
                        disabled={isRefreshing}
                        className="px-3 py-1.5 bg-helix-surface hover:bg-white/[0.07] rounded text-sm transition-colors disabled:opacity-50"
                    >
                        Refresh Now
                    </button>
                    <button
                        onClick={() => setUseMockData(!useMockData)}
                        className={`px-3 py-1.5 rounded text-sm transition-colors ${
                            useMockData ? 'bg-white/10 text-[#888]' : 'bg-white/10 text-white'
                        }`}
                    >
                        {useMockData ? 'Demo Mode' : 'Live Mode'}
                    </button>
                </div>
            </div>

            {/* Main Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
                {/* Left Column */}
                <div className="space-y-4">
                    <TrainingSessionsSection sessions={data.trainingSessions} />
                    <WorkerListSection workers={data.workers} />
                </div>

                {/* Middle Column */}
                <div className="space-y-4">
                    <LossDisplaySection metrics={data.metrics} />
                    <ProofSubmissionsSection proofs={data.proofs} />
                    <RoundStateSection modelId={BigInt(1)} />
                </div>

                {/* Right Column */}
                <div className="space-y-4">
                    <NetworkStatsSection stats={data.networkStats} />
                    <ContractStateSection />
                    <ErrorLogSection logs={errorLogs} />
                </div>
            </div>

            {/* Footer Status Bar */}
            <div className="fixed bottom-0 left-0 right-0 bg-helix-bg border-t border-helix-border px-6 py-2">
                <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-3">
                        <span className="flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full bg-white animate-pulse" />
                            <span className="text-[#888]">System Online</span>
                        </span>
                        <span className="text-[#555]">|</span>
                        <span className="text-[#888]">
                            Active Sessions: <span className="text-white">{data.trainingSessions.filter(s => s.status === 'training').length}</span>
                        </span>
                        <span className="text-[#555]">|</span>
                        <span className="text-[#888]">
                            Workers Online: <span className="text-white">{data.workers.filter(w => w.status !== 'offline').length}/{data.workers.length}</span>
                        </span>
                    </div>
                    <div className="text-[#666]">
                        HELIX Protocol v0.1.0
                    </div>
                </div>
            </div>
        </div>
    );
}
