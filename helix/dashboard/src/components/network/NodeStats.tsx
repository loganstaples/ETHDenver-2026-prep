'use client';

import React, { useState, useEffect } from 'react';
import { useContractState, useContractEvents } from '@/hooks/useContract';
import { useAccount } from 'wagmi';

interface WorkerNode {
    id: string;
    address: string;
    status: 'active' | 'idle' | 'offline' | 'syncing';
    lastSeen: number;
    stakedAmount: bigint;
    proofsSubmitted: number;
    roundsParticipated: number;
    reputation: number;
    capabilities: {
        canTrain: boolean;
        canAggregate: boolean;
        canProve: boolean;
        gpuMemoryMb: number;
    };
}

export default function NodeStats() {
    const { address: _address } = useAccount();
    const { slashingRecordCount: _slashingRecordCount, isLoading: stateLoading } = useContractState();
    const { proofEvents, stakedEvents, slashedEvents } = useContractEvents();
    const [workers, setWorkers] = useState<WorkerNode[]>([]);
    const [selectedWorker, setSelectedWorker] = useState<WorkerNode | null>(null);
    const [filter, setFilter] = useState<'all' | 'active' | 'idle' | 'offline'>('all');

    // Build worker list from contract events
    useEffect(() => {
        const workerMap = new Map<string, WorkerNode>();

        // Process staked events to get worker list
        stakedEvents.forEach((event) => {
            const addr = event.prover;
            if (!workerMap.has(addr)) {
                workerMap.set(addr, {
                    id: `worker-${addr.slice(2, 8)}`,
                    address: addr,
                    status: 'active',
                    lastSeen: event.timestamp,
                    stakedAmount: event.amount,
                    proofsSubmitted: 0,
                    roundsParticipated: 0,
                    reputation: 100,
                    capabilities: {
                        canTrain: true,
                        canAggregate: Math.random() > 0.5,
                        canProve: true,
                        gpuMemoryMb: Math.floor(Math.random() * 24000) + 8000,
                    },
                });
            } else {
                const worker = workerMap.get(addr)!;
                worker.stakedAmount += event.amount;
                worker.lastSeen = Math.max(worker.lastSeen, event.timestamp);
            }
        });

        // Process proof events to update stats
        proofEvents.forEach((event) => {
            const addr = event.prover;
            if (workerMap.has(addr)) {
                const worker = workerMap.get(addr)!;
                worker.proofsSubmitted++;
                worker.roundsParticipated++;
                worker.lastSeen = Math.max(worker.lastSeen, event.timestamp);
                worker.reputation = Math.min(100, worker.reputation + 1);
            }
        });

        // Process slashed events to update reputation
        slashedEvents.forEach((event) => {
            const addr = event.prover;
            if (workerMap.has(addr)) {
                const worker = workerMap.get(addr)!;
                worker.reputation = Math.max(0, worker.reputation - 20);
                worker.status = 'idle';
            }
        });

        // Update status based on last seen time
        const now = Date.now() / 1000;
        workerMap.forEach((worker) => {
            const timeSinceLastSeen = now - worker.lastSeen;
            if (timeSinceLastSeen > 3600) {
                worker.status = 'offline';
            } else if (timeSinceLastSeen > 300) {
                worker.status = 'idle';
            }
        });

        // If no real data, add demo workers
        if (workerMap.size === 0) {
            const demoWorkers: WorkerNode[] = [
                {
                    id: 'worker-demo-1',
                    address: '0x742d35Cc6634C0532925a3b844Bc9e7595f01231',
                    status: 'active',
                    lastSeen: Date.now() / 1000 - 30,
                    stakedAmount: BigInt('1000000000000000000'),
                    proofsSubmitted: 42,
                    roundsParticipated: 38,
                    reputation: 98,
                    capabilities: { canTrain: true, canAggregate: true, canProve: true, gpuMemoryMb: 24000 },
                },
                {
                    id: 'worker-demo-2',
                    address: '0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199',
                    status: 'active',
                    lastSeen: Date.now() / 1000 - 60,
                    stakedAmount: BigInt('500000000000000000'),
                    proofsSubmitted: 28,
                    roundsParticipated: 25,
                    reputation: 92,
                    capabilities: { canTrain: true, canAggregate: false, canProve: true, gpuMemoryMb: 16000 },
                },
                {
                    id: 'worker-demo-3',
                    address: '0xdD2FD4581271e230360230F9337D5c0430Bf44C0',
                    status: 'idle',
                    lastSeen: Date.now() / 1000 - 600,
                    stakedAmount: BigInt('250000000000000000'),
                    proofsSubmitted: 15,
                    roundsParticipated: 12,
                    reputation: 85,
                    capabilities: { canTrain: true, canAggregate: false, canProve: false, gpuMemoryMb: 8000 },
                },
                {
                    id: 'worker-demo-4',
                    address: '0xbDA5747bFD65F08deb54cb465eB87D40e51B197E',
                    status: 'syncing',
                    lastSeen: Date.now() / 1000 - 10,
                    stakedAmount: BigInt('750000000000000000'),
                    proofsSubmitted: 0,
                    roundsParticipated: 0,
                    reputation: 100,
                    capabilities: { canTrain: true, canAggregate: true, canProve: true, gpuMemoryMb: 32000 },
                },
            ];
            demoWorkers.forEach((w) => workerMap.set(w.address, w));
        }

        setWorkers(Array.from(workerMap.values()));
    }, [stakedEvents, proofEvents, slashedEvents]);

    const filteredWorkers = workers.filter((w) => filter === 'all' || w.status === filter);

    const getStatusColor = (status: string) => {
        const colors: Record<string, string> = {
            active: '#22c55e',
            idle: '#f59e0b',
            offline: '#ef4444',
            syncing: '#6366f1',
        };
        return colors[status] || '#6b7280';
    };

    const formatStake = (amount: bigint) => {
        const eth = Number(amount) / 1e18;
        return eth >= 1 ? `${eth.toFixed(2)} ETH` : `${(eth * 1000).toFixed(1)} mETH`;
    };

    const formatTimeSince = (timestamp: number) => {
        const diff = Date.now() / 1000 - timestamp;
        if (diff < 60) return `${Math.floor(diff)}s ago`;
        if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
        if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
        return `${Math.floor(diff / 86400)}d ago`;
    };

    const totalStaked = workers.reduce((sum, w) => sum + w.stakedAmount, BigInt(0));
    const activeWorkers = workers.filter((w) => w.status === 'active').length;
    const totalProofs = workers.reduce((sum, w) => sum + w.proofsSubmitted, 0);
    const avgReputation = workers.length > 0
        ? Math.round(workers.reduce((sum, w) => sum + w.reputation, 0) / workers.length)
        : 0;

    return (
        <div className="node-stats">
            <style jsx>{`
                .node-stats {
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    border-radius: 16px;
                    padding: 24px;
                    color: #fff;
                    font-family: 'Inter', -apple-system, sans-serif;
                }

                .header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 24px;
                }

                .title {
                    font-size: 24px;
                    font-weight: 700;
                    background: linear-gradient(90deg, #22c55e, #6366f1);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }

                .stats-grid {
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 16px;
                    margin-bottom: 24px;
                }

                .stat-card {
                    background: rgba(255, 255, 255, 0.03);
                    border-radius: 12px;
                    padding: 16px;
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    text-align: center;
                }

                .stat-value {
                    font-size: 28px;
                    font-weight: 700;
                    margin-bottom: 4px;
                }

                .stat-label {
                    font-size: 12px;
                    color: #9ca3af;
                    text-transform: uppercase;
                }

                .filters {
                    display: flex;
                    gap: 8px;
                    margin-bottom: 16px;
                }

                .filter-btn {
                    padding: 8px 16px;
                    border-radius: 8px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    background: rgba(255, 255, 255, 0.05);
                    color: #9ca3af;
                    font-size: 12px;
                    cursor: pointer;
                    transition: all 0.2s;
                }

                .filter-btn:hover {
                    background: rgba(255, 255, 255, 0.1);
                }

                .filter-btn.active {
                    background: rgba(34, 197, 94, 0.2);
                    border-color: rgba(34, 197, 94, 0.5);
                    color: #22c55e;
                }

                .worker-list {
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                }

                .worker-item {
                    display: grid;
                    grid-template-columns: 40px 1fr 120px 100px 100px 80px 60px;
                    align-items: center;
                    gap: 16px;
                    padding: 16px;
                    background: rgba(255, 255, 255, 0.03);
                    border-radius: 12px;
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    cursor: pointer;
                    transition: all 0.2s;
                }

                .worker-item:hover {
                    background: rgba(255, 255, 255, 0.06);
                    border-color: rgba(255, 255, 255, 0.15);
                }

                .worker-item.selected {
                    border-color: rgba(34, 197, 94, 0.5);
                    background: rgba(34, 197, 94, 0.1);
                }

                .status-dot {
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    animation: pulse 2s infinite;
                }

                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }

                .worker-address {
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 14px;
                }

                .worker-stake {
                    font-weight: 600;
                    color: #f59e0b;
                }

                .worker-proofs {
                    font-weight: 600;
                    color: #22c55e;
                }

                .worker-reputation {
                    font-weight: 600;
                }

                .worker-last-seen {
                    font-size: 12px;
                    color: #6b7280;
                }

                .capabilities {
                    display: flex;
                    gap: 4px;
                }

                .cap-badge {
                    padding: 2px 6px;
                    border-radius: 4px;
                    font-size: 10px;
                    font-weight: 600;
                }

                .detail-panel {
                    margin-top: 24px;
                    padding: 20px;
                    background: rgba(255, 255, 255, 0.03);
                    border-radius: 12px;
                    border: 1px solid rgba(255, 255, 255, 0.08);
                }

                .detail-title {
                    font-size: 18px;
                    font-weight: 600;
                    margin-bottom: 16px;
                }

                .detail-grid {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 16px;
                }

                .detail-item {
                    background: rgba(0, 0, 0, 0.2);
                    padding: 12px;
                    border-radius: 8px;
                }

                .detail-label {
                    font-size: 11px;
                    color: #6b7280;
                    text-transform: uppercase;
                    margin-bottom: 4px;
                }

                .detail-value {
                    font-size: 16px;
                    font-weight: 600;
                }

                .loading {
                    text-align: center;
                    padding: 40px;
                    color: #6b7280;
                }
            `}</style>

            <div className="header">
                <h2 className="title">Worker Nodes</h2>
                <span style={{ color: '#6b7280', fontSize: '14px' }}>
                    {workers.length} total workers
                </span>
            </div>

            <div className="stats-grid">
                <div className="stat-card">
                    <div className="stat-value" style={{ color: '#22c55e' }}>{activeWorkers}</div>
                    <div className="stat-label">Active Workers</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value" style={{ color: '#f59e0b' }}>{formatStake(totalStaked)}</div>
                    <div className="stat-label">Total Staked</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value" style={{ color: '#6366f1' }}>{totalProofs}</div>
                    <div className="stat-label">Total Proofs</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value" style={{ color: '#a855f7' }}>{avgReputation}%</div>
                    <div className="stat-label">Avg Reputation</div>
                </div>
            </div>

            <div className="filters">
                {(['all', 'active', 'idle', 'offline'] as const).map((f) => (
                    <button
                        key={f}
                        className={`filter-btn ${filter === f ? 'active' : ''}`}
                        onClick={() => setFilter(f)}
                    >
                        {f.charAt(0).toUpperCase() + f.slice(1)}
                    </button>
                ))}
            </div>

            {stateLoading ? (
                <div className="loading">Loading worker data...</div>
            ) : (
                <div className="worker-list">
                    {filteredWorkers.map((worker) => (
                        <div
                            key={worker.id}
                            className={`worker-item ${selectedWorker?.id === worker.id ? 'selected' : ''}`}
                            onClick={() => setSelectedWorker(worker)}
                        >
                            <div
                                className="status-dot"
                                style={{ backgroundColor: getStatusColor(worker.status) }}
                            />
                            <div className="worker-address">
                                {worker.address.slice(0, 6)}...{worker.address.slice(-4)}
                            </div>
                            <div className="worker-stake">{formatStake(worker.stakedAmount)}</div>
                            <div className="worker-proofs">{worker.proofsSubmitted} proofs</div>
                            <div
                                className="worker-reputation"
                                style={{
                                    color: worker.reputation >= 90 ? '#22c55e' :
                                           worker.reputation >= 70 ? '#f59e0b' : '#ef4444'
                                }}
                            >
                                {worker.reputation}%
                            </div>
                            <div className="worker-last-seen">{formatTimeSince(worker.lastSeen)}</div>
                            <div className="capabilities">
                                {worker.capabilities.canTrain && (
                                    <span className="cap-badge" style={{ background: 'rgba(99, 102, 241, 0.3)', color: '#818cf8' }}>T</span>
                                )}
                                {worker.capabilities.canAggregate && (
                                    <span className="cap-badge" style={{ background: 'rgba(168, 85, 247, 0.3)', color: '#c084fc' }}>A</span>
                                )}
                                {worker.capabilities.canProve && (
                                    <span className="cap-badge" style={{ background: 'rgba(34, 197, 94, 0.3)', color: '#4ade80' }}>P</span>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {selectedWorker && (
                <div className="detail-panel">
                    <div className="detail-title">
                        Worker Details: {selectedWorker.address.slice(0, 8)}...
                    </div>
                    <div className="detail-grid">
                        <div className="detail-item">
                            <div className="detail-label">Full Address</div>
                            <div className="detail-value" style={{ fontSize: '12px', fontFamily: 'monospace' }}>
                                {selectedWorker.address}
                            </div>
                        </div>
                        <div className="detail-item">
                            <div className="detail-label">Status</div>
                            <div className="detail-value" style={{ color: getStatusColor(selectedWorker.status) }}>
                                {selectedWorker.status.toUpperCase()}
                            </div>
                        </div>
                        <div className="detail-item">
                            <div className="detail-label">Staked Amount</div>
                            <div className="detail-value" style={{ color: '#f59e0b' }}>
                                {formatStake(selectedWorker.stakedAmount)}
                            </div>
                        </div>
                        <div className="detail-item">
                            <div className="detail-label">Proofs Submitted</div>
                            <div className="detail-value" style={{ color: '#22c55e' }}>
                                {selectedWorker.proofsSubmitted}
                            </div>
                        </div>
                        <div className="detail-item">
                            <div className="detail-label">Rounds Participated</div>
                            <div className="detail-value" style={{ color: '#6366f1' }}>
                                {selectedWorker.roundsParticipated}
                            </div>
                        </div>
                        <div className="detail-item">
                            <div className="detail-label">GPU Memory</div>
                            <div className="detail-value">
                                {(selectedWorker.capabilities.gpuMemoryMb / 1000).toFixed(0)} GB
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
