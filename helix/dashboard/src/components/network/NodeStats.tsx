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
            active: '#ffffff',
            idle: '#a3a3a3',
            offline: '#525252',
            syncing: '#d4d4d4',
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
        <div className="bg-helix-surface rounded-md p-4 text-white border border-helix-border">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-white font-semibold text-[15px]">Worker Nodes</h2>
                <span className="text-[#666] text-sm">
                    {workers.length} total workers
                </span>
            </div>

            <div className="grid grid-cols-4 gap-3 mb-4">
                <div className="bg-white/[0.03] rounded-md p-4 border border-helix-border text-center">
                    <div className="text-white text-xl font-bold mb-1">{activeWorkers}</div>
                    <div className="text-[#888] text-xs uppercase">Active Workers</div>
                </div>
                <div className="bg-white/[0.03] rounded-md p-4 border border-helix-border text-center">
                    <div className="text-white text-xl font-bold mb-1">{formatStake(totalStaked)}</div>
                    <div className="text-[#888] text-xs uppercase">Total Staked</div>
                </div>
                <div className="bg-white/[0.03] rounded-md p-4 border border-helix-border text-center">
                    <div className="text-white text-xl font-bold mb-1">{totalProofs}</div>
                    <div className="text-[#888] text-xs uppercase">Total Proofs</div>
                </div>
                <div className="bg-white/[0.03] rounded-md p-4 border border-helix-border text-center">
                    <div className="text-white text-xl font-bold mb-1">{avgReputation}%</div>
                    <div className="text-[#888] text-xs uppercase">Avg Reputation</div>
                </div>
            </div>

            <div className="flex gap-2 mb-4">
                {(['all', 'active', 'idle', 'offline'] as const).map((f) => (
                    <button
                        key={f}
                        className={`px-4 py-2 rounded-md border text-xs transition-all ${
                            filter === f
                                ? 'bg-white/10 border-white/10 text-white'
                                : 'bg-white/5 border-helix-border text-[#666] hover:bg-white/10'
                        }`}
                        onClick={() => setFilter(f)}
                    >
                        {f.charAt(0).toUpperCase() + f.slice(1)}
                    </button>
                ))}
            </div>

            {stateLoading ? (
                <div className="text-center py-10 text-[#666]">Loading worker data...</div>
            ) : (
                <div className="flex flex-col gap-2">
                    {filteredWorkers.map((worker) => (
                        <div
                            key={worker.id}
                            className={`grid grid-cols-[40px_1fr_120px_100px_100px_80px_60px] items-center gap-3 p-4 bg-white/[0.03] rounded-md border cursor-pointer transition-all ${
                                selectedWorker?.id === worker.id
                                    ? 'border-white/10 bg-white/10'
                                    : 'border-helix-border hover:bg-white/[0.06] hover:border-white/[0.15]'
                            }`}
                            onClick={() => setSelectedWorker(worker)}
                        >
                            <div
                                className="w-3 h-3 rounded-full animate-[pulse_2s_infinite]"
                                style={{ backgroundColor: getStatusColor(worker.status) }}
                            />
                            <div className="font-mono text-sm text-white">
                                {worker.address.slice(0, 6)}...{worker.address.slice(-4)}
                            </div>
                            <div className="text-white font-semibold">{formatStake(worker.stakedAmount)}</div>
                            <div className="text-white font-semibold">{worker.proofsSubmitted} proofs</div>
                            <div className="text-white font-semibold">
                                {worker.reputation}%
                            </div>
                            <div className="text-xs text-neutral-600">{formatTimeSince(worker.lastSeen)}</div>
                            <div className="flex gap-1">
                                {worker.capabilities.canTrain && (
                                    <span className="px-1.5 py-0.5 rounded bg-white/10 text-[#888] text-[10px] font-semibold">T</span>
                                )}
                                {worker.capabilities.canAggregate && (
                                    <span className="px-1.5 py-0.5 rounded bg-white/10 text-[#888] text-[10px] font-semibold">A</span>
                                )}
                                {worker.capabilities.canProve && (
                                    <span className="px-1.5 py-0.5 rounded bg-white/10 text-[#888] text-[10px] font-semibold">P</span>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {selectedWorker && (
                <div className="mt-4 p-4 bg-white/[0.03] rounded-md border border-helix-border">
                    <div className="text-lg font-semibold mb-4 text-white">
                        Worker Details: {selectedWorker.address.slice(0, 8)}...
                    </div>
                    <div className="grid grid-cols-3 gap-3">
                        <div className="bg-white/[0.03] rounded-md p-3">
                            <div className="text-[11px] text-neutral-600 uppercase mb-1">Full Address</div>
                            <div className="text-white text-xs font-semibold font-mono break-all">
                                {selectedWorker.address}
                            </div>
                        </div>
                        <div className="bg-white/[0.03] rounded-md p-3">
                            <div className="text-[11px] text-neutral-600 uppercase mb-1">Status</div>
                            <div className="text-white text-base font-semibold">
                                {selectedWorker.status.toUpperCase()}
                            </div>
                        </div>
                        <div className="bg-white/[0.03] rounded-md p-3">
                            <div className="text-[11px] text-neutral-600 uppercase mb-1">Staked Amount</div>
                            <div className="text-white text-base font-semibold">
                                {formatStake(selectedWorker.stakedAmount)}
                            </div>
                        </div>
                        <div className="bg-white/[0.03] rounded-md p-3">
                            <div className="text-[11px] text-neutral-600 uppercase mb-1">Proofs Submitted</div>
                            <div className="text-white text-base font-semibold">
                                {selectedWorker.proofsSubmitted}
                            </div>
                        </div>
                        <div className="bg-white/[0.03] rounded-md p-3">
                            <div className="text-[11px] text-neutral-600 uppercase mb-1">Rounds Participated</div>
                            <div className="text-white text-base font-semibold">
                                {selectedWorker.roundsParticipated}
                            </div>
                        </div>
                        <div className="bg-white/[0.03] rounded-md p-3">
                            <div className="text-[11px] text-neutral-600 uppercase mb-1">GPU Memory</div>
                            <div className="text-white text-base font-semibold">
                                {(selectedWorker.capabilities.gpuMemoryMb / 1000).toFixed(0)} GB
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
