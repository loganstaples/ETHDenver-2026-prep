'use client';

import React, { useState, useMemo } from 'react';
import { useContractState, useContractEvents } from '@/hooks/useContract';
import { useChainId } from 'wagmi';
import { getContractAddress } from '@/lib/contracts';

type EventType = 'ProofSubmitted' | 'RoundStarted' | 'RoundCompleted' | 'Staked' | 'Unstaked' | 'Slashed' | 'ModelRegistered';

interface UnifiedEvent {
    id: string;
    type: EventType;
    modelId: bigint;
    roundId?: bigint;
    prover?: string;
    amount?: bigint;
    reason?: string;
    commitment?: bigint;
    deadline?: bigint;
    blockNumber: number;
    transactionHash: string;
    timestamp: number;
}

export default function OnChainExplorer() {
    const chainId = useChainId();
    const { nextModelId, slashingRecordCount: _slashingRecordCount, isLoading: stateLoading } = useContractState();
    const {
        proofEvents,
        roundStartedEvents,
        roundCompletedEvents,
        stakedEvents,
        slashedEvents,
    } = useContractEvents();

    const [filter, setFilter] = useState<EventType | 'all'>('all');
    const [selectedEvent, setSelectedEvent] = useState<UnifiedEvent | null>(null);

    const contractAddress = getContractAddress(chainId, 'helixCoordinator');

    // Combine all events into unified format
    const allEvents = useMemo((): UnifiedEvent[] => {
        const events: UnifiedEvent[] = [];

        // Add proof events
        proofEvents.forEach((e, i) => {
            events.push({
                id: `proof-${i}-${e.transactionHash}`,
                type: 'ProofSubmitted',
                modelId: e.modelId,
                roundId: e.roundId,
                prover: e.prover,
                commitment: e.newCommitment,
                blockNumber: e.blockNumber,
                transactionHash: e.transactionHash,
                timestamp: e.timestamp,
            });
        });

        // Add round started events
        roundStartedEvents.forEach((e, i) => {
            events.push({
                id: `round-start-${i}-${e.transactionHash}`,
                type: 'RoundStarted',
                modelId: e.modelId,
                roundId: e.roundId,
                deadline: e.deadline,
                blockNumber: e.blockNumber,
                transactionHash: e.transactionHash,
                timestamp: e.timestamp,
            });
        });

        // Add round completed events
        roundCompletedEvents.forEach((e, i) => {
            events.push({
                id: `round-complete-${i}-${e.transactionHash}`,
                type: 'RoundCompleted',
                modelId: e.modelId,
                roundId: e.roundId,
                commitment: e.newCommitment,
                blockNumber: e.blockNumber,
                transactionHash: e.transactionHash,
                timestamp: e.timestamp,
            });
        });

        // Add staked events
        stakedEvents.forEach((e, i) => {
            events.push({
                id: `staked-${i}-${e.transactionHash}`,
                type: 'Staked',
                modelId: e.modelId,
                prover: e.prover,
                amount: e.amount,
                blockNumber: e.blockNumber,
                transactionHash: e.transactionHash,
                timestamp: e.timestamp,
            });
        });

        // Add slashed events
        slashedEvents.forEach((e, i) => {
            events.push({
                id: `slashed-${i}-${e.transactionHash}`,
                type: 'Slashed',
                modelId: e.modelId,
                prover: e.prover,
                amount: e.amount,
                reason: e.reason,
                blockNumber: e.blockNumber,
                transactionHash: e.transactionHash,
                timestamp: e.timestamp,
            });
        });

        // Sort by timestamp descending
        return events.sort((a, b) => b.timestamp - a.timestamp);
    }, [proofEvents, roundStartedEvents, roundCompletedEvents, stakedEvents, slashedEvents]);

    // Demo events if no real data
    const displayEvents = allEvents.length > 0 ? allEvents : [
        { id: 'demo-1', type: 'ProofSubmitted' as EventType, modelId: BigInt(0), roundId: BigInt(5), prover: '0x742d35Cc6634C0532925a3b844Bc9e7595f01231', commitment: BigInt(12456), blockNumber: 19234567, transactionHash: '0xabc...123', timestamp: Date.now() / 1000 - 60 },
        { id: 'demo-2', type: 'RoundCompleted' as EventType, modelId: BigInt(0), roundId: BigInt(4), commitment: BigInt(12345), blockNumber: 19234566, transactionHash: '0xdef...456', timestamp: Date.now() / 1000 - 180 },
        { id: 'demo-3', type: 'Staked' as EventType, modelId: BigInt(0), prover: '0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199', amount: BigInt('500000000000000000'), blockNumber: 19234565, transactionHash: '0xghi...789', timestamp: Date.now() / 1000 - 300 },
        { id: 'demo-4', type: 'RoundStarted' as EventType, modelId: BigInt(0), roundId: BigInt(5), deadline: BigInt(Date.now() / 1000 + 3600), blockNumber: 19234564, transactionHash: '0xjkl...012', timestamp: Date.now() / 1000 - 600 },
        { id: 'demo-5', type: 'ProofSubmitted' as EventType, modelId: BigInt(0), roundId: BigInt(4), prover: '0xdD2FD4581271e230360230F9337D5c0430Bf44C0', commitment: BigInt(12345), blockNumber: 19234563, transactionHash: '0xmno...345', timestamp: Date.now() / 1000 - 900 },
    ];

    const filteredEvents = filter === 'all'
        ? displayEvents
        : displayEvents.filter((e) => e.type === filter);

    const formatTime = (ts: number) => {
        const diff = Date.now() / 1000 - ts;
        if (diff < 60) return `${Math.floor(diff)}s ago`;
        if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
        if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
        return new Date(ts * 1000).toLocaleDateString();
    };

    const formatAmount = (amount: bigint) => {
        const eth = Number(amount) / 1e18;
        return eth >= 1 ? `${eth.toFixed(4)} ETH` : `${(eth * 1000).toFixed(2)} mETH`;
    };

    const formatAddress = (addr: string) => {
        if (addr.length > 12) return `${addr.slice(0, 6)}...${addr.slice(-4)}`;
        return addr;
    };

    const formatCommitment = (c: bigint) => {
        const hex = c.toString(16).padStart(8, '0');
        return `0x${hex.slice(0, 8)}`;
    };

    const getExplorerUrl = (hash: string) => {
        const explorers: Record<number, string> = {
            1: 'https://etherscan.io',
            11155111: 'https://sepolia.etherscan.io',
            31337: '', // Local network
        };
        const base = explorers[chainId];
        return base ? `${base}/tx/${hash}` : '';
    };

    // Calculate stats
    const totalProofs = displayEvents.filter((e) => e.type === 'ProofSubmitted').length;
    const totalRounds = displayEvents.filter((e) => e.type === 'RoundCompleted').length;
    const totalStaked = displayEvents
        .filter((e) => e.type === 'Staked' && e.amount)
        .reduce((sum, e) => sum + (e.amount || BigInt(0)), BigInt(0));

    return (
        <div className="bg-helix-surface rounded-md p-4 text-white font-sans">
            <div className="flex justify-between items-center mb-4">
                <div>
                    <h2 className="font-semibold text-[15px] text-white mb-2">On-Chain Explorer</h2>
                    <div className="flex items-center gap-2 text-xs text-white">
                        <span className="w-2 h-2 rounded-full bg-white animate-pulse" />
                        <span>Live - Watching for events</span>
                    </div>
                </div>
                <span className="px-4 py-2 bg-white/[0.03] rounded-md font-mono text-[13px] text-[#888]">{formatAddress(contractAddress)}</span>
            </div>

            <div className="grid grid-cols-4 gap-3 mb-4">
                <div className="bg-white/[0.03] rounded-md p-4 border border-helix-border text-center">
                    <div className="font-semibold text-[15px] mb-1 text-white">{nextModelId?.toString() || '0'}</div>
                    <div className="text-[11px] text-[#888] uppercase">Models Registered</div>
                </div>
                <div className="bg-white/[0.03] rounded-md p-4 border border-helix-border text-center">
                    <div className="font-semibold text-[15px] mb-1 text-white">{totalProofs}</div>
                    <div className="text-[11px] text-[#888] uppercase">Proofs Submitted</div>
                </div>
                <div className="bg-white/[0.03] rounded-md p-4 border border-helix-border text-center">
                    <div className="font-semibold text-[15px] mb-1 text-white">{totalRounds}</div>
                    <div className="text-[11px] text-[#888] uppercase">Rounds Completed</div>
                </div>
                <div className="bg-white/[0.03] rounded-md p-4 border border-helix-border text-center">
                    <div className="font-semibold text-[15px] mb-1 text-white">{formatAmount(totalStaked)}</div>
                    <div className="text-[11px] text-[#888] uppercase">Total Staked</div>
                </div>
            </div>

            <div className="flex gap-2 mb-4 flex-wrap">
                {(['all', 'ProofSubmitted', 'RoundStarted', 'RoundCompleted', 'Staked', 'Slashed'] as const).map((f) => (
                    <button
                        key={f}
                        className={`px-4 py-2 rounded-md border text-xs cursor-pointer transition-all ${
                            filter === f
                                ? 'bg-white/10 border-white/10 text-white'
                                : 'bg-white/[0.04] border-helix-border text-[#888] hover:bg-white/10'
                        }`}
                        onClick={() => setFilter(f)}
                    >
                        {f === 'all' ? 'All Events' : f.replace(/([A-Z])/g, ' $1').trim()}
                    </button>
                ))}
            </div>

            {stateLoading ? (
                <div className="text-center py-10 text-[#666]">Loading blockchain events...</div>
            ) : filteredEvents.length === 0 ? (
                <div className="text-center py-10 text-[#666]">
                    No events found. Events will appear here as they occur on-chain.
                </div>
            ) : (
                <div className="flex flex-col gap-2 max-h-[400px] overflow-y-auto">
                    {filteredEvents.map((event) => (
                        <div
                            key={event.id}
                            className={`flex items-center justify-between p-4 rounded-md border cursor-pointer transition-all ${
                                selectedEvent?.id === event.id
                                    ? 'border-white/10 bg-white/10'
                                    : 'bg-white/[0.03] border-helix-border hover:bg-white/[0.06] hover:border-white/15'
                            }`}
                            onClick={() => setSelectedEvent(event)}
                        >
                            <div className="flex items-center gap-3">
                                <span className="px-3 py-1.5 rounded-md text-[11px] font-semibold min-w-[120px] text-center bg-white/10 text-white">
                                    {event.type.replace(/([A-Z])/g, ' $1').trim()}
                                </span>
                                <div className="flex flex-col gap-1">
                                    <span className="text-sm font-medium text-white">
                                        Model #{event.modelId.toString()}
                                        {event.roundId !== undefined && ` / Round #${event.roundId.toString()}`}
                                    </span>
                                    <span className="font-mono text-xs text-[#666]">
                                        {event.prover && `Prover: ${formatAddress(event.prover)}`}
                                        {event.amount && `Amount: ${formatAmount(event.amount)}`}
                                        {event.commitment && `Commitment: ${formatCommitment(event.commitment)}`}
                                    </span>
                                </div>
                            </div>
                            <div className="flex items-center gap-3">
                                <div className="text-right">
                                    <div className="text-[13px] font-semibold font-mono text-white">#{event.blockNumber.toLocaleString()}</div>
                                    <div className="text-[10px] text-[#666]">Block</div>
                                </div>
                                <div className="text-right">
                                    <div className="text-[13px] font-semibold font-mono text-white">{formatTime(event.timestamp)}</div>
                                    <div className="text-[10px] text-[#666]">Time</div>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {selectedEvent && (
                <div className="mt-4 p-4 bg-white/[0.03] rounded-md border border-helix-border">
                    <div className="text-base font-semibold mb-4 flex items-center gap-3">
                        <span className="px-2.5 py-1 rounded-md bg-white/10 text-white text-xs">
                            {selectedEvent.type}
                        </span>
                        <span className="text-white">Event Details</span>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                        <div className="bg-white/[0.03] p-3 rounded-md">
                            <div className="text-[11px] text-[#666] uppercase mb-1">Transaction Hash</div>
                            <div className="font-mono text-[13px] text-[#aaa] break-all">{selectedEvent.transactionHash}</div>
                        </div>
                        <div className="bg-white/[0.03] p-3 rounded-md">
                            <div className="text-[11px] text-[#666] uppercase mb-1">Block Number</div>
                            <div className="font-mono text-[13px] text-[#aaa] break-all">{selectedEvent.blockNumber.toLocaleString()}</div>
                        </div>
                        <div className="bg-white/[0.03] p-3 rounded-md">
                            <div className="text-[11px] text-[#666] uppercase mb-1">Model ID</div>
                            <div className="font-mono text-[13px] text-[#aaa] break-all">{selectedEvent.modelId.toString()}</div>
                        </div>
                        {selectedEvent.roundId !== undefined && (
                            <div className="bg-white/[0.03] p-3 rounded-md">
                                <div className="text-[11px] text-[#666] uppercase mb-1">Round ID</div>
                                <div className="font-mono text-[13px] text-[#aaa] break-all">{selectedEvent.roundId.toString()}</div>
                            </div>
                        )}
                        {selectedEvent.prover && (
                            <div className="bg-white/[0.03] p-3 rounded-md">
                                <div className="text-[11px] text-[#666] uppercase mb-1">Prover Address</div>
                                <div className="font-mono text-[13px] text-[#aaa] break-all">{selectedEvent.prover}</div>
                            </div>
                        )}
                        {selectedEvent.amount && (
                            <div className="bg-white/[0.03] p-3 rounded-md">
                                <div className="text-[11px] text-[#666] uppercase mb-1">Amount</div>
                                <div className="font-mono text-[13px] text-[#aaa] break-all">{formatAmount(selectedEvent.amount)}</div>
                            </div>
                        )}
                        {selectedEvent.commitment && (
                            <div className="bg-white/[0.03] p-3 rounded-md">
                                <div className="text-[11px] text-[#666] uppercase mb-1">Commitment</div>
                                <div className="font-mono text-[13px] text-[#aaa] break-all">{selectedEvent.commitment.toString(16)}</div>
                            </div>
                        )}
                        {selectedEvent.deadline && (
                            <div className="bg-white/[0.03] p-3 rounded-md">
                                <div className="text-[11px] text-[#666] uppercase mb-1">Deadline</div>
                                <div className="font-mono text-[13px] text-[#aaa] break-all">
                                    {new Date(Number(selectedEvent.deadline) * 1000).toLocaleString()}
                                </div>
                            </div>
                        )}
                        {selectedEvent.reason && (
                            <div className="bg-white/[0.03] p-3 rounded-md">
                                <div className="text-[11px] text-[#666] uppercase mb-1">Slash Reason</div>
                                <div className="font-mono text-[13px] text-[#888] break-all">{selectedEvent.reason}</div>
                            </div>
                        )}
                        <div className="bg-white/[0.03] p-3 rounded-md">
                            <div className="text-[11px] text-[#666] uppercase mb-1">Timestamp</div>
                            <div className="font-mono text-[13px] text-[#aaa] break-all">
                                {new Date(selectedEvent.timestamp * 1000).toLocaleString()}
                            </div>
                        </div>
                    </div>
                    {getExplorerUrl(selectedEvent.transactionHash) && (
                        <a
                            className="inline-flex items-center gap-1.5 text-white text-[13px] no-underline mt-4 px-4 py-2 bg-white/10 rounded-md transition-all hover:bg-white/20"
                            href={getExplorerUrl(selectedEvent.transactionHash)}
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            View on Block Explorer →
                        </a>
                    )}
                </div>
            )}
        </div>
    );
}
