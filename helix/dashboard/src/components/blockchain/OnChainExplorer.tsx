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
    const { nextModelId, slashingRecordCount, isLoading: stateLoading } = useContractState();
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

    const getTypeColor = (type: EventType) => {
        const colors: Record<EventType, string> = {
            ProofSubmitted: '#22c55e',
            RoundStarted: '#6366f1',
            RoundCompleted: '#a855f7',
            Staked: '#f59e0b',
            Unstaked: '#94a3b8',
            Slashed: '#ef4444',
            ModelRegistered: '#06b6d4',
        };
        return colors[type];
    };

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
        <div className="onchain-explorer">
            <style jsx>{`
                .onchain-explorer {
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
                    background: linear-gradient(90deg, #f59e0b, #ef4444);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }

                .contract-address {
                    padding: 8px 16px;
                    background: rgba(0, 0, 0, 0.3);
                    border-radius: 8px;
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 13px;
                    color: #9ca3af;
                }

                .state-grid {
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 16px;
                    margin-bottom: 24px;
                }

                .state-card {
                    background: rgba(255, 255, 255, 0.03);
                    border-radius: 12px;
                    padding: 16px;
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    text-align: center;
                }

                .state-value {
                    font-size: 24px;
                    font-weight: 700;
                    margin-bottom: 4px;
                }

                .state-label {
                    font-size: 11px;
                    color: #9ca3af;
                    text-transform: uppercase;
                }

                .filters {
                    display: flex;
                    gap: 8px;
                    margin-bottom: 16px;
                    flex-wrap: wrap;
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
                    background: rgba(99, 102, 241, 0.2);
                    border-color: rgba(99, 102, 241, 0.5);
                    color: #818cf8;
                }

                .event-list {
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                    max-height: 400px;
                    overflow-y: auto;
                }

                .event-item {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: 16px;
                    background: rgba(255, 255, 255, 0.03);
                    border-radius: 12px;
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    cursor: pointer;
                    transition: all 0.2s;
                }

                .event-item:hover {
                    background: rgba(255, 255, 255, 0.06);
                    border-color: rgba(255, 255, 255, 0.15);
                }

                .event-item.selected {
                    border-color: rgba(99, 102, 241, 0.5);
                    background: rgba(99, 102, 241, 0.1);
                }

                .event-main {
                    display: flex;
                    align-items: center;
                    gap: 16px;
                }

                .event-type {
                    padding: 6px 12px;
                    border-radius: 6px;
                    font-size: 11px;
                    font-weight: 600;
                    min-width: 120px;
                    text-align: center;
                }

                .event-info {
                    display: flex;
                    flex-direction: column;
                    gap: 4px;
                }

                .event-primary {
                    font-size: 14px;
                    font-weight: 500;
                }

                .event-secondary {
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 12px;
                    color: #6b7280;
                }

                .event-meta {
                    display: flex;
                    align-items: center;
                    gap: 24px;
                }

                .event-stat {
                    text-align: right;
                }

                .event-stat-value {
                    font-size: 13px;
                    font-weight: 600;
                    font-family: 'JetBrains Mono', monospace;
                }

                .event-stat-label {
                    font-size: 10px;
                    color: #6b7280;
                }

                .detail-panel {
                    margin-top: 24px;
                    padding: 20px;
                    background: rgba(255, 255, 255, 0.03);
                    border-radius: 12px;
                    border: 1px solid rgba(255, 255, 255, 0.08);
                }

                .detail-title {
                    font-size: 16px;
                    font-weight: 600;
                    margin-bottom: 16px;
                    display: flex;
                    align-items: center;
                    gap: 12px;
                }

                .detail-grid {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
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
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 13px;
                    color: #d1d5db;
                    word-break: break-all;
                }

                .explorer-link {
                    display: inline-flex;
                    align-items: center;
                    gap: 6px;
                    color: #6366f1;
                    font-size: 13px;
                    text-decoration: none;
                    margin-top: 16px;
                    padding: 8px 16px;
                    background: rgba(99, 102, 241, 0.1);
                    border-radius: 8px;
                    transition: all 0.2s;
                }

                .explorer-link:hover {
                    background: rgba(99, 102, 241, 0.2);
                }

                .loading {
                    text-align: center;
                    padding: 40px;
                    color: #6b7280;
                }

                .no-events {
                    text-align: center;
                    padding: 40px;
                    color: #6b7280;
                }

                .live-indicator {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    font-size: 12px;
                    color: #22c55e;
                }

                .live-dot {
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: #22c55e;
                    animation: pulse 2s infinite;
                }

                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }
            `}</style>

            <div className="header">
                <div>
                    <h2 className="title">On-Chain Explorer</h2>
                    <div className="live-indicator" style={{ marginTop: '8px' }}>
                        <span className="live-dot" />
                        <span>Live - Watching for events</span>
                    </div>
                </div>
                <span className="contract-address">{formatAddress(contractAddress)}</span>
            </div>

            <div className="state-grid">
                <div className="state-card">
                    <div className="state-value" style={{ color: '#6366f1' }}>{nextModelId?.toString() || '0'}</div>
                    <div className="state-label">Models Registered</div>
                </div>
                <div className="state-card">
                    <div className="state-value" style={{ color: '#22c55e' }}>{totalProofs}</div>
                    <div className="state-label">Proofs Submitted</div>
                </div>
                <div className="state-card">
                    <div className="state-value" style={{ color: '#a855f7' }}>{totalRounds}</div>
                    <div className="state-label">Rounds Completed</div>
                </div>
                <div className="state-card">
                    <div className="state-value" style={{ color: '#f59e0b' }}>{formatAmount(totalStaked)}</div>
                    <div className="state-label">Total Staked</div>
                </div>
            </div>

            <div className="filters">
                {(['all', 'ProofSubmitted', 'RoundStarted', 'RoundCompleted', 'Staked', 'Slashed'] as const).map((f) => (
                    <button
                        key={f}
                        className={`filter-btn ${filter === f ? 'active' : ''}`}
                        onClick={() => setFilter(f)}
                    >
                        {f === 'all' ? 'All Events' : f.replace(/([A-Z])/g, ' $1').trim()}
                    </button>
                ))}
            </div>

            {stateLoading ? (
                <div className="loading">Loading blockchain events...</div>
            ) : filteredEvents.length === 0 ? (
                <div className="no-events">
                    No events found. Events will appear here as they occur on-chain.
                </div>
            ) : (
                <div className="event-list">
                    {filteredEvents.map((event) => (
                        <div
                            key={event.id}
                            className={`event-item ${selectedEvent?.id === event.id ? 'selected' : ''}`}
                            onClick={() => setSelectedEvent(event)}
                        >
                            <div className="event-main">
                                <span
                                    className="event-type"
                                    style={{
                                        background: `${getTypeColor(event.type)}22`,
                                        color: getTypeColor(event.type),
                                    }}
                                >
                                    {event.type.replace(/([A-Z])/g, ' $1').trim()}
                                </span>
                                <div className="event-info">
                                    <span className="event-primary">
                                        Model #{event.modelId.toString()}
                                        {event.roundId !== undefined && ` / Round #${event.roundId.toString()}`}
                                    </span>
                                    <span className="event-secondary">
                                        {event.prover && `Prover: ${formatAddress(event.prover)}`}
                                        {event.amount && `Amount: ${formatAmount(event.amount)}`}
                                        {event.commitment && `Commitment: ${formatCommitment(event.commitment)}`}
                                    </span>
                                </div>
                            </div>
                            <div className="event-meta">
                                <div className="event-stat">
                                    <div className="event-stat-value">#{event.blockNumber.toLocaleString()}</div>
                                    <div className="event-stat-label">Block</div>
                                </div>
                                <div className="event-stat">
                                    <div className="event-stat-value">{formatTime(event.timestamp)}</div>
                                    <div className="event-stat-label">Time</div>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {selectedEvent && (
                <div className="detail-panel">
                    <div className="detail-title">
                        <span
                            style={{
                                padding: '4px 10px',
                                borderRadius: '6px',
                                background: `${getTypeColor(selectedEvent.type)}22`,
                                color: getTypeColor(selectedEvent.type),
                                fontSize: '12px',
                            }}
                        >
                            {selectedEvent.type}
                        </span>
                        Event Details
                    </div>
                    <div className="detail-grid">
                        <div className="detail-item">
                            <div className="detail-label">Transaction Hash</div>
                            <div className="detail-value">{selectedEvent.transactionHash}</div>
                        </div>
                        <div className="detail-item">
                            <div className="detail-label">Block Number</div>
                            <div className="detail-value">{selectedEvent.blockNumber.toLocaleString()}</div>
                        </div>
                        <div className="detail-item">
                            <div className="detail-label">Model ID</div>
                            <div className="detail-value">{selectedEvent.modelId.toString()}</div>
                        </div>
                        {selectedEvent.roundId !== undefined && (
                            <div className="detail-item">
                                <div className="detail-label">Round ID</div>
                                <div className="detail-value">{selectedEvent.roundId.toString()}</div>
                            </div>
                        )}
                        {selectedEvent.prover && (
                            <div className="detail-item">
                                <div className="detail-label">Prover Address</div>
                                <div className="detail-value">{selectedEvent.prover}</div>
                            </div>
                        )}
                        {selectedEvent.amount && (
                            <div className="detail-item">
                                <div className="detail-label">Amount</div>
                                <div className="detail-value">{formatAmount(selectedEvent.amount)}</div>
                            </div>
                        )}
                        {selectedEvent.commitment && (
                            <div className="detail-item">
                                <div className="detail-label">Commitment</div>
                                <div className="detail-value">{selectedEvent.commitment.toString(16)}</div>
                            </div>
                        )}
                        {selectedEvent.deadline && (
                            <div className="detail-item">
                                <div className="detail-label">Deadline</div>
                                <div className="detail-value">
                                    {new Date(Number(selectedEvent.deadline) * 1000).toLocaleString()}
                                </div>
                            </div>
                        )}
                        {selectedEvent.reason && (
                            <div className="detail-item">
                                <div className="detail-label">Slash Reason</div>
                                <div className="detail-value" style={{ color: '#ef4444' }}>{selectedEvent.reason}</div>
                            </div>
                        )}
                        <div className="detail-item">
                            <div className="detail-label">Timestamp</div>
                            <div className="detail-value">
                                {new Date(selectedEvent.timestamp * 1000).toLocaleString()}
                            </div>
                        </div>
                    </div>
                    {getExplorerUrl(selectedEvent.transactionHash) && (
                        <a
                            className="explorer-link"
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
