'use client';

import { useState } from 'react';
import { useChainId, useAccount } from 'wagmi';
import { ConnectButton } from '@rainbow-me/rainbowkit';
import { useContractState, useContractEvents } from '@/hooks/useContract';
import LiveLossCurve from '@/components/training/LiveLossCurve';
import ProofStream from '@/components/proof/ProofStream';
import LiveTopology from '@/components/network/LiveTopology';

export default function DashboardPage() {
    const chainId = useChainId();
    const { isConnected: walletConnected } = useAccount();

    // Model ID - would come from URL params or user selection in production
    const [selectedModelId, setSelectedModelId] = useState<bigint>(BigInt(0));

    // Real contract state
    const {
        nextModelId,
        slashingRecordCount: _slashingRecordCount,
        defaultMinStake: _defaultMinStake,
        isLoading: _contractLoading,
    } = useContractState();

    // Real contract events
    const {
        proofEvents,
        roundStartedEvents,
        roundCompletedEvents,
        stakedEvents,
        slashedEvents,
    } = useContractEvents(selectedModelId);

    // Calculate real stats from contract events
    const totalProofsSubmitted = proofEvents.length;
    const totalRoundsCompleted = roundCompletedEvents.length;
    const totalSlashings = slashedEvents.length;
    const totalStaked = stakedEvents.reduce((sum, e) => sum + e.amount, BigInt(0));

    return (
        <div className="dashboard">
            <style jsx>{`
                .dashboard {
                    min-height: 100vh;
                    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
                    padding: 24px;
                    font-family: 'Inter', -apple-system, sans-serif;
                }

                .dashboard-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 32px;
                    padding-bottom: 24px;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                }

                .logo {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                }

                .logo-icon {
                    width: 48px;
                    height: 48px;
                    background: linear-gradient(135deg, #6366f1, #a855f7);
                    border-radius: 12px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 24px;
                    font-weight: 700;
                    color: white;
                }

                .logo-text {
                    font-size: 28px;
                    font-weight: 800;
                    background: linear-gradient(90deg, #6366f1, #a855f7, #22c55e);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }

                .logo-tagline {
                    font-size: 12px;
                    color: #6b7280;
                }

                .header-right {
                    display: flex;
                    align-items: center;
                    gap: 16px;
                }

                .chain-indicator {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    padding: 8px 16px;
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 8px;
                    font-size: 13px;
                    color: #9ca3af;
                }

                .stats-bar {
                    display: grid;
                    grid-template-columns: repeat(5, 1fr);
                    gap: 16px;
                    margin-bottom: 24px;
                }

                .stat-card {
                    background: rgba(255, 255, 255, 0.03);
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    border-radius: 12px;
                    padding: 20px;
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
                    letter-spacing: 0.5px;
                }

                .dashboard-grid {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 24px;
                }

                .full-width {
                    grid-column: span 2;
                }

                .section {
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    border-radius: 16px;
                    overflow: hidden;
                    color: #fff;
                }

                .section-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 20px 24px;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
                }

                .section-title {
                    font-size: 18px;
                    font-weight: 700;
                }

                .model-selector {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    padding: 8px 16px;
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 8px;
                    color: #d1d5db;
                    font-size: 14px;
                }

                .model-selector select {
                    background: transparent;
                    border: none;
                    color: #d1d5db;
                    font-size: 14px;
                    outline: none;
                    cursor: pointer;
                }

                .model-selector option {
                    background: #1a1a2e;
                }

                .events-section {
                    padding: 24px;
                    max-height: 400px;
                    overflow-y: auto;
                }

                .event-item {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: 12px 16px;
                    background: rgba(255, 255, 255, 0.03);
                    border-radius: 8px;
                    margin-bottom: 8px;
                    border: 1px solid rgba(255, 255, 255, 0.05);
                }

                .event-type {
                    padding: 4px 10px;
                    border-radius: 6px;
                    font-size: 11px;
                    font-weight: 600;
                }

                .event-info {
                    flex: 1;
                    margin-left: 12px;
                }

                .event-primary {
                    font-size: 14px;
                    color: #d1d5db;
                }

                .event-secondary {
                    font-size: 12px;
                    color: #6b7280;
                    font-family: monospace;
                }

                .event-meta {
                    text-align: right;
                }

                .no-data {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    padding: 60px 20px;
                    color: #6b7280;
                    text-align: center;
                }

                .no-data-icon {
                    font-size: 48px;
                    margin-bottom: 16px;
                    opacity: 0.5;
                }

                .no-data-text {
                    font-size: 16px;
                    margin-bottom: 8px;
                }

                .no-data-hint {
                    font-size: 13px;
                    color: #4b5563;
                }

                @media (max-width: 1400px) {
                    .dashboard-grid {
                        grid-template-columns: 1fr;
                    }
                    .full-width {
                        grid-column: span 1;
                    }
                    .stats-bar {
                        grid-template-columns: repeat(3, 1fr);
                    }
                }

                @media (max-width: 768px) {
                    .stats-bar {
                        grid-template-columns: repeat(2, 1fr);
                    }
                }
            `}</style>

            <header className="dashboard-header">
                <div className="logo">
                    <div className="logo-icon">H</div>
                    <div>
                        <div className="logo-text">HELIX Dashboard</div>
                        <div className="logo-tagline">Decentralized Verifiable ML Training</div>
                    </div>
                </div>
                <div className="header-right">
                    <div className="chain-indicator">
                        Chain ID: {chainId}
                    </div>
                    {nextModelId && Number(nextModelId) > 0 && (
                        <div className="model-selector">
                            <span>Model:</span>
                            <select
                                value={selectedModelId.toString()}
                                onChange={(e) => setSelectedModelId(BigInt(e.target.value))}
                            >
                                {Array.from({ length: Number(nextModelId) }, (_, i) => (
                                    <option key={i} value={i.toString()}>
                                        Model #{i}
                                    </option>
                                ))}
                            </select>
                        </div>
                    )}
                    <ConnectButton />
                </div>
            </header>

            {/* Stats from real on-chain data */}
            <div className="stats-bar">
                <div className="stat-card">
                    <div className="stat-value" style={{ color: '#6366f1' }}>
                        {nextModelId?.toString() || '0'}
                    </div>
                    <div className="stat-label">Models Registered</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value" style={{ color: '#22c55e' }}>
                        {totalProofsSubmitted}
                    </div>
                    <div className="stat-label">Proofs Submitted</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value" style={{ color: '#a855f7' }}>
                        {totalRoundsCompleted}
                    </div>
                    <div className="stat-label">Rounds Completed</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value" style={{ color: '#f59e0b' }}>
                        {totalStaked > BigInt(0) ? `${(Number(totalStaked) / 1e18).toFixed(2)}` : '0'} ETH
                    </div>
                    <div className="stat-label">Total Staked</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value" style={{ color: totalSlashings > 0 ? '#ef4444' : '#22c55e' }}>
                        {totalSlashings}
                    </div>
                    <div className="stat-label">Slashing Events</div>
                </div>
            </div>

            <div className="dashboard-grid">
                {/* Training Metrics - Uses its own hook for real-time data */}
                <LiveLossCurve modelId={selectedModelId} />

                {/* Proof Stream - Uses its own hook for real-time proofs */}
                <ProofStream modelId={selectedModelId} />

                {/* Network Topology - Uses its own hook for worker data */}
                <div className="full-width">
                    <LiveTopology modelId={selectedModelId} />
                </div>

                {/* On-Chain Events - Real blockchain data from contract events */}
                <div className="section full-width">
                    <div className="section-header">
                        <h2 className="section-title" style={{ color: '#f59e0b' }}>Recent On-Chain Events</h2>
                        <span style={{ color: '#9ca3af', fontSize: '13px' }}>
                            From HELIX Coordinator Contract
                        </span>
                    </div>

                    <div className="events-section">
                        {(proofEvents.length > 0 || roundStartedEvents.length > 0 || stakedEvents.length > 0) ? (
                            [
                                ...proofEvents.map(e => ({ ...e, eventType: 'ProofSubmitted' as const })),
                                ...roundStartedEvents.map(e => ({ ...e, eventType: 'RoundStarted' as const })),
                                ...roundCompletedEvents.map(e => ({ ...e, eventType: 'RoundCompleted' as const })),
                                ...stakedEvents.map(e => ({ ...e, eventType: 'Staked' as const })),
                                ...slashedEvents.map(e => ({ ...e, eventType: 'Slashed' as const })),
                            ]
                                .sort((a, b) => b.timestamp - a.timestamp)
                                .slice(0, 20)
                                .map((event, i) => (
                                    <div key={`${event.eventType}-${i}`} className="event-item">
                                        <span
                                            className="event-type"
                                            style={{
                                                background: event.eventType === 'ProofSubmitted' ? 'rgba(34, 197, 94, 0.2)' :
                                                    event.eventType === 'Slashed' ? 'rgba(239, 68, 68, 0.2)' :
                                                    event.eventType === 'Staked' ? 'rgba(245, 158, 11, 0.2)' :
                                                    'rgba(99, 102, 241, 0.2)',
                                                color: event.eventType === 'ProofSubmitted' ? '#22c55e' :
                                                    event.eventType === 'Slashed' ? '#ef4444' :
                                                    event.eventType === 'Staked' ? '#f59e0b' :
                                                    '#818cf8',
                                            }}
                                        >
                                            {event.eventType}
                                        </span>
                                        <div className="event-info">
                                            <div className="event-primary">
                                                Model #{event.modelId.toString()}
                                                {'roundId' in event && ` / Round #${event.roundId.toString()}`}
                                            </div>
                                            <div className="event-secondary">
                                                {'prover' in event && `${event.prover.slice(0, 10)}...`}
                                                {'amount' in event && ` • ${(Number(event.amount) / 1e18).toFixed(4)} ETH`}
                                            </div>
                                        </div>
                                        <div className="event-meta">
                                            <div style={{ fontSize: '13px', color: '#9ca3af' }}>
                                                Block #{event.blockNumber.toLocaleString()}
                                            </div>
                                            <a
                                                href={`https://etherscan.io/tx/${event.transactionHash}`}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                style={{ fontSize: '12px', color: '#6366f1' }}
                                            >
                                                View tx →
                                            </a>
                                        </div>
                                    </div>
                                ))
                        ) : (
                            <div className="no-data">
                                <div className="no-data-icon">⛓️</div>
                                <div className="no-data-text">No on-chain events yet</div>
                                <div className="no-data-hint">
                                    {walletConnected
                                        ? 'Events will appear here as they are emitted from the HELIX contracts'
                                        : 'Connect your wallet to view on-chain activity'}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
