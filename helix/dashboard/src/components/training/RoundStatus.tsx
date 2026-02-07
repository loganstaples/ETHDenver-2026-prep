'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { useModel, useRound, useContractEvents, useErrorBound } from '@/hooks/useContract';

interface RoundStatusProps {
    modelId?: bigint;
}

interface RoundData {
    roundId: bigint;
    modelCommitment: bigint;
    newCommitment: bigint;
    isCompleted: boolean;
    deadline: bigint;
    prover: string;
    proofCount: number;
    startTime: number;
}

export default function RoundStatus({ modelId = BigInt(0) }: RoundStatusProps) {
    const { model, isLoading: modelLoading } = useModel(modelId);
    const currentRoundId = model?.currentRound || BigInt(0);
    const { round, isLoading: roundLoading } = useRound(modelId, currentRoundId);
    const { errorBound } = useErrorBound(modelId);
    const { roundStartedEvents, roundCompletedEvents, proofEvents } = useContractEvents();

    const [selectedRound, setSelectedRound] = useState<bigint | null>(null);
    const [timeRemaining, setTimeRemaining] = useState<number>(0);

    // Calculate time remaining for current round
    useEffect(() => {
        if (!round?.deadline) return;

        const updateTimer = () => {
            const now = Math.floor(Date.now() / 1000);
            const deadline = Number(round.deadline);
            const remaining = Math.max(0, deadline - now);
            setTimeRemaining(remaining);
        };

        updateTimer();
        const interval = setInterval(updateTimer, 1000);
        return () => clearInterval(interval);
    }, [round?.deadline]);

    // Build round history from events
    const roundHistory = useMemo(() => {
        const _rounds: RoundData[] = [];
        const roundMap = new Map<string, RoundData>();

        // Process round started events
        roundStartedEvents.forEach((event) => {
            if (event.modelId === modelId) {
                const key = `${event.modelId}-${event.roundId}`;
                roundMap.set(key, {
                    roundId: event.roundId,
                    modelCommitment: BigInt(0),
                    newCommitment: BigInt(0),
                    isCompleted: false,
                    deadline: event.deadline,
                    prover: '',
                    proofCount: 0,
                    startTime: event.timestamp,
                });
            }
        });

        // Process round completed events
        roundCompletedEvents.forEach((event) => {
            if (event.modelId === modelId) {
                const key = `${event.modelId}-${event.roundId}`;
                const existing = roundMap.get(key);
                if (existing) {
                    existing.isCompleted = true;
                    existing.newCommitment = event.newCommitment;
                }
            }
        });

        // Count proofs per round
        proofEvents.forEach((event) => {
            if (event.modelId === modelId) {
                const key = `${event.modelId}-${event.roundId}`;
                const existing = roundMap.get(key);
                if (existing) {
                    existing.proofCount++;
                    existing.prover = event.prover;
                }
            }
        });

        // Sort by round ID descending
        return Array.from(roundMap.values())
            .sort((a, b) => Number(b.roundId) - Number(a.roundId))
            .slice(0, 10);
    }, [roundStartedEvents, roundCompletedEvents, proofEvents, modelId]);

    // Demo data if no real data
    const displayRounds = roundHistory.length > 0 ? roundHistory : [
        { roundId: BigInt(5), modelCommitment: BigInt(12345), newCommitment: BigInt(12456), isCompleted: true, deadline: BigInt(0), prover: '0x742d...1231', proofCount: 3, startTime: Date.now() / 1000 - 3600 },
        { roundId: BigInt(4), modelCommitment: BigInt(12234), newCommitment: BigInt(12345), isCompleted: true, deadline: BigInt(0), prover: '0x8626...1199', proofCount: 4, startTime: Date.now() / 1000 - 7200 },
        { roundId: BigInt(3), modelCommitment: BigInt(12123), newCommitment: BigInt(12234), isCompleted: true, deadline: BigInt(0), prover: '0xdD2F...44C0', proofCount: 2, startTime: Date.now() / 1000 - 10800 },
        { roundId: BigInt(2), modelCommitment: BigInt(12012), newCommitment: BigInt(12123), isCompleted: true, deadline: BigInt(0), prover: '0x742d...1231', proofCount: 5, startTime: Date.now() / 1000 - 14400 },
        { roundId: BigInt(1), modelCommitment: BigInt(11901), newCommitment: BigInt(12012), isCompleted: true, deadline: BigInt(0), prover: '0xbDA5...197E', proofCount: 3, startTime: Date.now() / 1000 - 18000 },
    ];

    const formatTime = (seconds: number) => {
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        if (hrs > 0) return `${hrs}h ${mins}m ${secs}s`;
        if (mins > 0) return `${mins}m ${secs}s`;
        return `${secs}s`;
    };

    const formatTimestamp = (ts: number) => {
        const date = new Date(ts * 1000);
        return date.toLocaleTimeString();
    };

    const formatCommitment = (c: bigint) => {
        const hex = c.toString(16).padStart(8, '0');
        return `0x${hex.slice(0, 8)}...`;
    };

    const getProgressPercent = () => {
        if (!round?.deadline || round.isCompleted) return 100;
        const now = Math.floor(Date.now() / 1000);
        const deadline = Number(round.deadline);
        const duration = 3600; // Assume 1 hour rounds
        const elapsed = duration - (deadline - now);
        return Math.min(100, Math.max(0, (elapsed / duration) * 100));
    };

    const isLoading = modelLoading || roundLoading;

    return (
        <div className="round-status">
            <style jsx>{`
                .round-status {
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
                    background: linear-gradient(90deg, #6366f1, #a855f7);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }

                .model-badge {
                    padding: 8px 16px;
                    background: rgba(99, 102, 241, 0.2);
                    border-radius: 8px;
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 13px;
                    color: #818cf8;
                }

                .current-round {
                    background: rgba(255, 255, 255, 0.03);
                    border-radius: 16px;
                    padding: 24px;
                    border: 1px solid rgba(99, 102, 241, 0.3);
                    margin-bottom: 24px;
                }

                .current-round-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                }

                .round-number {
                    font-size: 32px;
                    font-weight: 700;
                    color: #6366f1;
                }

                .round-status-badge {
                    padding: 6px 16px;
                    border-radius: 20px;
                    font-size: 12px;
                    font-weight: 600;
                    text-transform: uppercase;
                }

                .status-active {
                    background: rgba(34, 197, 94, 0.2);
                    color: #22c55e;
                }

                .status-completed {
                    background: rgba(99, 102, 241, 0.2);
                    color: #818cf8;
                }

                .status-pending {
                    background: rgba(245, 158, 11, 0.2);
                    color: #f59e0b;
                }

                .progress-container {
                    margin-bottom: 20px;
                }

                .progress-bar-bg {
                    width: 100%;
                    height: 8px;
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 4px;
                    overflow: hidden;
                }

                .progress-bar {
                    height: 100%;
                    background: linear-gradient(90deg, #6366f1, #a855f7);
                    border-radius: 4px;
                    transition: width 1s ease;
                }

                .progress-info {
                    display: flex;
                    justify-content: space-between;
                    margin-top: 8px;
                    font-size: 12px;
                    color: #9ca3af;
                }

                .timer {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }

                .timer-value {
                    font-size: 24px;
                    font-weight: 700;
                    font-family: 'JetBrains Mono', monospace;
                    color: ${timeRemaining < 300 ? '#ef4444' : timeRemaining < 900 ? '#f59e0b' : '#22c55e'};
                }

                .round-details {
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 16px;
                }

                .detail-card {
                    background: rgba(0, 0, 0, 0.2);
                    padding: 16px;
                    border-radius: 12px;
                    text-align: center;
                }

                .detail-value {
                    font-size: 18px;
                    font-weight: 700;
                    margin-bottom: 4px;
                    font-family: 'JetBrains Mono', monospace;
                }

                .detail-label {
                    font-size: 11px;
                    color: #6b7280;
                    text-transform: uppercase;
                }

                .history-section {
                    margin-top: 8px;
                }

                .history-title {
                    font-size: 16px;
                    font-weight: 600;
                    margin-bottom: 16px;
                    color: #d1d5db;
                }

                .history-list {
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                }

                .history-item {
                    display: grid;
                    grid-template-columns: 80px 1fr 100px 80px 120px;
                    align-items: center;
                    gap: 16px;
                    padding: 12px 16px;
                    background: rgba(255, 255, 255, 0.03);
                    border-radius: 8px;
                    border: 1px solid rgba(255, 255, 255, 0.05);
                    cursor: pointer;
                    transition: all 0.2s;
                }

                .history-item:hover {
                    background: rgba(255, 255, 255, 0.06);
                    border-color: rgba(255, 255, 255, 0.1);
                }

                .history-item.selected {
                    border-color: rgba(99, 102, 241, 0.5);
                    background: rgba(99, 102, 241, 0.1);
                }

                .history-round {
                    font-weight: 600;
                    color: #6366f1;
                }

                .history-commitment {
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 12px;
                    color: #9ca3af;
                }

                .history-proofs {
                    font-weight: 600;
                    color: #22c55e;
                }

                .history-prover {
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 12px;
                    color: #a855f7;
                }

                .history-time {
                    font-size: 12px;
                    color: #6b7280;
                    text-align: right;
                }

                .error-bound-section {
                    margin-top: 24px;
                    padding: 16px;
                    background: rgba(239, 68, 68, 0.1);
                    border: 1px solid rgba(239, 68, 68, 0.3);
                    border-radius: 12px;
                }

                .error-bound-title {
                    font-size: 14px;
                    font-weight: 600;
                    color: #ef4444;
                    margin-bottom: 8px;
                }

                .error-bound-value {
                    font-size: 24px;
                    font-weight: 700;
                    font-family: 'JetBrains Mono', monospace;
                    color: #fca5a5;
                }

                .error-bound-bar {
                    width: 100%;
                    height: 6px;
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 3px;
                    margin-top: 12px;
                    overflow: hidden;
                }

                .error-bound-fill {
                    height: 100%;
                    background: linear-gradient(90deg, #22c55e, #f59e0b, #ef4444);
                    border-radius: 3px;
                }

                .loading {
                    text-align: center;
                    padding: 40px;
                    color: #6b7280;
                }
            `}</style>

            <div className="header">
                <h2 className="title">Training Round Progress</h2>
                <span className="model-badge">Model #{modelId.toString()}</span>
            </div>

            {isLoading ? (
                <div className="loading">Loading round data...</div>
            ) : (
                <>
                    <div className="current-round">
                        <div className="current-round-header">
                            <div>
                                <span style={{ fontSize: '12px', color: '#6b7280', textTransform: 'uppercase' }}>Current Round</span>
                                <div className="round-number">#{currentRoundId.toString()}</div>
                            </div>
                            <div className="timer">
                                <span style={{ color: '#6b7280', fontSize: '12px' }}>Time Remaining:</span>
                                <span className="timer-value">{formatTime(timeRemaining)}</span>
                            </div>
                            <span className={`round-status-badge ${round?.isCompleted ? 'status-completed' : 'status-active'}`}>
                                {round?.isCompleted ? 'Completed' : 'Active'}
                            </span>
                        </div>

                        <div className="progress-container">
                            <div className="progress-bar-bg">
                                <div className="progress-bar" style={{ width: `${getProgressPercent()}%` }} />
                            </div>
                            <div className="progress-info">
                                <span>{getProgressPercent().toFixed(0)}% Complete</span>
                                <span>Deadline: {round?.deadline ? new Date(Number(round.deadline) * 1000).toLocaleTimeString() : 'N/A'}</span>
                            </div>
                        </div>

                        <div className="round-details">
                            <div className="detail-card">
                                <div className="detail-value" style={{ color: '#6366f1' }}>
                                    {model?.currentCommitment ? formatCommitment(model.currentCommitment) : '0x0000...'}
                                </div>
                                <div className="detail-label">Model Commitment</div>
                            </div>
                            <div className="detail-card">
                                <div className="detail-value" style={{ color: '#a855f7' }}>
                                    {round?.newCommitment ? formatCommitment(round.newCommitment) : 'Pending'}
                                </div>
                                <div className="detail-label">New Commitment</div>
                            </div>
                            <div className="detail-card">
                                <div className="detail-value" style={{ color: '#22c55e' }}>
                                    {round?.prover ? `${round.prover.slice(0, 6)}...` : 'None'}
                                </div>
                                <div className="detail-label">Current Prover</div>
                            </div>
                            <div className="detail-card">
                                <div className="detail-value" style={{ color: '#f59e0b' }}>
                                    {model?.minStake ? `${Number(model.minStake) / 1e18} ETH` : '0 ETH'}
                                </div>
                                <div className="detail-label">Min Stake</div>
                            </div>
                        </div>
                    </div>

                    {errorBound !== undefined && errorBound !== null && errorBound > BigInt(0) && (
                        <div className="error-bound-section">
                            <div className="error-bound-title">Accumulated Error Bound</div>
                            <div className="error-bound-value">{errorBound.toString()}</div>
                            <div className="error-bound-bar">
                                <div
                                    className="error-bound-fill"
                                    style={{ width: `${Math.min(100, Number(errorBound) / 10)}%` }}
                                />
                            </div>
                            <div style={{ fontSize: '11px', color: '#6b7280', marginTop: '8px' }}>
                                Maximum allowed: 1000 | Current: {errorBound.toString()}
                            </div>
                        </div>
                    )}

                    <div className="history-section">
                        <div className="history-title">Round History</div>
                        <div className="history-list">
                            {displayRounds.map((r) => (
                                <div
                                    key={r.roundId.toString()}
                                    className={`history-item ${selectedRound === r.roundId ? 'selected' : ''}`}
                                    onClick={() => setSelectedRound(r.roundId)}
                                >
                                    <span className="history-round">Round #{r.roundId.toString()}</span>
                                    <span className="history-commitment">
                                        {formatCommitment(r.modelCommitment)} → {formatCommitment(r.newCommitment)}
                                    </span>
                                    <span className="history-proofs">{r.proofCount} proofs</span>
                                    <span className="history-prover">{r.prover}</span>
                                    <span className="history-time">{formatTimestamp(r.startTime)}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}
