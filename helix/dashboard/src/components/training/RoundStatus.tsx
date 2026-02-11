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
        <div className="bg-helix-surface rounded-md p-4 text-white border border-helix-border">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-white font-semibold text-[15px]">Training Round Progress</h2>
                <span className="bg-white/5 border border-helix-border rounded-md px-4 py-2 font-mono text-sm text-[#888]">
                    Model #{modelId.toString()}
                </span>
            </div>

            {isLoading ? (
                <div className="text-center py-10 text-[#666]">Loading round data...</div>
            ) : (
                <>
                    <div className="bg-white/[0.03] rounded-md p-4 border border-helix-border mb-4">
                        <div className="flex justify-between items-center mb-4">
                            <div>
                                <span className="text-xs text-[#666] uppercase">Current Round</span>
                                <div className="text-xl font-bold text-white">#{currentRoundId.toString()}</div>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className="text-[#666] text-xs">Time Remaining:</span>
                                <span className="text-lg font-bold font-mono text-white">{formatTime(timeRemaining)}</span>
                            </div>
                            <span className={`px-4 py-1.5 rounded-md text-xs font-semibold uppercase ${round?.isCompleted ? 'bg-white/5 text-[#666]' : 'bg-white/10 text-white'}`}>
                                {round?.isCompleted ? 'Completed' : 'Active'}
                            </span>
                        </div>

                        <div className="mb-4">
                            <div className="w-full h-2 bg-white/10 rounded overflow-hidden">
                                <div
                                    className="h-full bg-white rounded transition-all duration-1000 ease-in-out"
                                    style={{ width: `${getProgressPercent()}%` }}
                                />
                            </div>
                            <div className="flex justify-between mt-2 text-xs text-[#888]">
                                <span>{getProgressPercent().toFixed(0)}% Complete</span>
                                <span>Deadline: {round?.deadline ? new Date(Number(round.deadline) * 1000).toLocaleTimeString() : 'N/A'}</span>
                            </div>
                        </div>

                        <div className="grid grid-cols-4 gap-3">
                            <div className="bg-white/[0.03] p-4 rounded-md text-center">
                                <div className="text-lg font-bold mb-1 font-mono text-white">
                                    {model?.currentCommitment ? formatCommitment(model.currentCommitment) : '0x0000...'}
                                </div>
                                <div className="text-[11px] text-[#666] uppercase">Model Commitment</div>
                            </div>
                            <div className="bg-white/[0.03] p-4 rounded-md text-center">
                                <div className="text-lg font-bold mb-1 font-mono text-white">
                                    {round?.newCommitment ? formatCommitment(round.newCommitment) : 'Pending'}
                                </div>
                                <div className="text-[11px] text-[#666] uppercase">New Commitment</div>
                            </div>
                            <div className="bg-white/[0.03] p-4 rounded-md text-center">
                                <div className="text-lg font-bold mb-1 font-mono text-white">
                                    {round?.prover ? `${round.prover.slice(0, 6)}...` : 'None'}
                                </div>
                                <div className="text-[11px] text-[#666] uppercase">Current Prover</div>
                            </div>
                            <div className="bg-white/[0.03] p-4 rounded-md text-center">
                                <div className="text-lg font-bold mb-1 font-mono text-white">
                                    {model?.minStake ? `${Number(model.minStake) / 1e18} ETH` : '0 ETH'}
                                </div>
                                <div className="text-[11px] text-[#666] uppercase">Min Stake</div>
                            </div>
                        </div>
                    </div>

                    {errorBound !== undefined && errorBound !== null && errorBound > BigInt(0) && (
                        <div className="mt-4 p-4 bg-white/5 border border-helix-border rounded-md">
                            <div className="text-sm font-semibold text-white mb-2">Accumulated Error Bound</div>
                            <div className="text-lg font-bold font-mono text-white">{errorBound.toString()}</div>
                            <div className="w-full h-1.5 bg-white/10 rounded mt-3 overflow-hidden">
                                <div
                                    className="h-full bg-white rounded"
                                    style={{ width: `${Math.min(100, Number(errorBound) / 10)}%` }}
                                />
                            </div>
                            <div className="text-[11px] text-[#666] mt-2">
                                Maximum allowed: 1000 | Current: {errorBound.toString()}
                            </div>
                        </div>
                    )}

                    <div className="mt-2">
                        <div className="text-base font-semibold mb-4 text-[#aaa]">Round History</div>
                        <div className="flex flex-col gap-2">
                            {displayRounds.map((r) => (
                                <div
                                    key={r.roundId.toString()}
                                    className={`grid grid-cols-[80px_1fr_100px_80px_120px] items-center gap-3 px-4 py-3 rounded-md border cursor-pointer transition-all ${
                                        selectedRound === r.roundId
                                            ? 'border-white/10 bg-white/10'
                                            : 'bg-white/[0.03] border-helix-border hover:bg-white/[0.06]'
                                    }`}
                                    onClick={() => setSelectedRound(r.roundId)}
                                >
                                    <span className="font-semibold text-white">Round #{r.roundId.toString()}</span>
                                    <span className="font-mono text-xs text-[#888]">
                                        {formatCommitment(r.modelCommitment)} → {formatCommitment(r.newCommitment)}
                                    </span>
                                    <span className="font-semibold text-white">{r.proofCount} proofs</span>
                                    <span className="font-mono text-xs text-[#888]">{r.prover}</span>
                                    <span className="text-xs text-[#666] text-right">{formatTimestamp(r.startTime)}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}
