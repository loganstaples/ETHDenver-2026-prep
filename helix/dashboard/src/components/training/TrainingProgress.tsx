'use client';

import React, { useState } from 'react';

interface TrainingMetrics {
    epoch: number;
    totalEpochs: number;
    loss: number;
    accuracy: number;
    learningRate: number;
    batchesCompleted: number;
    totalBatches: number;
    timestamp: number;
}

interface TrainingRound {
    roundId: number;
    status: 'pending' | 'in_progress' | 'completed' | 'failed';
    participants: number;
    startTime: number;
    endTime?: number;
    aggregatedGradients?: number;
    proofGenerated: boolean;
}

interface TrainingProgressProps {
    modelName?: string;
    initialMetrics?: TrainingMetrics;
    rounds?: TrainingRound[];
}

export default function TrainingProgress({
    modelName = 'HELIX Model',
    initialMetrics,
    rounds: _rounds = [],
}: TrainingProgressProps) {
    const [metrics, _setMetrics] = useState<TrainingMetrics>(
        initialMetrics || {
            epoch: 3,
            totalEpochs: 10,
            loss: 0.342,
            accuracy: 0.876,
            learningRate: 0.001,
            batchesCompleted: 156,
            totalBatches: 200,
            timestamp: Date.now(),
        }
    );

    const [lossHistory, _setLossHistory] = useState<number[]>([
        0.892, 0.654, 0.512, 0.423, 0.342,
    ]);

    const [accuracyHistory, _setAccuracyHistory] = useState<number[]>([
        0.543, 0.672, 0.745, 0.812, 0.876,
    ]);

    const epochProgress = (metrics.epoch / metrics.totalEpochs) * 100;
    const batchProgress = (metrics.batchesCompleted / metrics.totalBatches) * 100;

    const _formatTime = (timestamp: number) => {
        return new Date(timestamp).toLocaleTimeString();
    };

    const formatDuration = (start: number, end?: number) => {
        const duration = (end || Date.now()) - start;
        const minutes = Math.floor(duration / 60000);
        const seconds = Math.floor((duration % 60000) / 1000);
        return `${minutes}m ${seconds}s`;
    };

    return (
        <div className="bg-helix-surface border border-helix-border rounded-md p-5 text-white">
            {/* Header */}
            <div className="flex justify-between items-center mb-5">
                <h2 className="text-[15px] font-semibold text-white">{modelName}</h2>
                <span className="text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded-sm bg-white/[0.08] text-white border border-white/[0.12]">
                    Training Active
                </span>
            </div>

            {/* Metric Cards */}
            <div className="grid grid-cols-4 gap-3 mb-5">
                <div className="bg-white/[0.03] border border-helix-border rounded-md p-3">
                    <div className="text-[11px] text-[#666] uppercase tracking-wider mb-1.5">
                        Current Loss
                    </div>
                    <div className="text-xl font-mono font-semibold text-white">
                        {metrics.loss.toFixed(4)}
                    </div>
                </div>
                <div className="bg-white/[0.03] border border-helix-border rounded-md p-3">
                    <div className="text-[11px] text-[#666] uppercase tracking-wider mb-1.5">
                        Accuracy
                    </div>
                    <div className="text-xl font-mono font-semibold text-white">
                        {(metrics.accuracy * 100).toFixed(1)}%
                    </div>
                </div>
                <div className="bg-white/[0.03] border border-helix-border rounded-md p-3">
                    <div className="text-[11px] text-[#666] uppercase tracking-wider mb-1.5">
                        Learning Rate
                    </div>
                    <div className="text-xl font-mono font-semibold text-white">
                        {metrics.learningRate}
                    </div>
                </div>
                <div className="bg-white/[0.03] border border-helix-border rounded-md p-3">
                    <div className="text-[11px] text-[#666] uppercase tracking-wider mb-1.5">
                        Epoch
                    </div>
                    <div className="text-xl font-mono font-semibold text-white">
                        {metrics.epoch}/{metrics.totalEpochs}
                    </div>
                </div>
            </div>

            {/* Progress Bars */}
            <div className="mb-5">
                <div className="flex justify-between mb-1.5">
                    <span className="text-[11px] text-[#666] uppercase tracking-wider">Epoch Progress</span>
                    <span className="text-[13px] font-mono text-[#888]">{epochProgress.toFixed(0)}%</span>
                </div>
                <div className="h-1 bg-white/[0.06] rounded-sm overflow-hidden mb-4">
                    <div
                        className="h-1 bg-white rounded-sm transition-all duration-300 ease-in-out"
                        style={{ width: `${epochProgress}%` }}
                    />
                </div>

                <div className="flex justify-between mb-1.5">
                    <span className="text-[11px] text-[#666] uppercase tracking-wider">Batch Progress</span>
                    <span className="text-[13px] font-mono text-[#888]">
                        {metrics.batchesCompleted}/{metrics.totalBatches}
                    </span>
                </div>
                <div className="h-1 bg-white/[0.06] rounded-sm overflow-hidden">
                    <div
                        className="h-1 bg-white rounded-sm transition-all duration-300 ease-in-out"
                        style={{ width: `${batchProgress}%` }}
                    />
                </div>
            </div>

            {/* Charts */}
            <div className="grid grid-cols-2 gap-3 mb-5">
                <div className="bg-white/[0.03] border border-helix-border rounded-md p-3">
                    <div className="text-[13px] font-medium text-white mb-3">
                        Loss Over Epochs
                    </div>
                    <div className="flex items-end h-15 gap-1">
                        {lossHistory.map((loss, i) => (
                            <div
                                key={i}
                                className="flex-1 bg-white rounded-none transition-all duration-300 ease-in-out"
                                style={{
                                    height: `${loss * 100}%`,
                                    opacity: 0.4 + (i * 0.15)
                                }}
                            />
                        ))}
                    </div>
                </div>
                <div className="bg-white/[0.03] border border-helix-border rounded-md p-3">
                    <div className="text-[13px] font-medium text-white mb-3">
                        Accuracy Over Epochs
                    </div>
                    <div className="flex items-end h-15 gap-1">
                        {accuracyHistory.map((acc, i) => (
                            <div
                                key={i}
                                className="flex-1 bg-white rounded-none transition-all duration-300 ease-in-out"
                                style={{
                                    height: `${acc * 100}%`,
                                    opacity: 0.4 + (i * 0.15)
                                }}
                            />
                        ))}
                    </div>
                </div>
            </div>

            {/* Training Rounds */}
            <div className="border-t border-helix-border pt-5">
                <h3 className="text-[13px] font-medium text-white mb-3">Training Rounds</h3>
                <div className="flex flex-col">
                    {[
                        {
                            roundId: 3,
                            status: 'in_progress' as const,
                            participants: 8,
                            startTime: Date.now() - 120000,
                            proofGenerated: false,
                        },
                        {
                            roundId: 2,
                            status: 'completed' as const,
                            participants: 12,
                            startTime: Date.now() - 300000,
                            endTime: Date.now() - 180000,
                            proofGenerated: true,
                        },
                        {
                            roundId: 1,
                            status: 'completed' as const,
                            participants: 10,
                            startTime: Date.now() - 600000,
                            endTime: Date.now() - 480000,
                            proofGenerated: true,
                        },
                    ].map((round) => (
                        <div
                            key={round.roundId}
                            className="flex items-center justify-between px-0 py-2.5 border-b border-helix-border last:border-b-0"
                        >
                            <div className="flex items-center gap-3">
                                <span className="text-[13px] font-medium text-white">
                                    Round #{round.roundId}
                                </span>
                                <span
                                    className={`text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded-sm ${
                                        round.status === 'completed'
                                            ? 'bg-white/[0.06] text-[#666]'
                                            : round.status === 'in_progress'
                                            ? 'bg-white/[0.1] text-white'
                                            : 'bg-white/[0.06] text-[#666]'
                                    }`}
                                >
                                    {round.status.replace('_', ' ')}
                                </span>
                            </div>
                            <div className="flex gap-5 text-[13px] text-[#555]">
                                <span className="font-mono">{round.participants} participants</span>
                                <span className="font-mono">{formatDuration(round.startTime, round.endTime)}</span>
                                {round.proofGenerated && (
                                    <span className="flex items-center gap-1 text-white text-[13px]">
                                        &#x2713; Proof
                                    </span>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
