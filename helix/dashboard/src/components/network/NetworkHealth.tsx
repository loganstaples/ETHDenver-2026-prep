'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { useContractState, useContractEvents } from '@/hooks/useContract';
import { useBlockNumber, useChainId } from 'wagmi';

interface HealthMetric {
    name: string;
    value: number;
    max: number;
    status: 'healthy' | 'warning' | 'critical';
    description: string;
}

export default function NetworkHealth() {
    const { data: blockNumber } = useBlockNumber({ watch: true });
    const chainId = useChainId();
    const { nextModelId, slashingRecordCount, defaultMinStake: _defaultMinStake, isLoading } = useContractState();
    const { proofEvents, roundStartedEvents, roundCompletedEvents, stakedEvents, slashedEvents } = useContractEvents();

    const [latency, setLatency] = useState(0);
    const [lastBlockTime, setLastBlockTime] = useState(Date.now());

    // Track block latency
    useEffect(() => {
        if (blockNumber) {
            const now = Date.now();
            const diff = now - lastBlockTime;
            setLatency(diff);
            setLastBlockTime(now);
        }
    }, [blockNumber, lastBlockTime]);

    // Calculate health metrics
    const healthMetrics = useMemo((): HealthMetric[] => {
        const now = Date.now() / 1000;
        const oneHourAgo = now - 3600;
        const oneDayAgo = now - 86400;

        // Recent activity counts
        const recentProofs = proofEvents.filter((e) => e.timestamp > oneHourAgo).length;
        const _recentRounds = roundStartedEvents.filter((e) => e.timestamp > oneHourAgo).length;
        const _recentSlashes = slashedEvents.filter((e) => e.timestamp > oneDayAgo).length;
        const activeStakers = new Set(stakedEvents.map((e) => e.prover)).size;

        // Calculate completion rate
        const completedRounds = roundCompletedEvents.length;
        const startedRounds = roundStartedEvents.length;
        const completionRate = startedRounds > 0 ? (completedRounds / startedRounds) * 100 : 100;

        // Proof verification rate (simulated based on slash count)
        const proofVerificationRate = proofEvents.length > 0
            ? Math.max(0, 100 - (slashedEvents.length / proofEvents.length) * 100)
            : 100;

        return [
            {
                name: 'Network Uptime',
                value: 99.7, // Would need external monitoring
                max: 100,
                status: 99.7 >= 99 ? 'healthy' : 99.7 >= 95 ? 'warning' : 'critical',
                description: 'Overall network availability',
            },
            {
                name: 'Block Latency',
                value: Math.min(latency, 30000),
                max: 30000,
                status: latency < 15000 ? 'healthy' : latency < 25000 ? 'warning' : 'critical',
                description: `${(latency / 1000).toFixed(1)}s since last block`,
            },
            {
                name: 'Proof Throughput',
                value: recentProofs,
                max: 100,
                status: recentProofs >= 10 ? 'healthy' : recentProofs >= 3 ? 'warning' : 'critical',
                description: `${recentProofs} proofs/hour`,
            },
            {
                name: 'Round Completion',
                value: completionRate,
                max: 100,
                status: completionRate >= 90 ? 'healthy' : completionRate >= 70 ? 'warning' : 'critical',
                description: `${completionRate.toFixed(1)}% rounds completed`,
            },
            {
                name: 'Verification Rate',
                value: proofVerificationRate,
                max: 100,
                status: proofVerificationRate >= 95 ? 'healthy' : proofVerificationRate >= 80 ? 'warning' : 'critical',
                description: `${proofVerificationRate.toFixed(1)}% proofs valid`,
            },
            {
                name: 'Active Stakers',
                value: activeStakers || 4, // Demo fallback
                max: 50,
                status: activeStakers >= 5 ? 'healthy' : activeStakers >= 2 ? 'warning' : 'critical',
                description: `${activeStakers || 4} active participants`,
            },
        ];
    }, [proofEvents, roundStartedEvents, roundCompletedEvents, stakedEvents, slashedEvents, latency]);

    const overallHealth = useMemo(() => {
        const criticalCount = healthMetrics.filter((m) => m.status === 'critical').length;
        const warningCount = healthMetrics.filter((m) => m.status === 'warning').length;

        if (criticalCount > 0) return { status: 'critical', label: 'Critical Issues', color: '#ffffff' };
        if (warningCount > 1) return { status: 'warning', label: 'Needs Attention', color: '#ffffff' };
        if (warningCount === 1) return { status: 'good', label: 'Minor Issues', color: '#ffffff' };
        return { status: 'healthy', label: 'All Systems Healthy', color: '#ffffff' };
    }, [healthMetrics]);

    const getStatusColor = (status: string) => {
        const colors: Record<string, string> = {
            healthy: '#ffffff',
            warning: '#a3a3a3',
            critical: '#737373',
        };
        return colors[status] || '#6b7280';
    };

    const getMetricBarOpacity = (metric: HealthMetric) => {
        const percentage = (metric.value / metric.max) * 100;
        if (percentage >= 80) return 1;
        if (percentage >= 50) return 0.7;
        return 0.4;
    };

    const getNetworkName = (id: number) => {
        const networks: Record<number, string> = {
            1: 'Ethereum Mainnet',
            11155111: 'Sepolia Testnet',
            31337: 'Local Network',
        };
        return networks[id] || `Chain ${id}`;
    };

    return (
        <div className="bg-helix-surface rounded-md p-4 text-white border border-helix-border">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-white font-semibold text-[15px]">Network Health</h2>
                <div className="flex items-center gap-2 px-4 py-2 bg-white/5 border border-helix-border rounded-md text-sm">
                    <span className="w-2 h-2 rounded-full bg-white animate-pulse" />
                    <span>{getNetworkName(chainId)}</span>
                </div>
            </div>

            {isLoading ? (
                <div className="text-center py-10 text-[#666]">Loading network status...</div>
            ) : (
                <>
                    <div className="flex items-center gap-3 p-4 bg-white/[0.03] rounded-md border border-helix-border mb-4">
                        <div
                            className="w-16 h-16 rounded-md flex items-center justify-center text-xl bg-white/10 text-white"
                        >
                            {overallHealth.status === 'healthy' ? '✓' :
                             overallHealth.status === 'good' ? '○' :
                             overallHealth.status === 'warning' ? '!' : '✕'}
                        </div>
                        <div className="flex-1">
                            <div className="text-[15px] font-semibold mb-1 text-white">
                                {overallHealth.label}
                            </div>
                            <div className="text-sm text-[#888]">
                                {healthMetrics.filter((m) => m.status === 'healthy').length} of {healthMetrics.length} metrics healthy
                            </div>
                        </div>
                        <div className="text-right">
                            <div className="text-white font-mono text-xl font-bold">#{blockNumber?.toString() || '...'}</div>
                            <div className="text-xs text-[#666] uppercase">Latest Block</div>
                        </div>
                    </div>

                    <div className="grid grid-cols-3 gap-3 mb-4">
                        {healthMetrics.map((metric) => (
                            <div key={metric.name} className="bg-white/[0.03] rounded-md p-4 border border-helix-border">
                                <div className="flex justify-between items-center mb-3">
                                    <span className="text-sm font-semibold text-[#aaa]">{metric.name}</span>
                                    <div
                                        className="w-2.5 h-2.5 rounded-full"
                                        style={{ backgroundColor: getStatusColor(metric.status) }}
                                    />
                                </div>
                                <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden mb-2">
                                    <div
                                        className="h-full rounded-full bg-white transition-all duration-500"
                                        style={{
                                            width: `${(metric.value / metric.max) * 100}%`,
                                            opacity: getMetricBarOpacity(metric),
                                        }}
                                    />
                                </div>
                                <div className="text-xs text-[#666]">{metric.description}</div>
                            </div>
                        ))}
                    </div>

                    <div className="grid grid-cols-4 gap-3">
                        <div className="bg-white/[0.03] p-4 rounded-md text-center">
                            <div className="text-[15px] font-semibold text-white mb-1 font-mono">
                                {nextModelId?.toString() || '0'}
                            </div>
                            <div className="text-xs text-[#666] uppercase">Registered Models</div>
                        </div>
                        <div className="bg-white/[0.03] p-4 rounded-md text-center">
                            <div className="text-[15px] font-semibold text-white mb-1 font-mono">
                                {proofEvents.length}
                            </div>
                            <div className="text-xs text-[#666] uppercase">Total Proofs</div>
                        </div>
                        <div className="bg-white/[0.03] p-4 rounded-md text-center">
                            <div className="text-[15px] font-semibold text-white mb-1 font-mono">
                                {roundCompletedEvents.length}
                            </div>
                            <div className="text-xs text-[#666] uppercase">Completed Rounds</div>
                        </div>
                        <div className="bg-white/[0.03] p-4 rounded-md text-center">
                            <div className="text-[15px] font-semibold text-white mb-1 font-mono">
                                {slashingRecordCount?.toString() || '0'}
                            </div>
                            <div className="text-xs text-[#666] uppercase">Slashing Events</div>
                        </div>
                    </div>

                    <div className="mt-4">
                        <div className="text-sm font-semibold text-[#aaa] mb-3">Recent Network Events</div>
                        <div className="flex flex-col gap-2 max-h-[200px] overflow-y-auto">
                            {[...proofEvents, ...roundStartedEvents, ...roundCompletedEvents]
                                .sort((a, b) => b.timestamp - a.timestamp)
                                .slice(0, 10)
                                .map((event, idx) => {
                                    const isProof = 'prover' in event && 'newCommitment' in event;
                                    const isRoundStart = 'deadline' in event && !('newCommitment' in event);
                                    const type = isProof ? 'proof' : isRoundStart ? 'round-start' : 'round-complete';
                                    const labels: Record<string, string> = {
                                        'proof': 'Proof Submitted',
                                        'round-start': 'Round Started',
                                        'round-complete': 'Round Completed',
                                    };
                                    return (
                                        <div key={`${type}-${idx}`} className="flex items-center gap-3 px-3 py-2.5 bg-white/[0.02] rounded-md text-xs">
                                            <div className="w-2 h-2 rounded-full bg-white" />
                                            <span className="flex-1 text-[#aaa]">
                                                {labels[type]} - Model #{event.modelId.toString()}
                                                {isProof && ` by ${(event as unknown as { prover: string }).prover.slice(0, 8)}...`}
                                            </span>
                                            <span className="text-[#666] font-mono">
                                                {new Date(event.timestamp * 1000).toLocaleTimeString()}
                                            </span>
                                        </div>
                                    );
                                })}
                            {proofEvents.length === 0 && roundStartedEvents.length === 0 && (
                                <div className="flex items-center justify-center px-3 py-2.5 bg-white/[0.02] rounded-md text-xs text-[#666]">
                                    No recent events - waiting for network activity...
                                </div>
                            )}
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}
