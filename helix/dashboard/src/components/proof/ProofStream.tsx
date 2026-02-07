'use client';

/**
 * HELIX Proof Stream Component
 * Real-time proof generation and verification status with live updates.
 */

import React, { useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Shield,
    CheckCircle,
    XCircle,
    Clock,
    Loader2,
    ChevronRight,
    ExternalLink,
    Zap,
    Activity,
    AlertTriangle,
} from 'lucide-react';
import { useProofStream, type ProofStreamItem, type ProofStage } from '@/hooks/useProofStream';

// ============================================================================
// Types
// ============================================================================

interface ProofStreamProps {
    modelId?: bigint | number;
    maxDisplay?: number;
    showStats?: boolean;
    showOnChain?: boolean;
    compact?: boolean;
    enableSimulation?: boolean;
    className?: string;
}

// ============================================================================
// Helper Functions
// ============================================================================

function getStageInfo(stage: ProofStage): {
    label: string;
    color: string;
    bgColor: string;
    icon: React.ReactNode;
} {
    switch (stage) {
        case 'queued':
            return {
                label: 'Queued',
                color: 'text-neutral-400',
                bgColor: 'bg-neutral-500/20',
                icon: <Clock className="w-4 h-4" />,
            };
        case 'witness':
            return {
                label: 'Witness Generation',
                color: 'text-blue-400',
                bgColor: 'bg-blue-500/20',
                icon: <Loader2 className="w-4 h-4 animate-spin" />,
            };
        case 'setup':
            return {
                label: 'Circuit Setup',
                color: 'text-purple-400',
                bgColor: 'bg-purple-500/20',
                icon: <Loader2 className="w-4 h-4 animate-spin" />,
            };
        case 'proving':
            return {
                label: 'Generating Proof',
                color: 'text-amber-400',
                bgColor: 'bg-amber-500/20',
                icon: <Zap className="w-4 h-4 animate-pulse" />,
            };
        case 'submitting':
            return {
                label: 'Submitting',
                color: 'text-cyan-400',
                bgColor: 'bg-cyan-500/20',
                icon: <Loader2 className="w-4 h-4 animate-spin" />,
            };
        case 'verifying':
            return {
                label: 'Verifying',
                color: 'text-indigo-400',
                bgColor: 'bg-indigo-500/20',
                icon: <Shield className="w-4 h-4 animate-pulse" />,
            };
        case 'verified':
            return {
                label: 'Verified',
                color: 'text-emerald-400',
                bgColor: 'bg-emerald-500/20',
                icon: <CheckCircle className="w-4 h-4" />,
            };
        case 'failed':
            return {
                label: 'Failed',
                color: 'text-red-400',
                bgColor: 'bg-red-500/20',
                icon: <XCircle className="w-4 h-4" />,
            };
        default:
            return {
                label: 'Unknown',
                color: 'text-neutral-400',
                bgColor: 'bg-neutral-500/20',
                icon: <Clock className="w-4 h-4" />,
            };
    }
}

function formatDuration(ms: number): string {
    if (ms < 1000) return `${Math.round(ms)}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
}

function formatTimestamp(timestamp: number): string {
    const now = Date.now();
    const diff = now - timestamp;

    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return new Date(timestamp).toLocaleDateString();
}

// ============================================================================
// Sub-Components
// ============================================================================

function ProofCard({
    proof,
    compact,
    onSelect,
}: {
    proof: ProofStreamItem;
    compact?: boolean;
    onSelect?: (proof: ProofStreamItem) => void;
}) {
    const stageInfo = getStageInfo(proof.stage);
    const isActive = !['verified', 'failed'].includes(proof.stage);

    return (
        <motion.div
            layout
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, x: -100 }}
            transition={{ duration: 0.2 }}
            className={`bg-neutral-800/50 rounded-lg border ${
                isActive ? 'border-neutral-600' : 'border-neutral-800'
            } hover:border-neutral-600 transition-colors cursor-pointer ${compact ? 'p-3' : 'p-4'}`}
            onClick={() => onSelect?.(proof)}
        >
            <div className="flex items-center justify-between gap-4">
                {/* Left: Status and ID */}
                <div className="flex items-center gap-3 min-w-0">
                    <div className={`w-10 h-10 rounded-lg ${stageInfo.bgColor} flex items-center justify-center flex-shrink-0`}>
                        <span className={stageInfo.color}>{stageInfo.icon}</span>
                    </div>
                    <div className="min-w-0">
                        <div className="flex items-center gap-2">
                            <span className="text-sm font-medium text-white truncate">
                                {proof.id.slice(0, 20)}...
                            </span>
                            <span className={`px-2 py-0.5 rounded text-xs font-medium ${stageInfo.bgColor} ${stageInfo.color}`}>
                                {stageInfo.label}
                            </span>
                        </div>
                        <div className="flex items-center gap-2 text-xs text-neutral-400">
                            <span>Round #{proof.roundId.toString()}</span>
                            <span>·</span>
                            <span>{proof.type}</span>
                            <span>·</span>
                            <span>{formatTimestamp(proof.startedAt)}</span>
                        </div>
                    </div>
                </div>

                {/* Right: Progress or Stats */}
                <div className="flex items-center gap-4 flex-shrink-0">
                    {isActive ? (
                        <div className="flex items-center gap-3">
                            <div className="w-32 h-2 bg-neutral-700 rounded-full overflow-hidden">
                                <motion.div
                                    className={`h-full rounded-full ${
                                        proof.stage === 'proving' ? 'bg-amber-500' : 'bg-blue-500'
                                    }`}
                                    initial={{ width: 0 }}
                                    animate={{ width: `${proof.progress}%` }}
                                    transition={{ duration: 0.3 }}
                                />
                            </div>
                            <span className="text-sm font-medium text-neutral-300 w-12 text-right">
                                {proof.progress.toFixed(0)}%
                            </span>
                        </div>
                    ) : (
                        <div className="flex items-center gap-4 text-sm">
                            {proof.metrics.totalTime && (
                                <div className="text-neutral-400">
                                    <span className="text-neutral-500">Time:</span>{' '}
                                    <span className="text-white">{formatDuration(proof.metrics.totalTime)}</span>
                                </div>
                            )}
                            {proof.gasUsed && (
                                <div className="text-neutral-400">
                                    <span className="text-neutral-500">Gas:</span>{' '}
                                    <span className="text-white">{(Number(proof.gasUsed) / 1000).toFixed(0)}k</span>
                                </div>
                            )}
                        </div>
                    )}

                    <ChevronRight className="w-5 h-5 text-neutral-500" />
                </div>
            </div>

            {/* On-chain info for verified proofs */}
            {proof.stage === 'verified' && proof.transactionHash && !compact && (
                <div className="mt-3 pt-3 border-t border-neutral-700/50 flex items-center justify-between">
                    <div className="flex items-center gap-2 text-xs text-neutral-400">
                        <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                        <span>On-chain verified</span>
                        <span>·</span>
                        <span>Block #{proof.blockNumber}</span>
                    </div>
                    <a
                        href={`https://etherscan.io/tx/${proof.transactionHash}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300"
                        onClick={(e) => e.stopPropagation()}
                    >
                        View transaction
                        <ExternalLink className="w-3 h-3" />
                    </a>
                </div>
            )}
        </motion.div>
    );
}

function StatsBar({ stats }: { stats: ReturnType<typeof useProofStream>['stats'] }) {
    return (
        <div className="grid grid-cols-4 gap-4 p-4 bg-neutral-800/30 rounded-lg border border-neutral-800">
            <div className="text-center">
                <div className="text-2xl font-bold text-emerald-400">{stats.verified}</div>
                <div className="text-xs text-neutral-400">Verified</div>
            </div>
            <div className="text-center">
                <div className="text-2xl font-bold text-amber-400">{stats.generating}</div>
                <div className="text-xs text-neutral-400">Generating</div>
            </div>
            <div className="text-center">
                <div className="text-2xl font-bold text-blue-400">{stats.submitting}</div>
                <div className="text-xs text-neutral-400">Submitting</div>
            </div>
            <div className="text-center">
                <div className="text-2xl font-bold text-neutral-400">{stats.queued}</div>
                <div className="text-xs text-neutral-400">Queued</div>
            </div>
        </div>
    );
}

function CurrentProofProgress({ proof }: { proof: ProofStreamItem }) {
    const stageInfo = getStageInfo(proof.stage);
    const stages: ProofStage[] = ['queued', 'witness', 'setup', 'proving', 'submitting', 'verifying', 'verified'];
    const currentIndex = stages.indexOf(proof.stage);

    return (
        <div className="bg-neutral-800/50 rounded-lg border border-neutral-700 p-4">
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                    <div className={`w-10 h-10 rounded-lg ${stageInfo.bgColor} flex items-center justify-center`}>
                        <span className={stageInfo.color}>{stageInfo.icon}</span>
                    </div>
                    <div>
                        <div className="text-sm font-medium text-white">{stageInfo.label}</div>
                        <div className="text-xs text-neutral-400">
                            {proof.circuitType} · {(proof.constraintCount / 1000).toFixed(0)}k constraints
                        </div>
                    </div>
                </div>
                <div className="text-right">
                    <div className="text-lg font-bold text-white">{proof.progress.toFixed(0)}%</div>
                    <div className="text-xs text-neutral-400">
                        {formatDuration(Date.now() - proof.startedAt)} elapsed
                    </div>
                </div>
            </div>

            {/* Stage progress */}
            <div className="flex items-center gap-1 mb-3">
                {stages.slice(0, -1).map((stage, i) => {
                    const isComplete = i < currentIndex;
                    const isCurrent = i === currentIndex;
                    const info = getStageInfo(stage);

                    return (
                        <div
                            key={stage}
                            className={`flex-1 h-2 rounded-full transition-colors ${
                                isComplete
                                    ? 'bg-emerald-500'
                                    : isCurrent
                                        ? info.bgColor.replace('/20', '')
                                        : 'bg-neutral-700'
                            }`}
                        >
                            {isCurrent && (
                                <motion.div
                                    className={`h-full rounded-full ${info.bgColor.replace('/20', '')}`}
                                    style={{ width: `${proof.progress}%` }}
                                    animate={{ width: `${proof.progress}%` }}
                                    transition={{ duration: 0.3 }}
                                />
                            )}
                        </div>
                    );
                })}
            </div>

            {/* Resource usage */}
            <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                    <div className="text-neutral-400 text-xs mb-1">Memory Usage</div>
                    <div className="flex items-center gap-2">
                        <div className="flex-1 h-1.5 bg-neutral-700 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-purple-500 rounded-full"
                                style={{ width: `${Math.min((proof.metrics.memoryPeak || 0) / 80, 100)}%` }}
                            />
                        </div>
                        <span className="text-neutral-300 text-xs">
                            {((proof.metrics.memoryPeak || 0) / 1000).toFixed(1)} GB
                        </span>
                    </div>
                </div>
                <div>
                    <div className="text-neutral-400 text-xs mb-1">CPU Usage</div>
                    <div className="flex items-center gap-2">
                        <div className="flex-1 h-1.5 bg-neutral-700 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-blue-500 rounded-full"
                                style={{ width: `${proof.metrics.cpuPeak || 0}%` }}
                            />
                        </div>
                        <span className="text-neutral-300 text-xs">
                            {(proof.metrics.cpuPeak || 0).toFixed(0)}%
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
}

// ============================================================================
// Main Component
// ============================================================================

export default function ProofStream({
    modelId,
    maxDisplay = 10,
    showStats = true,
    showOnChain: _showOnChain = true,
    compact = false,
    enableSimulation = true,
    className = '',
}: ProofStreamProps) {
    const [_selectedProof, setSelectedProof] = useState<ProofStreamItem | null>(null);
    const [filter, setFilter] = useState<'all' | 'active' | 'verified' | 'failed'>('all');

    const {
        proofs,
        activeProofs,
        stats,
        currentGenerating,
        isLoading,
        error,
    } = useProofStream({
        modelId,
        maxProofs: 100,
        enableWebSocket: true,
        enableSimulation,
    });

    const filteredProofs = useMemo(() => {
        let result = proofs;

        switch (filter) {
            case 'active':
                result = result.filter(p => !['verified', 'failed'].includes(p.stage));
                break;
            case 'verified':
                result = result.filter(p => p.stage === 'verified');
                break;
            case 'failed':
                result = result.filter(p => p.stage === 'failed');
                break;
        }

        return result.slice(0, maxDisplay);
    }, [proofs, filter, maxDisplay]);

    if (isLoading) {
        return (
            <div className={`bg-neutral-900/50 rounded-xl border border-neutral-800 p-6 ${className}`}>
                <div className="flex items-center justify-center h-48">
                    <Loader2 className="w-8 h-8 text-neutral-500 animate-spin" />
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className={`bg-neutral-900/50 rounded-xl border border-neutral-800 p-6 ${className}`}>
                <div className="flex items-center justify-center h-48 text-red-400">
                    <AlertTriangle className="w-6 h-6 mr-2" />
                    <span>{error}</span>
                </div>
            </div>
        );
    }

    return (
        <div className={`bg-neutral-900/50 rounded-xl border border-neutral-800 overflow-hidden ${className}`}>
            {/* Header */}
            <div className="p-4 border-b border-neutral-800">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-indigo-500/20 flex items-center justify-center">
                            <Shield className="w-5 h-5 text-indigo-400" />
                        </div>
                        <div>
                            <h3 className="font-semibold text-white">Proof Stream</h3>
                            <p className="text-sm text-neutral-400">
                                {activeProofs.length} active · {stats.verified} verified
                            </p>
                        </div>
                    </div>

                    {/* Filter tabs */}
                    <div className="flex items-center gap-1 bg-neutral-800 rounded-lg p-1">
                        {(['all', 'active', 'verified', 'failed'] as const).map((f) => (
                            <button
                                key={f}
                                onClick={() => setFilter(f)}
                                className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                                    filter === f
                                        ? 'bg-neutral-700 text-white'
                                        : 'text-neutral-400 hover:text-white'
                                }`}
                            >
                                {f.charAt(0).toUpperCase() + f.slice(1)}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* Stats */}
            {showStats && (
                <div className="p-4 border-b border-neutral-800">
                    <StatsBar stats={stats} />
                </div>
            )}

            {/* Current generating proof */}
            {currentGenerating && filter !== 'verified' && filter !== 'failed' && (
                <div className="p-4 border-b border-neutral-800">
                    <div className="text-xs text-neutral-400 uppercase tracking-wide mb-2">
                        Currently Generating
                    </div>
                    <CurrentProofProgress proof={currentGenerating} />
                </div>
            )}

            {/* Proof list */}
            <div className="p-4">
                <div className="space-y-3">
                    <AnimatePresence mode="popLayout">
                        {filteredProofs.map((proof) => (
                            <ProofCard
                                key={proof.id}
                                proof={proof}
                                compact={compact}
                                onSelect={setSelectedProof}
                            />
                        ))}
                    </AnimatePresence>

                    {filteredProofs.length === 0 && (
                        <div className="text-center py-8 text-neutral-500">
                            No proofs match the current filter
                        </div>
                    )}
                </div>
            </div>

            {/* Footer stats */}
            <div className="px-4 py-3 border-t border-neutral-800 bg-neutral-800/30">
                <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-4 text-neutral-400">
                        <div className="flex items-center gap-1">
                            <Activity className="w-4 h-4" />
                            <span>Throughput:</span>
                            <span className="text-white font-medium">{stats.throughputPerMinute}/min</span>
                        </div>
                        <div className="flex items-center gap-1">
                            <Zap className="w-4 h-4" />
                            <span>Avg Time:</span>
                            <span className="text-white font-medium">{formatDuration(stats.averageGenerationTime)}</span>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="text-neutral-400">Success Rate:</span>
                        <span className={`font-medium ${
                            stats.successRate > 0.95 ? 'text-emerald-400' :
                            stats.successRate > 0.85 ? 'text-amber-400' : 'text-red-400'
                        }`}>
                            {(stats.successRate * 100).toFixed(1)}%
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
}
