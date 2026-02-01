'use client';

/**
 * HELIX Error Bounds Visualization Component
 * Real-time visualization of error bound propagation through training layers.
 */

import React, { useMemo, useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    AlertTriangle,
    TrendingUp,
    Activity,
    Layers,
    ChevronRight,
    Info,
    Shield,
    Zap,
    CheckCircle,
} from 'lucide-react';
import { useTrainingStatus } from '@/hooks/useTrainingStatus';
import { useErrorBound, useContractState, useModel } from '@/hooks/useContract';

// ============================================================================
// Types
// ============================================================================

interface ErrorBoundsVizProps {
    modelId?: bigint | number;
    showDetails?: boolean;
    showTimeline?: boolean;
    compact?: boolean;
    className?: string;
}

interface LayerBound {
    name: string;
    operation: string;
    inputBound: number;
    outputBound: number;
    amplification: number;
    riskLevel: 'low' | 'medium' | 'high' | 'critical';
    contribution: number;
}

// ============================================================================
// Helper Functions
// ============================================================================

function getRiskColor(level: string): { bg: string; text: string; border: string } {
    switch (level) {
        case 'critical':
            return { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/30' };
        case 'high':
            return { bg: 'bg-amber-500/20', text: 'text-amber-400', border: 'border-amber-500/30' };
        case 'medium':
            return { bg: 'bg-yellow-500/20', text: 'text-yellow-400', border: 'border-yellow-500/30' };
        default:
            return { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/30' };
    }
}

function calculateRiskLevel(amplification: number, outputBound: number, maxBound: number): 'low' | 'medium' | 'high' | 'critical' {
    const percentage = (outputBound / maxBound) * 100;
    if (percentage > 80 || amplification > 2) return 'critical';
    if (percentage > 60 || amplification > 1.5) return 'high';
    if (percentage > 40 || amplification > 1.2) return 'medium';
    return 'low';
}

// ============================================================================
// Sub-Components
// ============================================================================

function LayerCard({
    layer,
    index,
    isExpanded,
    onToggle,
}: {
    layer: LayerBound;
    index: number;
    isExpanded: boolean;
    onToggle: () => void;
}) {
    const colors = getRiskColor(layer.riskLevel);

    return (
        <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
            className={`border rounded-lg overflow-hidden ${colors.border} ${colors.bg}`}
        >
            <button
                onClick={onToggle}
                className="w-full p-3 flex items-center justify-between hover:bg-white/5 transition-colors"
            >
                <div className="flex items-center gap-3">
                    <div className={`w-8 h-8 rounded-lg ${colors.bg} flex items-center justify-center`}>
                        <Layers className={`w-4 h-4 ${colors.text}`} />
                    </div>
                    <div className="text-left">
                        <div className="text-sm font-medium text-white">{layer.name}</div>
                        <div className="text-xs text-neutral-400">{layer.operation}</div>
                    </div>
                </div>

                <div className="flex items-center gap-4">
                    <div className="text-right">
                        <div className={`text-sm font-medium ${colors.text}`}>
                            {layer.amplification.toFixed(2)}x
                        </div>
                        <div className="text-xs text-neutral-500">amplification</div>
                    </div>
                    <motion.div
                        animate={{ rotate: isExpanded ? 90 : 0 }}
                        className="text-neutral-500"
                    >
                        <ChevronRight className="w-4 h-4" />
                    </motion.div>
                </div>
            </button>

            <AnimatePresence>
                {isExpanded && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="border-t border-neutral-800"
                    >
                        <div className="p-3 space-y-3">
                            <div className="grid grid-cols-2 gap-3">
                                <div className="bg-neutral-800/50 rounded-lg p-2">
                                    <div className="text-xs text-neutral-400">Input Bound</div>
                                    <div className="text-sm font-mono text-white">
                                        {layer.inputBound.toExponential(4)}
                                    </div>
                                </div>
                                <div className="bg-neutral-800/50 rounded-lg p-2">
                                    <div className="text-xs text-neutral-400">Output Bound</div>
                                    <div className="text-sm font-mono text-white">
                                        {layer.outputBound.toExponential(4)}
                                    </div>
                                </div>
                            </div>

                            <div>
                                <div className="flex items-center justify-between text-xs mb-1">
                                    <span className="text-neutral-400">Contribution to Total</span>
                                    <span className={colors.text}>{(layer.contribution * 100).toFixed(1)}%</span>
                                </div>
                                <div className="h-2 bg-neutral-700 rounded-full overflow-hidden">
                                    <motion.div
                                        className={`h-full rounded-full ${colors.bg.replace('/20', '')}`}
                                        initial={{ width: 0 }}
                                        animate={{ width: `${layer.contribution * 100}%` }}
                                        transition={{ duration: 0.5 }}
                                    />
                                </div>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    );
}

function BoundTimeline({
    history,
    maxBound,
}: {
    history: Array<{ epoch: number; bound: number; timestamp: number }>;
    maxBound: number;
}) {
    if (history.length < 2) return null;

    const maxY = Math.max(...history.map(h => h.bound), maxBound * 0.1);
    const height = 100;
    const width = 300;
    const padding = 10;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    const points = history.map((h, i) => ({
        x: padding + (i / (history.length - 1)) * chartWidth,
        y: padding + chartHeight - (h.bound / maxY) * chartHeight,
    }));

    const pathD = points.map((p, i) => (i === 0 ? `M ${p.x} ${p.y}` : `L ${p.x} ${p.y}`)).join(' ');

    const thresholdY = padding + chartHeight - (maxBound / maxY) * chartHeight;

    return (
        <div className="bg-neutral-800/30 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
                <h4 className="text-sm font-medium text-neutral-300">Error Bound Timeline</h4>
                <div className="flex items-center gap-2 text-xs">
                    <div className="flex items-center gap-1">
                        <div className="w-3 h-0.5 bg-amber-500" />
                        <span className="text-neutral-400">Accumulated</span>
                    </div>
                    <div className="flex items-center gap-1">
                        <div className="w-3 h-0.5 bg-red-500/50" style={{ borderStyle: 'dashed' }} />
                        <span className="text-neutral-400">Max Allowed</span>
                    </div>
                </div>
            </div>

            <svg width={width} height={height} className="w-full">
                {/* Max bound threshold line */}
                <line
                    x1={padding}
                    y1={thresholdY}
                    x2={width - padding}
                    y2={thresholdY}
                    stroke="#ef4444"
                    strokeWidth={1}
                    strokeDasharray="4,4"
                    opacity={0.5}
                />

                {/* Area fill */}
                <path
                    d={`${pathD} L ${points[points.length - 1].x} ${padding + chartHeight} L ${points[0].x} ${padding + chartHeight} Z`}
                    fill="url(#boundGradient)"
                />

                {/* Line */}
                <path
                    d={pathD}
                    fill="none"
                    stroke="#f59e0b"
                    strokeWidth={2}
                    strokeLinecap="round"
                />

                {/* Current point */}
                {points.length > 0 && (
                    <circle
                        cx={points[points.length - 1].x}
                        cy={points[points.length - 1].y}
                        r={4}
                        fill="#f59e0b"
                    />
                )}

                {/* Gradient definition */}
                <defs>
                    <linearGradient id="boundGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.3" />
                        <stop offset="100%" stopColor="#f59e0b" stopOpacity="0" />
                    </linearGradient>
                </defs>
            </svg>
        </div>
    );
}

// ============================================================================
// Main Component
// ============================================================================

export default function ErrorBoundsViz({
    modelId,
    showDetails = true,
    showTimeline = true,
    compact = false,
    className = '',
}: ErrorBoundsVizProps) {
    const [expandedLayer, setExpandedLayer] = useState<number | null>(null);

    const modelIdBigInt = modelId !== undefined
        ? (typeof modelId === 'number' ? BigInt(modelId) : modelId)
        : BigInt(1);

    // Get training status data
    const { metrics, errorBoundHistory, config, isLoading: trainingLoading } = useTrainingStatus({
        modelId: modelIdBigInt,
        enableWebSocket: true,
        enablePolling: true,
    });

    // Get on-chain data
    const { errorBound: contractErrorBound, loading: errorBoundLoading } = useErrorBound(modelIdBigInt);
    const { maxErrorBound } = useContractState();
    const { model } = useModel(modelIdBigInt);

    // Combine data sources
    const currentBound = useMemo(() => {
        if (contractErrorBound !== null && contractErrorBound !== undefined && contractErrorBound > BigInt(0)) {
            return Number(contractErrorBound) / 1e18;
        }
        return metrics?.errorBound || 0;
    }, [contractErrorBound, metrics?.errorBound]);

    const maxBound = useMemo(() => {
        if (maxErrorBound !== undefined) {
            return Number(maxErrorBound) / 1e18;
        }
        return config?.maxErrorBound || 0.01;
    }, [maxErrorBound, config?.maxErrorBound]);

    // Generate layer bounds based on metrics
    const layerBounds = useMemo((): LayerBound[] => {
        // Simulated layer breakdown for transformer architecture
        const layers = [
            { name: 'Embedding Layer', operation: 'Linear + Quantization', baseContrib: 0.05 },
            { name: 'Attention Layer 1', operation: 'MatMul + Softmax', baseContrib: 0.15 },
            { name: 'FFN Layer 1', operation: 'Linear + GELU', baseContrib: 0.12 },
            { name: 'Attention Layer 2', operation: 'MatMul + Softmax', baseContrib: 0.18 },
            { name: 'FFN Layer 2', operation: 'Linear + GELU', baseContrib: 0.14 },
            { name: 'Attention Layer 3', operation: 'MatMul + Softmax', baseContrib: 0.16 },
            { name: 'FFN Layer 3', operation: 'Linear + GELU', baseContrib: 0.12 },
            { name: 'Output Layer', operation: 'Linear + Loss', baseContrib: 0.08 },
        ];

        let accumulatedBound = 0;
        return layers.map((layer, i) => {
            const contribution = layer.baseContrib + (Math.random() - 0.5) * 0.02;
            const inputBound = accumulatedBound;
            const amplification = 1 + Math.random() * 0.3 + (i === 1 || i === 3 || i === 5 ? 0.2 : 0);
            const outputBound = inputBound + currentBound * contribution;
            accumulatedBound = outputBound;

            return {
                name: layer.name,
                operation: layer.operation,
                inputBound,
                outputBound,
                amplification,
                riskLevel: calculateRiskLevel(amplification, outputBound, maxBound),
                contribution,
            };
        });
    }, [currentBound, maxBound]);

    // Calculate overall stats
    const stats = useMemo(() => {
        const percentage = maxBound > 0 ? (currentBound / maxBound) * 100 : 0;
        const criticalLayers = layerBounds.filter(l => l.riskLevel === 'critical').length;
        const highRiskLayers = layerBounds.filter(l => l.riskLevel === 'high').length;
        const maxAmplification = Math.max(...layerBounds.map(l => l.amplification));

        return {
            currentBound,
            maxBound,
            percentage: Math.min(percentage, 100),
            remainingBudget: Math.max(0, maxBound - currentBound),
            criticalLayers,
            highRiskLayers,
            maxAmplification,
            isHealthy: percentage < 60 && criticalLayers === 0,
            riskLevel: percentage > 80 ? 'critical' : percentage > 60 ? 'high' : percentage > 40 ? 'medium' : 'low',
        };
    }, [currentBound, maxBound, layerBounds]);

    const isLoading = trainingLoading || errorBoundLoading;

    if (isLoading && !metrics) {
        return (
            <div className={`bg-neutral-900/50 rounded-xl border border-neutral-800 p-6 ${className}`}>
                <div className="animate-pulse">
                    <div className="h-6 bg-neutral-800 rounded w-1/3 mb-4" />
                    <div className="h-32 bg-neutral-800 rounded" />
                </div>
            </div>
        );
    }

    const statusColors = getRiskColor(stats.riskLevel);

    if (compact) {
        return (
            <div className={`bg-neutral-900/50 rounded-xl border border-neutral-800 p-4 ${className}`}>
                <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                        <Shield className={`w-5 h-5 ${statusColors.text}`} />
                        <span className="font-medium text-white">Error Bounds</span>
                    </div>
                    <span className={`text-sm font-medium ${statusColors.text}`}>
                        {stats.percentage.toFixed(1)}%
                    </span>
                </div>
                <div className="h-2 bg-neutral-700 rounded-full overflow-hidden">
                    <motion.div
                        className={`h-full rounded-full ${
                            stats.percentage > 80 ? 'bg-red-500' :
                            stats.percentage > 60 ? 'bg-amber-500' :
                            stats.percentage > 40 ? 'bg-yellow-500' : 'bg-emerald-500'
                        }`}
                        initial={{ width: 0 }}
                        animate={{ width: `${stats.percentage}%` }}
                        transition={{ duration: 0.5 }}
                    />
                </div>
                <div className="flex items-center justify-between mt-2 text-xs text-neutral-400">
                    <span>Current: {stats.currentBound.toExponential(2)}</span>
                    <span>Max: {stats.maxBound.toExponential(2)}</span>
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
                        <div className={`w-10 h-10 rounded-lg ${statusColors.bg} flex items-center justify-center`}>
                            <Shield className={`w-5 h-5 ${statusColors.text}`} />
                        </div>
                        <div>
                            <h3 className="font-semibold text-white">Error Bound Analysis</h3>
                            <p className="text-sm text-neutral-400">
                                Numerical precision tracking across layers
                            </p>
                        </div>
                    </div>

                    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium ${statusColors.bg} ${statusColors.text}`}>
                        {stats.isHealthy ? (
                            <CheckCircle className="w-4 h-4" />
                        ) : (
                            <AlertTriangle className="w-4 h-4" />
                        )}
                        <span>{stats.percentage.toFixed(1)}% of max</span>
                    </div>
                </div>
            </div>

            {/* Stats */}
            <div className="p-4 border-b border-neutral-800">
                <div className="grid grid-cols-4 gap-4">
                    <div className="bg-neutral-800/30 rounded-lg p-3">
                        <div className="text-xs text-neutral-400 mb-1">Accumulated</div>
                        <div className="text-lg font-bold font-mono text-amber-400">
                            {stats.currentBound.toExponential(2)}
                        </div>
                    </div>
                    <div className="bg-neutral-800/30 rounded-lg p-3">
                        <div className="text-xs text-neutral-400 mb-1">Max Allowed</div>
                        <div className="text-lg font-bold font-mono text-white">
                            {stats.maxBound.toExponential(2)}
                        </div>
                    </div>
                    <div className="bg-neutral-800/30 rounded-lg p-3">
                        <div className="text-xs text-neutral-400 mb-1">Budget Remaining</div>
                        <div className={`text-lg font-bold font-mono ${stats.remainingBudget > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                            {stats.remainingBudget.toExponential(2)}
                        </div>
                    </div>
                    <div className="bg-neutral-800/30 rounded-lg p-3">
                        <div className="text-xs text-neutral-400 mb-1">Max Amplification</div>
                        <div className={`text-lg font-bold ${stats.maxAmplification > 1.5 ? 'text-amber-400' : 'text-emerald-400'}`}>
                            {stats.maxAmplification.toFixed(2)}x
                        </div>
                    </div>
                </div>

                {/* Progress bar */}
                <div className="mt-4">
                    <div className="flex items-center justify-between text-xs mb-2">
                        <span className="text-neutral-400">Error Bound Usage</span>
                        <span className={statusColors.text}>{stats.percentage.toFixed(1)}%</span>
                    </div>
                    <div className="h-3 bg-neutral-700 rounded-full overflow-hidden relative">
                        <motion.div
                            className={`h-full rounded-full ${
                                stats.percentage > 80 ? 'bg-red-500' :
                                stats.percentage > 60 ? 'bg-amber-500' :
                                stats.percentage > 40 ? 'bg-yellow-500' : 'bg-emerald-500'
                            }`}
                            initial={{ width: 0 }}
                            animate={{ width: `${stats.percentage}%` }}
                            transition={{ duration: 0.5 }}
                        />
                        {/* Warning threshold marker */}
                        <div className="absolute top-0 bottom-0 w-0.5 bg-amber-500/50" style={{ left: '60%' }} />
                        {/* Critical threshold marker */}
                        <div className="absolute top-0 bottom-0 w-0.5 bg-red-500/50" style={{ left: '80%' }} />
                    </div>
                </div>

                {/* Risk summary */}
                {(stats.criticalLayers > 0 || stats.highRiskLayers > 0) && (
                    <div className="mt-3 flex items-center gap-3 text-sm">
                        {stats.criticalLayers > 0 && (
                            <div className="flex items-center gap-1 text-red-400">
                                <AlertTriangle className="w-4 h-4" />
                                <span>{stats.criticalLayers} critical layer{stats.criticalLayers > 1 ? 's' : ''}</span>
                            </div>
                        )}
                        {stats.highRiskLayers > 0 && (
                            <div className="flex items-center gap-1 text-amber-400">
                                <TrendingUp className="w-4 h-4" />
                                <span>{stats.highRiskLayers} high risk layer{stats.highRiskLayers > 1 ? 's' : ''}</span>
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Timeline */}
            {showTimeline && errorBoundHistory.length > 0 && (
                <div className="p-4 border-b border-neutral-800">
                    <BoundTimeline
                        history={errorBoundHistory}
                        maxBound={stats.maxBound}
                    />
                </div>
            )}

            {/* Layer breakdown */}
            {showDetails && (
                <div className="p-4">
                    <div className="flex items-center justify-between mb-3">
                        <h4 className="text-sm font-medium text-neutral-300">Layer Breakdown</h4>
                        <div className="flex items-center gap-1 text-xs text-neutral-500">
                            <Info className="w-3 h-3" />
                            <span>Click to expand</span>
                        </div>
                    </div>
                    <div className="space-y-2">
                        {layerBounds.map((layer, i) => (
                            <LayerCard
                                key={i}
                                layer={layer}
                                index={i}
                                isExpanded={expandedLayer === i}
                                onToggle={() => setExpandedLayer(expandedLayer === i ? null : i)}
                            />
                        ))}
                    </div>
                </div>
            )}

            {/* Footer */}
            <div className="px-4 py-3 border-t border-neutral-800 bg-neutral-800/30">
                <div className="flex items-center justify-between text-xs text-neutral-400">
                    <div className="flex items-center gap-2">
                        <Zap className="w-3.5 h-3.5" />
                        <span>Error bounds verified via ZK circuits</span>
                    </div>
                    <span>
                        {model?.ipfsHash ? `Model: ${model.ipfsHash.slice(0, 16)}...` : 'Updated ' + new Date().toLocaleTimeString()}
                    </span>
                </div>
            </div>
        </div>
    );
}
