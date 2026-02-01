'use client';

/**
 * HELIX Live Loss Curve Component
 * Real-time loss and accuracy visualization with smooth animations and trend analysis.
 */

import React, { useMemo, useRef, useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TrendingDown, TrendingUp, Minus, Activity, Zap, AlertTriangle } from 'lucide-react';
import { useTrainingStatus } from '@/hooks/useTrainingStatus';

// ============================================================================
// Types
// ============================================================================

interface LiveLossCurveProps {
    modelId?: bigint | number;
    height?: number;
    showAccuracy?: boolean;
    showErrorBound?: boolean;
    showThroughput?: boolean;
    className?: string;
}

interface DataPoint {
    x: number;
    y: number;
    timestamp: number;
}

// ============================================================================
// SVG Path Utilities
// ============================================================================

function createSmoothPath(points: DataPoint[], xScale: number, yScale: number, height: number, yMin: number): string {
    if (points.length < 2) return '';

    const scaled = points.map(p => ({
        x: p.x * xScale,
        y: height - ((p.y - yMin) * yScale),
    }));

    let path = `M ${scaled[0].x} ${scaled[0].y}`;

    for (let i = 1; i < scaled.length; i++) {
        const prev = scaled[i - 1];
        const curr = scaled[i];
        const cpx1 = prev.x + (curr.x - prev.x) * 0.5;
        const cpy1 = prev.y;
        const cpx2 = curr.x - (curr.x - prev.x) * 0.5;
        const cpy2 = curr.y;
        path += ` C ${cpx1} ${cpy1}, ${cpx2} ${cpy2}, ${curr.x} ${curr.y}`;
    }

    return path;
}

function createAreaPath(points: DataPoint[], xScale: number, yScale: number, height: number, yMin: number): string {
    if (points.length < 2) return '';

    const linePath = createSmoothPath(points, xScale, yScale, height, yMin);
    const firstX = points[0].x * xScale;
    const lastX = points[points.length - 1].x * xScale;

    return `${linePath} L ${lastX} ${height} L ${firstX} ${height} Z`;
}

// ============================================================================
// Component
// ============================================================================

export default function LiveLossCurve({
    modelId,
    height = 300,
    showAccuracy = true,
    showErrorBound = true,
    showThroughput = false,
    className = '',
}: LiveLossCurveProps) {
    const svgRef = useRef<SVGSVGElement>(null);
    const [dimensions, setDimensions] = useState({ width: 600, height });
    const [hoveredPoint, setHoveredPoint] = useState<{ x: number; y: number; data: DataPoint; type: string } | null>(null);

    const {
        metrics,
        lossHistory,
        accuracyHistory,
        errorBoundHistory,
        throughputHistory,
        status,
        config,
        isLoading,
    } = useTrainingStatus({
        modelId: modelId || BigInt(1),
        enableWebSocket: true,
        enablePolling: false, // Only use real data from WebSocket/contract
    });

    // Update dimensions on resize
    useEffect(() => {
        const updateDimensions = () => {
            if (svgRef.current?.parentElement) {
                const { width } = svgRef.current.parentElement.getBoundingClientRect();
                setDimensions({ width: Math.max(400, width), height });
            }
        };

        updateDimensions();
        window.addEventListener('resize', updateDimensions);
        return () => window.removeEventListener('resize', updateDimensions);
    }, [height]);

    // Prepare data
    const chartData = useMemo(() => {
        const padding = { top: 20, right: 60, bottom: 40, left: 60 };
        const chartWidth = dimensions.width - padding.left - padding.right;
        const chartHeight = dimensions.height - padding.top - padding.bottom;

        // Loss data points
        const lossPoints: DataPoint[] = lossHistory.map((h, i) => ({
            x: i,
            y: h.loss,
            timestamp: h.timestamp,
        }));

        // Accuracy data points (scale to match loss range for visualization)
        const accuracyPoints: DataPoint[] = accuracyHistory.map((h, i) => ({
            x: i,
            y: h.accuracy,
            timestamp: h.timestamp,
        }));

        // Error bound points
        const errorPoints: DataPoint[] = errorBoundHistory.map((h, i) => ({
            x: i,
            y: h.bound * 1000, // Scale up for visibility
            timestamp: h.timestamp,
        }));

        // Calculate scales
        const maxX = Math.max(lossPoints.length - 1, 1);
        const xScale = chartWidth / maxX;

        // Loss scale
        const lossValues = lossPoints.map(p => p.y);
        const lossMin = Math.min(...lossValues, 0);
        const lossMax = Math.max(...lossValues, 1) * 1.1;
        const lossYScale = chartHeight / (lossMax - lossMin);

        // Accuracy is always 0-1
        const accYScale = chartHeight;

        // Error bound scale
        const errorValues = errorPoints.map(p => p.y);
        const errorMax = Math.max(...errorValues, 0.01) * 1.2;
        const errorYScale = chartHeight / errorMax;

        return {
            padding,
            chartWidth,
            chartHeight,
            lossPoints,
            accuracyPoints,
            errorPoints,
            xScale,
            lossYScale,
            lossMin,
            lossMax,
            accYScale,
            errorYScale,
            errorMax,
            maxX,
        };
    }, [dimensions, lossHistory, accuracyHistory, errorBoundHistory]);

    // Calculate trend
    const trend = useMemo(() => {
        if (lossHistory.length < 5) return { direction: 'stable' as const, change: 0 };

        const recent = lossHistory.slice(-5);
        const older = lossHistory.slice(-10, -5);

        if (older.length === 0) return { direction: 'stable' as const, change: 0 };

        const recentAvg = recent.reduce((s, p) => s + p.loss, 0) / recent.length;
        const olderAvg = older.reduce((s, p) => s + p.loss, 0) / older.length;
        const change = ((recentAvg - olderAvg) / olderAvg) * 100;

        if (change < -5) return { direction: 'improving' as const, change };
        if (change > 5) return { direction: 'degrading' as const, change };
        return { direction: 'stable' as const, change };
    }, [lossHistory]);

    // Convergence rate
    const convergenceRate = useMemo(() => {
        if (lossHistory.length < 3) return 0;
        const recent = lossHistory.slice(-3);
        const older = lossHistory.slice(-6, -3);
        if (older.length === 0) return 0;
        const recentAvg = recent.reduce((s, p) => s + p.loss, 0) / recent.length;
        const olderAvg = older.reduce((s, p) => s + p.loss, 0) / older.length;
        return Math.max(0, (olderAvg - recentAvg) / olderAvg);
    }, [lossHistory]);

    const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
        if (!svgRef.current) return;

        const rect = svgRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left - chartData.padding.left;
        const y = e.clientY - rect.top - chartData.padding.top;

        if (x < 0 || x > chartData.chartWidth || y < 0 || y > chartData.chartHeight) {
            setHoveredPoint(null);
            return;
        }

        const pointIndex = Math.round(x / chartData.xScale);
        if (pointIndex >= 0 && pointIndex < chartData.lossPoints.length) {
            const point = chartData.lossPoints[pointIndex];
            setHoveredPoint({
                x: chartData.padding.left + pointIndex * chartData.xScale,
                y: chartData.padding.top + chartData.chartHeight - ((point.y - chartData.lossMin) * chartData.lossYScale),
                data: point,
                type: 'loss',
            });
        }
    };

    if (isLoading) {
        return (
            <div className={`bg-neutral-900/50 rounded-xl border border-neutral-800 p-6 ${className}`}>
                <div className="animate-pulse">
                    <div className="h-6 bg-neutral-800 rounded w-1/3 mb-4" />
                    <div className="h-64 bg-neutral-800 rounded" />
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
                        <div className="w-10 h-10 rounded-lg bg-emerald-500/20 flex items-center justify-center">
                            <Activity className="w-5 h-5 text-emerald-400" />
                        </div>
                        <div>
                            <h3 className="font-semibold text-white">Training Progress</h3>
                            <p className="text-sm text-neutral-400">
                                Epoch {metrics?.epoch || 0}/{config?.epochs || 10} | Batch {metrics?.batch || 0}
                            </p>
                        </div>
                    </div>

                    <div className="flex items-center gap-4">
                        {/* Trend Indicator */}
                        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium ${
                            trend.direction === 'improving'
                                ? 'bg-emerald-500/20 text-emerald-400'
                                : trend.direction === 'degrading'
                                    ? 'bg-red-500/20 text-red-400'
                                    : 'bg-neutral-700/50 text-neutral-400'
                        }`}>
                            {trend.direction === 'improving' && <TrendingDown className="w-4 h-4" />}
                            {trend.direction === 'degrading' && <TrendingUp className="w-4 h-4" />}
                            {trend.direction === 'stable' && <Minus className="w-4 h-4" />}
                            <span>{Math.abs(trend.change).toFixed(1)}%</span>
                        </div>

                        {/* Status */}
                        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
                            status?.status === 'training'
                                ? 'bg-blue-500/20 text-blue-400'
                                : status?.status === 'paused'
                                    ? 'bg-yellow-500/20 text-yellow-400'
                                    : 'bg-neutral-700/50 text-neutral-400'
                        }`}>
                            {status?.status === 'training' && (
                                <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
                            )}
                            {status?.status || 'Initializing'}
                        </div>
                    </div>
                </div>
            </div>

            {/* Chart */}
            <div className="p-4">
                <svg
                    ref={svgRef}
                    width={dimensions.width}
                    height={dimensions.height}
                    className="cursor-crosshair"
                    onMouseMove={handleMouseMove}
                    onMouseLeave={() => setHoveredPoint(null)}
                >
                    <defs>
                        {/* Loss gradient */}
                        <linearGradient id="lossGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                            <stop offset="0%" stopColor="#f43f5e" stopOpacity="0.3" />
                            <stop offset="100%" stopColor="#f43f5e" stopOpacity="0" />
                        </linearGradient>

                        {/* Accuracy gradient */}
                        <linearGradient id="accuracyGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                            <stop offset="0%" stopColor="#22c55e" stopOpacity="0.3" />
                            <stop offset="100%" stopColor="#22c55e" stopOpacity="0" />
                        </linearGradient>

                        {/* Error bound gradient */}
                        <linearGradient id="errorGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                            <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.2" />
                            <stop offset="100%" stopColor="#f59e0b" stopOpacity="0" />
                        </linearGradient>
                    </defs>

                    <g transform={`translate(${chartData.padding.left}, ${chartData.padding.top})`}>
                        {/* Grid lines */}
                        {[0, 0.25, 0.5, 0.75, 1].map((ratio, i) => {
                            const y = chartData.chartHeight * ratio;
                            return (
                                <g key={i}>
                                    <line
                                        x1={0}
                                        y1={y}
                                        x2={chartData.chartWidth}
                                        y2={y}
                                        stroke="#374151"
                                        strokeWidth={1}
                                        strokeDasharray={i === 0 || i === 4 ? undefined : "4,4"}
                                    />
                                    <text
                                        x={-8}
                                        y={y + 4}
                                        fill="#6b7280"
                                        fontSize={11}
                                        textAnchor="end"
                                    >
                                        {((1 - ratio) * (chartData.lossMax - chartData.lossMin) + chartData.lossMin).toFixed(2)}
                                    </text>
                                </g>
                            );
                        })}

                        {/* Error bound area */}
                        {showErrorBound && chartData.errorPoints.length > 1 && (
                            <motion.path
                                d={createAreaPath(
                                    chartData.errorPoints,
                                    chartData.xScale,
                                    chartData.errorYScale,
                                    chartData.chartHeight,
                                    0
                                )}
                                fill="url(#errorGradient)"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ duration: 0.5 }}
                            />
                        )}

                        {/* Loss area */}
                        {chartData.lossPoints.length > 1 && (
                            <motion.path
                                d={createAreaPath(
                                    chartData.lossPoints,
                                    chartData.xScale,
                                    chartData.lossYScale,
                                    chartData.chartHeight,
                                    chartData.lossMin
                                )}
                                fill="url(#lossGradient)"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ duration: 0.5 }}
                            />
                        )}

                        {/* Loss line */}
                        {chartData.lossPoints.length > 1 && (
                            <motion.path
                                d={createSmoothPath(
                                    chartData.lossPoints,
                                    chartData.xScale,
                                    chartData.lossYScale,
                                    chartData.chartHeight,
                                    chartData.lossMin
                                )}
                                fill="none"
                                stroke="#f43f5e"
                                strokeWidth={2.5}
                                strokeLinecap="round"
                                initial={{ pathLength: 0 }}
                                animate={{ pathLength: 1 }}
                                transition={{ duration: 1 }}
                            />
                        )}

                        {/* Accuracy line */}
                        {showAccuracy && chartData.accuracyPoints.length > 1 && (
                            <motion.path
                                d={createSmoothPath(
                                    chartData.accuracyPoints,
                                    chartData.xScale,
                                    chartData.accYScale,
                                    chartData.chartHeight,
                                    0
                                )}
                                fill="none"
                                stroke="#22c55e"
                                strokeWidth={2.5}
                                strokeLinecap="round"
                                initial={{ pathLength: 0 }}
                                animate={{ pathLength: 1 }}
                                transition={{ duration: 1, delay: 0.2 }}
                            />
                        )}

                        {/* Error bound line */}
                        {showErrorBound && chartData.errorPoints.length > 1 && (
                            <motion.path
                                d={createSmoothPath(
                                    chartData.errorPoints,
                                    chartData.xScale,
                                    chartData.errorYScale,
                                    chartData.chartHeight,
                                    0
                                )}
                                fill="none"
                                stroke="#f59e0b"
                                strokeWidth={2}
                                strokeLinecap="round"
                                strokeDasharray="6,3"
                                initial={{ pathLength: 0 }}
                                animate={{ pathLength: 1 }}
                                transition={{ duration: 1, delay: 0.4 }}
                            />
                        )}

                        {/* Current point indicators */}
                        {chartData.lossPoints.length > 0 && (
                            <motion.circle
                                cx={chartData.lossPoints[chartData.lossPoints.length - 1].x * chartData.xScale}
                                cy={chartData.chartHeight - ((chartData.lossPoints[chartData.lossPoints.length - 1].y - chartData.lossMin) * chartData.lossYScale)}
                                r={6}
                                fill="#f43f5e"
                                initial={{ scale: 0 }}
                                animate={{ scale: 1 }}
                                transition={{ type: "spring", stiffness: 500 }}
                            />
                        )}

                        {showAccuracy && chartData.accuracyPoints.length > 0 && (
                            <motion.circle
                                cx={chartData.accuracyPoints[chartData.accuracyPoints.length - 1].x * chartData.xScale}
                                cy={chartData.chartHeight - (chartData.accuracyPoints[chartData.accuracyPoints.length - 1].y * chartData.accYScale)}
                                r={6}
                                fill="#22c55e"
                                initial={{ scale: 0 }}
                                animate={{ scale: 1 }}
                                transition={{ type: "spring", stiffness: 500, delay: 0.2 }}
                            />
                        )}

                        {/* Hover indicator */}
                        <AnimatePresence>
                            {hoveredPoint && (
                                <motion.g
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    exit={{ opacity: 0 }}
                                    transition={{ duration: 0.1 }}
                                >
                                    <line
                                        x1={hoveredPoint.x - chartData.padding.left}
                                        y1={0}
                                        x2={hoveredPoint.x - chartData.padding.left}
                                        y2={chartData.chartHeight}
                                        stroke="#6b7280"
                                        strokeWidth={1}
                                        strokeDasharray="4,4"
                                    />
                                    <circle
                                        cx={hoveredPoint.x - chartData.padding.left}
                                        cy={hoveredPoint.y - chartData.padding.top}
                                        r={8}
                                        fill="#f43f5e"
                                        fillOpacity={0.3}
                                        stroke="#f43f5e"
                                        strokeWidth={2}
                                    />
                                </motion.g>
                            )}
                        </AnimatePresence>
                    </g>

                    {/* X-axis labels */}
                    <g transform={`translate(${chartData.padding.left}, ${chartData.padding.top + chartData.chartHeight + 20})`}>
                        {chartData.lossPoints.length > 0 && [0, 0.5, 1].map((ratio, i) => {
                            const index = Math.floor(ratio * (chartData.lossPoints.length - 1));
                            const point = chartData.lossPoints[index];
                            if (!point) return null;
                            return (
                                <text
                                    key={i}
                                    x={index * chartData.xScale}
                                    y={0}
                                    fill="#6b7280"
                                    fontSize={11}
                                    textAnchor="middle"
                                >
                                    Epoch {Math.floor(index / 10)}
                                </text>
                            );
                        })}
                    </g>

                    {/* Right axis (accuracy) */}
                    {showAccuracy && (
                        <g transform={`translate(${chartData.padding.left + chartData.chartWidth + 10}, ${chartData.padding.top})`}>
                            {[0, 0.25, 0.5, 0.75, 1].map((ratio, i) => {
                                const y = chartData.chartHeight * (1 - ratio);
                                return (
                                    <text
                                        key={i}
                                        x={0}
                                        y={y + 4}
                                        fill="#22c55e"
                                        fontSize={11}
                                    >
                                        {(ratio * 100).toFixed(0)}%
                                    </text>
                                );
                            })}
                        </g>
                    )}
                </svg>

                {/* Tooltip */}
                <AnimatePresence>
                    {hoveredPoint && (
                        <motion.div
                            className="absolute bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-sm pointer-events-none z-10"
                            style={{
                                left: hoveredPoint.x + 10,
                                top: hoveredPoint.y - 40,
                            }}
                            initial={{ opacity: 0, y: 5 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: 5 }}
                        >
                            <div className="text-neutral-400">Loss</div>
                            <div className="text-white font-medium">{hoveredPoint.data.y.toFixed(4)}</div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>

            {/* Legend & Stats */}
            <div className="px-4 pb-4">
                <div className="flex items-center justify-between">
                    {/* Legend */}
                    <div className="flex items-center gap-6">
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-rose-500" />
                            <span className="text-sm text-neutral-400">Loss</span>
                            <span className="text-sm font-medium text-white">{metrics?.loss.toFixed(4)}</span>
                        </div>
                        {showAccuracy && (
                            <div className="flex items-center gap-2">
                                <div className="w-3 h-3 rounded-full bg-emerald-500" />
                                <span className="text-sm text-neutral-400">Accuracy</span>
                                <span className="text-sm font-medium text-white">{((metrics?.accuracy || 0) * 100).toFixed(1)}%</span>
                            </div>
                        )}
                        {showErrorBound && (
                            <div className="flex items-center gap-2">
                                <div className="w-3 h-0.5 bg-amber-500" style={{ borderStyle: 'dashed' }} />
                                <span className="text-sm text-neutral-400">Error Bound</span>
                                <span className="text-sm font-medium text-white">{((metrics?.errorBound || 0) * 1000).toFixed(3)}</span>
                            </div>
                        )}
                    </div>

                    {/* Stats */}
                    <div className="flex items-center gap-4">
                        <div className="flex items-center gap-2 text-sm">
                            <Zap className="w-4 h-4 text-blue-400" />
                            <span className="text-neutral-400">Throughput:</span>
                            <span className="font-medium text-white">{metrics?.throughput.toFixed(1)} samples/s</span>
                        </div>
                        <div className="flex items-center gap-2 text-sm">
                            <Activity className="w-4 h-4 text-purple-400" />
                            <span className="text-neutral-400">Convergence:</span>
                            <span className="font-medium text-white">{(convergenceRate * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
