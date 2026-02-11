'use client';

/**
 * HELIX Live Network Topology Component
 * Real-time network visualization with worker nodes, connections, and health status.
 */

import React, { useMemo, useRef, useEffect, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Server,
    Cpu,
    Shield,
    Layers,
    Wifi,
    WifiOff,
    Activity,
    AlertTriangle,
    ChevronRight,
    Globe,
    Zap,
} from 'lucide-react';
import { useWorkerHealth, type WorkerHealth, type HealthStatus } from '@/hooks/useWorkerHealth';

// ============================================================================
// Types
// ============================================================================

interface LiveTopologyProps {
    modelId?: bigint | number;
    height?: number;
    interactive?: boolean;
    showDetails?: boolean;
    enablePolling?: boolean;
    className?: string;
}

interface NodePosition {
    x: number;
    y: number;
    worker: WorkerHealth;
}

interface Connection {
    from: NodePosition;
    to: NodePosition;
    active: boolean;
}

// ============================================================================
// Helper Functions
// ============================================================================

function getStatusColor(status: HealthStatus): string {
    switch (status) {
        case 'healthy': return '#ffffff';
        case 'degraded': return '#a3a3a3';
        case 'unhealthy': return '#737373';
        case 'offline': return '#525252';
        default: return '#9ca3af';
    }
}

function getRoleIcon(role: string) {
    switch (role) {
        case 'compute': return <Cpu className="w-4 h-4" />;
        case 'aggregator': return <Layers className="w-4 h-4" />;
        case 'verifier': return <Shield className="w-4 h-4" />;
        default: return <Server className="w-4 h-4" />;
    }
}

function formatAddress(address: string): string {
    return `${address.slice(0, 6)}...${address.slice(-4)}`;
}

// ============================================================================
// Sub-Components
// ============================================================================

function NetworkNode({
    position,
    isSelected,
    onSelect,
    isHovered,
    onHover,
}: {
    position: NodePosition;
    isSelected: boolean;
    onSelect: () => void;
    isHovered: boolean;
    onHover: (hover: boolean) => void;
}) {
    const { worker } = position;
    const statusColor = getStatusColor(worker.status);
    const isActive = worker.status !== 'offline';

    return (
        <motion.g
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: 'spring', stiffness: 500, delay: Math.random() * 0.3 }}
            onClick={onSelect}
            onMouseEnter={() => onHover(true)}
            onMouseLeave={() => onHover(false)}
            style={{ cursor: 'pointer' }}
        >
            {/* Pulse effect for active nodes */}
            {isActive && (
                <motion.circle
                    cx={position.x}
                    cy={position.y}
                    r={28}
                    fill={statusColor}
                    fillOpacity={0.2}
                    animate={{
                        r: [28, 35, 28],
                        opacity: [0.2, 0.1, 0.2],
                    }}
                    transition={{
                        duration: 2,
                        repeat: Infinity,
                        ease: 'easeInOut',
                    }}
                />
            )}

            {/* Selection ring */}
            {isSelected && (
                <motion.circle
                    cx={position.x}
                    cy={position.y}
                    r={32}
                    fill="none"
                    stroke={statusColor}
                    strokeWidth={2}
                    strokeDasharray="4,4"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1, rotate: 360 }}
                    transition={{ duration: 10, repeat: Infinity, ease: 'linear' }}
                />
            )}

            {/* Outer ring */}
            <circle
                cx={position.x}
                cy={position.y}
                r={24}
                fill="#1f2937"
                stroke={statusColor}
                strokeWidth={isHovered || isSelected ? 3 : 2}
            />

            {/* Inner background */}
            <circle
                cx={position.x}
                cy={position.y}
                r={20}
                fill={`${statusColor}15`}
            />

            {/* Role icon */}
            <foreignObject
                x={position.x - 10}
                y={position.y - 10}
                width={20}
                height={20}
            >
                <div
                    className="w-full h-full flex items-center justify-center"
                    style={{ color: statusColor }}
                >
                    {getRoleIcon(worker.role)}
                </div>
            </foreignObject>

            {/* Status indicator */}
            <circle
                cx={position.x + 16}
                cy={position.y - 16}
                r={6}
                fill={statusColor}
            />
            {worker.activity !== 'idle' && worker.status !== 'offline' && (
                <motion.circle
                    cx={position.x + 16}
                    cy={position.y - 16}
                    r={6}
                    fill="none"
                    stroke={statusColor}
                    strokeWidth={2}
                    animate={{ r: [6, 10], opacity: [1, 0] }}
                    transition={{ duration: 1, repeat: Infinity }}
                />
            )}

            {/* Label */}
            <text
                x={position.x}
                y={position.y + 38}
                textAnchor="middle"
                fill="#9ca3af"
                fontSize={10}
                fontWeight={500}
            >
                {formatAddress(worker.address)}
            </text>
        </motion.g>
    );
}

function ConnectionLine({ connection }: { connection: Connection }) {
    const midX = (connection.from.x + connection.to.x) / 2;
    const midY = (connection.from.y + connection.to.y) / 2;
    const offset = 30;

    const controlX = midX + (Math.random() - 0.5) * offset;
    const controlY = midY + (Math.random() - 0.5) * offset;

    const path = `M ${connection.from.x} ${connection.from.y} Q ${controlX} ${controlY} ${connection.to.x} ${connection.to.y}`;

    return (
        <g>
            <path
                d={path}
                fill="none"
                stroke={connection.active ? '#374151' : '#1f2937'}
                strokeWidth={2}
            />
            {connection.active && (
                <motion.circle
                    r={3}
                    fill="#ffffff"
                    initial={{ offsetDistance: '0%' }}
                    animate={{ offsetDistance: '100%' }}
                    transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                    style={{ offsetPath: `path('${path}')` }}
                />
            )}
        </g>
    );
}

function WorkerDetails({ worker, onClose }: { worker: WorkerHealth; onClose: () => void }) {
    const statusColor = getStatusColor(worker.status);

    return (
        <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            className="absolute right-0 top-0 bottom-0 w-80 bg-neutral-900 border-l border-helix-border overflow-y-auto"
        >
            {/* Header */}
            <div className="sticky top-0 bg-neutral-900 border-b border-helix-border p-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div
                            className="w-10 h-10 rounded-md flex items-center justify-center"
                            style={{ backgroundColor: `${statusColor}20`, color: statusColor }}
                        >
                            {getRoleIcon(worker.role)}
                        </div>
                        <div>
                            <h3 className="font-semibold text-white capitalize">{worker.role}</h3>
                            <p className="text-xs text-[#888]">{formatAddress(worker.address)}</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-1 text-[#888] hover:text-white"
                    >
                        <ChevronRight className="w-5 h-5" />
                    </button>
                </div>
            </div>

            {/* Status */}
            <div className="p-4 border-b border-helix-border">
                <div className="flex items-center justify-between mb-3">
                    <span className="text-sm text-[#888]">Status</span>
                    <span
                        className="px-2 py-1 rounded text-xs font-medium capitalize"
                        style={{ backgroundColor: `${statusColor}20`, color: statusColor }}
                    >
                        {worker.status}
                    </span>
                </div>
                <div className="flex items-center justify-between">
                    <span className="text-sm text-[#888]">Activity</span>
                    <span className="text-sm text-white capitalize">{worker.activity}</span>
                </div>
            </div>

            {/* Metrics */}
            <div className="p-4 border-b border-helix-border">
                <h4 className="text-sm font-medium text-[#aaa] mb-3">System Metrics</h4>
                <div className="space-y-3">
                    <MetricBar label="CPU" value={worker.metrics.cpu} color="#ffffff" />
                    <MetricBar label="Memory" value={worker.metrics.memory} color="#d4d4d4" />
                    {worker.metrics.gpu && (
                        <MetricBar label="GPU" value={worker.metrics.gpu} color="#a3a3a3" />
                    )}
                    <div className="flex items-center justify-between text-sm">
                        <span className="text-[#888]">Latency</span>
                        <span className="text-white">{worker.metrics.latency.toFixed(0)}ms</span>
                    </div>
                    {worker.metrics.temperature && (
                        <div className="flex items-center justify-between text-sm">
                            <span className="text-[#888]">Temperature</span>
                            <span className="text-white">
                                {worker.metrics.temperature.toFixed(0)}°C
                            </span>
                        </div>
                    )}
                </div>
            </div>

            {/* Performance */}
            <div className="p-4 border-b border-helix-border">
                <h4 className="text-sm font-medium text-[#aaa] mb-3">Performance</h4>
                <div className="grid grid-cols-2 gap-3">
                    <div className="bg-neutral-800/50 rounded-md p-3">
                        <div className="text-lg font-bold text-white">{worker.performance.proofsVerified}</div>
                        <div className="text-xs text-[#888]">Proofs Verified</div>
                    </div>
                    <div className="bg-neutral-800/50 rounded-md p-3">
                        <div className="text-lg font-bold text-white">{(worker.performance.successRate * 100).toFixed(1)}%</div>
                        <div className="text-xs text-[#888]">Success Rate</div>
                    </div>
                    <div className="bg-neutral-800/50 rounded-md p-3">
                        <div className="text-lg font-bold text-white">{worker.performance.averageProofTime.toFixed(0)}ms</div>
                        <div className="text-xs text-[#888]">Avg Proof Time</div>
                    </div>
                    <div className="bg-neutral-800/50 rounded-md p-3">
                        <div className="text-lg font-bold text-white">{(worker.performance.uptime * 100).toFixed(1)}%</div>
                        <div className="text-xs text-[#888]">Uptime</div>
                    </div>
                </div>
            </div>

            {/* Stake */}
            <div className="p-4 border-b border-helix-border">
                <h4 className="text-sm font-medium text-[#aaa] mb-3">Stake</h4>
                <div className="space-y-2 text-sm">
                    <div className="flex items-center justify-between">
                        <span className="text-[#888]">Amount</span>
                        <span className="text-white font-medium">{worker.stake.amountFormatted} HELIX</span>
                    </div>
                    <div className="flex items-center justify-between">
                        <span className="text-[#888]">Status</span>
                        <span className={worker.stake.slashed ? 'text-[#666]' : worker.stake.isLocked ? 'text-[#666]' : 'text-white'}>
                            {worker.stake.slashed ? 'Slashed' : worker.stake.isLocked ? 'Locked' : 'Active'}
                        </span>
                    </div>
                </div>
            </div>

            {/* Issues */}
            {worker.issues.length > 0 && (
                <div className="p-4">
                    <h4 className="text-sm font-medium text-[#aaa] mb-3">Issues</h4>
                    <div className="space-y-2">
                        {worker.issues.map((issue) => (
                            <div
                                key={issue.id}
                                className={`p-2 rounded text-xs ${
                                    issue.severity === 'critical'
                                        ? 'bg-white/10 text-white'
                                        : issue.severity === 'error'
                                            ? 'bg-white/5 text-[#888]'
                                            : 'bg-white/5 text-[#888]'
                                }`}
                            >
                                {issue.message}
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </motion.div>
    );
}

function MetricBar({ label, value, color }: { label: string; value: number; color: string }) {
    return (
        <div>
            <div className="flex items-center justify-between mb-1">
                <span className="text-sm text-[#888]">{label}</span>
                <span className="text-sm text-white">{value.toFixed(0)}%</span>
            </div>
            <div className="h-2 bg-neutral-700 rounded-full overflow-hidden">
                <motion.div
                    className="h-full rounded-full"
                    style={{ backgroundColor: color }}
                    initial={{ width: 0 }}
                    animate={{ width: `${value}%` }}
                    transition={{ duration: 0.5 }}
                />
            </div>
        </div>
    );
}

// ============================================================================
// Main Component
// ============================================================================

export default function LiveTopology({
    modelId,
    height = 500,
    interactive = true,
    showDetails = true,
    enablePolling = true,
    className = '',
}: LiveTopologyProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const [dimensions, setDimensions] = useState({ width: 800, height });
    const [selectedWorkerId, setSelectedWorkerId] = useState<string | null>(null);
    const [hoveredWorkerId, setHoveredWorkerId] = useState<string | null>(null);

    const {
        workers: _workers,
        computeWorkers,
        aggregators,
        verifiers,
        networkSummary,
        selectedWorker,
        selectWorker,
        isLoading,
    } = useWorkerHealth({
        modelId,
        enableWebSocket: true,
        enablePolling,
    });

    // Update dimensions
    useEffect(() => {
        const updateDimensions = () => {
            if (containerRef.current) {
                const { width } = containerRef.current.getBoundingClientRect();
                setDimensions({ width: Math.max(600, width), height });
            }
        };

        updateDimensions();
        window.addEventListener('resize', updateDimensions);
        return () => window.removeEventListener('resize', updateDimensions);
    }, [height]);

    // Calculate node positions
    const { nodePositions, connections } = useMemo(() => {
        const positions: NodePosition[] = [];
        const conns: Connection[] = [];

        const centerX = dimensions.width / 2;
        const centerY = dimensions.height / 2;
        const radius = Math.min(dimensions.width, dimensions.height) * 0.35;

        // Position aggregators in the center
        aggregators.forEach((w, i) => {
            const angle = (i / Math.max(aggregators.length, 1)) * Math.PI * 2 - Math.PI / 2;
            const r = radius * 0.3;
            positions.push({
                x: centerX + Math.cos(angle) * r,
                y: centerY + Math.sin(angle) * r,
                worker: w,
            });
        });

        // Position compute workers in outer ring
        computeWorkers.forEach((w, i) => {
            const angle = (i / Math.max(computeWorkers.length, 1)) * Math.PI * 2 - Math.PI / 2;
            positions.push({
                x: centerX + Math.cos(angle) * radius,
                y: centerY + Math.sin(angle) * radius,
                worker: w,
            });
        });

        // Position verifiers in middle ring
        verifiers.forEach((w, i) => {
            const angle = (i / Math.max(verifiers.length, 1)) * Math.PI * 2 + Math.PI / 4;
            const r = radius * 0.65;
            positions.push({
                x: centerX + Math.cos(angle) * r,
                y: centerY + Math.sin(angle) * r,
                worker: w,
            });
        });

        // Create connections (compute -> aggregator)
        const aggregatorPositions = positions.filter(p => p.worker.role === 'aggregator');
        positions.forEach((pos) => {
            if (pos.worker.role === 'compute' && aggregatorPositions.length > 0) {
                const nearestAgg = aggregatorPositions[0];
                conns.push({
                    from: pos,
                    to: nearestAgg,
                    active: pos.worker.status === 'healthy' && nearestAgg.worker.status === 'healthy',
                });
            }
        });

        return { nodePositions: positions, connections: conns };
    }, [computeWorkers, aggregators, verifiers, dimensions]);

    const handleSelectWorker = useCallback((workerId: string) => {
        setSelectedWorkerId(workerId);
        selectWorker(workerId);
    }, [selectWorker]);

    if (isLoading) {
        return (
            <div className={`bg-helix-surface rounded-md border border-helix-border p-4 ${className}`}>
                <div className="flex items-center justify-center h-96">
                    <Activity className="w-8 h-8 text-[#666] animate-pulse" />
                </div>
            </div>
        );
    }

    return (
        <div
            ref={containerRef}
            className={`bg-helix-surface rounded-md border border-helix-border overflow-hidden relative ${className}`}
        >
            {/* Header */}
            <div className="p-4 border-b border-helix-border">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-md bg-white/10 flex items-center justify-center">
                            <Globe className="w-5 h-5 text-white" />
                        </div>
                        <div>
                            <h3 className="font-semibold text-white">Network Topology</h3>
                            <p className="text-sm text-[#888]">
                                {networkSummary.totalWorkers} nodes · {networkSummary.healthyWorkers} healthy
                            </p>
                        </div>
                    </div>

                    {/* Network health indicator */}
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium bg-white/10 text-white">
                        {networkSummary.networkHealth === 'healthy' ? (
                            <Wifi className="w-4 h-4" />
                        ) : networkSummary.networkHealth === 'offline' ? (
                            <WifiOff className="w-4 h-4" />
                        ) : (
                            <AlertTriangle className="w-4 h-4" />
                        )}
                        <span>Score: {networkSummary.healthScore}</span>
                    </div>
                </div>
            </div>

            {/* Topology visualization */}
            <div className="relative" style={{ height }}>
                <svg
                    width={dimensions.width - (selectedWorkerId ? 320 : 0)}
                    height={height}
                    className="transition-all duration-300"
                >
                    {/* Grid pattern */}
                    <defs>
                        <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#1f2937" strokeWidth="1" />
                        </pattern>
                    </defs>
                    <rect width="100%" height="100%" fill="url(#grid)" />

                    {/* Connections */}
                    <g>
                        {connections.map((conn, i) => (
                            <ConnectionLine key={i} connection={conn} />
                        ))}
                    </g>

                    {/* Nodes */}
                    <g>
                        {nodePositions.map((pos) => (
                            <NetworkNode
                                key={pos.worker.id}
                                position={pos}
                                isSelected={selectedWorkerId === pos.worker.id}
                                onSelect={() => interactive && handleSelectWorker(pos.worker.id)}
                                isHovered={hoveredWorkerId === pos.worker.id}
                                onHover={(hover) => setHoveredWorkerId(hover ? pos.worker.id : null)}
                            />
                        ))}
                    </g>
                </svg>

                {/* Worker details panel */}
                <AnimatePresence>
                    {showDetails && selectedWorker && (
                        <WorkerDetails
                            worker={selectedWorker}
                            onClose={() => {
                                setSelectedWorkerId(null);
                                selectWorker(null);
                            }}
                        />
                    )}
                </AnimatePresence>
            </div>

            {/* Legend */}
            <div className="p-4 border-t border-helix-border bg-neutral-800/30">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="flex items-center gap-2 text-sm">
                            <div className="w-6 h-6 rounded bg-neutral-800 flex items-center justify-center">
                                <Cpu className="w-3.5 h-3.5 text-[#888]" />
                            </div>
                            <span className="text-[#888]">Compute ({computeWorkers.length})</span>
                        </div>
                        <div className="flex items-center gap-2 text-sm">
                            <div className="w-6 h-6 rounded bg-neutral-800 flex items-center justify-center">
                                <Layers className="w-3.5 h-3.5 text-[#888]" />
                            </div>
                            <span className="text-[#888]">Aggregator ({aggregators.length})</span>
                        </div>
                        <div className="flex items-center gap-2 text-sm">
                            <div className="w-6 h-6 rounded bg-neutral-800 flex items-center justify-center">
                                <Shield className="w-3.5 h-3.5 text-[#888]" />
                            </div>
                            <span className="text-[#888]">Verifier ({verifiers.length})</span>
                        </div>
                    </div>

                    <div className="flex items-center gap-3 text-sm">
                        <div className="flex items-center gap-2">
                            <Zap className="w-4 h-4 text-white/50" />
                            <span className="text-[#888]">Avg Latency:</span>
                            <span className="text-white">{networkSummary.averageLatency.toFixed(0)}ms</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <Activity className="w-4 h-4 text-white/50" />
                            <span className="text-[#888]">Uptime:</span>
                            <span className="text-white">{(networkSummary.averageUptime * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
