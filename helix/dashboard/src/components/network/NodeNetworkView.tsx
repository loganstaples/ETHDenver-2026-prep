'use client';

import React, { useState } from 'react';

interface NetworkNode {
    id: string;
    type: 'compute' | 'aggregator' | 'verifier';
    status: 'online' | 'offline' | 'syncing';
    address: string;
    peers: number;
    lastSeen: number;
    metrics: {
        cpu: number;
        memory: number;
        gpu?: number;
    };
    position: { x: number; y: number };
}

interface Connection {
    from: string;
    to: string;
    latency: number;
    bandwidth: number;
}

interface NodeNetworkViewProps {
    nodes?: NetworkNode[];
    connections?: Connection[];
}

export default function NodeNetworkView({
    nodes,
    connections,
}: NodeNetworkViewProps) {
    const [selectedNode, setSelectedNode] = useState<NetworkNode | null>(null);
    const [viewMode, setViewMode] = useState<'graph' | 'list'>('graph');

    const mockNodes: NetworkNode[] = nodes || [
        { id: 'node_01', type: 'aggregator', status: 'online', address: '0x1234...5678', peers: 12, lastSeen: Date.now(), metrics: { cpu: 45, memory: 62, gpu: 78 }, position: { x: 50, y: 50 } },
        { id: 'node_02', type: 'compute', status: 'online', address: '0xabcd...ef01', peers: 8, lastSeen: Date.now() - 5000, metrics: { cpu: 78, memory: 45, gpu: 92 }, position: { x: 25, y: 30 } },
        { id: 'node_03', type: 'compute', status: 'online', address: '0x2345...6789', peers: 6, lastSeen: Date.now() - 3000, metrics: { cpu: 34, memory: 58 }, position: { x: 75, y: 30 } },
        { id: 'node_04', type: 'verifier', status: 'online', address: '0x3456...7890', peers: 10, lastSeen: Date.now() - 1000, metrics: { cpu: 23, memory: 41 }, position: { x: 25, y: 70 } },
        { id: 'node_05', type: 'compute', status: 'syncing', address: '0x4567...8901', peers: 4, lastSeen: Date.now() - 15000, metrics: { cpu: 89, memory: 72, gpu: 45 }, position: { x: 75, y: 70 } },
        { id: 'node_06', type: 'compute', status: 'offline', address: '0x5678...9012', peers: 0, lastSeen: Date.now() - 300000, metrics: { cpu: 0, memory: 0 }, position: { x: 50, y: 85 } },
    ];

    const _mockConnections: Connection[] = connections || [
        { from: 'node_01', to: 'node_02', latency: 12, bandwidth: 150 },
        { from: 'node_01', to: 'node_03', latency: 8, bandwidth: 200 },
        { from: 'node_01', to: 'node_04', latency: 15, bandwidth: 100 },
        { from: 'node_01', to: 'node_05', latency: 25, bandwidth: 80 },
        { from: 'node_02', to: 'node_03', latency: 20, bandwidth: 120 },
        { from: 'node_04', to: 'node_05', latency: 18, bandwidth: 90 },
    ];

    const getTypeColor = (_type: string) => {
        return '#ffffff';
    };

    const getStatusColor = (status: string) => {
        const colors: Record<string, string> = {
            online: '#ffffff',
            offline: '#525252',
            syncing: '#a3a3a3',
        };
        return colors[status] || '#ffffff';
    };

    const formatTime = (ts: number) => {
        const diff = Date.now() - ts;
        if (diff < 60000) return `${Math.floor(diff / 1000)}s ago`;
        if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
        return 'Offline';
    };

    const onlineNodes = mockNodes.filter(n => n.status !== 'offline').length;
    const totalPeers = mockNodes.reduce((sum, n) => sum + n.peers, 0);

    return (
        <div className="bg-helix-surface rounded-md p-4 text-white border border-helix-border">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-white font-semibold text-[15px]">Network Nodes</h2>
                <div className="flex gap-2">
                    <button
                        className={`px-4 py-2 rounded-md border text-[13px] cursor-pointer transition-all ${
                            viewMode === 'graph'
                                ? 'bg-white/10 border-white/10 text-white'
                                : 'bg-white/5 border-helix-border text-[#666]'
                        }`}
                        onClick={() => setViewMode('graph')}
                    >
                        Graph
                    </button>
                    <button
                        className={`px-4 py-2 rounded-md border text-[13px] cursor-pointer transition-all ${
                            viewMode === 'list'
                                ? 'bg-white/10 border-white/10 text-white'
                                : 'bg-white/5 border-helix-border text-[#666]'
                        }`}
                        onClick={() => setViewMode('list')}
                    >
                        List
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-4 gap-3 mb-4">
                <div className="bg-white/[0.03] rounded-md p-4 border border-helix-border text-center">
                    <div className="text-lg font-bold mb-1 text-white">
                        {onlineNodes}
                    </div>
                    <div className="text-xs text-[#888] uppercase">Online Nodes</div>
                </div>
                <div className="bg-white/[0.03] rounded-md p-4 border border-helix-border text-center">
                    <div className="text-lg font-bold mb-1 text-white">
                        {mockNodes.filter(n => n.type === 'compute').length}
                    </div>
                    <div className="text-xs text-[#888] uppercase">Compute Nodes</div>
                </div>
                <div className="bg-white/[0.03] rounded-md p-4 border border-helix-border text-center">
                    <div className="text-lg font-bold mb-1 text-white">
                        {mockNodes.filter(n => n.type === 'aggregator').length}
                    </div>
                    <div className="text-xs text-[#888] uppercase">Aggregators</div>
                </div>
                <div className="bg-white/[0.03] rounded-md p-4 border border-helix-border text-center">
                    <div className="text-lg font-bold mb-1 text-white">
                        {totalPeers}
                    </div>
                    <div className="text-xs text-[#888] uppercase">Total Peers</div>
                </div>
            </div>

            {viewMode === 'graph' ? (
                <div className="relative h-[400px] bg-white/[0.03] rounded-md overflow-hidden mb-4">
                    {mockNodes.map((node) => (
                        <div
                            key={node.id}
                            className="absolute -translate-x-1/2 -translate-y-1/2 cursor-pointer transition-all hover:scale-110"
                            style={{
                                left: `${node.position.x}%`,
                                top: `${node.position.y}%`,
                            }}
                            onClick={() => setSelectedNode(node)}
                        >
                            <div
                                className="w-12 h-12 rounded-full flex items-center justify-center font-semibold text-xs bg-white/10 border-[3px] border-white shadow-[0_4px_12px_rgba(0,0,0,0.3)] text-white"
                            >
                                {node.type.charAt(0).toUpperCase()}
                            </div>
                            <div
                                className="absolute -top-1 -right-1 w-3 h-3 rounded-full border-2 border-[#141414]"
                                style={{
                                    background: getStatusColor(node.status),
                                    opacity: node.status === 'online' ? 1 : (node.status === 'syncing' ? 0.7 : 0.4)
                                }}
                            />
                            <div className="absolute -bottom-5 left-1/2 -translate-x-1/2 text-[10px] whitespace-nowrap text-[#888]">
                                {node.id}
                            </div>
                        </div>
                    ))}
                </div>
            ) : (
                <div className="flex flex-col gap-2">
                    {mockNodes.map((node) => (
                        <div
                            key={node.id}
                            className={`flex items-center justify-between p-4 bg-white/[0.03] rounded-md border cursor-pointer transition-all hover:bg-white/[0.06] ${
                                selectedNode?.id === node.id
                                    ? 'border-white/10 bg-white/10'
                                    : 'border-helix-border'
                            }`}
                            onClick={() => setSelectedNode(node)}
                        >
                            <div className="flex items-center gap-3">
                                <span className="px-3 py-1.5 rounded-md text-[11px] font-semibold uppercase bg-white/5 text-[#888]">
                                    {node.type}
                                </span>
                                <div>
                                    <div className="font-semibold">{node.id}</div>
                                    <div className="font-mono text-xs text-[#666]">{node.address}</div>
                                </div>
                            </div>
                            <div className="flex gap-3 items-center">
                                <div className="text-right">
                                    <div className="text-sm font-semibold">{node.peers}</div>
                                    <div className="text-[11px] text-[#666]">Peers</div>
                                </div>
                                <div className="text-right">
                                    <div className="text-sm font-semibold">{formatTime(node.lastSeen)}</div>
                                    <div className="text-[11px] text-[#666]">Last Seen</div>
                                </div>
                                <div
                                    className="w-2.5 h-2.5 rounded-full"
                                    style={{
                                        background: getStatusColor(node.status),
                                        opacity: node.status === 'online' ? 1 : (node.status === 'syncing' ? 0.7 : 0.4)
                                    }}
                                />
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {selectedNode && (
                <div className="p-4 bg-white/[0.03] rounded-md border border-helix-border">
                    <h3 className="text-base font-semibold mb-4">{selectedNode.id} Metrics</h3>
                    <div className="grid grid-cols-3 gap-3">
                        <div>
                            <div className="flex justify-between">
                                <span className="text-xs text-[#888]">CPU</span>
                                <span className="text-sm font-semibold">{selectedNode.metrics.cpu}%</span>
                            </div>
                            <div className="bg-white/[0.03] rounded h-2 overflow-hidden mt-2">
                                <div
                                    className="h-full rounded bg-white"
                                    style={{
                                        width: `${selectedNode.metrics.cpu}%`,
                                        opacity: selectedNode.metrics.cpu > 80 ? 1 : 0.7
                                    }}
                                />
                            </div>
                        </div>
                        <div>
                            <div className="flex justify-between">
                                <span className="text-xs text-[#888]">Memory</span>
                                <span className="text-sm font-semibold">{selectedNode.metrics.memory}%</span>
                            </div>
                            <div className="bg-white/[0.03] rounded h-2 overflow-hidden mt-2">
                                <div
                                    className="h-full rounded bg-white"
                                    style={{
                                        width: `${selectedNode.metrics.memory}%`,
                                        opacity: selectedNode.metrics.memory > 80 ? 1 : 0.7
                                    }}
                                />
                            </div>
                        </div>
                        {selectedNode.metrics.gpu !== undefined && (
                            <div>
                                <div className="flex justify-between">
                                    <span className="text-xs text-[#888]">GPU</span>
                                    <span className="text-sm font-semibold">{selectedNode.metrics.gpu}%</span>
                                </div>
                                <div className="bg-white/[0.03] rounded h-2 overflow-hidden mt-2">
                                    <div
                                        className="h-full rounded bg-white"
                                        style={{
                                            width: `${selectedNode.metrics.gpu}%`,
                                            opacity: selectedNode.metrics.gpu > 80 ? 1 : 0.7
                                        }}
                                    />
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
