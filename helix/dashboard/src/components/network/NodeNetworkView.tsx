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

    const getTypeColor = (type: string) => {
        const colors: Record<string, string> = {
            aggregator: '#a855f7',
            compute: '#6366f1',
            verifier: '#22c55e',
        };
        return colors[type] || '#6b7280';
    };

    const getStatusColor = (status: string) => {
        const colors: Record<string, string> = {
            online: '#22c55e',
            offline: '#ef4444',
            syncing: '#f59e0b',
        };
        return colors[status] || '#6b7280';
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
        <div className="node-network">
            <style jsx>{`
        .node-network {
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

        .controls {
          display: flex;
          gap: 8px;
        }

        .view-btn {
          padding: 8px 16px;
          border-radius: 8px;
          border: 1px solid rgba(255, 255, 255, 0.1);
          background: rgba(255, 255, 255, 0.05);
          color: #9ca3af;
          font-size: 13px;
          cursor: pointer;
          transition: all 0.2s;
        }

        .view-btn.active {
          background: rgba(99, 102, 241, 0.2);
          border-color: rgba(99, 102, 241, 0.5);
          color: #818cf8;
        }

        .stats-row {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 16px;
          margin-bottom: 24px;
        }

        .stat-card {
          background: rgba(255, 255, 255, 0.03);
          border-radius: 12px;
          padding: 16px;
          border: 1px solid rgba(255, 255, 255, 0.08);
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
        }

        .network-graph {
          position: relative;
          height: 400px;
          background: rgba(0, 0, 0, 0.2);
          border-radius: 12px;
          overflow: hidden;
          margin-bottom: 24px;
        }

        .graph-node {
          position: absolute;
          transform: translate(-50%, -50%);
          cursor: pointer;
          transition: all 0.2s;
        }

        .graph-node:hover {
          transform: translate(-50%, -50%) scale(1.1);
        }

        .node-circle {
          width: 48px;
          height: 48px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: 600;
          font-size: 12px;
          border: 3px solid;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }

        .node-label {
          position: absolute;
          bottom: -20px;
          left: 50%;
          transform: translateX(-50%);
          font-size: 10px;
          white-space: nowrap;
          color: #9ca3af;
        }

        .node-status {
          position: absolute;
          top: -4px;
          right: -4px;
          width: 12px;
          height: 12px;
          border-radius: 50%;
          border: 2px solid #1a1a2e;
        }

        .connection-line {
          position: absolute;
          height: 2px;
          background: rgba(99, 102, 241, 0.3);
          transform-origin: left center;
        }

        .nodes-list {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .node-item {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 16px;
          background: rgba(255, 255, 255, 0.03);
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.08);
          cursor: pointer;
          transition: all 0.2s;
        }

        .node-item:hover {
          background: rgba(255, 255, 255, 0.06);
        }

        .node-item.selected {
          border-color: rgba(99, 102, 241, 0.5);
          background: rgba(99, 102, 241, 0.1);
        }

        .node-info {
          display: flex;
          align-items: center;
          gap: 16px;
        }

        .node-type-badge {
          padding: 6px 12px;
          border-radius: 6px;
          font-size: 11px;
          font-weight: 600;
          text-transform: uppercase;
        }

        .node-id {
          font-weight: 600;
        }

        .node-address {
          font-family: 'JetBrains Mono', monospace;
          font-size: 12px;
          color: #6b7280;
        }

        .node-meta {
          display: flex;
          gap: 24px;
          align-items: center;
        }

        .node-stat {
          text-align: right;
        }

        .node-stat-value {
          font-size: 14px;
          font-weight: 600;
        }

        .node-stat-label {
          font-size: 11px;
          color: #6b7280;
        }

        .status-dot {
          width: 10px;
          height: 10px;
          border-radius: 50%;
        }

        .detail-panel {
          padding: 20px;
          background: rgba(255, 255, 255, 0.03);
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .detail-title {
          font-size: 16px;
          font-weight: 600;
          margin-bottom: 16px;
        }

        .metrics-row {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 16px;
        }

        .metric-bar {
          background: rgba(0, 0, 0, 0.3);
          border-radius: 4px;
          height: 8px;
          overflow: hidden;
          margin-top: 8px;
        }

        .metric-fill {
          height: 100%;
          border-radius: 4px;
        }
      `}</style>

            <div className="header">
                <h2 className="title">Network Nodes</h2>
                <div className="controls">
                    <button
                        className={`view-btn ${viewMode === 'graph' ? 'active' : ''}`}
                        onClick={() => setViewMode('graph')}
                    >
                        Graph
                    </button>
                    <button
                        className={`view-btn ${viewMode === 'list' ? 'active' : ''}`}
                        onClick={() => setViewMode('list')}
                    >
                        List
                    </button>
                </div>
            </div>

            <div className="stats-row">
                <div className="stat-card">
                    <div className="stat-value" style={{ color: '#22c55e' }}>
                        {onlineNodes}
                    </div>
                    <div className="stat-label">Online Nodes</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value" style={{ color: '#6366f1' }}>
                        {mockNodes.filter(n => n.type === 'compute').length}
                    </div>
                    <div className="stat-label">Compute Nodes</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value" style={{ color: '#a855f7' }}>
                        {mockNodes.filter(n => n.type === 'aggregator').length}
                    </div>
                    <div className="stat-label">Aggregators</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value" style={{ color: '#f59e0b' }}>
                        {totalPeers}
                    </div>
                    <div className="stat-label">Total Peers</div>
                </div>
            </div>

            {viewMode === 'graph' ? (
                <div className="network-graph">
                    {mockNodes.map((node) => (
                        <div
                            key={node.id}
                            className="graph-node"
                            style={{
                                left: `${node.position.x}%`,
                                top: `${node.position.y}%`,
                            }}
                            onClick={() => setSelectedNode(node)}
                        >
                            <div
                                className="node-circle"
                                style={{
                                    background: `${getTypeColor(node.type)}33`,
                                    borderColor: getTypeColor(node.type),
                                    color: getTypeColor(node.type),
                                }}
                            >
                                {node.type.charAt(0).toUpperCase()}
                            </div>
                            <div
                                className="node-status"
                                style={{ background: getStatusColor(node.status) }}
                            />
                            <div className="node-label">{node.id}</div>
                        </div>
                    ))}
                </div>
            ) : (
                <div className="nodes-list">
                    {mockNodes.map((node) => (
                        <div
                            key={node.id}
                            className={`node-item ${selectedNode?.id === node.id ? 'selected' : ''}`}
                            onClick={() => setSelectedNode(node)}
                        >
                            <div className="node-info">
                                <span
                                    className="node-type-badge"
                                    style={{
                                        background: `${getTypeColor(node.type)}22`,
                                        color: getTypeColor(node.type),
                                    }}
                                >
                                    {node.type}
                                </span>
                                <div>
                                    <div className="node-id">{node.id}</div>
                                    <div className="node-address">{node.address}</div>
                                </div>
                            </div>
                            <div className="node-meta">
                                <div className="node-stat">
                                    <div className="node-stat-value">{node.peers}</div>
                                    <div className="node-stat-label">Peers</div>
                                </div>
                                <div className="node-stat">
                                    <div className="node-stat-value">{formatTime(node.lastSeen)}</div>
                                    <div className="node-stat-label">Last Seen</div>
                                </div>
                                <div
                                    className="status-dot"
                                    style={{ background: getStatusColor(node.status) }}
                                />
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {selectedNode && (
                <div className="detail-panel">
                    <h3 className="detail-title">{selectedNode.id} Metrics</h3>
                    <div className="metrics-row">
                        <div>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span style={{ fontSize: '12px', color: '#9ca3af' }}>CPU</span>
                                <span style={{ fontSize: '14px', fontWeight: 600 }}>{selectedNode.metrics.cpu}%</span>
                            </div>
                            <div className="metric-bar">
                                <div className="metric-fill" style={{
                                    width: `${selectedNode.metrics.cpu}%`,
                                    background: selectedNode.metrics.cpu > 80 ? '#ef4444' : '#6366f1'
                                }} />
                            </div>
                        </div>
                        <div>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span style={{ fontSize: '12px', color: '#9ca3af' }}>Memory</span>
                                <span style={{ fontSize: '14px', fontWeight: 600 }}>{selectedNode.metrics.memory}%</span>
                            </div>
                            <div className="metric-bar">
                                <div className="metric-fill" style={{
                                    width: `${selectedNode.metrics.memory}%`,
                                    background: selectedNode.metrics.memory > 80 ? '#ef4444' : '#22c55e'
                                }} />
                            </div>
                        </div>
                        {selectedNode.metrics.gpu !== undefined && (
                            <div>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <span style={{ fontSize: '12px', color: '#9ca3af' }}>GPU</span>
                                    <span style={{ fontSize: '14px', fontWeight: 600 }}>{selectedNode.metrics.gpu}%</span>
                                </div>
                                <div className="metric-bar">
                                    <div className="metric-fill" style={{
                                        width: `${selectedNode.metrics.gpu}%`,
                                        background: selectedNode.metrics.gpu > 80 ? '#ef4444' : '#a855f7'
                                    }} />
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
