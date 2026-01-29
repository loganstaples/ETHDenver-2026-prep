'use client';

import React, { useState } from 'react';

interface Transaction {
    hash: string;
    type: 'TrainingRoundSubmitted' | 'GradientAggregated' | 'ProofVerified' | 'RewardDistributed';
    blockNumber: number;
    timestamp: number;
    from: string;
    value?: string;
    gasUsed: number;
    status: 'success' | 'pending' | 'failed';
}

interface ContractState {
    totalRounds: number;
    activeParticipants: number;
    totalRewardsDistributed: string;
    lastUpdateBlock: number;
    modelHash: string;
}

interface OnChainExplorerProps {
    transactions?: Transaction[];
    contractState?: ContractState;
    contractAddress?: string;
}

export default function OnChainExplorer({
    transactions,
    contractState,
    contractAddress = '0x1234...5678',
}: OnChainExplorerProps) {
    const [filter, setFilter] = useState<string>('all');
    const [selectedTx, setSelectedTx] = useState<Transaction | null>(null);

    const mockState: ContractState = contractState || {
        totalRounds: 156,
        activeParticipants: 24,
        totalRewardsDistributed: '12,450.5 HELIX',
        lastUpdateBlock: 19234567,
        modelHash: '0xabcdef123456789...',
    };

    const mockTxs: Transaction[] = transactions || [
        { hash: '0xabc...123', type: 'ProofVerified', blockNumber: 19234567, timestamp: Date.now() - 60000, from: '0x1234...5678', gasUsed: 245000, status: 'success' },
        { hash: '0xdef...456', type: 'GradientAggregated', blockNumber: 19234566, timestamp: Date.now() - 180000, from: '0xabcd...ef01', gasUsed: 312000, status: 'success' },
        { hash: '0xghi...789', type: 'RewardDistributed', blockNumber: 19234565, timestamp: Date.now() - 300000, from: '0x1234...5678', value: '125.5 HELIX', gasUsed: 156000, status: 'success' },
        { hash: '0xjkl...012', type: 'TrainingRoundSubmitted', blockNumber: 19234564, timestamp: Date.now() - 600000, from: '0x2345...6789', gasUsed: 423000, status: 'success' },
        { hash: '0xmno...345', type: 'ProofVerified', blockNumber: 19234563, timestamp: Date.now() - 900000, from: '0x1234...5678', gasUsed: 267000, status: 'pending' },
    ];

    const filteredTxs = mockTxs.filter(tx => filter === 'all' || tx.type === filter);

    const getTypeColor = (type: string) => {
        const colors: Record<string, string> = {
            TrainingRoundSubmitted: '#6366f1',
            GradientAggregated: '#a855f7',
            ProofVerified: '#22c55e',
            RewardDistributed: '#f59e0b',
        };
        return colors[type] || '#6b7280';
    };

    const formatTime = (ts: number) => {
        const diff = Date.now() - ts;
        if (diff < 60000) return `${Math.floor(diff / 1000)}s ago`;
        if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
        return new Date(ts).toLocaleTimeString();
    };

    const formatGas = (gas: number) => {
        return `${(gas / 1000).toFixed(1)}K gas`;
    };

    return (
        <div className="onchain-explorer">
            <style jsx>{`
        .onchain-explorer {
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
          background: linear-gradient(90deg, #f59e0b, #ef4444);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .contract-address {
          padding: 8px 16px;
          background: rgba(0, 0, 0, 0.3);
          border-radius: 8px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 13px;
          color: #9ca3af;
        }

        .state-grid {
          display: grid;
          grid-template-columns: repeat(5, 1fr);
          gap: 16px;
          margin-bottom: 24px;
        }

        .state-card {
          background: rgba(255, 255, 255, 0.03);
          border-radius: 12px;
          padding: 16px;
          border: 1px solid rgba(255, 255, 255, 0.08);
          text-align: center;
        }

        .state-value {
          font-size: 24px;
          font-weight: 700;
          margin-bottom: 4px;
        }

        .state-label {
          font-size: 11px;
          color: #9ca3af;
          text-transform: uppercase;
        }

        .filters {
          display: flex;
          gap: 8px;
          margin-bottom: 16px;
          flex-wrap: wrap;
        }

        .filter-btn {
          padding: 8px 16px;
          border-radius: 8px;
          border: 1px solid rgba(255, 255, 255, 0.1);
          background: rgba(255, 255, 255, 0.05);
          color: #9ca3af;
          font-size: 12px;
          cursor: pointer;
          transition: all 0.2s;
        }

        .filter-btn:hover {
          background: rgba(255, 255, 255, 0.1);
        }

        .filter-btn.active {
          background: rgba(99, 102, 241, 0.2);
          border-color: rgba(99, 102, 241, 0.5);
          color: #818cf8;
        }

        .tx-list {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .tx-item {
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

        .tx-item:hover {
          background: rgba(255, 255, 255, 0.06);
          border-color: rgba(255, 255, 255, 0.15);
        }

        .tx-item.selected {
          border-color: rgba(99, 102, 241, 0.5);
          background: rgba(99, 102, 241, 0.1);
        }

        .tx-main {
          display: flex;
          align-items: center;
          gap: 16px;
        }

        .tx-type {
          padding: 6px 12px;
          border-radius: 6px;
          font-size: 11px;
          font-weight: 600;
        }

        .tx-hash {
          font-family: 'JetBrains Mono', monospace;
          font-size: 14px;
          font-weight: 500;
        }

        .tx-from {
          font-family: 'JetBrains Mono', monospace;
          font-size: 12px;
          color: #6b7280;
        }

        .tx-meta {
          display: flex;
          align-items: center;
          gap: 24px;
        }

        .tx-stat {
          text-align: right;
        }

        .tx-stat-value {
          font-size: 13px;
          font-weight: 600;
        }

        .tx-stat-label {
          font-size: 10px;
          color: #6b7280;
        }

        .status-badge {
          padding: 4px 10px;
          border-radius: 12px;
          font-size: 10px;
          font-weight: 600;
          text-transform: uppercase;
        }

        .status-badge.success {
          background: rgba(34, 197, 94, 0.2);
          color: #22c55e;
        }

        .status-badge.pending {
          background: rgba(245, 158, 11, 0.2);
          color: #f59e0b;
        }

        .status-badge.failed {
          background: rgba(239, 68, 68, 0.2);
          color: #ef4444;
        }

        .detail-panel {
          margin-top: 24px;
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

        .detail-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 16px;
        }

        .detail-item {
          background: rgba(0, 0, 0, 0.2);
          padding: 12px;
          border-radius: 8px;
        }

        .detail-label {
          font-size: 11px;
          color: #6b7280;
          text-transform: uppercase;
          margin-bottom: 4px;
        }

        .detail-value {
          font-family: 'JetBrains Mono', monospace;
          font-size: 13px;
          color: #d1d5db;
          word-break: break-all;
        }

        .etherscan-link {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          color: #6366f1;
          font-size: 13px;
          text-decoration: none;
          margin-top: 16px;
        }

        .etherscan-link:hover {
          text-decoration: underline;
        }
      `}</style>

            <div className="header">
                <h2 className="title">On-Chain Explorer</h2>
                <span className="contract-address">{contractAddress}</span>
            </div>

            <div className="state-grid">
                <div className="state-card">
                    <div className="state-value" style={{ color: '#6366f1' }}>
                        {mockState.totalRounds}
                    </div>
                    <div className="state-label">Training Rounds</div>
                </div>
                <div className="state-card">
                    <div className="state-value" style={{ color: '#22c55e' }}>
                        {mockState.activeParticipants}
                    </div>
                    <div className="state-label">Active Participants</div>
                </div>
                <div className="state-card">
                    <div className="state-value" style={{ color: '#f59e0b' }}>
                        {mockState.totalRewardsDistributed.split(' ')[0]}
                    </div>
                    <div className="state-label">Rewards (HELIX)</div>
                </div>
                <div className="state-card">
                    <div className="state-value" style={{ color: '#a855f7' }}>
                        #{mockState.lastUpdateBlock.toLocaleString()}
                    </div>
                    <div className="state-label">Last Block</div>
                </div>
                <div className="state-card">
                    <div className="state-value" style={{ fontSize: 14, color: '#9ca3af' }}>
                        {mockState.modelHash.slice(0, 12)}...
                    </div>
                    <div className="state-label">Model Hash</div>
                </div>
            </div>

            <div className="filters">
                {['all', 'TrainingRoundSubmitted', 'GradientAggregated', 'ProofVerified', 'RewardDistributed'].map((f) => (
                    <button
                        key={f}
                        className={`filter-btn ${filter === f ? 'active' : ''}`}
                        onClick={() => setFilter(f)}
                    >
                        {f === 'all' ? 'All' : f.replace(/([A-Z])/g, ' $1').trim()}
                    </button>
                ))}
            </div>

            <div className="tx-list">
                {filteredTxs.map((tx) => (
                    <div
                        key={tx.hash}
                        className={`tx-item ${selectedTx?.hash === tx.hash ? 'selected' : ''}`}
                        onClick={() => setSelectedTx(tx)}
                    >
                        <div className="tx-main">
                            <span
                                className="tx-type"
                                style={{
                                    background: `${getTypeColor(tx.type)}22`,
                                    color: getTypeColor(tx.type),
                                }}
                            >
                                {tx.type.replace(/([A-Z])/g, ' $1').trim()}
                            </span>
                            <div>
                                <div className="tx-hash">{tx.hash}</div>
                                <div className="tx-from">From: {tx.from}</div>
                            </div>
                        </div>
                        <div className="tx-meta">
                            <div className="tx-stat">
                                <div className="tx-stat-value">#{tx.blockNumber.toLocaleString()}</div>
                                <div className="tx-stat-label">Block</div>
                            </div>
                            <div className="tx-stat">
                                <div className="tx-stat-value">{formatGas(tx.gasUsed)}</div>
                                <div className="tx-stat-label">Gas Used</div>
                            </div>
                            <div className="tx-stat">
                                <div className="tx-stat-value">{formatTime(tx.timestamp)}</div>
                                <div className="tx-stat-label">Time</div>
                            </div>
                            <span className={`status-badge ${tx.status}`}>{tx.status}</span>
                        </div>
                    </div>
                ))}
            </div>

            {selectedTx && (
                <div className="detail-panel">
                    <h3 className="detail-title">Transaction Details</h3>
                    <div className="detail-grid">
                        <div className="detail-item">
                            <div className="detail-label">Transaction Hash</div>
                            <div className="detail-value">{selectedTx.hash}</div>
                        </div>
                        <div className="detail-item">
                            <div className="detail-label">From Address</div>
                            <div className="detail-value">{selectedTx.from}</div>
                        </div>
                        <div className="detail-item">
                            <div className="detail-label">Block Number</div>
                            <div className="detail-value">{selectedTx.blockNumber.toLocaleString()}</div>
                        </div>
                        <div className="detail-item">
                            <div className="detail-label">Gas Used</div>
                            <div className="detail-value">{selectedTx.gasUsed.toLocaleString()}</div>
                        </div>
                        {selectedTx.value && (
                            <div className="detail-item">
                                <div className="detail-label">Value</div>
                                <div className="detail-value">{selectedTx.value}</div>
                            </div>
                        )}
                    </div>
                    <a className="etherscan-link" href="#" onClick={(e) => e.preventDefault()}>
                        View on Etherscan →
                    </a>
                </div>
            )}
        </div>
    );
}
