'use client';

import React, { useState } from 'react';

interface ProofData {
    id: string;
    type: 'training' | 'aggregation' | 'gradient' | 'computation';
    status: 'verified' | 'pending' | 'failed';
    roundId: number;
    timestamp: number;
    size: number;
    verificationTime: number;
    hash: string;
    modelHash: string;
    errorBound: number;
}

interface ProofExplorerProps {
    proofs?: ProofData[];
}

export default function ProofExplorer({ proofs }: ProofExplorerProps) {
    const [selectedProof, setSelectedProof] = useState<ProofData | null>(null);
    const [filter, setFilter] = useState<string>('all');

    const mockProofs: ProofData[] = proofs || [
        {
            id: 'proof_001',
            type: 'training',
            status: 'verified',
            roundId: 3,
            timestamp: Date.now() - 60000,
            size: 2048,
            verificationTime: 1.23,
            hash: '0x7a8b9c...3d4e5f',
            modelHash: '0xabcdef...123456',
            errorBound: 0.0001,
        },
        {
            id: 'proof_002',
            type: 'aggregation',
            status: 'verified',
            roundId: 3,
            timestamp: Date.now() - 120000,
            size: 4096,
            verificationTime: 2.45,
            hash: '0x1a2b3c...7d8e9f',
            modelHash: '0xabcdef...123456',
            errorBound: 0.00015,
        },
        {
            id: 'proof_003',
            type: 'gradient',
            status: 'pending',
            roundId: 3,
            timestamp: Date.now() - 30000,
            size: 1024,
            verificationTime: 0,
            hash: '0x4e5f6g...0a1b2c',
            modelHash: '0xabcdef...123456',
            errorBound: 0.00008,
        },
        {
            id: 'proof_004',
            type: 'computation',
            status: 'verified',
            roundId: 2,
            timestamp: Date.now() - 300000,
            size: 3072,
            verificationTime: 1.87,
            hash: '0x9h0i1j...4k5l6m',
            modelHash: '0xfedcba...654321',
            errorBound: 0.00012,
        },
    ];

    const filteredProofs = mockProofs.filter((p) =>
        filter === 'all' ? true : p.type === filter
    );

    const formatTime = (ts: number) => {
        const diff = Date.now() - ts;
        if (diff < 60000) return `${Math.floor(diff / 1000)}s ago`;
        if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
        return new Date(ts).toLocaleTimeString();
    };

    const formatSize = (bytes: number) => {
        if (bytes < 1024) return `${bytes} B`;
        return `${(bytes / 1024).toFixed(1)} KB`;
    };

    const getTypeColor = (type: string) => {
        const colors: Record<string, string> = {
            training: '#6366f1',
            aggregation: '#a855f7',
            gradient: '#22c55e',
            computation: '#f59e0b',
        };
        return colors[type] || '#6b7280';
    };

    return (
        <div className="proof-explorer">
            <style jsx>{`
        .proof-explorer {
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
          background: linear-gradient(90deg, #22c55e, #14b8a6);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .filters {
          display: flex;
          gap: 8px;
        }

        .filter-btn {
          padding: 8px 16px;
          border-radius: 8px;
          border: 1px solid rgba(255, 255, 255, 0.1);
          background: rgba(255, 255, 255, 0.05);
          color: #9ca3af;
          font-size: 13px;
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

        .proofs-list {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .proof-card {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 16px 20px;
          background: rgba(255, 255, 255, 0.03);
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.08);
          cursor: pointer;
          transition: all 0.2s;
        }

        .proof-card:hover {
          background: rgba(255, 255, 255, 0.06);
          border-color: rgba(255, 255, 255, 0.15);
        }

        .proof-card.selected {
          border-color: rgba(99, 102, 241, 0.5);
          background: rgba(99, 102, 241, 0.1);
        }

        .proof-main {
          display: flex;
          align-items: center;
          gap: 16px;
        }

        .proof-type {
          padding: 6px 12px;
          border-radius: 6px;
          font-size: 11px;
          font-weight: 600;
          text-transform: uppercase;
        }

        .proof-id {
          font-family: 'JetBrains Mono', monospace;
          font-size: 14px;
          font-weight: 500;
        }

        .proof-hash {
          font-family: 'JetBrains Mono', monospace;
          font-size: 12px;
          color: #6b7280;
        }

        .proof-meta {
          display: flex;
          align-items: center;
          gap: 24px;
        }

        .proof-stat {
          text-align: right;
        }

        .proof-stat-value {
          font-size: 14px;
          font-weight: 600;
        }

        .proof-stat-label {
          font-size: 11px;
          color: #6b7280;
        }

        .status-badge {
          padding: 4px 12px;
          border-radius: 12px;
          font-size: 11px;
          font-weight: 600;
          text-transform: uppercase;
        }

        .status-badge.verified {
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
          color: #d1d5db;
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
      `}</style>

            <div className="header">
                <h2 className="title">Proof Explorer</h2>
                <div className="filters">
                    {['all', 'training', 'aggregation', 'gradient', 'computation'].map(
                        (f) => (
                            <button
                                key={f}
                                className={`filter-btn ${filter === f ? 'active' : ''}`}
                                onClick={() => setFilter(f)}
                            >
                                {f.charAt(0).toUpperCase() + f.slice(1)}
                            </button>
                        )
                    )}
                </div>
            </div>

            <div className="stats-row">
                <div className="stat-card">
                    <div className="stat-value" style={{ color: '#22c55e' }}>
                        {mockProofs.filter((p) => p.status === 'verified').length}
                    </div>
                    <div className="stat-label">Verified</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value" style={{ color: '#f59e0b' }}>
                        {mockProofs.filter((p) => p.status === 'pending').length}
                    </div>
                    <div className="stat-label">Pending</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value" style={{ color: '#6366f1' }}>
                        {formatSize(mockProofs.reduce((sum, p) => sum + p.size, 0))}
                    </div>
                    <div className="stat-label">Total Size</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value" style={{ color: '#a855f7' }}>
                        {(
                            mockProofs
                                .filter((p) => p.verificationTime > 0)
                                .reduce((sum, p) => sum + p.verificationTime, 0) /
                            mockProofs.filter((p) => p.verificationTime > 0).length
                        ).toFixed(2)}
                        s
                    </div>
                    <div className="stat-label">Avg Verify Time</div>
                </div>
            </div>

            <div className="proofs-list">
                {filteredProofs.map((proof) => (
                    <div
                        key={proof.id}
                        className={`proof-card ${selectedProof?.id === proof.id ? 'selected' : ''}`}
                        onClick={() => setSelectedProof(proof)}
                    >
                        <div className="proof-main">
                            <span
                                className="proof-type"
                                style={{
                                    background: `${getTypeColor(proof.type)}22`,
                                    color: getTypeColor(proof.type),
                                }}
                            >
                                {proof.type}
                            </span>
                            <div>
                                <div className="proof-id">{proof.id}</div>
                                <div className="proof-hash">{proof.hash}</div>
                            </div>
                        </div>
                        <div className="proof-meta">
                            <div className="proof-stat">
                                <div className="proof-stat-value">Round #{proof.roundId}</div>
                                <div className="proof-stat-label">Training Round</div>
                            </div>
                            <div className="proof-stat">
                                <div className="proof-stat-value">{formatSize(proof.size)}</div>
                                <div className="proof-stat-label">Size</div>
                            </div>
                            <div className="proof-stat">
                                <div className="proof-stat-value">{formatTime(proof.timestamp)}</div>
                                <div className="proof-stat-label">Created</div>
                            </div>
                            <span className={`status-badge ${proof.status}`}>
                                {proof.status}
                            </span>
                        </div>
                    </div>
                ))}
            </div>

            {selectedProof && (
                <div className="detail-panel">
                    <h3 className="detail-title">Proof Details: {selectedProof.id}</h3>
                    <div className="detail-grid">
                        <div className="detail-item">
                            <div className="detail-label">Proof Hash</div>
                            <div className="detail-value">{selectedProof.hash}</div>
                        </div>
                        <div className="detail-item">
                            <div className="detail-label">Model Hash</div>
                            <div className="detail-value">{selectedProof.modelHash}</div>
                        </div>
                        <div className="detail-item">
                            <div className="detail-label">Error Bound</div>
                            <div className="detail-value">{selectedProof.errorBound}</div>
                        </div>
                        <div className="detail-item">
                            <div className="detail-label">Verification Time</div>
                            <div className="detail-value">
                                {selectedProof.verificationTime > 0
                                    ? `${selectedProof.verificationTime}s`
                                    : 'Pending'}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
