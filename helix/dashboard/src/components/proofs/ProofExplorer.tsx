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
        return '#ffffff';
    };

    return (
        <div className="bg-helix-surface rounded-md p-4 text-white border border-helix-border">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-white font-semibold text-[15px]">Proof Explorer</h2>
                <div className="flex gap-2">
                    {['all', 'training', 'aggregation', 'gradient', 'computation'].map(
                        (f) => (
                            <button
                                key={f}
                                className={`px-4 py-2 rounded-md border text-[13px] cursor-pointer transition-all duration-200 ${
                                    filter === f
                                        ? 'bg-white/10 border-white/10 text-white'
                                        : 'bg-white/[0.04] border-helix-border text-[#666] hover:bg-white/10'
                                }`}
                                onClick={() => setFilter(f)}
                            >
                                {f.charAt(0).toUpperCase() + f.slice(1)}
                            </button>
                        )
                    )}
                </div>
            </div>

            <div className="grid grid-cols-4 gap-3 mb-4">
                <div className="bg-white/[0.03] rounded-md p-4 border border-helix-border text-center">
                    <div className="text-lg font-bold mb-1 text-white">
                        {mockProofs.filter((p) => p.status === 'verified').length}
                    </div>
                    <div className="text-xs text-[#888] uppercase">Verified</div>
                </div>
                <div className="bg-white/[0.03] rounded-md p-4 border border-helix-border text-center">
                    <div className="text-lg font-bold mb-1 text-white">
                        {mockProofs.filter((p) => p.status === 'pending').length}
                    </div>
                    <div className="text-xs text-[#888] uppercase">Pending</div>
                </div>
                <div className="bg-white/[0.03] rounded-md p-4 border border-helix-border text-center">
                    <div className="text-lg font-bold mb-1 text-white">
                        {formatSize(mockProofs.reduce((sum, p) => sum + p.size, 0))}
                    </div>
                    <div className="text-xs text-[#888] uppercase">Total Size</div>
                </div>
                <div className="bg-white/[0.03] rounded-md p-4 border border-helix-border text-center">
                    <div className="text-lg font-bold mb-1 text-white">
                        {(
                            mockProofs
                                .filter((p) => p.verificationTime > 0)
                                .reduce((sum, p) => sum + p.verificationTime, 0) /
                            mockProofs.filter((p) => p.verificationTime > 0).length
                        ).toFixed(2)}
                        s
                    </div>
                    <div className="text-xs text-[#888] uppercase">Avg Verify Time</div>
                </div>
            </div>

            <div className="flex flex-col gap-3">
                {filteredProofs.map((proof) => (
                    <div
                        key={proof.id}
                        className={`flex items-center justify-between px-5 py-4 rounded-md border cursor-pointer transition-all duration-200 ${
                            selectedProof?.id === proof.id
                                ? 'border-white/10 bg-white/10'
                                : 'bg-white/[0.03] border-helix-border hover:bg-white/[0.06]'
                        }`}
                        onClick={() => setSelectedProof(proof)}
                    >
                        <div className="flex items-center gap-3">
                            <span className="px-3 py-1.5 rounded-md text-[11px] font-semibold uppercase bg-white/[0.04] text-[#888]">
                                {proof.type}
                            </span>
                            <div>
                                <div className="font-mono text-sm font-medium text-white">{proof.id}</div>
                                <div className="font-mono text-xs text-[#666]">{proof.hash}</div>
                            </div>
                        </div>
                        <div className="flex items-center gap-3">
                            <div className="text-right">
                                <div className="text-sm font-semibold text-white">Round #{proof.roundId}</div>
                                <div className="text-[11px] text-[#666]">Training Round</div>
                            </div>
                            <div className="text-right">
                                <div className="text-sm font-semibold text-white">{formatSize(proof.size)}</div>
                                <div className="text-[11px] text-[#666]">Size</div>
                            </div>
                            <div className="text-right">
                                <div className="text-sm font-semibold text-white">{formatTime(proof.timestamp)}</div>
                                <div className="text-[11px] text-[#666]">Created</div>
                            </div>
                            <span
                                className={`px-3 py-1 rounded-md text-[11px] font-semibold uppercase ${
                                    proof.status === 'verified'
                                        ? 'bg-white/10 text-white'
                                        : proof.status === 'pending'
                                        ? 'bg-white/[0.04] text-[#888]'
                                        : 'bg-white/[0.04] text-[#666]'
                                }`}
                            >
                                {proof.status}
                            </span>
                        </div>
                    </div>
                ))}
            </div>

            {selectedProof && (
                <div className="mt-4 p-4 bg-white/[0.03] rounded-md border border-helix-border">
                    <h3 className="text-base font-semibold mb-4 text-[#aaa]">
                        Proof Details: {selectedProof.id}
                    </h3>
                    <div className="grid grid-cols-2 gap-3">
                        <div className="bg-white/[0.03] rounded-md p-3">
                            <div className="text-[11px] text-[#666] uppercase mb-1">Proof Hash</div>
                            <div className="font-mono text-[13px] text-[#aaa] break-all">
                                {selectedProof.hash}
                            </div>
                        </div>
                        <div className="bg-white/[0.03] rounded-md p-3">
                            <div className="text-[11px] text-[#666] uppercase mb-1">Model Hash</div>
                            <div className="font-mono text-[13px] text-[#aaa] break-all">
                                {selectedProof.modelHash}
                            </div>
                        </div>
                        <div className="bg-white/[0.03] rounded-md p-3">
                            <div className="text-[11px] text-[#666] uppercase mb-1">Error Bound</div>
                            <div className="font-mono text-[13px] text-[#aaa] break-all">
                                {selectedProof.errorBound}
                            </div>
                        </div>
                        <div className="bg-white/[0.03] rounded-md p-3">
                            <div className="text-[11px] text-[#666] uppercase mb-1">Verification Time</div>
                            <div className="font-mono text-[13px] text-[#aaa] break-all">
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
