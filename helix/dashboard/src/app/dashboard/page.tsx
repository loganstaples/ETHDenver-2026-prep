'use client';

import { useState } from 'react';
import { useChainId, useAccount } from 'wagmi';
import { ConnectButton } from '@rainbow-me/rainbowkit';
import { useContractState, useContractEvents } from '@/hooks/useContract';
import LiveLossCurve from '@/components/training/LiveLossCurve';
import ProofStream from '@/components/proof/ProofStream';
import LiveTopology from '@/components/network/LiveTopology';

export default function DashboardPage() {
    const chainId = useChainId();
    const { isConnected: walletConnected } = useAccount();

    const [selectedModelId, setSelectedModelId] = useState<bigint>(BigInt(0));

    const {
        nextModelId,
        slashingRecordCount: _slashingRecordCount,
        defaultMinStake: _defaultMinStake,
        isLoading: _contractLoading,
    } = useContractState();

    const {
        proofEvents,
        roundStartedEvents,
        roundCompletedEvents,
        stakedEvents,
        slashedEvents,
    } = useContractEvents(selectedModelId);

    const totalProofsSubmitted = proofEvents.length;
    const totalRoundsCompleted = roundCompletedEvents.length;
    const totalSlashings = slashedEvents.length;
    const totalStaked = stakedEvents.reduce((sum, e) => sum + e.amount, BigInt(0));

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between pb-6 border-b border-helix-border">
                <div className="flex items-center gap-3">
                    <div className="flex items-center gap-2 px-3 py-1.5 bg-white/5 border border-helix-border rounded-md text-sm text-[#888]">
                        Chain: {chainId}
                    </div>
                    {nextModelId && Number(nextModelId) > 0 && (
                        <div className="flex items-center gap-2 px-3 py-1.5 bg-white/5 border border-helix-border rounded-md text-sm">
                            <span className="text-[#888]">Model:</span>
                            <select
                                value={selectedModelId.toString()}
                                onChange={(e) => setSelectedModelId(BigInt(e.target.value))}
                                className="bg-transparent border-none text-white text-sm outline-none cursor-pointer"
                            >
                                {Array.from({ length: Number(nextModelId) }, (_, i) => (
                                    <option key={i} value={i.toString()} className="bg-helix-surface">
                                        Model #{i}
                                    </option>
                                ))}
                            </select>
                        </div>
                    )}
                </div>
                <ConnectButton />
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
                {[
                    { label: 'Models Registered', value: nextModelId?.toString() || '0' },
                    { label: 'Proofs Submitted', value: totalProofsSubmitted.toString() },
                    { label: 'Rounds Completed', value: totalRoundsCompleted.toString() },
                    { label: 'Total Staked', value: totalStaked > BigInt(0) ? `${(Number(totalStaked) / 1e18).toFixed(2)} ETH` : '0 ETH' },
                    { label: 'Slashing Events', value: totalSlashings.toString() },
                ].map((stat) => (
                    <div key={stat.label} className="bg-helix-surface border border-helix-border rounded-md p-4 text-center">
                        <div className="text-[15px] font-semibold text-white mb-1">{stat.value}</div>
                        <div className="text-xs text-[#666] uppercase tracking-wide">{stat.label}</div>
                    </div>
                ))}
            </div>

            {/* Charts Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                <LiveLossCurve modelId={selectedModelId} />
                <ProofStream modelId={selectedModelId} />
            </div>

            {/* Network Topology */}
            <LiveTopology modelId={selectedModelId} />

            {/* On-Chain Events */}
            <div className="bg-helix-surface border border-helix-border rounded-md overflow-hidden">
                <div className="flex items-center justify-between p-4 border-b border-helix-border">
                    <h2 className="text-lg font-semibold">Recent On-Chain Events</h2>
                    <span className="text-sm text-[#666]">From HELIX Coordinator Contract</span>
                </div>

                <div className="p-4 max-h-[400px] overflow-y-auto space-y-2">
                    {(proofEvents.length > 0 || roundStartedEvents.length > 0 || stakedEvents.length > 0) ? (
                        [
                            ...proofEvents.map(e => ({ ...e, eventType: 'ProofSubmitted' as const })),
                            ...roundStartedEvents.map(e => ({ ...e, eventType: 'RoundStarted' as const })),
                            ...roundCompletedEvents.map(e => ({ ...e, eventType: 'RoundCompleted' as const })),
                            ...stakedEvents.map(e => ({ ...e, eventType: 'Staked' as const })),
                            ...slashedEvents.map(e => ({ ...e, eventType: 'Slashed' as const })),
                        ]
                            .sort((a, b) => b.timestamp - a.timestamp)
                            .slice(0, 20)
                            .map((event, i) => (
                                <div key={`${event.eventType}-${i}`} className="flex items-center justify-between p-3 bg-white/[0.02] border border-helix-border rounded-md">
                                    <div className="flex items-center gap-3">
                                        <span className="px-2.5 py-1 rounded text-[11px] font-semibold bg-white/5 border border-helix-border text-white">
                                            {event.eventType}
                                        </span>
                                        <div>
                                            <div className="text-sm text-[#aaa]">
                                                Model #{event.modelId.toString()}
                                                {'roundId' in event && ` / Round #${event.roundId.toString()}`}
                                            </div>
                                            <div className="text-xs text-[#666] font-mono">
                                                {'prover' in event && `${event.prover.slice(0, 10)}...`}
                                                {'amount' in event && ` ${(Number(event.amount) / 1e18).toFixed(4)} ETH`}
                                            </div>
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <div className="text-xs text-[#888]">
                                            Block #{event.blockNumber.toLocaleString()}
                                        </div>
                                        <a
                                            href={`https://etherscan.io/tx/${event.transactionHash}`}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="text-xs text-white/60 hover:text-white transition-colors"
                                        >
                                            View tx
                                        </a>
                                    </div>
                                </div>
                            ))
                    ) : (
                        <div className="flex flex-col items-center justify-center py-16 text-[#666]">
                            <p className="text-base mb-2">No on-chain events yet</p>
                            <p className="text-sm text-[#555]">
                                {walletConnected
                                    ? 'Events will appear here as they are emitted from the HELIX contracts'
                                    : 'Connect your wallet to view on-chain activity'}
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
