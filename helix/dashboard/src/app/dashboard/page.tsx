'use client';

import { useState } from 'react';
import { useChainId, useAccount } from 'wagmi';
import { ConnectButton } from '@rainbow-me/rainbowkit';
import { Activity, Box, Database, Layers, Shield, ShieldAlert, Cpu, Network } from 'lucide-react';
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

    const stats = [
        {
            label: 'Models Registered',
            value: nextModelId?.toString() || '0',
            icon: Box,
            desc: 'Active models on network'
        },
        {
            label: 'Total Staked',
            value: totalStaked > BigInt(0) ? `${(Number(totalStaked) / 1e18).toFixed(2)} ETH` : '0 ETH',
            icon: Database,
            desc: 'Total value locked'
        },
        {
            label: 'Rounds Completed',
            value: totalRoundsCompleted.toString(),
            icon: Activity,
            desc: 'Training cycles finished'
        },
        {
            label: 'Proofs Verified',
            value: totalProofsSubmitted.toString(),
            icon: Shield,
            desc: 'ZK proofs submitted'
        },
        {
            label: 'Slashing Events',
            value: totalSlashings.toString(),
            icon: ShieldAlert,
            desc: 'Malicious actors caught',
            alert: totalSlashings > 0
        },
    ];

    const allEvents = [
        ...proofEvents.map(e => ({ ...e, eventType: 'ProofSubmitted' as const, label: 'Proof Submitted' })),
        ...roundStartedEvents.map(e => ({ ...e, eventType: 'RoundStarted' as const, label: 'Round Started' })),
        ...roundCompletedEvents.map(e => ({ ...e, eventType: 'RoundCompleted' as const, label: 'Round Completed' })),
        ...stakedEvents.map(e => ({ ...e, eventType: 'Staked' as const, label: 'Stake Deposited' })),
        ...slashedEvents.map(e => ({ ...e, eventType: 'Slashed' as const, label: 'Slashing Event' })),
    ].sort((a, b) => b.timestamp - a.timestamp).slice(0, 20);

    return (
        <div className="min-h-screen bg-[#09090b] text-zinc-100 font-sans selection:bg-zinc-800">
            <div className="max-w-[1600px] mx-auto p-6 lg:p-10 space-y-10">

                {/* Header */}
                <header className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                    <div className="space-y-1">
                        <h1 className="text-3xl font-medium tracking-tight text-white">Dashboard</h1>
                        <p className="text-zinc-400 text-sm">Monitor training, proofs, and network status.</p>
                    </div>

                    <div className="flex items-center gap-4">
                        <div className="flex items-center gap-3 px-1">
                            <div className="px-3 py-1.5 rounded-full bg-zinc-900 border border-zinc-800 text-xs text-zinc-400 flex items-center gap-2">
                                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.4)]"></span>
                                Chain ID: {chainId}
                            </div>

                            {nextModelId && Number(nextModelId) > 0 && (
                                <div className="relative group">
                                    <select
                                        value={selectedModelId.toString()}
                                        onChange={(e) => setSelectedModelId(BigInt(e.target.value))}
                                        className="appearance-none bg-zinc-900 border border-zinc-800 text-zinc-300 text-xs px-4 py-1.5 rounded-full pr-8 cursor-pointer hover:border-zinc-700 transition-colors focus:outline-none focus:ring-1 focus:ring-zinc-700"
                                    >
                                        {Array.from({ length: Number(nextModelId) }, (_, i) => (
                                            <option key={i} value={i.toString()}>Model #{i}</option>
                                        ))}
                                    </select>
                                    <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none">
                                        <svg className="w-3 h-3 text-zinc-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                        </svg>
                                    </div>
                                </div>
                            )}
                        </div>

                        <div className="bg-zinc-900 rounded-xl border border-zinc-800/50 shadow-sm">
                            <ConnectButton showBalance={false} accountStatus="address" chainStatus="icon" />
                        </div>
                    </div>
                </header>

                {/* Stats Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
                    {stats.map((stat) => (
                        <div key={stat.label} className="group relative p-5 rounded-xl border border-zinc-800/60 bg-zinc-900/20 hover:bg-zinc-900/40 transition-all duration-300">
                            <div className="absolute inset-x-0 -top-px h-px bg-gradient-to-r from-transparent via-zinc-700/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

                            <div className="flex items-start justify-between mb-4">
                                <div className={`p-2 rounded-lg ${stat.alert ? 'bg-red-500/10 text-red-400' : 'bg-zinc-800/50 text-zinc-400 group-hover:text-zinc-200'} transition-colors`}>
                                    <stat.icon size={18} strokeWidth={1.5} />
                                </div>
                            </div>

                            <div>
                                <div className="text-2xl font-semibold text-zinc-100 tracking-tight mb-1">{stat.value}</div>
                                <div className="text-xs font-medium text-zinc-500 uppercase tracking-wider">{stat.label}</div>
                            </div>
                        </div>
                    ))}
                </div>

                {/* Main Content Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 auto-rows-min">

                    {/* Charts Column (2/3 width) */}
                    <div className="lg:col-span-2 space-y-6">

                        {/* Training Loss */}
                        <div className="rounded-xl border border-zinc-800/60 bg-zinc-900/20 overflow-hidden">
                            <div className="px-6 py-4 border-b border-zinc-800/60 flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <Activity size={16} className="text-zinc-400" />
                                    <h3 className="text-sm font-medium text-zinc-200">Live Training Loss</h3>
                                </div>
                                <span className="text-xs text-zinc-500">Real-time updates</span>
                            </div>
                            <div className="p-6">
                                <LiveLossCurve modelId={selectedModelId} />
                            </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {/* Proof Stream */}
                            <div className="rounded-xl border border-zinc-800/60 bg-zinc-900/20 overflow-hidden">
                                <div className="px-6 py-4 border-b border-zinc-800/60 flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                        <Shield size={16} className="text-zinc-400" />
                                        <h3 className="text-sm font-medium text-zinc-200">Proof Stream</h3>
                                    </div>
                                </div>
                                <div className="p-6 h-[300px]">
                                    <ProofStream modelId={selectedModelId} />
                                </div>
                            </div>

                            {/* Network Topology */}
                            <div className="rounded-xl border border-zinc-800/60 bg-zinc-900/20 overflow-hidden">
                                <div className="px-6 py-4 border-b border-zinc-800/60 flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                        <Network size={16} className="text-zinc-400" />
                                        <h3 className="text-sm font-medium text-zinc-200">Network Topology</h3>
                                    </div>
                                </div>
                                <div className="p-6 h-[300px]">
                                    <LiveTopology modelId={selectedModelId} />
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Activity Feed Column (1/3 width) */}
                    <div className="lg:col-span-1">
                        <div className="rounded-xl border border-zinc-800/60 bg-zinc-900/20 h-full flex flex-col">
                            <div className="px-6 py-4 border-b border-zinc-800/60 flex items-center justify-between">
                                <h3 className="text-sm font-medium text-zinc-200">Recent Activity</h3>
                                <span className="text-xs text-zinc-500">Global Events</span>
                            </div>

                            <div className="flex-1 p-0 overflow-hidden flex flex-col">
                                {allEvents.length > 0 ? (
                                    <div className="overflow-y-auto custom-scrollbar max-h-[800px] divide-y divide-zinc-800/40">
                                        {allEvents.map((event, i) => (
                                            <div key={`${event.eventType}-${i}`} className="p-4 hover:bg-zinc-800/30 transition-colors group">
                                                <div className="flex items-start gap-3">
                                                    <div className={`mt-0.5 w-1.5 h-1.5 rounded-full shrink-0 ${event.eventType === 'Slashed' ? 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.4)]' :
                                                            event.eventType === 'ProofSubmitted' ? 'bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.4)]' :
                                                                event.eventType === 'RoundCompleted' ? 'bg-purple-500 shadow-[0_0_8px_rgba(168,85,247,0.4)]' :
                                                                    'bg-zinc-500'
                                                        }`} />

                                                    <div className="flex-1 min-w-0 space-y-1">
                                                        <div className="flex items-center justify-between">
                                                            <span className="text-xs font-medium text-zinc-200">{event.label}</span>
                                                            <span className="text-[10px] text-zinc-600 font-mono">
                                                                Block {event.blockNumber.toLocaleString()}
                                                            </span>
                                                        </div>

                                                        <div className="text-xs text-zinc-500 flex flex-col gap-0.5">
                                                            <span>Model #{event.modelId.toString()}</span>
                                                            {'amount' in event && (
                                                                <span className="text-zinc-400 font-mono">
                                                                    {(Number(event.amount) / 1e18).toFixed(4)} ETH
                                                                </span>
                                                            )}
                                                            {'prover' in event && (
                                                                <span className="font-mono text-[10px] opacity-60 truncate">
                                                                    {event.prover}
                                                                </span>
                                                            )}
                                                        </div>

                                                        <div className="pt-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                                            <a
                                                                href={`https://etherscan.io/tx/${event.transactionHash}`}
                                                                target="_blank"
                                                                rel="noopener noreferrer"
                                                                className="inline-flex items-center gap-1 text-[10px] text-zinc-500 hover:text-zinc-300 transition-colors"
                                                            >
                                                                View Transaction
                                                                <svg className="w-2.5 h-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                                                                </svg>
                                                            </a>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <div className="flex flex-col items-center justify-center py-20 text-zinc-600 gap-2">
                                        <Activity size={24} className="opacity-20" />
                                        <p className="text-sm">No recent activity</p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                </div>
            </div>
        </div>
    );
}
