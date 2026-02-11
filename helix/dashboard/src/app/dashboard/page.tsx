'use client';

import { useState } from 'react';
import { useAccount, useChainId } from 'wagmi';
import { motion } from 'framer-motion';
import {
    LayoutDashboard,
    Settings,
    Activity,
    Database,
    Globe,
    TerminalSquare,
    Cpu,
    Share2
} from 'lucide-react';
import { ConnectButton } from '@rainbow-me/rainbowkit';

import MetricTicker from '@/components/dashboard/MetricTicker';
import EventConsole from '@/components/dashboard/EventConsole';
import NetworkHealthMatrix from '@/components/dashboard/NetworkHealthMatrix';
import ModelPerformanceCharts from '@/components/dashboard/ModelPerformanceCharts';
import { useDashboardMetrics } from '@/hooks/dashboard/useDashboardMetrics';

export default function DashboardPage() {
    const chainId = useChainId();
    const [activeTab, setActiveTab] = useState('overview');
    const [selectedModelId, setSelectedModelId] = useState<bigint>(BigInt(0));

    const { metrics, nextModelId } = useDashboardMetrics(selectedModelId);

    const tabs = [
        { id: 'overview', label: 'OVERVIEW', icon: LayoutDashboard },
        { id: 'network', label: 'LIVE NETWORK', icon: Globe },
        { id: 'nodes', label: 'NODE STATUS', icon: Database },
        { id: 'logs', label: 'SYS LOGS', icon: TerminalSquare },
    ];

    return (
        <div className="h-screen w-screen bg-[#050505] text-zinc-100 flex flex-col font-sans overflow-hidden">

            {/* Top Ticker Consumer */}
            <MetricTicker />

            {/* Main Workspace */}
            <div className="flex-1 flex overflow-hidden">

                {/* Sidebar */}
                <div className="w-16 lg:w-64 border-r border-zinc-800/50 bg-zinc-950 flex flex-col justify-between z-20">
                    <div>
                        <div className="h-16 flex items-center px-6 border-b border-zinc-800/50">
                            <div className="w-8 h-8 rounded bg-gradient-to-tr from-cyan-600 to-violet-600 flex items-center justify-center font-bold text-white">
                                H
                            </div>
                            <span className="ml-3 font-semibold tracking-tight hidden lg:block">HELIX <span className="text-zinc-600 font-light">TERMINAL</span></span>
                        </div>

                        <nav className="p-2 space-y-1">
                            {tabs.map(tab => (
                                <button
                                    key={tab.id}
                                    onClick={() => setActiveTab(tab.id)}
                                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm transition-all duration-200 ${activeTab === tab.id
                                            ? 'bg-zinc-900 text-white border border-zinc-800 shadow-sm'
                                            : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900/50'
                                        }`}
                                >
                                    <tab.icon size={18} strokeWidth={2} />
                                    <span className="hidden lg:block font-medium">{tab.label}</span>
                                    {activeTab === tab.id && (
                                        <motion.div
                                            layoutId="activeTabIndicator"
                                            className="absolute left-0 w-1 h-8 bg-cyan-500 rounded-r-full"
                                        />
                                    )}
                                </button>
                            ))}
                        </nav>
                    </div>

                    <div className="p-4 border-t border-zinc-800/50">
                        <div className="hidden lg:flex items-center justify-between text-xs text-zinc-500 mb-4 px-1">
                            <span>v2.4.0-stable</span>
                            <div className="flex items-center gap-1.5">
                                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                                <span>ONLINE</span>
                            </div>
                        </div>
                        <div className="flex justify-center lg:justify-start">
                            <ConnectButton showBalance={false} chainStatus="none" accountStatus="avatar" />
                        </div>
                    </div>
                </div>

                {/* Dashboard Stage */}
                <div className="flex-1 flex flex-col bg-[#050505] relative">

                    {/* Stage Header */}
                    <header className="h-16 border-b border-zinc-800/50 flex items-center justify-between px-6 bg-zinc-950/50 backdrop-blur-sm z-10">
                        <div className="flex items-center gap-4">
                            <h2 className="text-lg font-medium text-white tracking-tight">{tabs.find(t => t.id === activeTab)?.label}</h2>
                            <div className="h-4 w-px bg-zinc-800" />

                            {/* Model Switcher */}
                            <div className="hidden md:flex items-center bg-zinc-900 border border-zinc-800/50 rounded-md px-3 py-1">
                                <Settings size={14} className="text-zinc-500 mr-2" />
                                <select
                                    value={selectedModelId.toString()}
                                    onChange={(e) => setSelectedModelId(BigInt(e.target.value))}
                                    className="bg-transparent border-none text-xs text-zinc-300 focus:outline-none"
                                >
                                    {Array.from({ length: Number(nextModelId || 0) }, (_, i) => (
                                        <option key={i} value={i.toString()}>Model #{i} (Production)</option>
                                    ))}
                                </select>
                            </div>
                        </div>

                        <div className="flex items-center gap-4">
                            <div className="flex items-center gap-2 px-3 py-1.5 bg-cyan-950/30 border border-cyan-500/20 rounded text-xs text-cyan-400">
                                <Share2 size={12} />
                                <span>Connected to RPC: {chainId}</span>
                            </div>
                        </div>
                    </header>

                    {/* Stage Content */}
                    <main className="flex-1 overflow-auto p-6 scroll-smooth">
                        <div className="max-w-[1800px] mx-auto space-y-6 h-full flex flex-col">

                            {/* Top Stats Row */}
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                                {[
                                    { label: 'Total Stake', value: `${(Number(metrics?.totalStake || 0) / 1e18).toFixed(0)} ETH`, sub: '+12% this week' },
                                    { label: 'Active Provers', value: metrics?.activeProvers.toString() || '0', sub: '98% uptime' },
                                    { label: 'Proof Latency', value: '~1.2s', sub: '-240ms improvement' },
                                    { label: 'Network Load', value: `${metrics?.networkLoad}%`, sub: 'Optimal' },
                                ].map((stat, i) => (
                                    <div key={i} className="bg-zinc-900/40 border border-zinc-800/60 p-5 rounded-lg">
                                        <p className="text-xs text-zinc-500 uppercase tracking-wider font-mono mb-1">{stat.label}</p>
                                        <div className="flex items-end justify-between">
                                            <h3 className="text-2xl font-light text-white">{stat.value}</h3>
                                            <span className="text-[10px] text-emerald-500 font-mono bg-emerald-500/10 px-1.5 py-0.5 rounded">{stat.sub}</span>
                                        </div>
                                    </div>
                                ))}
                            </div>

                            {/* Main Split View */}
                            <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-6 min-h-[400px]">

                                {/* Center Stage: Charts */}
                                <div className="lg:col-span-2 flex flex-col gap-6">
                                    <div className="flex-1 bg-zinc-900/20 border border-zinc-800/50 rounded-xl p-1 overflow-hidden">
                                        <ModelPerformanceCharts />
                                    </div>

                                    <div className="h-64 bg-zinc-900/20 border border-zinc-800/50 rounded-xl p-4 overflow-hidden relative">
                                        <NetworkHealthMatrix />
                                    </div>
                                </div>

                                {/* Right Panel: Event Console & Topology */}
                                <div className="lg:col-span-1 flex flex-col gap-6">
                                    <div className="flex-1 min-h-[400px]">
                                        <EventConsole events={metrics?.recentActivity || []} />
                                    </div>

                                    <div className="h-64 bg-zinc-900/20 border border-zinc-800/50 rounded-xl p-6 flex flex-col justify-center items-center text-center">
                                        <Cpu size={48} className="text-zinc-700 mb-4" />
                                        <h3 className="text-zinc-400 font-medium">System Topology</h3>
                                        <p className="text-xs text-zinc-600 mt-2 max-w-[200px]">
                                            3D Visualization disabled in lite mode. Switch back to ThreeJS renderer to view.
                                        </p>
                                    </div>
                                </div>

                            </div>
                        </div>
                    </main>
                </div>
            </div>
        </div>
    );
}
