'use client';

import { motion } from 'framer-motion';
import { Users, Cpu, Activity, Zap } from 'lucide-react';
import classNames from 'clsx';
import { useAccount } from 'wagmi';

const stats = [
    { name: 'Active Models', value: '12', change: '+2.5%', icon: Boxes, color: 'text-blue-500', bg: 'bg-blue-500/10' },
    { name: 'Total Rounds', value: '843', change: '+18.2%', icon: Activity, color: 'text-emerald-500', bg: 'bg-emerald-500/10' },
    { name: 'Compute Nodes', value: '24', change: '-1.4%', icon: Cpu, color: 'text-purple-500', bg: 'bg-purple-500/10' },
    { name: 'proofs Verified', value: '1,294', change: '+4.3%', icon: Zap, color: 'text-amber-500', bg: 'bg-amber-500/10' },
];

import { Boxes } from 'lucide-react';

export default function Page() {
    const { isConnected } = useAccount();

    return (
        <div className="space-y-8">
            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {stats.map((stat, i) => (
                    <motion.div
                        key={stat.name}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.1 }}
                        className="bg-neutral-900/50 backdrop-blur-md border border-neutral-800 p-6 rounded-2xl hover:border-neutral-700 transition-colors"
                    >
                        <div className="flex items-center justify-between mb-4">
                            <div className={classNames("p-3 rounded-xl", stat.bg, stat.color)}>
                                <stat.icon className="w-6 h-6" />
                            </div>
                            <span className={classNames("text-sm font-medium", stat.change.startsWith('+') ? "text-emerald-400" : "text-rose-400")}>
                                {stat.change}
                            </span>
                        </div>
                        <div className="text-3xl font-bold mb-1">{stat.value}</div>
                        <div className="text-neutral-500 text-sm">{stat.name}</div>
                    </motion.div>
                ))}
            </div>

            {/* Main Panel */}
            <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.4 }}
                className="bg-neutral-900/50 backdrop-blur-md border border-neutral-800 rounded-3xl p-8 min-h-[400px]"
            >
                <h2 className="text-xl font-bold mb-6">Network Activity</h2>

                {isConnected ? (
                    <div className="flex items-center justify-center h-64 text-neutral-500">
                        <p>Graph Placeholder (Use Recharts or Chart.js here)</p>
                    </div>
                ) : (
                    <div className="flex flex-col items-center justify-center h-64 text-neutral-500 space-y-4">
                        <div className="p-4 bg-neutral-800 rounded-full">
                            <Activity className="w-8 h-8 text-neutral-400" />
                        </div>
                        <p>Connect wallet to view network stats</p>
                    </div>
                )}
            </motion.div>
        </div>
    );
}
