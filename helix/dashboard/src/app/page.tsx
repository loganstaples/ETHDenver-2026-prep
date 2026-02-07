'use client';

import { motion } from 'framer-motion';
import { Cpu, Activity, Zap, Boxes } from 'lucide-react';
import classNames from 'clsx';
import { useAccount } from 'wagmi';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

const stats = [
    { name: 'Active Models', value: '12', change: '+2.5%', icon: Boxes, color: 'text-blue-500', bg: 'bg-blue-500/10' },
    { name: 'Total Rounds', value: '843', change: '+18.2%', icon: Activity, color: 'text-emerald-500', bg: 'bg-emerald-500/10' },
    { name: 'Compute Nodes', value: '24', change: '-1.4%', icon: Cpu, color: 'text-purple-500', bg: 'bg-purple-500/10' },
    { name: 'Proofs Verified', value: '1,294', change: '+4.3%', icon: Zap, color: 'text-amber-500', bg: 'bg-amber-500/10' },
];

const networkActivityData = Array.from({ length: 24 }, (_, i) => ({
    time: `${String(i).padStart(2, '0')}:00`,
    proofs: Math.floor(Math.random() * 40) + 10,
    rounds: Math.floor(Math.random() * 8) + 2,
}));

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
                    <ResponsiveContainer width="100%" height={300}>
                        <AreaChart data={networkActivityData}>
                            <defs>
                                <linearGradient id="proofGradient" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                                </linearGradient>
                                <linearGradient id="roundGradient" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <XAxis
                                dataKey="time"
                                stroke="#525252"
                                fontSize={12}
                                tickLine={false}
                                axisLine={false}
                            />
                            <YAxis
                                stroke="#525252"
                                fontSize={12}
                                tickLine={false}
                                axisLine={false}
                            />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: '#171717',
                                    border: '1px solid #404040',
                                    borderRadius: '8px',
                                    color: '#fff',
                                }}
                            />
                            <Area
                                type="monotone"
                                dataKey="proofs"
                                stroke="#10b981"
                                fill="url(#proofGradient)"
                                strokeWidth={2}
                                name="Proof Submissions"
                            />
                            <Area
                                type="monotone"
                                dataKey="rounds"
                                stroke="#6366f1"
                                fill="url(#roundGradient)"
                                strokeWidth={2}
                                name="Rounds Completed"
                            />
                        </AreaChart>
                    </ResponsiveContainer>
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
