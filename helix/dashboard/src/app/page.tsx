'use client';

import { motion } from 'framer-motion';
import { Boxes, Activity, Cpu, Zap } from 'lucide-react';
import { useAccount } from 'wagmi';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

const stats = [
    { label: 'Active Models', value: '12', delta: '+2', icon: Boxes },
    { label: 'Total Rounds', value: '843', delta: '+18', icon: Activity },
    { label: 'Compute Nodes', value: '24', delta: '-1', icon: Cpu },
    { label: 'Proofs Verified', value: '1,294', delta: '+43', icon: Zap },
];

const chartData = Array.from({ length: 24 }, (_, i) => ({
    t: `${String(i).padStart(2, '0')}:00`,
    proofs: Math.floor(Math.random() * 40) + 10,
    rounds: Math.floor(Math.random() * 8) + 2,
}));

export default function Page() {
    const { isConnected } = useAccount();

    return (
        <div className="space-y-4">
            {/* Stats */}
            <div className="grid grid-cols-4 gap-3">
                {stats.map((s, i) => (
                    <motion.div
                        key={s.label}
                        initial={{ opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.05, duration: 0.3 }}
                        className="bg-helix-surface border border-helix-border rounded-md p-4 hover:border-white/[0.08] transition-colors"
                    >
                        <div className="flex items-center justify-between mb-3">
                            <span className="text-[11px] text-[#666] uppercase tracking-wider">{s.label}</span>
                            <s.icon className="w-3.5 h-3.5 text-[#444]" strokeWidth={1.5} />
                        </div>
                        <div className="flex items-baseline gap-2">
                            <span className="text-2xl font-semibold tracking-tight font-mono">{s.value}</span>
                            <span className="text-[11px] text-[#555] font-mono">{s.delta}</span>
                        </div>
                    </motion.div>
                ))}
            </div>

            {/* Chart */}
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}
                className="bg-helix-surface border border-helix-border rounded-md p-5"
            >
                <div className="flex items-center justify-between mb-5">
                    <span className="text-[13px] font-medium text-white">Network Activity</span>
                    <span className="text-[11px] text-[#555] font-mono">24h</span>
                </div>

                {isConnected ? (
                    <ResponsiveContainer width="100%" height={260}>
                        <AreaChart data={chartData}>
                            <defs>
                                <linearGradient id="gProofs" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="0%" stopColor="#fff" stopOpacity={0.06} />
                                    <stop offset="100%" stopColor="#fff" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <XAxis
                                dataKey="t"
                                stroke="#333"
                                fontSize={10}
                                tickLine={false}
                                axisLine={false}
                                tick={{ fill: '#555', fontFamily: 'var(--font-geist-mono)' }}
                                interval={5}
                            />
                            <YAxis
                                stroke="#333"
                                fontSize={10}
                                tickLine={false}
                                axisLine={false}
                                tick={{ fill: '#555', fontFamily: 'var(--font-geist-mono)' }}
                                width={30}
                            />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: '#111113',
                                    border: '1px solid #1e1e22',
                                    borderRadius: '4px',
                                    color: '#fff',
                                    fontSize: '11px',
                                    fontFamily: 'var(--font-geist-mono)',
                                    padding: '6px 10px',
                                }}
                                itemStyle={{ color: '#999', fontSize: '11px' }}
                                labelStyle={{ color: '#fff', fontSize: '11px', marginBottom: '4px' }}
                            />
                            <Area
                                type="monotone"
                                dataKey="proofs"
                                stroke="#fff"
                                fill="url(#gProofs)"
                                strokeWidth={1}
                                name="Proofs"
                            />
                            <Area
                                type="monotone"
                                dataKey="rounds"
                                stroke="#555"
                                fill="none"
                                strokeWidth={1}
                                strokeDasharray="3 3"
                                name="Rounds"
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                ) : (
                    <div className="flex flex-col items-center justify-center h-[260px] text-[#444]">
                        <Activity className="w-6 h-6 mb-3" strokeWidth={1} />
                        <p className="text-[13px]">Connect wallet to view activity</p>
                    </div>
                )}
            </motion.div>
        </div>
    );
}
