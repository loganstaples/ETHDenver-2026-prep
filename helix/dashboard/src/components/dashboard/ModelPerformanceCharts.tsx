'use client';

import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid, BarChart, Bar, Legend } from 'recharts';

export default function ModelPerformanceCharts() {
    // Mock data generation
    const data = Array.from({ length: 24 }, (_, i) => ({
        time: `${i}:00`,
        proofs: Math.floor(Math.random() * 50) + 10,
        latency: Math.floor(Math.random() * 200) + 800,
        failures: Math.floor(Math.random() * 3),
    }));

    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full">
            {/* Chart 1: Proof Volume & Latency */}
            <div className="bg-zinc-900/20 border border-zinc-800/50 rounded-lg p-4 flex flex-col">
                <h3 className="text-xs font-mono text-zinc-400 mb-4 flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-cyan-500"></span>
                    24H PROOF VOLUME & LATENCY
                </h3>
                <div className="flex-1 min-h-[200px]">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={data}>
                            <defs>
                                <linearGradient id="colorProofs" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                                </linearGradient>
                                <linearGradient id="colorLatency" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                            <XAxis dataKey="time" stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} />
                            <YAxis yAxisId="left" stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} />
                            <YAxis yAxisId="right" orientation="right" stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#09090b', borderColor: '#27272a', fontSize: '12px' }}
                                itemStyle={{ color: '#aeaeae' }}
                            />
                            <Area yAxisId="left" type="monotone" dataKey="proofs" stroke="#06b6d4" strokeWidth={2} fillOpacity={1} fill="url(#colorProofs)" />
                            <Area yAxisId="right" type="monotone" dataKey="latency" stroke="#8b5cf6" strokeWidth={2} fillOpacity={1} fill="url(#colorLatency)" />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Chart 2: Success vs Failure */}
            <div className="bg-zinc-900/20 border border-zinc-800/50 rounded-lg p-4 flex flex-col">
                <h3 className="text-xs font-mono text-zinc-400 mb-4 flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
                    VERIFICATION SUCCESS RATE
                </h3>
                <div className="flex-1 min-h-[200px]">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={data}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                            <XAxis dataKey="time" stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} />
                            <YAxis stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} />
                            <Tooltip
                                cursor={{ fill: '#27272a', opacity: 0.4 }}
                                contentStyle={{ backgroundColor: '#09090b', borderColor: '#27272a', fontSize: '12px' }}
                            />
                            <Bar dataKey="proofs" name="Success" stackId="a" fill="#10b981" barSize={8} />
                            <Bar dataKey="failures" name="Failure" stackId="a" fill="#ef4444" barSize={8} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
}
