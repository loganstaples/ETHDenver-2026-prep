'use client';

import { motion } from 'framer-motion';

export default function MetricTicker() {
    const metrics = [
        { label: 'HELIX/USD', value: '$1.00', change: '+0.0%' },
        { label: 'ETH/USD', value: '$2,450.20', change: '+1.2%' },
        { label: 'GAS', value: '12 Gwei', change: '-4.5%' },
        { label: 'AVG BLOCK TIME', value: '12.1s', change: '0.0%' },
        { label: 'NETWORK HASHRATE', value: '450 TH/s', change: '+2.1%' },
        { label: 'ACTIVE NODES', value: '1,240', change: '+5.3%' },
        { label: 'TOTAL PROOFS', value: '1.2M', change: '+12%' },
    ];

    return (
        <div className="h-8 bg-zinc-950 border-b border-zinc-800/50 flex items-center overflow-hidden whitespace-nowrap relative">
            <div className="absolute inset-y-0 left-0 w-8 bg-gradient-to-r from-zinc-950 to-transparent z-10" />
            <div className="absolute inset-y-0 right-0 w-8 bg-gradient-to-l from-zinc-950 to-transparent z-10" />

            <motion.div
                className="flex items-center gap-8 px-4"
                animate={{ x: [0, -1000] }}
                transition={{ repeat: Infinity, duration: 45, ease: "linear" }}
            >
                {[...metrics, ...metrics, ...metrics].map((m, i) => (
                    <div key={i} className="flex items-center gap-2 text-xs font-mono">
                        <span className="text-zinc-500">{m.label}</span>
                        <span className="text-zinc-300">{m.value}</span>
                        <span className={m.change.startsWith('+') ? 'text-emerald-500' : 'text-red-500'}>
                            {m.change}
                        </span>
                    </div>
                ))}
            </motion.div>
        </div>
    );
}
