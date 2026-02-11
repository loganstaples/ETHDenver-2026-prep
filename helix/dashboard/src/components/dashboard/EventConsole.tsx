'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Filter, AlertTriangle, Shield, CheckCircle } from 'lucide-react';
import { DashboardMetrics } from '@/hooks/dashboard/useDashboardMetrics';

interface EventConsoleProps {
    events: DashboardMetrics['recentActivity'];
}

export default function EventConsole({ events }: EventConsoleProps) {
    const [filter, setFilter] = useState('');
    const [typeFilter, setTypeFilter] = useState<string | null>(null);

    const filteredEvents = events.filter(e => {
        const matchesSearch = e.message.toLowerCase().includes(filter.toLowerCase()) ||
            e.transactionHash?.toLowerCase().includes(filter.toLowerCase());
        const matchesType = typeFilter ? e.type === typeFilter : true;
        return matchesSearch && matchesType;
    });

    return (
        <div className="flex flex-col h-full bg-zinc-950 border border-zinc-800/50 rounded-lg overflow-hidden">
            {/* Console Toolbar */}
            <div className="h-10 border-b border-zinc-800/50 flex items-center px-3 gap-3 bg-zinc-900/30">
                <div className="flex items-center gap-2 text-zinc-500">
                    <div className="w-3 h-3 rounded-full bg-red-500/20 border border-red-500/50" />
                    <div className="w-3 h-3 rounded-full bg-amber-500/20 border border-amber-500/50" />
                    <div className="w-3 h-3 rounded-full bg-emerald-500/20 border border-emerald-500/50" />
                </div>
                <div className="h-4 w-px bg-zinc-800" />
                <div className="flex-1 flex items-center relative">
                    <Search className="w-3.5 h-3.5 text-zinc-600 absolute left-2" />
                    <input
                        type="text"
                        value={filter}
                        onChange={(e) => setFilter(e.target.value)}
                        placeholder="grep events..."
                        className="w-full bg-transparent border-none text-xs font-mono text-zinc-300 pl-8 focus:outline-none placeholder:text-zinc-700"
                    />
                </div>
                <div className="flex gap-1">
                    {['PROOF', 'SLASH', 'STAKE'].map(type => (
                        <button
                            key={type}
                            onClick={() => setTypeFilter(typeFilter === type ? null : type)}
                            className={`px-2 py-0.5 text-[10px] font-mono border rounded ${typeFilter === type
                                    ? 'bg-zinc-800 border-zinc-600 text-zinc-200'
                                    : 'border-transparent text-zinc-600 hover:bg-zinc-900'
                                }`}
                        >
                            {type}
                        </button>
                    ))}
                </div>
            </div>

            {/* Console Output */}
            <div className="flex-1 overflow-y-auto p-2 space-y-0.5 font-mono text-xs custom-scrollbar">
                <AnimatePresence initial={false}>
                    {filteredEvents.map((event, i) => (
                        <motion.div
                            key={`${event.transactionHash}-${i}`}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0 }}
                            className="flex items-start gap-2 hover:bg-zinc-900/50 p-1 rounded group"
                        >
                            <span className="text-zinc-600 min-w-[70px]">
                                {new Date(Number(event.timestamp) * 1000).toLocaleTimeString([], { hour12: false })}
                            </span>
                            <span className={`
                        ${event.severity === 'error' ? 'text-red-400' :
                                    event.severity === 'warning' ? 'text-amber-400' :
                                        event.severity === 'success' ? 'text-emerald-400' :
                                            'text-blue-400'}
                    `}>
                                [{event.type}]
                            </span>
                            <span className="text-zinc-300 flex-1 break-all">
                                {event.message}
                            </span>
                            <span className="text-zinc-700 opacity-0 group-hover:opacity-100 transition-opacity">
                                {event.transactionHash.slice(0, 8)}...
                            </span>
                        </motion.div>
                    ))}
                    {filteredEvents.length === 0 && (
                        <div className="text-zinc-700 p-2 italic">-- No events matching filter --</div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
}
