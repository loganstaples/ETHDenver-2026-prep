'use client';

import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';

export default function NetworkHealthMatrix() {
    const [nodes, setNodes] = useState<any[]>([]);

    useEffect(() => {
        // Generate mock node status
        const mockNodes = Array.from({ length: 64 }, (_, i) => ({
            id: i,
            status: Math.random() > 0.9 ? 'error' : Math.random() > 0.7 ? 'busy' : 'idle',
            latency: Math.floor(Math.random() * 100),
        }));
        setNodes(mockNodes);

        const interval = setInterval(() => {
            setNodes(prev => prev.map(n => ({
                ...n,
                status: Math.random() > 0.95 ? 'error' : Math.random() > 0.8 ? 'busy' : 'idle',
                latency: Math.floor(Math.random() * 100),
            })));
        }, 2000);

        return () => clearInterval(interval);
    }, []);

    return (
        <div className="h-full flex flex-col">
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-xs font-mono text-zinc-400 uppercase tracking-wider">Node Status Matrix</h3>
                <div className="flex gap-2">
                    <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-sm bg-emerald-500/20 border border-emerald-500/50"></span><span className="text-[10px] text-zinc-500">IDLE</span></div>
                    <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-sm bg-amber-500/20 border border-amber-500/50"></span><span className="text-[10px] text-zinc-500">BUSY</span></div>
                    <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-sm bg-red-500/20 border border-red-500/50"></span><span className="text-[10px] text-zinc-500">ERR</span></div>
                </div>
            </div>

            <div className="grid grid-cols-8 md:grid-cols-16 gap-1 flex-1 content-start">
                {nodes.map((node) => (
                    <motion.div
                        key={node.id}
                        initial={false}
                        animate={{
                            backgroundColor:
                                node.status === 'error' ? 'rgba(239, 68, 68, 0.2)' :
                                    node.status === 'busy' ? 'rgba(245, 158, 11, 0.2)' :
                                        'rgba(16, 185, 129, 0.1)',
                            borderColor:
                                node.status === 'error' ? 'rgba(239, 68, 68, 0.5)' :
                                    node.status === 'busy' ? 'rgba(245, 158, 11, 0.5)' :
                                        'rgba(16, 185, 129, 0.3)',
                        }}
                        className="aspect-square rounded-[2px] border cursor-pointer hover:border-white/50 transition-colors relative group"
                    >
                        <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                            <span className="text-[8px] font-mono text-white/90">{node.latency}ms</span>
                        </div>
                    </motion.div>
                ))}
            </div>
        </div>
    );
}
