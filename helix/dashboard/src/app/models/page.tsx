'use client';

import { motion } from 'framer-motion';
import { Clock } from 'lucide-react';
import Link from 'next/link';

const mockModels = [
    { id: '0x1a...4f2', name: 'GPT-2-Small', type: 'NLP', rounds: 42, status: 'Training', accuracy: '89.4%' },
    { id: '0x3b...9a1', name: 'ResNet-50', type: 'Vision', rounds: 156, status: 'Idle', accuracy: '94.1%' },
    { id: '0x7c...2e9', name: 'Helix-Audio-v1', type: 'Audio', rounds: 12, status: 'Training', accuracy: '76.8%' },
];

export default function ModelsPage() {
    return (
        <div className="space-y-4">
            <div className="flex justify-between items-center">
                <h2 className="text-[13px] font-medium text-white">Model Registry</h2>
                <Link
                    href="/submit"
                    className="border border-white/10 text-[13px] px-3 py-1.5 rounded-[4px] text-[#888] hover:text-white hover:border-white/20 transition-colors"
                >
                    Submit New Model
                </Link>
            </div>

            <div className="bg-helix-surface border border-helix-border rounded-md">
                {mockModels.map((model, i) => (
                    <motion.div
                        key={model.id}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.06, duration: 0.3 }}
                        className={`p-4 flex items-center justify-between group hover:bg-white/[0.02] transition-colors ${
                            i < mockModels.length - 1 ? 'border-b border-helix-border' : ''
                        }`}
                    >
                        <div className="flex items-center gap-3">
                            <div className="w-8 h-8 bg-white/[0.04] rounded-md flex items-center justify-center font-mono text-[11px] text-[#555] group-hover:text-[#888] transition-colors">
                                {model.name[0]}
                            </div>
                            <div>
                                <h3 className="text-[13px] font-medium text-white">{model.name}</h3>
                                <div className="flex items-center gap-2 mt-0.5">
                                    <span className="font-mono text-[11px] text-[#555]">{model.id}</span>
                                    <span className="w-[3px] h-[3px] bg-[#333] rounded-full"></span>
                                    <span className="text-[11px] text-[#666] uppercase tracking-wider">{model.type}</span>
                                </div>
                            </div>
                        </div>

                        <div className="flex items-center gap-6">
                            <div className="text-right">
                                <div className="text-[11px] text-[#666] uppercase tracking-wider">Rounds</div>
                                <div className="font-mono text-[13px] text-[#888] mt-0.5">{model.rounds}</div>
                            </div>
                            <div className="text-right">
                                <div className="text-[11px] text-[#666] uppercase tracking-wider">Accuracy</div>
                                <div className="font-mono text-[13px] text-[#888] mt-0.5">{model.accuracy}</div>
                            </div>
                            <div className="w-20 flex justify-end">
                                {model.status === 'Training' ? (
                                    <span className="px-2 py-0.5 text-[10px] rounded-sm font-mono border border-white/10 text-[#888] flex items-center gap-1.5">
                                        <span className="relative flex h-[5px] w-[5px]">
                                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white/60"></span>
                                            <span className="relative inline-flex rounded-full h-[5px] w-[5px] bg-white/80"></span>
                                        </span>
                                        TRAINING
                                    </span>
                                ) : (
                                    <span className="px-2 py-0.5 text-[10px] rounded-sm font-mono text-[#555]">
                                        IDLE
                                    </span>
                                )}
                            </div>
                        </div>
                    </motion.div>
                ))}
            </div>
        </div>
    );
}
