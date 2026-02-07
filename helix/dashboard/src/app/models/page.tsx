'use client';

import { motion } from 'framer-motion';
import { Clock } from 'lucide-react';

const mockModels = [
    { id: '0x1a...4f2', name: 'GPT-2-Small', type: 'NLP', rounds: 42, status: 'Training', accuracy: '89.4%' },
    { id: '0x3b...9a1', name: 'ResNet-50', type: 'Vision', rounds: 156, status: 'Idle', accuracy: '94.1%' },
    { id: '0x7c...2e9', name: 'Helix-Audio-v1', type: 'Audio', rounds: 12, status: 'Training', accuracy: '76.8%' },
];

export default function ModelsPage() {
    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <h2 className="text-xl font-bold">Model Registry</h2>
                <button className="bg-emerald-500 hover:bg-emerald-600 text-black px-4 py-2 rounded-lg font-medium transition-colors">
                    Register New Model
                </button>
            </div>

            <div className="grid gap-4">
                {mockModels.map((model, i) => (
                    <motion.div
                        key={model.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.1 }}
                        className="bg-neutral-900/50 backdrop-blur-md border border-neutral-800 p-6 rounded-2xl flex items-center justify-between group hover:border-neutral-700 transition-all"
                    >
                        <div className="flex items-center gap-6">
                            <div className="w-12 h-12 bg-neutral-800 rounded-xl flex items-center justify-center font-bold text-neutral-400 group-hover:bg-neutral-700 group-hover:text-white transition-colors">
                                {model.name[0]}
                            </div>
                            <div>
                                <h3 className="font-bold text-lg">{model.name}</h3>
                                <div className="flex items-center gap-2 text-sm text-neutral-500">
                                    <span className="font-mono">{model.id}</span>
                                    <span className="w-1 h-1 bg-neutral-700 rounded-full"></span>
                                    <span>{model.type}</span>
                                </div>
                            </div>
                        </div>

                        <div className="flex items-center gap-8">
                            <div className="text-right">
                                <div className="text-xs text-neutral-500 uppercase font-semibold">Rounds</div>
                                <div className="font-mono">{model.rounds}</div>
                            </div>
                            <div className="text-right">
                                <div className="text-xs text-neutral-500 uppercase font-semibold">Accuracy</div>
                                <div className="font-mono text-emerald-400">{model.accuracy}</div>
                            </div>
                            <div className="w-24 flex justify-end">
                                {model.status === 'Training' ? (
                                    <span className="px-3 py-1 bg-amber-500/10 text-amber-500 rounded-full text-xs font-medium border border-amber-500/20 flex items-center gap-1">
                                        <Clock className="w-3 h-3 animate-spin-slow" />
                                        Training
                                    </span>
                                ) : (
                                    <span className="px-3 py-1 bg-neutral-800 text-neutral-400 rounded-full text-xs font-medium border border-neutral-700">
                                        Idle
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
