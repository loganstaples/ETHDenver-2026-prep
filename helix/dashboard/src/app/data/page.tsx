'use client';

import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Database, Upload, FileSpreadsheet, Trash2, Eye } from 'lucide-react';
import { useAccount } from 'wagmi';

interface Dataset {
    id: string;
    name: string;
    size: number;
    format: string;
    rows: number;
    modelId: string | null;
    uploadedAt: number;
}

const mockDatasets: Dataset[] = [
    { id: '1', name: 'imagenet-subset-10k.tar.gz', size: 2_400_000_000, format: 'Images', rows: 10000, modelId: '0x1a...4f2', uploadedAt: Date.now() - 86400000 },
    { id: '2', name: 'wiki-corpus-en.jsonl', size: 890_000_000, format: 'JSONL', rows: 500000, modelId: '0x7c...2e9', uploadedAt: Date.now() - 172800000 },
    { id: '3', name: 'audio-samples-v2.csv', size: 45_000_000, format: 'CSV', rows: 8500, modelId: null, uploadedAt: Date.now() - 259200000 },
];

export default function DatasetsPage() {
    const { isConnected } = useAccount();
    const [datasets, setDatasets] = useState<Dataset[]>(mockDatasets);
    const [dragOver, setDragOver] = useState(false);
    const [selectedDataset, setSelectedDataset] = useState<string | null>(null);

    const formatSize = (bytes: number) => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
        return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
    };

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer.files[0];
        if (file) {
            const ext = file.name.split('.').pop()?.toLowerCase() || '';
            const format = ['csv', 'tsv'].includes(ext) ? 'CSV' :
                          ['json', 'jsonl'].includes(ext) ? 'JSONL' :
                          ['tar', 'gz', 'zip'].includes(ext) ? 'Archive' : 'Other';
            setDatasets(prev => [{
                id: Date.now().toString(),
                name: file.name,
                size: file.size,
                format,
                rows: 0,
                modelId: null,
                uploadedAt: Date.now(),
            }, ...prev]);
        }
    }, []);

    const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            const ext = file.name.split('.').pop()?.toLowerCase() || '';
            const format = ['csv', 'tsv'].includes(ext) ? 'CSV' :
                          ['json', 'jsonl'].includes(ext) ? 'JSONL' :
                          ['tar', 'gz', 'zip'].includes(ext) ? 'Archive' : 'Other';
            setDatasets(prev => [{
                id: Date.now().toString(),
                name: file.name,
                size: file.size,
                format,
                rows: 0,
                modelId: null,
                uploadedAt: Date.now(),
            }, ...prev]);
        }
    }, []);

    const removeDataset = (id: string) => {
        setDatasets(prev => prev.filter(d => d.id !== id));
    };

    if (!isConnected) {
        return (
            <div className="flex flex-col items-center justify-center py-24 text-[#555]">
                <Database className="w-10 h-10 mb-3 text-[#444]" />
                <p className="text-[13px] mb-1">Connect your wallet</p>
                <p className="text-[11px] text-[#666] uppercase tracking-wider">Wallet required to manage datasets</p>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Upload Zone */}
            <div
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                className={`border border-dashed rounded-md p-8 text-center transition-colors ${
                    dragOver ? 'border-[#555] bg-white/[0.03]' : 'border-[#333] hover:border-[#555]'
                }`}
            >
                <input
                    type="file"
                    id="dataset-file"
                    className="hidden"
                    accept=".csv,.tsv,.json,.jsonl,.tar,.gz,.zip,.parquet"
                    onChange={handleFileSelect}
                />
                <label htmlFor="dataset-file" className="cursor-pointer">
                    <Upload className="w-8 h-8 mx-auto mb-3 text-[#444]" />
                    <p className="text-[13px] text-[#888] mb-1">Drag and drop training data here</p>
                    <p className="text-[11px] text-[#666] uppercase tracking-wider">.csv, .json, .jsonl, .parquet, .tar.gz</p>
                </label>
            </div>

            {/* Dataset List */}
            <div className="bg-helix-surface border border-helix-border rounded-md">
                {datasets.length === 0 ? (
                    <div className="text-center py-12 text-[#555]">
                        <Database className="w-8 h-8 mx-auto mb-3 text-[#444]" />
                        <p className="text-[13px]">No datasets uploaded yet</p>
                    </div>
                ) : (
                    datasets.map((dataset, i) => (
                        <motion.div
                            key={dataset.id}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: i * 0.05 }}
                            className={`p-4 transition-colors cursor-pointer ${
                                i < datasets.length - 1 ? 'border-b border-helix-border' : ''
                            } ${
                                selectedDataset === dataset.id
                                    ? 'bg-white/[0.02]'
                                    : 'hover:bg-white/[0.02]'
                            }`}
                            onClick={() => setSelectedDataset(selectedDataset === dataset.id ? null : dataset.id)}
                        >
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                    <FileSpreadsheet className="w-4 h-4 text-[#555] flex-shrink-0" />
                                    <div>
                                        <h3 className="font-mono text-[13px] text-white">{dataset.name}</h3>
                                        <div className="flex items-center gap-2 mt-0.5">
                                            <span className="font-mono text-[13px] text-[#666]">{formatSize(dataset.size)}</span>
                                            <span className="text-[#333]">/</span>
                                            <span className="text-[10px] font-mono text-[#666] bg-white/[0.04] px-1.5 py-0.5 rounded-sm">{dataset.format}</span>
                                            {dataset.rows > 0 && (
                                                <>
                                                    <span className="text-[#333]">/</span>
                                                    <span className="font-mono text-[13px] text-[#666]">{dataset.rows.toLocaleString()} rows</span>
                                                </>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                <div className="flex items-center gap-3">
                                    {dataset.modelId ? (
                                        <span className="font-mono text-[11px] text-[#555]">{dataset.modelId}</span>
                                    ) : (
                                        <span className="text-[11px] text-[#666] uppercase tracking-wider">Unassigned</span>
                                    )}
                                    <div className="flex items-center gap-0.5">
                                        <button className="p-1.5 rounded-[4px] hover:bg-white/[0.05] text-[#555] hover:text-white transition-colors">
                                            <Eye className="w-3.5 h-3.5" />
                                        </button>
                                        <button
                                            onClick={(e) => { e.stopPropagation(); removeDataset(dataset.id); }}
                                            className="p-1.5 rounded-[4px] hover:bg-white/[0.05] text-[#555] hover:text-white transition-colors"
                                        >
                                            <Trash2 className="w-3.5 h-3.5" />
                                        </button>
                                    </div>
                                </div>
                            </div>

                            {/* Expanded Preview */}
                            {selectedDataset === dataset.id && (
                                <motion.div
                                    initial={{ height: 0, opacity: 0 }}
                                    animate={{ height: 'auto', opacity: 1 }}
                                    className="mt-3 pt-3 border-t border-helix-border"
                                >
                                    <div className="grid grid-cols-3 gap-4">
                                        <div>
                                            <span className="text-[11px] text-[#666] uppercase tracking-wider block mb-1">Uploaded</span>
                                            <span className="font-mono text-[13px] text-white">{new Date(dataset.uploadedAt).toLocaleDateString()}</span>
                                        </div>
                                        <div>
                                            <span className="text-[11px] text-[#666] uppercase tracking-wider block mb-1">Format</span>
                                            <span className="text-[10px] font-mono text-[#666] bg-white/[0.04] px-1.5 py-0.5 rounded-sm">{dataset.format}</span>
                                        </div>
                                        <div>
                                            <span className="text-[11px] text-[#666] uppercase tracking-wider block mb-1">Associated Model</span>
                                            <span className="font-mono text-[13px] text-white">{dataset.modelId || 'None'}</span>
                                        </div>
                                    </div>

                                    <div className="mt-3 bg-black/30 rounded-[4px] p-3 font-mono text-[11px] text-[#666]">
                                        <p className="text-[#444] mb-1.5">// Preview (first 3 rows)</p>
                                        <p>{`{ "text": "Sample training data row 1...", "label": 0 }`}</p>
                                        <p>{`{ "text": "Sample training data row 2...", "label": 1 }`}</p>
                                        <p>{`{ "text": "Sample training data row 3...", "label": 0 }`}</p>
                                    </div>
                                </motion.div>
                            )}
                        </motion.div>
                    ))
                )}
            </div>
        </div>
    );
}
