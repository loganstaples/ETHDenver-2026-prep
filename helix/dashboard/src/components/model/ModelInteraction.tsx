'use client';

import React, { useState } from 'react';

interface ModelVersion {
    version: string;
    accuracy: number;
    loss: number;
    timestamp: number;
    proofHash: string;
    participants: number;
}

interface LayerConfig {
    name: string;
    type: string;
    params: number;
    quantized: boolean;
    errorBound: number;
}

interface ModelInteractionProps {
    modelName?: string;
    versions?: ModelVersion[];
    layers?: LayerConfig[];
}

export default function ModelInteraction({
    modelName = 'helix-gpt-mini',
    versions,
    layers,
}: ModelInteractionProps) {
    const [activeTab, setActiveTab] = useState<'overview' | 'architecture' | 'inference'>('overview');
    const [prompt, setPrompt] = useState('');
    const [response, setResponse] = useState('');
    const [isGenerating, setIsGenerating] = useState(false);

    const mockVersions: ModelVersion[] = versions || [
        { version: 'v1.2.0', accuracy: 0.876, loss: 0.342, timestamp: Date.now() - 3600000, proofHash: '0xabc...123', participants: 12 },
        { version: 'v1.1.0', accuracy: 0.854, loss: 0.398, timestamp: Date.now() - 86400000, proofHash: '0xdef...456', participants: 10 },
        { version: 'v1.0.0', accuracy: 0.812, loss: 0.467, timestamp: Date.now() - 172800000, proofHash: '0xghi...789', participants: 8 },
    ];

    const mockLayers: LayerConfig[] = layers || [
        { name: 'embedding', type: 'Embedding', params: 12582912, quantized: false, errorBound: 1e-7 },
        { name: 'attention_0', type: 'MultiHeadAttention', params: 2359296, quantized: true, errorBound: 2.3e-6 },
        { name: 'ffn_0', type: 'FeedForward', params: 3145728, quantized: true, errorBound: 4.1e-6 },
        { name: 'attention_1', type: 'MultiHeadAttention', params: 2359296, quantized: true, errorBound: 8.7e-6 },
        { name: 'ffn_1', type: 'FeedForward', params: 3145728, quantized: true, errorBound: 1.2e-5 },
        { name: 'lm_head', type: 'Linear', params: 12582912, quantized: false, errorBound: 1.5e-5 },
    ];

    const totalParams = mockLayers.reduce((sum, l) => sum + l.params, 0);

    const formatParams = (n: number) => {
        if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
        if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
        if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
        return n.toString();
    };

    const handleInference = async () => {
        if (!prompt.trim()) return;
        setIsGenerating(true);
        setResponse('');

        // Simulate streaming response
        const mockResponse = "The distributed training process ensures verifiable computation through zero-knowledge proofs. Each gradient update is cryptographically verified before aggregation.";
        for (let i = 0; i <= mockResponse.length; i++) {
            await new Promise(r => setTimeout(r, 20));
            setResponse(mockResponse.slice(0, i));
        }
        setIsGenerating(false);
    };

    return (
        <div className="bg-helix-surface rounded-md p-4 text-white font-sans">
            <div className="flex justify-between items-center mb-4">
                <h2 className="font-semibold text-[15px] text-white">Model Interaction</h2>
                <span className="px-4 py-2 bg-white/10 border border-white/10 rounded-md font-mono text-sm text-white">
                    {modelName}
                </span>
            </div>

            <div className="flex gap-1 mb-4 bg-white/[0.03] p-1 rounded-md">
                <button
                    className={`flex-1 px-3 py-3 border-none text-sm font-medium cursor-pointer rounded-md transition-all ${
                        activeTab === 'overview'
                            ? 'bg-white/10 text-white'
                            : 'bg-transparent text-[#888] hover:text-white'
                    }`}
                    onClick={() => setActiveTab('overview')}
                >
                    Overview
                </button>
                <button
                    className={`flex-1 px-3 py-3 border-none text-sm font-medium cursor-pointer rounded-md transition-all ${
                        activeTab === 'architecture'
                            ? 'bg-white/10 text-white'
                            : 'bg-transparent text-[#888] hover:text-white'
                    }`}
                    onClick={() => setActiveTab('architecture')}
                >
                    Architecture
                </button>
                <button
                    className={`flex-1 px-3 py-3 border-none text-sm font-medium cursor-pointer rounded-md transition-all ${
                        activeTab === 'inference'
                            ? 'bg-white/10 text-white'
                            : 'bg-transparent text-[#888] hover:text-white'
                    }`}
                    onClick={() => setActiveTab('inference')}
                >
                    Inference
                </button>
            </div>

            {activeTab === 'overview' && (
                <div className="grid grid-cols-2 gap-3">
                    <div className="bg-white/[0.03] rounded-md p-4 border border-helix-border">
                        <h3 className="text-sm font-semibold mb-4 text-[#aaa]">Model Versions</h3>
                        <div className="flex flex-col gap-2">
                            {mockVersions.map((v) => (
                                <div key={v.version} className="flex justify-between items-center p-3 bg-white/[0.03] rounded-md">
                                    <span className="font-semibold text-white">{v.version}</span>
                                    <div className="flex gap-3 text-xs text-[#888]">
                                        <span>Acc: <span className="font-semibold text-[#aaa]">{(v.accuracy * 100).toFixed(1)}%</span></span>
                                        <span>Loss: <span className="font-semibold text-[#aaa]">{v.loss.toFixed(3)}</span></span>
                                        <span>{v.participants} nodes</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                    <div className="bg-white/[0.03] rounded-md p-4 border border-helix-border">
                        <h3 className="text-sm font-semibold mb-4 text-[#aaa]">Model Stats</h3>
                        <div className="flex flex-col gap-3">
                            <div className="flex justify-between items-center p-3 bg-white/[0.03] rounded-md">
                                <span className="text-white">Total Parameters</span>
                                <span className="font-semibold text-white">{formatParams(totalParams)}</span>
                            </div>
                            <div className="flex justify-between items-center p-3 bg-white/[0.03] rounded-md">
                                <span className="text-white">Quantized Layers</span>
                                <span className="font-semibold text-white">{mockLayers.filter(l => l.quantized).length}/{mockLayers.length}</span>
                            </div>
                            <div className="flex justify-between items-center p-3 bg-white/[0.03] rounded-md">
                                <span className="text-white">Max Error Bound</span>
                                <span className="font-semibold text-[#888]">{Math.max(...mockLayers.map(l => l.errorBound)).toExponential(1)}</span>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {activeTab === 'architecture' && (
                <div className="bg-white/[0.03] rounded-md p-4 border border-helix-border">
                    <h3 className="text-sm font-semibold mb-4 text-[#aaa]">Layer Configuration</h3>
                    <div className="flex flex-col gap-2">
                        {mockLayers.map((layer) => (
                            <div key={layer.name} className="flex items-center justify-between p-3 bg-white/[0.03] rounded-md">
                                <div>
                                    <div className="font-semibold text-[13px] text-white">{layer.name}</div>
                                    <div className="text-[11px] text-[#666]">{layer.type}</div>
                                </div>
                                <div className="flex gap-3 items-center">
                                    <span className="font-mono text-xs text-white">{formatParams(layer.params)}</span>
                                    {layer.quantized && (
                                        <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-white/10 text-white">
                                            INT8
                                        </span>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {activeTab === 'inference' && (
                <div className="flex flex-col gap-3">
                    <div className="flex gap-3">
                        <textarea
                            className="flex-1 p-4 bg-white/[0.03] border border-helix-border rounded-md text-white text-sm resize-none focus:outline-none focus:border-white/30 placeholder:text-[#666]"
                            placeholder="Enter your prompt..."
                            rows={3}
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                        />
                        <button
                            className="px-8 py-4 bg-white text-black border-none rounded-md font-semibold cursor-pointer transition-all hover:-translate-y-0.5 hover:bg-neutral-200 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                            onClick={handleInference}
                            disabled={isGenerating || !prompt.trim()}
                        >
                            {isGenerating ? 'Generating...' : 'Generate'}
                        </button>
                    </div>
                    <div className="p-4 bg-white/[0.03] rounded-md min-h-[120px] text-sm leading-relaxed text-[#aaa]">
                        {response || 'Response will appear here...'}
                        {isGenerating && (
                            <span className="inline-block w-2 h-4 bg-white ml-0.5 animate-[blink_1s_infinite]" />
                        )}
                    </div>
                    <div className="flex items-center gap-2 px-4 py-3 bg-white/[0.04] border border-helix-border rounded-md text-[13px] text-white">
                        ✓ All inference computations are verified with ZK proofs
                    </div>
                </div>
            )}
        </div>
    );
}
