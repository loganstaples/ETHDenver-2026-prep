'use client';

import React, { useState } from 'react';

interface ErrorBound {
    layer: string;
    operation: string;
    inputError: number;
    outputError: number;
    amplification: number;
    riskLevel: 'low' | 'medium' | 'high' | 'critical';
}

interface ErrorBoundsVisualizationProps {
    bounds?: ErrorBound[];
    modelName?: string;
}

export default function ErrorBoundsVisualization({
    bounds,
    modelName = 'Transformer Model',
}: ErrorBoundsVisualizationProps) {
    const [selectedLayer, setSelectedLayer] = useState<string | null>(null);

    const mockBounds: ErrorBound[] = bounds || [
        { layer: 'embed', operation: 'Embedding', inputError: 0, outputError: 1e-7, amplification: 1.0, riskLevel: 'low' },
        { layer: 'attn_0', operation: 'Attention', inputError: 1e-7, outputError: 2.3e-6, amplification: 23.0, riskLevel: 'medium' },
        { layer: 'norm_0', operation: 'LayerNorm', inputError: 2.3e-6, outputError: 4.1e-6, amplification: 1.78, riskLevel: 'low' },
        { layer: 'ffn_0', operation: 'FFN', inputError: 4.1e-6, outputError: 8.7e-6, amplification: 2.12, riskLevel: 'medium' },
        { layer: 'attn_1', operation: 'Attention', inputError: 8.7e-6, outputError: 1.8e-4, amplification: 20.7, riskLevel: 'high' },
        { layer: 'norm_1', operation: 'LayerNorm', inputError: 1.8e-4, outputError: 2.1e-4, amplification: 1.17, riskLevel: 'low' },
        { layer: 'ffn_1', operation: 'FFN', inputError: 2.1e-4, outputError: 4.4e-4, amplification: 2.1, riskLevel: 'medium' },
        { layer: 'softmax', operation: 'Softmax', inputError: 4.4e-4, outputError: 1.2e-3, amplification: 2.73, riskLevel: 'high' },
    ];

    const totalError = mockBounds[mockBounds.length - 1]?.outputError || 0;
    const _maxAmp = Math.max(...mockBounds.map(b => b.amplification));

    const getRiskColor = (risk: string) => {
        const colors: Record<string, string> = {
            low: '#ffffff',
            medium: '#a3a3a3',
            high: '#737373',
            critical: '#525252',
        };
        return colors[risk] || '#a3a3a3';
    };

    const formatError = (e: number) => {
        if (e === 0) return '0';
        if (e < 1e-9) return e.toExponential(1);
        if (e < 1e-6) return `${(e * 1e9).toFixed(1)}e-9`;
        if (e < 1e-3) return `${(e * 1e6).toFixed(1)}e-6`;
        return e.toExponential(2);
    };

    return (
        <div className="bg-helix-surface rounded-md p-4 text-white font-sans">
            <div className="flex justify-between items-center mb-4">
                <h2 className="font-semibold text-[15px] text-white">
                    Error Bounds: {modelName}
                </h2>
                <div className="px-4 py-2 bg-white/[0.04] border border-helix-border rounded-md font-mono text-sm">
                    Total Output Error: <span className="text-[#888] font-semibold">{formatError(totalError)}</span>
                </div>
            </div>

            <div className="flex gap-3 justify-center p-4 bg-white/[0.04] rounded-md mb-4">
                <div className="flex items-center gap-2 text-xs text-[#888]">
                    <div className="w-3 h-3 rounded-sm bg-white" />
                    <span>Low Risk</span>
                </div>
                <div className="flex items-center gap-2 text-xs text-[#888]">
                    <div className="w-3 h-3 rounded-sm bg-neutral-400" />
                    <span>Medium Risk</span>
                </div>
                <div className="flex items-center gap-2 text-xs text-[#888]">
                    <div className="w-3 h-3 rounded-sm bg-neutral-500" />
                    <span>High Risk</span>
                </div>
                <div className="flex items-center gap-2 text-xs text-[#888]">
                    <div className="w-3 h-3 rounded-sm bg-neutral-300" />
                    <span>Amplification</span>
                </div>
            </div>

            <div className="flex flex-col gap-2 mb-4">
                {mockBounds.map((bound, _idx) => {
                    const width = Math.min(
                        (Math.log10(bound.outputError + 1e-10) + 10) * 10,
                        100
                    );
                    return (
                        <div
                            key={bound.layer}
                            className={`flex items-center gap-3 px-4 py-3 bg-white/[0.04] rounded-md cursor-pointer transition-all duration-200 border border-transparent hover:bg-white/10 ${
                                selectedLayer === bound.layer ? 'bg-white/10 border-white/10' : ''
                            }`}
                            onClick={() => setSelectedLayer(bound.layer)}
                        >
                            <div className="w-25 font-semibold text-[13px]">{bound.layer}</div>
                            <div className="w-25 text-xs text-[#888]">{bound.operation}</div>
                            <div className="flex-1 h-6 bg-white/[0.03] rounded relative overflow-hidden">
                                <div
                                    className="h-full rounded flex items-center justify-end pr-2 text-[10px] font-semibold text-white/90 transition-all duration-300"
                                    style={{
                                        width: `${width}%`,
                                        background: getRiskColor(bound.riskLevel),
                                    }}
                                >
                                    {formatError(bound.outputError)}
                                </div>
                            </div>
                            <div className="min-w-[60px] px-2 py-1 rounded text-[11px] font-semibold text-center bg-white/10 text-[#aaa]">
                                {bound.amplification.toFixed(1)}x
                            </div>
                            <div
                                className="w-2 h-2 rounded-full"
                                style={{ background: getRiskColor(bound.riskLevel) }}
                            />
                        </div>
                    );
                })}
            </div>

            {selectedLayer && (
                <div className="p-4 bg-white/[0.04] rounded-md border border-helix-border">
                    <h3 className="text-base font-semibold mb-4">
                        Layer: {mockBounds.find((b) => b.layer === selectedLayer)?.layer}
                    </h3>
                    {(() => {
                        const bound = mockBounds.find((b) => b.layer === selectedLayer);
                        if (!bound) return null;
                        return (
                            <>
                                <div className="grid grid-cols-4 gap-3">
                                    <div className="bg-white/[0.03] p-3 rounded-md text-center">
                                        <div className="text-lg font-bold font-mono mb-1 text-white">
                                            {formatError(bound.inputError)}
                                        </div>
                                        <div className="text-[11px] text-[#666] uppercase">Input Error</div>
                                    </div>
                                    <div className="bg-white/[0.03] p-3 rounded-md text-center">
                                        <div
                                            className="text-lg font-bold font-mono mb-1"
                                            style={{ color: getRiskColor(bound.riskLevel) }}
                                        >
                                            {formatError(bound.outputError)}
                                        </div>
                                        <div className="text-[11px] text-[#666] uppercase">Output Error</div>
                                    </div>
                                    <div className="bg-white/[0.03] p-3 rounded-md text-center">
                                        <div className="text-lg font-bold font-mono mb-1 text-[#aaa]">
                                            {bound.amplification.toFixed(2)}x
                                        </div>
                                        <div className="text-[11px] text-[#666] uppercase">Amplification</div>
                                    </div>
                                    <div className="bg-white/[0.03] p-3 rounded-md text-center">
                                        <div
                                            className="text-lg font-bold font-mono mb-1"
                                            style={{ color: getRiskColor(bound.riskLevel) }}
                                        >
                                            {bound.riskLevel.toUpperCase()}
                                        </div>
                                        <div className="text-[11px] text-[#666] uppercase">Risk Level</div>
                                    </div>
                                </div>
                                <div className="flex items-center gap-2 mt-4 p-3 bg-white/[0.03] rounded-md overflow-x-auto">
                                    {mockBounds.slice(0, mockBounds.findIndex(b => b.layer === selectedLayer) + 1).map((b, i, arr) => (
                                        <React.Fragment key={b.layer}>
                                            <div
                                                className="px-3 py-2 rounded-md text-[11px] font-medium whitespace-nowrap"
                                                style={{
                                                    background: b.layer === selectedLayer ? 'rgba(255, 255, 255, 0.15)' : 'rgba(255, 255, 255, 0.05)'
                                                }}
                                            >
                                                {b.layer}
                                            </div>
                                            {i < arr.length - 1 && <span className="text-[#666]">→</span>}
                                        </React.Fragment>
                                    ))}
                                </div>
                            </>
                        );
                    })()}
                </div>
            )}
        </div>
    );
}
