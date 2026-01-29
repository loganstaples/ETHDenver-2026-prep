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
    const maxAmp = Math.max(...mockBounds.map(b => b.amplification));

    const getRiskColor = (risk: string) => {
        const colors: Record<string, string> = {
            low: '#22c55e',
            medium: '#f59e0b',
            high: '#ef4444',
            critical: '#dc2626',
        };
        return colors[risk] || '#6b7280';
    };

    const formatError = (e: number) => {
        if (e === 0) return '0';
        if (e < 1e-9) return e.toExponential(1);
        if (e < 1e-6) return `${(e * 1e9).toFixed(1)}e-9`;
        if (e < 1e-3) return `${(e * 1e6).toFixed(1)}e-6`;
        return e.toExponential(2);
    };

    return (
        <div className="error-bounds-viz">
            <style jsx>{`
        .error-bounds-viz {
          background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
          border-radius: 16px;
          padding: 24px;
          color: #fff;
          font-family: 'Inter', -apple-system, sans-serif;
        }

        .header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 24px;
        }

        .title {
          font-size: 24px;
          font-weight: 700;
          background: linear-gradient(90deg, #f59e0b, #ef4444);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .total-error {
          padding: 8px 16px;
          background: rgba(239, 68, 68, 0.1);
          border: 1px solid rgba(239, 68, 68, 0.3);
          border-radius: 8px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 14px;
        }

        .total-error span {
          color: #ef4444;
          font-weight: 600;
        }

        .visualization {
          display: flex;
          flex-direction: column;
          gap: 8px;
          margin-bottom: 24px;
        }

        .layer-row {
          display: flex;
          align-items: center;
          gap: 16px;
          padding: 12px 16px;
          background: rgba(255, 255, 255, 0.03);
          border-radius: 8px;
          cursor: pointer;
          transition: all 0.2s;
          border: 1px solid transparent;
        }

        .layer-row:hover {
          background: rgba(255, 255, 255, 0.06);
        }

        .layer-row.selected {
          background: rgba(99, 102, 241, 0.1);
          border-color: rgba(99, 102, 241, 0.3);
        }

        .layer-name {
          width: 100px;
          font-weight: 600;
          font-size: 13px;
        }

        .operation {
          width: 100px;
          font-size: 12px;
          color: #9ca3af;
        }

        .error-bar-container {
          flex: 1;
          height: 24px;
          background: rgba(0, 0, 0, 0.3);
          border-radius: 4px;
          position: relative;
          overflow: hidden;
        }

        .error-bar {
          height: 100%;
          border-radius: 4px;
          transition: width 0.3s ease;
          display: flex;
          align-items: center;
          justify-content: flex-end;
          padding-right: 8px;
          font-size: 10px;
          font-weight: 600;
          color: rgba(255, 255, 255, 0.9);
        }

        .amp-badge {
          min-width: 60px;
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 11px;
          font-weight: 600;
          text-align: center;
          background: rgba(168, 85, 247, 0.2);
          color: #a855f7;
        }

        .risk-indicator {
          width: 8px;
          height: 8px;
          border-radius: 50%;
        }

        .legend {
          display: flex;
          gap: 24px;
          justify-content: center;
          padding: 16px;
          background: rgba(255, 255, 255, 0.03);
          border-radius: 8px;
          margin-bottom: 24px;
        }

        .legend-item {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 12px;
          color: #9ca3af;
        }

        .legend-color {
          width: 12px;
          height: 12px;
          border-radius: 2px;
        }

        .detail-panel {
          padding: 20px;
          background: rgba(255, 255, 255, 0.03);
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .detail-title {
          font-size: 16px;
          font-weight: 600;
          margin-bottom: 16px;
        }

        .detail-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 16px;
        }

        .detail-item {
          background: rgba(0, 0, 0, 0.2);
          padding: 12px;
          border-radius: 8px;
          text-align: center;
        }

        .detail-value {
          font-size: 18px;
          font-weight: 700;
          font-family: 'JetBrains Mono', monospace;
          margin-bottom: 4px;
        }

        .detail-label {
          font-size: 11px;
          color: #6b7280;
          text-transform: uppercase;
        }

        .propagation-chain {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-top: 16px;
          padding: 12px;
          background: rgba(0, 0, 0, 0.2);
          border-radius: 8px;
          overflow-x: auto;
        }

        .chain-node {
          padding: 8px 12px;
          background: rgba(99, 102, 241, 0.2);
          border-radius: 6px;
          font-size: 11px;
          font-weight: 500;
          white-space: nowrap;
        }

        .chain-arrow {
          color: #6b7280;
        }
      `}</style>

            <div className="header">
                <h2 className="title">Error Bounds: {modelName}</h2>
                <div className="total-error">
                    Total Output Error: <span>{formatError(totalError)}</span>
                </div>
            </div>

            <div className="legend">
                <div className="legend-item">
                    <div className="legend-color" style={{ background: '#22c55e' }} />
                    <span>Low Risk</span>
                </div>
                <div className="legend-item">
                    <div className="legend-color" style={{ background: '#f59e0b' }} />
                    <span>Medium Risk</span>
                </div>
                <div className="legend-item">
                    <div className="legend-color" style={{ background: '#ef4444' }} />
                    <span>High Risk</span>
                </div>
                <div className="legend-item">
                    <div className="legend-color" style={{ background: '#a855f7' }} />
                    <span>Amplification</span>
                </div>
            </div>

            <div className="visualization">
                {mockBounds.map((bound, idx) => {
                    const width = Math.min(
                        (Math.log10(bound.outputError + 1e-10) + 10) * 10,
                        100
                    );
                    return (
                        <div
                            key={bound.layer}
                            className={`layer-row ${selectedLayer === bound.layer ? 'selected' : ''}`}
                            onClick={() => setSelectedLayer(bound.layer)}
                        >
                            <div className="layer-name">{bound.layer}</div>
                            <div className="operation">{bound.operation}</div>
                            <div className="error-bar-container">
                                <div
                                    className="error-bar"
                                    style={{
                                        width: `${width}%`,
                                        background: `linear-gradient(90deg, ${getRiskColor(bound.riskLevel)}66, ${getRiskColor(bound.riskLevel)})`,
                                    }}
                                >
                                    {formatError(bound.outputError)}
                                </div>
                            </div>
                            <div className="amp-badge">{bound.amplification.toFixed(1)}×</div>
                            <div
                                className="risk-indicator"
                                style={{ background: getRiskColor(bound.riskLevel) }}
                            />
                        </div>
                    );
                })}
            </div>

            {selectedLayer && (
                <div className="detail-panel">
                    <h3 className="detail-title">
                        Layer: {mockBounds.find((b) => b.layer === selectedLayer)?.layer}
                    </h3>
                    {(() => {
                        const bound = mockBounds.find((b) => b.layer === selectedLayer);
                        if (!bound) return null;
                        return (
                            <>
                                <div className="detail-grid">
                                    <div className="detail-item">
                                        <div className="detail-value" style={{ color: '#6366f1' }}>
                                            {formatError(bound.inputError)}
                                        </div>
                                        <div className="detail-label">Input Error</div>
                                    </div>
                                    <div className="detail-item">
                                        <div className="detail-value" style={{ color: getRiskColor(bound.riskLevel) }}>
                                            {formatError(bound.outputError)}
                                        </div>
                                        <div className="detail-label">Output Error</div>
                                    </div>
                                    <div className="detail-item">
                                        <div className="detail-value" style={{ color: '#a855f7' }}>
                                            {bound.amplification.toFixed(2)}×
                                        </div>
                                        <div className="detail-label">Amplification</div>
                                    </div>
                                    <div className="detail-item">
                                        <div className="detail-value" style={{ color: getRiskColor(bound.riskLevel) }}>
                                            {bound.riskLevel.toUpperCase()}
                                        </div>
                                        <div className="detail-label">Risk Level</div>
                                    </div>
                                </div>
                                <div className="propagation-chain">
                                    {mockBounds.slice(0, mockBounds.findIndex(b => b.layer === selectedLayer) + 1).map((b, i, arr) => (
                                        <React.Fragment key={b.layer}>
                                            <div className="chain-node" style={{
                                                background: b.layer === selectedLayer ? 'rgba(99, 102, 241, 0.4)' : 'rgba(99, 102, 241, 0.15)'
                                            }}>
                                                {b.layer}
                                            </div>
                                            {i < arr.length - 1 && <span className="chain-arrow">→</span>}
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
