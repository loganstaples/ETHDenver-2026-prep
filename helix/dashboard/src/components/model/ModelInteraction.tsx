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
        <div className="model-interaction">
            <style jsx>{`
        .model-interaction {
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
          background: linear-gradient(90deg, #14b8a6, #22c55e);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .model-badge {
          padding: 8px 16px;
          background: rgba(20, 184, 166, 0.2);
          border: 1px solid rgba(20, 184, 166, 0.3);
          border-radius: 8px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 14px;
          color: #14b8a6;
        }

        .tabs {
          display: flex;
          gap: 4px;
          margin-bottom: 24px;
          background: rgba(0, 0, 0, 0.2);
          padding: 4px;
          border-radius: 10px;
        }

        .tab {
          flex: 1;
          padding: 12px;
          border: none;
          background: transparent;
          color: #9ca3af;
          font-size: 14px;
          font-weight: 500;
          cursor: pointer;
          border-radius: 8px;
          transition: all 0.2s;
        }

        .tab.active {
          background: rgba(99, 102, 241, 0.2);
          color: #818cf8;
        }

        .overview-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 24px;
        }

        .section {
          background: rgba(255, 255, 255, 0.03);
          border-radius: 12px;
          padding: 20px;
          border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .section-title {
          font-size: 14px;
          font-weight: 600;
          margin-bottom: 16px;
          color: #d1d5db;
        }

        .version-list {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .version-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 12px;
          background: rgba(0, 0, 0, 0.2);
          border-radius: 8px;
        }

        .version-tag {
          font-weight: 600;
          color: #14b8a6;
        }

        .version-meta {
          display: flex;
          gap: 16px;
          font-size: 12px;
          color: #9ca3af;
        }

        .stat-value {
          font-weight: 600;
          color: #d1d5db;
        }

        .layer-list {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .layer-item {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 12px;
          background: rgba(0, 0, 0, 0.2);
          border-radius: 8px;
        }

        .layer-name {
          font-weight: 600;
          font-size: 13px;
        }

        .layer-type {
          font-size: 11px;
          color: #6b7280;
        }

        .layer-meta {
          display: flex;
          gap: 12px;
          align-items: center;
        }

        .param-count {
          font-family: 'JetBrains Mono', monospace;
          font-size: 12px;
          color: #a855f7;
        }

        .quantized-badge {
          padding: 2px 8px;
          border-radius: 4px;
          font-size: 10px;
          font-weight: 600;
          background: rgba(34, 197, 94, 0.2);
          color: #22c55e;
        }

        .inference-section {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .input-group {
          display: flex;
          gap: 12px;
        }

        .prompt-input {
          flex: 1;
          padding: 16px;
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          color: #fff;
          font-size: 14px;
          resize: none;
        }

        .prompt-input:focus {
          outline: none;
          border-color: rgba(99, 102, 241, 0.5);
        }

        .generate-btn {
          padding: 16px 32px;
          background: linear-gradient(135deg, #6366f1, #a855f7);
          border: none;
          border-radius: 12px;
          color: #fff;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
        }

        .generate-btn:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
        }

        .generate-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
          transform: none;
        }

        .response-box {
          padding: 20px;
          background: rgba(0, 0, 0, 0.3);
          border-radius: 12px;
          min-height: 120px;
          font-size: 14px;
          line-height: 1.6;
          color: #d1d5db;
        }

        .cursor {
          display: inline-block;
          width: 8px;
          height: 16px;
          background: #6366f1;
          animation: blink 1s infinite;
          margin-left: 2px;
        }

        @keyframes blink {
          0%, 50% { opacity: 1; }
          51%, 100% { opacity: 0; }
        }

        .proof-notice {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 12px 16px;
          background: rgba(34, 197, 94, 0.1);
          border: 1px solid rgba(34, 197, 94, 0.2);
          border-radius: 8px;
          font-size: 13px;
          color: #22c55e;
        }
      `}</style>

            <div className="header">
                <h2 className="title">Model Interaction</h2>
                <span className="model-badge">{modelName}</span>
            </div>

            <div className="tabs">
                <button
                    className={`tab ${activeTab === 'overview' ? 'active' : ''}`}
                    onClick={() => setActiveTab('overview')}
                >
                    Overview
                </button>
                <button
                    className={`tab ${activeTab === 'architecture' ? 'active' : ''}`}
                    onClick={() => setActiveTab('architecture')}
                >
                    Architecture
                </button>
                <button
                    className={`tab ${activeTab === 'inference' ? 'active' : ''}`}
                    onClick={() => setActiveTab('inference')}
                >
                    Inference
                </button>
            </div>

            {activeTab === 'overview' && (
                <div className="overview-grid">
                    <div className="section">
                        <h3 className="section-title">Model Versions</h3>
                        <div className="version-list">
                            {mockVersions.map((v) => (
                                <div key={v.version} className="version-item">
                                    <span className="version-tag">{v.version}</span>
                                    <div className="version-meta">
                                        <span>Acc: <span className="stat-value">{(v.accuracy * 100).toFixed(1)}%</span></span>
                                        <span>Loss: <span className="stat-value">{v.loss.toFixed(3)}</span></span>
                                        <span>{v.participants} nodes</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                    <div className="section">
                        <h3 className="section-title">Model Stats</h3>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                            <div className="version-item">
                                <span>Total Parameters</span>
                                <span className="stat-value" style={{ color: '#a855f7' }}>{formatParams(totalParams)}</span>
                            </div>
                            <div className="version-item">
                                <span>Quantized Layers</span>
                                <span className="stat-value" style={{ color: '#22c55e' }}>{mockLayers.filter(l => l.quantized).length}/{mockLayers.length}</span>
                            </div>
                            <div className="version-item">
                                <span>Max Error Bound</span>
                                <span className="stat-value" style={{ color: '#f59e0b' }}>{Math.max(...mockLayers.map(l => l.errorBound)).toExponential(1)}</span>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {activeTab === 'architecture' && (
                <div className="section">
                    <h3 className="section-title">Layer Configuration</h3>
                    <div className="layer-list">
                        {mockLayers.map((layer) => (
                            <div key={layer.name} className="layer-item">
                                <div>
                                    <div className="layer-name">{layer.name}</div>
                                    <div className="layer-type">{layer.type}</div>
                                </div>
                                <div className="layer-meta">
                                    <span className="param-count">{formatParams(layer.params)}</span>
                                    {layer.quantized && <span className="quantized-badge">INT8</span>}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {activeTab === 'inference' && (
                <div className="inference-section">
                    <div className="input-group">
                        <textarea
                            className="prompt-input"
                            placeholder="Enter your prompt..."
                            rows={3}
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                        />
                        <button
                            className="generate-btn"
                            onClick={handleInference}
                            disabled={isGenerating || !prompt.trim()}
                        >
                            {isGenerating ? 'Generating...' : 'Generate'}
                        </button>
                    </div>
                    <div className="response-box">
                        {response || 'Response will appear here...'}
                        {isGenerating && <span className="cursor" />}
                    </div>
                    <div className="proof-notice">
                        ✓ All inference computations are verified with ZK proofs
                    </div>
                </div>
            )}
        </div>
    );
}
