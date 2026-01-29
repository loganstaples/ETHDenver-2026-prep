'use client';

import React, { useState, useEffect } from 'react';

interface TrainingMetrics {
    epoch: number;
    totalEpochs: number;
    loss: number;
    accuracy: number;
    learningRate: number;
    batchesCompleted: number;
    totalBatches: number;
    timestamp: number;
}

interface TrainingRound {
    roundId: number;
    status: 'pending' | 'in_progress' | 'completed' | 'failed';
    participants: number;
    startTime: number;
    endTime?: number;
    aggregatedGradients?: number;
    proofGenerated: boolean;
}

interface TrainingProgressProps {
    modelName?: string;
    initialMetrics?: TrainingMetrics;
    rounds?: TrainingRound[];
}

export default function TrainingProgress({
    modelName = 'HELIX Model',
    initialMetrics,
    rounds = [],
}: TrainingProgressProps) {
    const [metrics, setMetrics] = useState<TrainingMetrics>(
        initialMetrics || {
            epoch: 3,
            totalEpochs: 10,
            loss: 0.342,
            accuracy: 0.876,
            learningRate: 0.001,
            batchesCompleted: 156,
            totalBatches: 200,
            timestamp: Date.now(),
        }
    );

    const [lossHistory, setLossHistory] = useState<number[]>([
        0.892, 0.654, 0.512, 0.423, 0.342,
    ]);

    const [accuracyHistory, setAccuracyHistory] = useState<number[]>([
        0.543, 0.672, 0.745, 0.812, 0.876,
    ]);

    const epochProgress = (metrics.epoch / metrics.totalEpochs) * 100;
    const batchProgress = (metrics.batchesCompleted / metrics.totalBatches) * 100;

    const formatTime = (timestamp: number) => {
        return new Date(timestamp).toLocaleTimeString();
    };

    const formatDuration = (start: number, end?: number) => {
        const duration = (end || Date.now()) - start;
        const minutes = Math.floor(duration / 60000);
        const seconds = Math.floor((duration % 60000) / 1000);
        return `${minutes}m ${seconds}s`;
    };

    return (
        <div className="training-progress">
            <style jsx>{`
        .training-progress {
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

        .model-name {
          font-size: 24px;
          font-weight: 700;
          background: linear-gradient(90deg, #6366f1, #a855f7);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .status-badge {
          padding: 6px 16px;
          border-radius: 20px;
          font-size: 12px;
          font-weight: 600;
          text-transform: uppercase;
          background: rgba(34, 197, 94, 0.2);
          color: #22c55e;
          border: 1px solid rgba(34, 197, 94, 0.3);
        }

        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 16px;
          margin-bottom: 24px;
        }

        .metric-card {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 12px;
          padding: 16px;
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .metric-label {
          font-size: 12px;
          color: #9ca3af;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          margin-bottom: 8px;
        }

        .metric-value {
          font-size: 28px;
          font-weight: 700;
        }

        .metric-value.loss {
          color: #f59e0b;
        }

        .metric-value.accuracy {
          color: #22c55e;
        }

        .metric-value.lr {
          color: #6366f1;
        }

        .metric-value.epoch {
          color: #a855f7;
        }

        .progress-section {
          margin-bottom: 24px;
        }

        .progress-header {
          display: flex;
          justify-content: space-between;
          margin-bottom: 8px;
          font-size: 14px;
        }

        .progress-bar {
          height: 8px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 4px;
          overflow: hidden;
          margin-bottom: 16px;
        }

        .progress-fill {
          height: 100%;
          border-radius: 4px;
          transition: width 0.3s ease;
        }

        .progress-fill.epoch {
          background: linear-gradient(90deg, #6366f1, #a855f7);
        }

        .progress-fill.batch {
          background: linear-gradient(90deg, #22c55e, #16a34a);
        }

        .charts-row {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 16px;
          margin-bottom: 24px;
        }

        .chart-card {
          background: rgba(255, 255, 255, 0.03);
          border-radius: 12px;
          padding: 16px;
          border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .chart-title {
          font-size: 14px;
          font-weight: 600;
          margin-bottom: 12px;
          color: #d1d5db;
        }

        .mini-chart {
          display: flex;
          align-items: flex-end;
          height: 60px;
          gap: 4px;
        }

        .bar {
          flex: 1;
          border-radius: 2px;
          transition: height 0.3s ease;
        }

        .bar.loss {
          background: linear-gradient(0deg, #f59e0b, #fbbf24);
        }

        .bar.accuracy {
          background: linear-gradient(0deg, #22c55e, #4ade80);
        }

        .rounds-section {
          border-top: 1px solid rgba(255, 255, 255, 0.1);
          padding-top: 24px;
        }

        .section-title {
          font-size: 16px;
          font-weight: 600;
          margin-bottom: 16px;
        }

        .rounds-list {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .round-item {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 12px 16px;
          background: rgba(255, 255, 255, 0.03);
          border-radius: 8px;
          border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .round-info {
          display: flex;
          align-items: center;
          gap: 16px;
        }

        .round-id {
          font-weight: 600;
          color: #a855f7;
        }

        .round-status {
          padding: 4px 10px;
          border-radius: 12px;
          font-size: 11px;
          font-weight: 600;
          text-transform: uppercase;
        }

        .round-status.completed {
          background: rgba(34, 197, 94, 0.2);
          color: #22c55e;
        }

        .round-status.in_progress {
          background: rgba(99, 102, 241, 0.2);
          color: #818cf8;
        }

        .round-status.pending {
          background: rgba(156, 163, 175, 0.2);
          color: #9ca3af;
        }

        .round-meta {
          display: flex;
          gap: 24px;
          font-size: 13px;
          color: #9ca3af;
        }

        .proof-badge {
          display: flex;
          align-items: center;
          gap: 4px;
          color: #22c55e;
        }
      `}</style>

            <div className="header">
                <h2 className="model-name">{modelName}</h2>
                <span className="status-badge">Training Active</span>
            </div>

            <div className="metrics-grid">
                <div className="metric-card">
                    <div className="metric-label">Current Loss</div>
                    <div className="metric-value loss">{metrics.loss.toFixed(4)}</div>
                </div>
                <div className="metric-card">
                    <div className="metric-label">Accuracy</div>
                    <div className="metric-value accuracy">
                        {(metrics.accuracy * 100).toFixed(1)}%
                    </div>
                </div>
                <div className="metric-card">
                    <div className="metric-label">Learning Rate</div>
                    <div className="metric-value lr">{metrics.learningRate}</div>
                </div>
                <div className="metric-card">
                    <div className="metric-label">Epoch</div>
                    <div className="metric-value epoch">
                        {metrics.epoch}/{metrics.totalEpochs}
                    </div>
                </div>
            </div>

            <div className="progress-section">
                <div className="progress-header">
                    <span>Epoch Progress</span>
                    <span>{epochProgress.toFixed(0)}%</span>
                </div>
                <div className="progress-bar">
                    <div
                        className="progress-fill epoch"
                        style={{ width: `${epochProgress}%` }}
                    />
                </div>

                <div className="progress-header">
                    <span>Batch Progress</span>
                    <span>
                        {metrics.batchesCompleted}/{metrics.totalBatches}
                    </span>
                </div>
                <div className="progress-bar">
                    <div
                        className="progress-fill batch"
                        style={{ width: `${batchProgress}%` }}
                    />
                </div>
            </div>

            <div className="charts-row">
                <div className="chart-card">
                    <div className="chart-title">Loss Over Epochs</div>
                    <div className="mini-chart">
                        {lossHistory.map((loss, i) => (
                            <div
                                key={i}
                                className="bar loss"
                                style={{ height: `${loss * 100}%` }}
                            />
                        ))}
                    </div>
                </div>
                <div className="chart-card">
                    <div className="chart-title">Accuracy Over Epochs</div>
                    <div className="mini-chart">
                        {accuracyHistory.map((acc, i) => (
                            <div
                                key={i}
                                className="bar accuracy"
                                style={{ height: `${acc * 100}%` }}
                            />
                        ))}
                    </div>
                </div>
            </div>

            <div className="rounds-section">
                <h3 className="section-title">Training Rounds</h3>
                <div className="rounds-list">
                    {[
                        {
                            roundId: 3,
                            status: 'in_progress' as const,
                            participants: 8,
                            startTime: Date.now() - 120000,
                            proofGenerated: false,
                        },
                        {
                            roundId: 2,
                            status: 'completed' as const,
                            participants: 12,
                            startTime: Date.now() - 300000,
                            endTime: Date.now() - 180000,
                            proofGenerated: true,
                        },
                        {
                            roundId: 1,
                            status: 'completed' as const,
                            participants: 10,
                            startTime: Date.now() - 600000,
                            endTime: Date.now() - 480000,
                            proofGenerated: true,
                        },
                    ].map((round) => (
                        <div key={round.roundId} className="round-item">
                            <div className="round-info">
                                <span className="round-id">Round #{round.roundId}</span>
                                <span className={`round-status ${round.status}`}>
                                    {round.status.replace('_', ' ')}
                                </span>
                            </div>
                            <div className="round-meta">
                                <span>{round.participants} participants</span>
                                <span>{formatDuration(round.startTime, round.endTime)}</span>
                                {round.proofGenerated && (
                                    <span className="proof-badge">✓ Proof</span>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
