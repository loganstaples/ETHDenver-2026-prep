'use client';

import TrainingProgress from '@/components/training/TrainingProgress';
import ProofExplorer from '@/components/proofs/ProofExplorer';
import ErrorBoundsVisualization from '@/components/analysis/ErrorBoundsVisualization';
import NodeNetworkView from '@/components/network/NodeNetworkView';
import ModelInteraction from '@/components/model/ModelInteraction';
import OnChainExplorer from '@/components/blockchain/OnChainExplorer';

export default function DashboardPage() {
    return (
        <div className="dashboard">
            <style jsx>{`
        .dashboard {
          min-height: 100vh;
          background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
          padding: 24px;
          font-family: 'Inter', -apple-system, sans-serif;
        }

        .dashboard-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 32px;
          padding-bottom: 24px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .logo {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .logo-icon {
          width: 48px;
          height: 48px;
          background: linear-gradient(135deg, #6366f1, #a855f7);
          border-radius: 12px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 24px;
          font-weight: 700;
          color: white;
        }

        .logo-text {
          font-size: 28px;
          font-weight: 800;
          background: linear-gradient(90deg, #6366f1, #a855f7, #22c55e);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .logo-tagline {
          font-size: 12px;
          color: #6b7280;
        }

        .header-actions {
          display: flex;
          gap: 12px;
        }

        .header-btn {
          padding: 12px 24px;
          border-radius: 10px;
          border: 1px solid rgba(255, 255, 255, 0.1);
          background: rgba(255, 255, 255, 0.05);
          color: #d1d5db;
          font-size: 14px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
        }

        .header-btn:hover {
          background: rgba(255, 255, 255, 0.1);
        }

        .header-btn.primary {
          background: linear-gradient(135deg, #6366f1, #a855f7);
          border: none;
          color: white;
        }

        .header-btn.primary:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
        }

        .dashboard-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 24px;
        }

        .full-width {
          grid-column: span 2;
        }

        @media (max-width: 1400px) {
          .dashboard-grid {
            grid-template-columns: 1fr;
          }
          .full-width {
            grid-column: span 1;
          }
        }
      `}</style>

            <header className="dashboard-header">
                <div className="logo">
                    <div className="logo-icon">H</div>
                    <div>
                        <div className="logo-text">HELIX Dashboard</div>
                        <div className="logo-tagline">Decentralized Verifiable ML Training</div>
                    </div>
                </div>
                <div className="header-actions">
                    <button className="header-btn">Documentation</button>
                    <button className="header-btn">Settings</button>
                    <button className="header-btn primary">Connect Wallet</button>
                </div>
            </header>

            <div className="dashboard-grid">
                <TrainingProgress modelName="helix-llm-7b" />
                <ProofExplorer />
                <ErrorBoundsVisualization modelName="helix-llm-7b" />
                <NodeNetworkView />
                <div className="full-width">
                    <ModelInteraction modelName="helix-llm-7b" />
                </div>
                <div className="full-width">
                    <OnChainExplorer contractAddress="0xHelix...Coordinator" />
                </div>
            </div>
        </div>
    );
}
