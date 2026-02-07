'use client';

import { useState, useEffect } from 'react';
import { useAccount } from 'wagmi';
import { formatEther } from 'viem';
import { useStake, useStaking, useModel, useContractState } from '@/hooks/useContract';

interface StakingInterfaceProps {
    modelId: number;
}

export default function StakingInterface({ modelId }: StakingInterfaceProps) {
    const { isConnected, address: _address } = useAccount();
    const { stake, loading: stakeLoading, refetch: refetchStake } = useStake(modelId);
    const { model, isLoading: _modelLoading } = useModel(modelId);
    const contractState = useContractState();
    const { stakeTokens, unstakeTokens, pending, txHash, error } = useStaking(modelId);

    const [stakeAmount, setStakeAmount] = useState('0.1');
    const [activeTab, setActiveTab] = useState<'stake' | 'unstake' | 'info'>('stake');

    // Refresh stake after transaction
    useEffect(() => {
        if (txHash && !pending) {
            refetchStake();
        }
    }, [txHash, pending, refetchStake]);

    const handleStake = async () => {
        await stakeTokens(stakeAmount);
    };

    const handleUnstake = async () => {
        await unstakeTokens();
    };

    const formatDuration = (seconds: number) => {
        const days = Math.floor(seconds / 86400);
        const hours = Math.floor((seconds % 86400) / 3600);
        if (days > 0) return `${days}d ${hours}h`;
        return `${hours}h`;
    };

    const isLocked = stake && Number(stake.lockedUntil) > Date.now() / 1000;
    const lockTimeRemaining = stake
        ? Math.max(0, Number(stake.lockedUntil) - Math.floor(Date.now() / 1000))
        : 0;

    return (
        <div className="staking-container">
            <div className="staking-header">
                <h2>Staking</h2>
                <div className="model-info">
                    <span className="model-label">Model #{modelId}</span>
                    {model?.active && <span className="active-badge">Active</span>}
                </div>
            </div>

            {!isConnected ? (
                <div className="connect-prompt">
                    <div className="icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                            <path d="M7 11V7a5 5 0 0110 0v4" />
                        </svg>
                    </div>
                    <p>Connect your wallet to stake</p>
                </div>
            ) : (
                <>
                    {/* Current Stake Info */}
                    <div className="stake-info-panel">
                        <div className="info-row">
                            <span className="label">Your Stake</span>
                            <span className="value">
                                {stakeLoading
                                    ? '...'
                                    : stake
                                    ? `${formatEther(stake.amount)} ETH`
                                    : '0 ETH'}
                            </span>
                        </div>
                        <div className="info-row">
                            <span className="label">Lock Status</span>
                            <span className={`value ${stake?.slashed ? 'slashed' : isLocked ? 'locked' : 'unlocked'}`}>
                                {stake?.slashed
                                    ? 'Slashed'
                                    : isLocked
                                    ? `Locked (${formatDuration(lockTimeRemaining)})`
                                    : 'Unlocked'}
                            </span>
                        </div>
                        <div className="info-row">
                            <span className="label">Min Stake</span>
                            <span className="value">
                                {model ? `${formatEther(model.minStake)} ETH` : '...'}
                            </span>
                        </div>
                    </div>

                    {/* Tabs */}
                    <div className="tabs">
                        <button
                            className={`tab ${activeTab === 'stake' ? 'active' : ''}`}
                            onClick={() => setActiveTab('stake')}
                        >
                            Stake
                        </button>
                        <button
                            className={`tab ${activeTab === 'unstake' ? 'active' : ''}`}
                            onClick={() => setActiveTab('unstake')}
                        >
                            Unstake
                        </button>
                        <button
                            className={`tab ${activeTab === 'info' ? 'active' : ''}`}
                            onClick={() => setActiveTab('info')}
                        >
                            Info
                        </button>
                    </div>

                    {/* Tab Content */}
                    <div className="tab-content">
                        {activeTab === 'stake' && (
                            <div className="stake-form">
                                <div className="input-group">
                                    <label>Amount to Stake</label>
                                    <div className="input-wrapper">
                                        <input
                                            type="number"
                                            step="0.01"
                                            min="0"
                                            value={stakeAmount}
                                            onChange={(e) => setStakeAmount(e.target.value)}
                                            disabled={pending}
                                        />
                                        <span className="suffix">ETH</span>
                                    </div>
                                </div>

                                <div className="quick-amounts">
                                    {['0.1', '0.5', '1.0', '2.0'].map((amount) => (
                                        <button
                                            key={amount}
                                            className="quick-btn"
                                            onClick={() => setStakeAmount(amount)}
                                            disabled={pending}
                                        >
                                            {amount} ETH
                                        </button>
                                    ))}
                                </div>

                                <button
                                    className="action-btn stake-btn"
                                    onClick={handleStake}
                                    disabled={pending || parseFloat(stakeAmount) <= 0}
                                >
                                    {pending ? (
                                        <span className="loading">
                                            <span className="spinner" />
                                            Staking...
                                        </span>
                                    ) : (
                                        <>
                                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                <path d="M12 2v20M2 12h20" />
                                            </svg>
                                            Stake {stakeAmount} ETH
                                        </>
                                    )}
                                </button>
                            </div>
                        )}

                        {activeTab === 'unstake' && (
                            <div className="unstake-form">
                                {stake?.slashed ? (
                                    <div className="warning-box">
                                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                            <path d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                        </svg>
                                        <div>
                                            <strong>Stake Slashed</strong>
                                            <p>Your stake was slashed due to an invalid proof submission.</p>
                                        </div>
                                    </div>
                                ) : isLocked ? (
                                    <div className="info-box">
                                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                            <circle cx="12" cy="12" r="10" />
                                            <path d="M12 6v6l4 2" />
                                        </svg>
                                        <div>
                                            <strong>Stake Locked</strong>
                                            <p>
                                                Your stake will unlock in {formatDuration(lockTimeRemaining)}.
                                            </p>
                                        </div>
                                    </div>
                                ) : null}

                                <div className="unstake-amount">
                                    <span className="label">Amount to Unstake</span>
                                    <span className="value">
                                        {stake ? formatEther(stake.amount) : '0'} ETH
                                    </span>
                                </div>

                                <button
                                    className="action-btn unstake-btn"
                                    onClick={handleUnstake}
                                    disabled={pending || !stake || stake.amount === BigInt(0) || isLocked || stake.slashed}
                                >
                                    {pending ? (
                                        <span className="loading">
                                            <span className="spinner" />
                                            Unstaking...
                                        </span>
                                    ) : (
                                        <>
                                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                <path d="M5 12h14" />
                                            </svg>
                                            Unstake All
                                        </>
                                    )}
                                </button>
                            </div>
                        )}

                        {activeTab === 'info' && (
                            <div className="info-content">
                                <div className="info-section">
                                    <h3>Staking Parameters</h3>
                                    <div className="param-grid">
                                        <div className="param">
                                            <span className="param-label">Lock Period</span>
                                            <span className="param-value">
                                                {contractState.stakeLockPeriod !== undefined
                                                    ? formatDuration(Number(contractState.stakeLockPeriod))
                                                    : '...'}
                                            </span>
                                        </div>
                                        <div className="param">
                                            <span className="param-label">Slash Percentage</span>
                                            <span className="param-value">
                                                {contractState.slashPercentage !== undefined
                                                    ? `${Number(contractState.slashPercentage) / 100}%`
                                                    : '...'}
                                            </span>
                                        </div>
                                        <div className="param">
                                            <span className="param-label">Default Min Stake</span>
                                            <span className="param-value">
                                                {contractState.defaultMinStake !== undefined
                                                    ? `${formatEther(contractState.defaultMinStake)} ETH`
                                                    : '...'}
                                            </span>
                                        </div>
                                    </div>
                                </div>

                                <div className="info-section">
                                    <h3>How Staking Works</h3>
                                    <ul>
                                        <li>Stake ETH to participate in model training</li>
                                        <li>Stakes are locked for {contractState.stakeLockPeriod !== undefined ? formatDuration(Number(contractState.stakeLockPeriod)) : '7 days'} after staking</li>
                                        <li>Invalid proofs result in {contractState.slashPercentage !== undefined ? `${Number(contractState.slashPercentage) / 100}%` : '50%'} stake slashing</li>
                                        <li>Successfully verified proofs unlock your stake immediately</li>
                                    </ul>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Error Display */}
                    {error && (
                        <div className="error-box">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <circle cx="12" cy="12" r="10" />
                                <path d="M15 9l-6 6M9 9l6 6" />
                            </svg>
                            <span>{error}</span>
                        </div>
                    )}

                    {/* Transaction Status */}
                    {txHash && (
                        <div className="tx-status">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M20 6L9 17l-5-5" />
                            </svg>
                            <span>Transaction submitted</span>
                            <a
                                href={`https://etherscan.io/tx/${txHash}`}
                                target="_blank"
                                rel="noopener noreferrer"
                            >
                                View on Etherscan
                            </a>
                        </div>
                    )}
                </>
            )}

            <style jsx>{`
                .staking-container {
                    background: linear-gradient(135deg, #1a1a2e 0%, #16162a 100%);
                    border-radius: 16px;
                    padding: 24px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }

                .staking-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                }

                .staking-header h2 {
                    font-size: 20px;
                    font-weight: 600;
                    color: #fff;
                    margin: 0;
                }

                .model-info {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }

                .model-label {
                    color: rgba(255, 255, 255, 0.6);
                    font-size: 14px;
                }

                .active-badge {
                    background: rgba(34, 197, 94, 0.2);
                    color: #22c55e;
                    padding: 2px 8px;
                    border-radius: 4px;
                    font-size: 12px;
                }

                .connect-prompt {
                    text-align: center;
                    padding: 40px 20px;
                }

                .connect-prompt .icon {
                    width: 48px;
                    height: 48px;
                    margin: 0 auto 16px;
                    color: rgba(255, 255, 255, 0.3);
                }

                .connect-prompt p {
                    color: rgba(255, 255, 255, 0.6);
                    margin: 0;
                }

                .stake-info-panel {
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 12px;
                    padding: 16px;
                    margin-bottom: 20px;
                }

                .info-row {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 8px 0;
                }

                .info-row:not(:last-child) {
                    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                }

                .info-row .label {
                    color: rgba(255, 255, 255, 0.6);
                    font-size: 14px;
                }

                .info-row .value {
                    color: #fff;
                    font-weight: 500;
                }

                .info-row .value.locked {
                    color: #f59e0b;
                }

                .info-row .value.unlocked {
                    color: #22c55e;
                }

                .info-row .value.slashed {
                    color: #ef4444;
                }

                .tabs {
                    display: flex;
                    gap: 4px;
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 8px;
                    padding: 4px;
                    margin-bottom: 20px;
                }

                .tab {
                    flex: 1;
                    padding: 10px 16px;
                    border: none;
                    border-radius: 6px;
                    background: transparent;
                    color: rgba(255, 255, 255, 0.6);
                    font-size: 14px;
                    cursor: pointer;
                    transition: all 0.2s;
                }

                .tab:hover {
                    color: #fff;
                }

                .tab.active {
                    background: rgba(99, 102, 241, 0.2);
                    color: #6366f1;
                }

                .tab-content {
                    min-height: 200px;
                }

                .input-group {
                    margin-bottom: 16px;
                }

                .input-group label {
                    display: block;
                    color: rgba(255, 255, 255, 0.6);
                    font-size: 14px;
                    margin-bottom: 8px;
                }

                .input-wrapper {
                    display: flex;
                    align-items: center;
                    background: rgba(0, 0, 0, 0.2);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 8px;
                    overflow: hidden;
                }

                .input-wrapper input {
                    flex: 1;
                    padding: 12px 16px;
                    background: transparent;
                    border: none;
                    color: #fff;
                    font-size: 18px;
                    outline: none;
                }

                .input-wrapper .suffix {
                    padding: 0 16px;
                    color: rgba(255, 255, 255, 0.4);
                    font-size: 14px;
                }

                .quick-amounts {
                    display: flex;
                    gap: 8px;
                    margin-bottom: 20px;
                }

                .quick-btn {
                    flex: 1;
                    padding: 8px;
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 6px;
                    color: rgba(255, 255, 255, 0.6);
                    font-size: 13px;
                    cursor: pointer;
                    transition: all 0.2s;
                }

                .quick-btn:hover:not(:disabled) {
                    background: rgba(255, 255, 255, 0.1);
                    color: #fff;
                }

                .action-btn {
                    width: 100%;
                    padding: 14px 24px;
                    border: none;
                    border-radius: 8px;
                    font-size: 16px;
                    font-weight: 500;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 8px;
                    transition: all 0.2s;
                }

                .action-btn svg {
                    width: 18px;
                    height: 18px;
                }

                .stake-btn {
                    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                    color: #fff;
                }

                .stake-btn:hover:not(:disabled) {
                    transform: translateY(-1px);
                    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
                }

                .unstake-btn {
                    background: rgba(239, 68, 68, 0.2);
                    color: #ef4444;
                    border: 1px solid rgba(239, 68, 68, 0.3);
                }

                .unstake-btn:hover:not(:disabled) {
                    background: rgba(239, 68, 68, 0.3);
                }

                .action-btn:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                }

                .loading {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }

                .spinner {
                    width: 16px;
                    height: 16px;
                    border: 2px solid rgba(255, 255, 255, 0.3);
                    border-top-color: #fff;
                    border-radius: 50%;
                    animation: spin 0.8s linear infinite;
                }

                @keyframes spin {
                    to {
                        transform: rotate(360deg);
                    }
                }

                .warning-box,
                .info-box {
                    display: flex;
                    gap: 12px;
                    padding: 16px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                }

                .warning-box {
                    background: rgba(239, 68, 68, 0.1);
                    border: 1px solid rgba(239, 68, 68, 0.2);
                }

                .warning-box svg {
                    flex-shrink: 0;
                    width: 24px;
                    height: 24px;
                    color: #ef4444;
                }

                .info-box {
                    background: rgba(245, 158, 11, 0.1);
                    border: 1px solid rgba(245, 158, 11, 0.2);
                }

                .info-box svg {
                    flex-shrink: 0;
                    width: 24px;
                    height: 24px;
                    color: #f59e0b;
                }

                .warning-box strong,
                .info-box strong {
                    display: block;
                    color: #fff;
                    margin-bottom: 4px;
                }

                .warning-box p,
                .info-box p {
                    color: rgba(255, 255, 255, 0.6);
                    font-size: 14px;
                    margin: 0;
                }

                .unstake-amount {
                    text-align: center;
                    padding: 24px;
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 12px;
                    margin-bottom: 20px;
                }

                .unstake-amount .label {
                    display: block;
                    color: rgba(255, 255, 255, 0.6);
                    font-size: 14px;
                    margin-bottom: 8px;
                }

                .unstake-amount .value {
                    font-size: 28px;
                    font-weight: 600;
                    color: #fff;
                }

                .info-content {
                    color: rgba(255, 255, 255, 0.8);
                }

                .info-section {
                    margin-bottom: 24px;
                }

                .info-section h3 {
                    font-size: 14px;
                    font-weight: 600;
                    color: #fff;
                    margin: 0 0 12px 0;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }

                .param-grid {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 12px;
                }

                .param {
                    background: rgba(255, 255, 255, 0.05);
                    padding: 12px;
                    border-radius: 8px;
                    text-align: center;
                }

                .param-label {
                    display: block;
                    color: rgba(255, 255, 255, 0.5);
                    font-size: 12px;
                    margin-bottom: 4px;
                }

                .param-value {
                    color: #fff;
                    font-weight: 500;
                }

                .info-content ul {
                    margin: 0;
                    padding-left: 20px;
                }

                .info-content li {
                    color: rgba(255, 255, 255, 0.7);
                    font-size: 14px;
                    margin-bottom: 8px;
                }

                .error-box {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    padding: 12px;
                    background: rgba(239, 68, 68, 0.1);
                    border: 1px solid rgba(239, 68, 68, 0.2);
                    border-radius: 8px;
                    margin-top: 16px;
                }

                .error-box svg {
                    flex-shrink: 0;
                    width: 18px;
                    height: 18px;
                    color: #ef4444;
                }

                .error-box span {
                    color: #ef4444;
                    font-size: 14px;
                }

                .tx-status {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    padding: 12px;
                    background: rgba(34, 197, 94, 0.1);
                    border: 1px solid rgba(34, 197, 94, 0.2);
                    border-radius: 8px;
                    margin-top: 16px;
                }

                .tx-status svg {
                    flex-shrink: 0;
                    width: 18px;
                    height: 18px;
                    color: #22c55e;
                }

                .tx-status span {
                    color: #22c55e;
                    font-size: 14px;
                }

                .tx-status a {
                    margin-left: auto;
                    color: #6366f1;
                    font-size: 14px;
                    text-decoration: none;
                }

                .tx-status a:hover {
                    text-decoration: underline;
                }

                @media (max-width: 640px) {
                    .param-grid {
                        grid-template-columns: 1fr;
                    }

                    .quick-amounts {
                        flex-wrap: wrap;
                    }

                    .quick-btn {
                        flex: 1 1 calc(50% - 4px);
                    }
                }
            `}</style>
        </div>
    );
}
