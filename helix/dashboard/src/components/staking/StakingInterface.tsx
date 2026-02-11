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
        <div className="bg-helix-surface border border-helix-border rounded-md p-4">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold text-white m-0">Staking</h2>
                <div className="flex items-center gap-2">
                    <span className="text-[#666] text-sm">Model #{modelId}</span>
                    {model?.active && (
                        <span className="bg-white/10 text-white px-2 py-0.5 rounded text-xs">
                            Active
                        </span>
                    )}
                </div>
            </div>

            {!isConnected ? (
                <div className="text-center py-10 px-5">
                    <div className="w-12 h-12 mx-auto mb-4 text-white/50">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                            <path d="M7 11V7a5 5 0 0110 0v4" />
                        </svg>
                    </div>
                    <p className="text-[#666] m-0">Connect your wallet to stake</p>
                </div>
            ) : (
                <>
                    {/* Current Stake Info */}
                    <div className="bg-white/5 rounded-md p-4 mb-5">
                        <div className="flex justify-between items-center py-2">
                            <span className="text-[#666] text-sm">Your Stake</span>
                            <span className="text-white font-medium">
                                {stakeLoading
                                    ? '...'
                                    : stake
                                    ? `${formatEther(stake.amount)} ETH`
                                    : '0 ETH'}
                            </span>
                        </div>
                        <div className="flex justify-between items-center py-2 border-t border-helix-border">
                            <span className="text-[#666] text-sm">Lock Status</span>
                            <span className={`font-medium ${stake?.slashed || isLocked ? 'text-[#888]' : 'text-white'}`}>
                                {stake?.slashed
                                    ? 'Slashed'
                                    : isLocked
                                    ? `Locked (${formatDuration(lockTimeRemaining)})`
                                    : 'Unlocked'}
                            </span>
                        </div>
                        <div className="flex justify-between items-center py-2 border-t border-helix-border">
                            <span className="text-[#666] text-sm">Min Stake</span>
                            <span className="text-white font-medium">
                                {model ? `${formatEther(model.minStake)} ETH` : '...'}
                            </span>
                        </div>
                    </div>

                    {/* Tabs */}
                    <div className="flex gap-1 bg-white/5 rounded-md p-1 mb-5">
                        <button
                            className={`flex-1 py-2.5 px-4 border-none rounded-md text-sm cursor-pointer transition-all ${
                                activeTab === 'stake'
                                    ? 'bg-white/10 text-white'
                                    : 'bg-transparent text-[#666] hover:text-white'
                            }`}
                            onClick={() => setActiveTab('stake')}
                        >
                            Stake
                        </button>
                        <button
                            className={`flex-1 py-2.5 px-4 border-none rounded-md text-sm cursor-pointer transition-all ${
                                activeTab === 'unstake'
                                    ? 'bg-white/10 text-white'
                                    : 'bg-transparent text-[#666] hover:text-white'
                            }`}
                            onClick={() => setActiveTab('unstake')}
                        >
                            Unstake
                        </button>
                        <button
                            className={`flex-1 py-2.5 px-4 border-none rounded-md text-sm cursor-pointer transition-all ${
                                activeTab === 'info'
                                    ? 'bg-white/10 text-white'
                                    : 'bg-transparent text-[#666] hover:text-white'
                            }`}
                            onClick={() => setActiveTab('info')}
                        >
                            Info
                        </button>
                    </div>

                    {/* Tab Content */}
                    <div className="min-h-[200px]">
                        {activeTab === 'stake' && (
                            <div>
                                <div className="mb-4">
                                    <label className="block text-[#666] text-sm mb-2">
                                        Amount to Stake
                                    </label>
                                    <div className="flex items-center bg-black border border-helix-border rounded-md overflow-hidden focus-within:border-white/30">
                                        <input
                                            type="number"
                                            step="0.01"
                                            min="0"
                                            value={stakeAmount}
                                            onChange={(e) => setStakeAmount(e.target.value)}
                                            disabled={pending}
                                            className="flex-1 py-3 px-4 bg-transparent border-none text-white text-lg outline-none"
                                        />
                                        <span className="px-4 text-white/40 text-sm">ETH</span>
                                    </div>
                                </div>

                                <div className="flex gap-2 mb-5">
                                    {['0.1', '0.5', '1.0', '2.0'].map((amount) => (
                                        <button
                                            key={amount}
                                            className="flex-1 py-2 px-2 bg-white/5 border border-helix-border rounded-md text-[#888] text-xs cursor-pointer transition-all hover:text-white hover:bg-white/10 disabled:opacity-50 disabled:cursor-not-allowed"
                                            onClick={() => setStakeAmount(amount)}
                                            disabled={pending}
                                        >
                                            {amount} ETH
                                        </button>
                                    ))}
                                </div>

                                <button
                                    className="w-full py-3.5 px-6 border-none rounded-md text-base font-medium cursor-pointer flex items-center justify-center gap-2 transition-all bg-white text-black hover:bg-neutral-200 disabled:opacity-50 disabled:cursor-not-allowed"
                                    onClick={handleStake}
                                    disabled={pending || parseFloat(stakeAmount) <= 0}
                                >
                                    {pending ? (
                                        <span className="flex items-center gap-2">
                                            <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                            Staking...
                                        </span>
                                    ) : (
                                        <>
                                            <svg className="w-[18px] h-[18px]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                <path d="M12 2v20M2 12h20" />
                                            </svg>
                                            Stake {stakeAmount} ETH
                                        </>
                                    )}
                                </button>
                            </div>
                        )}

                        {activeTab === 'unstake' && (
                            <div>
                                {stake?.slashed ? (
                                    <div className="flex gap-3 p-4 rounded-md mb-5 bg-white/5 border border-helix-border">
                                        <svg className="flex-shrink-0 w-6 h-6 text-white/50" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                            <path d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                        </svg>
                                        <div>
                                            <strong className="block text-white mb-1">Stake Slashed</strong>
                                            <p className="text-[#666] text-sm m-0">
                                                Your stake was slashed due to an invalid proof submission.
                                            </p>
                                        </div>
                                    </div>
                                ) : isLocked ? (
                                    <div className="flex gap-3 p-4 rounded-md mb-5 bg-white/5 border border-helix-border">
                                        <svg className="flex-shrink-0 w-6 h-6 text-white/50" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                            <circle cx="12" cy="12" r="10" />
                                            <path d="M12 6v6l4 2" />
                                        </svg>
                                        <div>
                                            <strong className="block text-white mb-1">Stake Locked</strong>
                                            <p className="text-[#666] text-sm m-0">
                                                Your stake will unlock in {formatDuration(lockTimeRemaining)}.
                                            </p>
                                        </div>
                                    </div>
                                ) : null}

                                <div className="text-center py-6 px-4 bg-white/5 rounded-md mb-5">
                                    <span className="block text-[#666] text-sm mb-2">
                                        Amount to Unstake
                                    </span>
                                    <span className="text-lg font-semibold text-white">
                                        {stake ? formatEther(stake.amount) : '0'} ETH
                                    </span>
                                </div>

                                <button
                                    className="w-full py-3.5 px-6 rounded-md text-base font-medium cursor-pointer flex items-center justify-center gap-2 transition-all border border-white/10 text-white hover:bg-white/10 disabled:opacity-50 disabled:cursor-not-allowed"
                                    onClick={handleUnstake}
                                    disabled={pending || !stake || stake.amount === BigInt(0) || isLocked || stake.slashed}
                                >
                                    {pending ? (
                                        <span className="flex items-center gap-2">
                                            <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                            Unstaking...
                                        </span>
                                    ) : (
                                        <>
                                            <svg className="w-[18px] h-[18px]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                <path d="M5 12h14" />
                                            </svg>
                                            Unstake All
                                        </>
                                    )}
                                </button>
                            </div>
                        )}

                        {activeTab === 'info' && (
                            <div className="text-white/80">
                                <div className="mb-4">
                                    <h3 className="text-sm font-semibold text-white m-0 mb-3 uppercase tracking-wider">
                                        Staking Parameters
                                    </h3>
                                    <div className="grid grid-cols-3 gap-3 max-sm:grid-cols-1">
                                        <div className="bg-white/5 p-3 rounded-md text-center">
                                            <span className="block text-white/50 text-xs mb-1">
                                                Lock Period
                                            </span>
                                            <span className="text-white font-medium">
                                                {contractState.stakeLockPeriod !== undefined
                                                    ? formatDuration(Number(contractState.stakeLockPeriod))
                                                    : '...'}
                                            </span>
                                        </div>
                                        <div className="bg-white/5 p-3 rounded-md text-center">
                                            <span className="block text-white/50 text-xs mb-1">
                                                Slash Percentage
                                            </span>
                                            <span className="text-white font-medium">
                                                {contractState.slashPercentage !== undefined
                                                    ? `${Number(contractState.slashPercentage) / 100}%`
                                                    : '...'}
                                            </span>
                                        </div>
                                        <div className="bg-white/5 p-3 rounded-md text-center">
                                            <span className="block text-white/50 text-xs mb-1">
                                                Default Min Stake
                                            </span>
                                            <span className="text-white font-medium">
                                                {contractState.defaultMinStake !== undefined
                                                    ? `${formatEther(contractState.defaultMinStake)} ETH`
                                                    : '...'}
                                            </span>
                                        </div>
                                    </div>
                                </div>

                                <div className="mb-4">
                                    <h3 className="text-sm font-semibold text-white m-0 mb-3 uppercase tracking-wider">
                                        How Staking Works
                                    </h3>
                                    <ul className="m-0 pl-5">
                                        <li className="text-white/70 text-sm mb-2">
                                            Stake ETH to participate in model training
                                        </li>
                                        <li className="text-white/70 text-sm mb-2">
                                            Stakes are locked for {contractState.stakeLockPeriod !== undefined ? formatDuration(Number(contractState.stakeLockPeriod)) : '7 days'} after staking
                                        </li>
                                        <li className="text-white/70 text-sm mb-2">
                                            Invalid proofs result in {contractState.slashPercentage !== undefined ? `${Number(contractState.slashPercentage) / 100}%` : '50%'} stake slashing
                                        </li>
                                        <li className="text-white/70 text-sm mb-2">
                                            Successfully verified proofs unlock your stake immediately
                                        </li>
                                    </ul>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Error Display */}
                    {error && (
                        <div className="flex items-center gap-2 p-3 rounded-md mt-4 bg-white/5 border border-helix-border">
                            <svg className="flex-shrink-0 w-[18px] h-[18px] text-white/50" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <circle cx="12" cy="12" r="10" />
                                <path d="M15 9l-6 6M9 9l6 6" />
                            </svg>
                            <span className="text-[#666] text-sm">{error}</span>
                        </div>
                    )}

                    {/* Transaction Status */}
                    {txHash && (
                        <div className="flex items-center gap-2 p-3 rounded-md mt-4 bg-white/5 border border-helix-border">
                            <svg className="flex-shrink-0 w-[18px] h-[18px] text-white/50" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M20 6L9 17l-5-5" />
                            </svg>
                            <span className="text-white text-sm">Transaction submitted</span>
                            <a
                                href={`https://etherscan.io/tx/${txHash}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="ml-auto text-white text-sm underline"
                            >
                                View on Etherscan
                            </a>
                        </div>
                    )}
                </>
            )}
        </div>
    );
}
