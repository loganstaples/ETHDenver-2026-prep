'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Wallet, ArrowUpRight, ArrowDownLeft, Clock, Lock, Loader2 } from 'lucide-react';
import { useAccount } from 'wagmi';
import { formatEther } from 'viem';
import { useStake, useStaking, useContractState, useContractEvents } from '@/hooks/useContract';

export default function BillingPage() {
    const { isConnected } = useAccount();
    const contractState = useContractState();
    const { stake, loading: stakeLoading } = useStake(0);
    const { stakeTokens, unstakeTokens, pending, txHash, error } = useStaking(0);
    const { stakedEvents, slashedEvents } = useContractEvents();

    const [stakeAmount, setStakeAmount] = useState('0.1');
    const [activeTab, setActiveTab] = useState<'stake' | 'unstake'>('stake');

    const isLocked = stake && Number(stake.lockedUntil) > Date.now() / 1000;
    const lockTimeRemaining = stake
        ? Math.max(0, Number(stake.lockedUntil) - Math.floor(Date.now() / 1000))
        : 0;

    const formatDuration = (seconds: number) => {
        const days = Math.floor(seconds / 86400);
        const hours = Math.floor((seconds % 86400) / 3600);
        if (days > 0) return `${days}d ${hours}h`;
        return `${hours}h`;
    };

    // Estimated cost calculation
    const estimatedCostPerEpoch = 0.005; // ETH per epoch (mock)
    const [estimateEpochs, setEstimateEpochs] = useState('10');
    const [estimateBatchSize, setEstimateBatchSize] = useState('32');
    const estimatedTotal = parseFloat(estimateEpochs) * estimatedCostPerEpoch * (parseInt(estimateBatchSize) / 32);

    if (!isConnected) {
        return (
            <div className="flex flex-col items-center justify-center py-24 text-[#555]">
                <Wallet className="w-10 h-10 mb-3 text-[#444]" />
                <p className="text-[13px] mb-1 text-[#666]">Connect your wallet</p>
                <p className="text-[11px] text-[#555]">You need a connected wallet to manage billing</p>
            </div>
        );
    }

    return (
        <div className="space-y-5">
            {/* Balance Cards */}
            <div className="grid grid-cols-3 gap-3">
                <motion.div
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="bg-helix-surface border border-helix-border rounded-md p-4"
                >
                    <span className="text-[11px] text-[#666] uppercase tracking-wider block mb-2">Current Stake</span>
                    <div className="text-xl font-mono font-semibold text-white">
                        {stakeLoading ? '...' : stake ? formatEther(stake.amount) : '0'}
                        <span className="text-[11px] text-[#555] ml-1.5 font-sans font-normal uppercase tracking-wider">ETH</span>
                    </div>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.05 }}
                    className="bg-helix-surface border border-helix-border rounded-md p-4"
                >
                    <span className="text-[11px] text-[#666] uppercase tracking-wider block mb-2">Lock Status</span>
                    <div className="text-xl font-mono font-semibold text-white">
                        {stake?.slashed ? 'SLASHED' : isLocked ? (
                            <>{formatDuration(lockTimeRemaining)}<span className="text-[11px] text-[#555] ml-1.5 font-sans font-normal uppercase tracking-wider">locked</span></>
                        ) : 'Unlocked'}
                    </div>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="bg-helix-surface border border-helix-border rounded-md p-4"
                >
                    <span className="text-[11px] text-[#666] uppercase tracking-wider block mb-2">Min Required</span>
                    <div className="text-xl font-mono font-semibold text-white">
                        {contractState.defaultMinStake !== undefined
                            ? <>{formatEther(contractState.defaultMinStake)}<span className="text-[11px] text-[#555] ml-1.5 font-sans font-normal uppercase tracking-wider">ETH</span></>
                            : '...'}
                    </div>
                </motion.div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
                {/* Stake / Unstake */}
                <div className="bg-helix-surface border border-helix-border rounded-md p-4 space-y-4">
                    {/* Tab switcher */}
                    <div className="flex gap-4 border-b border-helix-border">
                        <button
                            onClick={() => setActiveTab('stake')}
                            className={`pb-2.5 text-[13px] font-medium transition-colors border-b ${
                                activeTab === 'stake'
                                    ? 'text-white border-white'
                                    : 'text-[#555] border-transparent hover:text-[#888]'
                            }`}
                        >
                            Stake
                        </button>
                        <button
                            onClick={() => setActiveTab('unstake')}
                            className={`pb-2.5 text-[13px] font-medium transition-colors border-b ${
                                activeTab === 'unstake'
                                    ? 'text-white border-white'
                                    : 'text-[#555] border-transparent hover:text-[#888]'
                            }`}
                        >
                            Unstake
                        </button>
                    </div>

                    {activeTab === 'stake' ? (
                        <div className="space-y-3">
                            <div>
                                <label className="text-[11px] text-[#666] uppercase tracking-wider block mb-1.5">Amount</label>
                                <div className="flex items-center gap-2">
                                    <input
                                        type="number"
                                        step="0.01"
                                        min="0"
                                        value={stakeAmount}
                                        onChange={(e) => setStakeAmount(e.target.value)}
                                        disabled={pending}
                                        className="flex-1 bg-transparent border border-helix-border rounded-[4px] px-3 py-2 text-[13px] font-mono text-white placeholder:text-[#444] focus:border-white/20 outline-none disabled:opacity-30"
                                    />
                                    <span className="text-[11px] text-[#555] uppercase tracking-wider">ETH</span>
                                </div>
                            </div>

                            <div className="flex gap-1.5">
                                {['0.1', '0.5', '1.0', '2.0'].map((amt) => (
                                    <button
                                        key={amt}
                                        onClick={() => setStakeAmount(amt)}
                                        disabled={pending}
                                        className="flex-1 text-[11px] font-mono text-[#666] border border-helix-border rounded-[4px] px-2 py-1 hover:text-white hover:border-white/10 transition-colors disabled:opacity-30"
                                    >
                                        {amt}
                                    </button>
                                ))}
                            </div>

                            <button
                                onClick={() => stakeTokens(stakeAmount)}
                                disabled={pending || parseFloat(stakeAmount) <= 0}
                                className="w-full bg-white text-black text-[13px] font-medium px-4 py-2 rounded-[4px] hover:bg-white/90 disabled:opacity-30 flex items-center justify-center gap-2 transition-colors"
                            >
                                {pending ? (
                                    <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Staking...</>
                                ) : (
                                    <><ArrowUpRight className="w-3.5 h-3.5" /> Stake {stakeAmount} ETH</>
                                )}
                            </button>
                        </div>
                    ) : (
                        <div className="space-y-3">
                            {isLocked && (
                                <div className="border border-helix-border rounded-[4px] px-3 py-2 text-[13px] text-[#888]">
                                    Stake locked for <span className="font-mono text-white">{formatDuration(lockTimeRemaining)}</span>
                                </div>
                            )}

                            <div className="py-4">
                                <span className="text-[11px] text-[#666] uppercase tracking-wider block mb-1.5">Amount to Unstake</span>
                                <span className="text-xl font-mono font-semibold text-white">
                                    {stake ? formatEther(stake.amount) : '0'}
                                    <span className="text-[11px] text-[#555] ml-1.5 font-sans font-normal uppercase tracking-wider">ETH</span>
                                </span>
                            </div>

                            <button
                                onClick={() => unstakeTokens()}
                                disabled={pending || !stake || stake.amount === BigInt(0) || isLocked || stake.slashed}
                                className="w-full border border-helix-border text-white text-[13px] font-medium px-4 py-2 rounded-[4px] hover:border-white/20 transition-colors disabled:opacity-30 flex items-center justify-center gap-2"
                            >
                                {pending ? (
                                    <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Unstaking...</>
                                ) : (
                                    <><ArrowDownLeft className="w-3.5 h-3.5" /> Unstake All</>
                                )}
                            </button>
                        </div>
                    )}

                    {error && (
                        <div className="border border-helix-border rounded-[4px] px-3 py-2 text-[13px] text-[#888]">
                            {error}
                        </div>
                    )}
                    {txHash && (
                        <div className="border border-helix-border rounded-[4px] px-3 py-2">
                            <span className="text-[13px] text-white">Transaction submitted</span>
                            <a
                                href={`https://etherscan.io/tx/${txHash}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="block text-[11px] text-[#555] hover:text-white transition-colors font-mono mt-0.5"
                            >
                                {txHash.slice(0, 20)}...
                            </a>
                        </div>
                    )}
                </div>

                {/* Cost Estimator */}
                <div className="bg-helix-surface border border-helix-border rounded-md p-4 space-y-4">
                    <span className="text-[11px] text-[#666] uppercase tracking-wider">Cost Estimator</span>

                    <div className="grid grid-cols-2 gap-3">
                        <div>
                            <label className="text-[11px] text-[#666] uppercase tracking-wider block mb-1.5">Epochs</label>
                            <input
                                type="number"
                                value={estimateEpochs}
                                onChange={(e) => setEstimateEpochs(e.target.value)}
                                className="w-full bg-transparent border border-helix-border rounded-[4px] px-3 py-2 text-[13px] font-mono text-white placeholder:text-[#444] focus:border-white/20 outline-none"
                            />
                        </div>
                        <div>
                            <label className="text-[11px] text-[#666] uppercase tracking-wider block mb-1.5">Batch Size</label>
                            <select
                                value={estimateBatchSize}
                                onChange={(e) => setEstimateBatchSize(e.target.value)}
                                className="w-full bg-transparent border border-helix-border rounded-[4px] px-3 py-2 text-[13px] font-mono text-white focus:border-white/20 outline-none cursor-pointer"
                            >
                                <option value="16">16</option>
                                <option value="32">32</option>
                                <option value="64">64</option>
                                <option value="128">128</option>
                            </select>
                        </div>
                    </div>

                    <div className="space-y-2">
                        <div className="flex justify-between items-center border-b border-helix-border pb-2">
                            <span className="text-[11px] text-[#666] uppercase tracking-wider">Cost per epoch</span>
                            <span className="text-[13px] font-mono text-white">{estimatedCostPerEpoch}</span>
                        </div>
                        <div className="flex justify-between items-center border-b border-helix-border pb-2">
                            <span className="text-[11px] text-[#666] uppercase tracking-wider">Batch multiplier</span>
                            <span className="text-[13px] font-mono text-white">{parseInt(estimateBatchSize) / 32}x</span>
                        </div>
                        <div className="flex justify-between items-center pt-1">
                            <span className="text-[11px] text-[#888] uppercase tracking-wider font-medium">Estimated Total</span>
                            <span className="text-xl font-mono font-semibold text-white">{estimatedTotal.toFixed(4)}<span className="text-[11px] text-[#555] ml-1 font-sans font-normal uppercase tracking-wider">ETH</span></span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Transaction History */}
            <div className="bg-helix-surface border border-helix-border rounded-md">
                <div className="px-4 py-3 border-b border-helix-border">
                    <span className="text-[11px] text-[#666] uppercase tracking-wider">Transaction History</span>
                </div>
                <div>
                    {[...stakedEvents, ...slashedEvents].length > 0 ? (
                        [...stakedEvents.map(e => ({ ...e, type: 'Staked' as const })),
                         ...slashedEvents.map(e => ({ ...e, type: 'Slashed' as const }))]
                            .sort((a, b) => b.timestamp - a.timestamp)
                            .slice(0, 10)
                            .map((event, i) => (
                                <div key={i} className="px-4 py-2.5 flex items-center justify-between border-b border-helix-border last:border-b-0">
                                    <div className="flex items-center gap-3">
                                        {event.type === 'Staked' ? (
                                            <ArrowUpRight className="w-3.5 h-3.5 text-[#555]" />
                                        ) : (
                                            <ArrowDownLeft className="w-3.5 h-3.5 text-[#555]" />
                                        )}
                                        <div>
                                            <span className="text-[13px] text-white">{event.type}</span>
                                            <span className="block text-[11px] text-[#555] font-mono">
                                                {new Date(event.timestamp * 1000).toLocaleString()}
                                            </span>
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <span className="text-[13px] text-white font-mono">
                                            {(Number(event.amount) / 1e18).toFixed(4)} ETH
                                        </span>
                                        <a
                                            href={`https://etherscan.io/tx/${event.transactionHash}`}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="block text-[11px] text-[#555] hover:text-white transition-colors font-mono"
                                        >
                                            View tx
                                        </a>
                                    </div>
                                </div>
                            ))
                    ) : (
                        <div className="px-4 py-6 text-center text-[#555] text-[13px]">
                            No transactions yet
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
