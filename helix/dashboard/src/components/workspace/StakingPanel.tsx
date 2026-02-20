'use client';

import { useState, useCallback } from 'react';
import { Lock, Unlock, AlertTriangle } from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Skeleton } from '@/components/ui/Skeleton';
import { useStake, useStaking } from '@/hooks/useContract';
import { cn } from '@/lib/utils';

interface StakingPanelProps {
  modelId: bigint;
}

function formatEth(wei: bigint): string {
  const eth = Number(wei) / 1e18;
  if (eth === 0) return '0';
  if (eth < 0.0001) return eth.toExponential(2);
  if (eth < 1) return eth.toFixed(4);
  return eth.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function isLocked(lockedUntil: bigint): boolean {
  return Number(lockedUntil) * 1000 > Date.now();
}

function formatLockTime(lockedUntil: bigint): string {
  const lockTimestamp = Number(lockedUntil) * 1000;
  if (lockTimestamp <= Date.now()) return 'Unlocked';
  const remaining = lockTimestamp - Date.now();
  const hours = Math.floor(remaining / 3600000);
  const minutes = Math.floor((remaining % 3600000) / 60000);
  if (hours > 24) {
    const days = Math.floor(hours / 24);
    return `${days}d ${hours % 24}h remaining`;
  }
  if (hours > 0) return `${hours}h ${minutes}m remaining`;
  return `${minutes}m remaining`;
}

export function StakingPanel({ modelId }: StakingPanelProps) {
  const { stake, isLoading } = useStake(Number(modelId));
  const { stakeTokens, unstakeTokens, pending, txHash, error } = useStaking(Number(modelId));
  const [amount, setAmount] = useState('');
  const [actionError, setActionError] = useState<string | null>(null);

  const handleStake = useCallback(async () => {
    setActionError(null);
    const trimmed = amount.trim();
    if (!trimmed || isNaN(Number(trimmed)) || Number(trimmed) <= 0) {
      setActionError('Enter a valid amount');
      return;
    }
    try {
      await stakeTokens(trimmed);
      setAmount('');
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Staking failed');
    }
  }, [amount, stakeTokens]);

  const handleUnstake = useCallback(async () => {
    setActionError(null);
    try {
      await unstakeTokens();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Unstaking failed');
    }
  }, [unstakeTokens]);

  const displayError = actionError || error;

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton height="8rem" />
        <Skeleton height="10rem" />
        <Skeleton height="4rem" />
      </div>
    );
  }

  const locked = stake ? isLocked(stake.lockedUntil) : false;

  return (
    <div className="space-y-6">
      {/* Stake Summary */}
      <Card variant="glass" className="space-y-4">
        <h3 className="text-2xs font-mono uppercase tracking-wider text-helix-muted">
          Stake Summary
        </h3>

        <div className="grid grid-cols-3 gap-6">
          {/* Current Stake */}
          <div className="space-y-1">
            <p className="text-2xs text-helix-muted uppercase tracking-wider">
              Current Stake
            </p>
            <p className="text-2xl font-light tracking-tight text-helix-text">
              {stake ? formatEth(stake.amount) : '0'}
            </p>
            <p className="text-2xs text-helix-dim font-mono">ADI</p>
          </div>

          {/* Lock Status */}
          <div className="space-y-1">
            <p className="text-2xs text-helix-muted uppercase tracking-wider">
              Lock Status
            </p>
            <div className="flex items-center gap-2 mt-1">
              {locked ? (
                <>
                  <Lock size={14} className="text-helix-muted" />
                  <Badge variant="outline">Locked</Badge>
                </>
              ) : (
                <>
                  <Unlock size={14} className="text-helix-dim" />
                  <Badge variant="default">Unlocked</Badge>
                </>
              )}
            </div>
            {stake && (
              <p className="text-2xs text-helix-dim font-mono mt-1">
                {formatLockTime(stake.lockedUntil)}
              </p>
            )}
          </div>

          {/* Slashed Status */}
          <div className="space-y-1">
            <p className="text-2xs text-helix-muted uppercase tracking-wider">
              Slashed
            </p>
            {stake?.slashed ? (
              <div className="flex items-center gap-2 mt-1">
                <AlertTriangle size={14} className="text-helix-muted" />
                <Badge className="bg-white/[0.06] text-helix-text2">
                  Slashed
                </Badge>
              </div>
            ) : (
              <p className="text-sm text-helix-text2 mt-1">None</p>
            )}
          </div>
        </div>
      </Card>

      {/* Stake/Unstake Actions */}
      <Card variant="default" className="space-y-4">
        <h3 className="text-2xs font-mono uppercase tracking-wider text-helix-muted">
          Manage Stake
        </h3>

        {/* Input */}
        <div className="space-y-2">
          <label
            htmlFor="stake-amount"
            className="text-2xs text-helix-muted uppercase tracking-wider"
          >
            Amount (ADI)
          </label>
          <input
            id="stake-amount"
            type="text"
            value={amount}
            onChange={(e) => setAmount(e.target.value)}
            placeholder="0.0"
            disabled={pending}
            className={cn(
              'w-full bg-transparent border border-helix-border rounded-lg px-3 py-2',
              'text-sm font-mono text-helix-text placeholder:text-helix-dim',
              'focus:outline-none focus:border-helix-border2 transition-colors',
              'disabled:opacity-50',
            )}
          />
        </div>

        {/* Buttons */}
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={handleStake}
            disabled={pending || !amount.trim()}
            className={cn(
              'flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
              'bg-white text-black',
              'hover:bg-white/90',
              'disabled:opacity-40 disabled:cursor-not-allowed',
            )}
          >
            {pending ? 'Processing...' : 'Stake'}
          </button>
          <button
            type="button"
            onClick={handleUnstake}
            disabled={pending || locked || !stake || stake.amount === BigInt(0)}
            className={cn(
              'flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
              'border border-helix-border text-white',
              'hover:border-helix-border2',
              'disabled:opacity-40 disabled:cursor-not-allowed',
            )}
          >
            {pending ? 'Processing...' : 'Unstake'}
          </button>
        </div>

        {/* TX Hash */}
        {txHash && (
          <p className="text-2xs text-helix-dim font-mono break-all">
            tx: {txHash}
          </p>
        )}

        {/* Error */}
        {displayError && (
          <p className="text-2xs text-helix-muted">
            {displayError}
          </p>
        )}
      </Card>

      {/* Staking Rewards */}
      <Card variant="default" className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-2xs font-mono uppercase tracking-wider text-helix-muted">
            Staking Rewards
          </h3>
          <span className="text-2xs font-mono px-2 py-0.5 rounded-full bg-white/[0.06] text-helix-dim">
            Coming Soon
          </span>
        </div>
        <p className="text-2xs text-helix-dim py-4">
          Earn rewards for staking on models. Reward distribution and claiming will be available in a future update.
        </p>
      </Card>

      {/* Staking History Placeholder */}
      <Card variant="default" className="space-y-3">
        <h3 className="text-2xs font-mono uppercase tracking-wider text-helix-muted">
          Staking History
        </h3>
        <p className="text-2xs text-helix-dim py-4">
          Transaction history coming soon
        </p>
      </Card>
    </div>
  );
}
