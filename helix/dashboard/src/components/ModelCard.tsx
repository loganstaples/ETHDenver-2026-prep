'use client';

import Link from 'next/link';
import { cn } from '@/lib/utils';

// ─── Types ───────────────────────────────────────────────────────────────────

export interface ModelCardVersion {
  semver: string;
  accuracy: number;
  weightsStored?: boolean;
  rootHash?: string;
  sessionId?: string;
  timestamp?: number;
}

export interface ModelCardProps {
  /** Unique key for selection tracking */
  id: string;
  name: string;
  accuracy: number | null;
  isSelected: boolean;
  onSelect: () => void;
  /** On-chain model versions (omit for trained sessions) */
  versions?: ModelCardVersion[];
  /** Currently selected version index */
  selectedVersionIndex?: number;
  /** Called when version changes */
  onVersionChange?: (index: number) => void;
  /** Owner address of the on-chain model */
  ownerAddress?: string;
  /** Current user address */
  userAddress?: string;
  /** On-chain token ID for linking */
  tokenId?: number;
}

// ─── Trained Session Card ────────────────────────────────────────────────────

export interface TrainedSessionCardProps {
  sessionId: string;
  name: string;
  accuracy: number | null;
  isSelected: boolean;
  onSelect: () => void;
}

export function TrainedSessionCard({
  name, accuracy, isSelected, onSelect,
}: TrainedSessionCardProps) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        'w-full text-left rounded-2xl transition-all',
        isSelected
          ? 'bg-white/[0.07] ring-1 ring-white/20 px-6 py-6'
          : 'bg-helix-surface border border-helix-border hover:border-helix-border2 px-5 py-4',
      )}
    >
      <div className="flex items-baseline justify-between gap-4">
        <h3 className={cn(
          'font-semibold text-white truncate tracking-tight',
          isSelected ? 'text-xl' : 'text-[15px]',
        )}>
          {name || 'Trained Model'}
        </h3>
        {accuracy != null && (
          <span className={cn(
            'font-mono tabular-nums text-white shrink-0',
            isSelected ? 'text-xl font-semibold' : 'text-sm font-medium text-white/60',
          )}>
            {(accuracy * 100).toFixed(1)}%
          </span>
        )}
      </div>
    </button>
  );
}

// ─── On-Chain Model Card ─────────────────────────────────────────────────────

export function ModelCard({
  name, accuracy, isSelected, onSelect,
  versions, selectedVersionIndex, onVersionChange,
  ownerAddress, userAddress, tokenId,
}: ModelCardProps) {
  const isOwner = ownerAddress && userAddress
    ? ownerAddress.toLowerCase() === userAddress.toLowerCase()
    : false;

  const sv = versions && selectedVersionIndex != null
    ? versions[selectedVersionIndex] ?? null
    : null;
  const displayAccuracy = isSelected && sv ? sv.accuracy : (accuracy ?? 0);
  const hasMultipleVersions = versions && versions.length > 1;

  return (
    <div
      className={cn(
        'w-full rounded-2xl transition-all overflow-hidden',
        isSelected
          ? 'bg-white/[0.07] ring-1 ring-white/20'
          : 'bg-helix-surface border border-helix-border hover:border-helix-border2',
      )}
    >
      <button
        type="button"
        onClick={onSelect}
        className={cn(
          'w-full text-left',
          isSelected ? 'px-6 py-6' : 'px-5 py-4',
        )}
      >
        <div className="flex items-baseline justify-between gap-4">
          <h3 className={cn(
            'font-semibold text-white truncate tracking-tight',
            isSelected ? 'text-xl' : 'text-[15px]',
          )}>
            {name}
          </h3>
          {displayAccuracy > 0 && (
            <span className={cn(
              'font-mono tabular-nums text-white shrink-0',
              isSelected ? 'text-xl font-semibold' : 'text-sm font-medium text-white/60',
            )}>
              {(displayAccuracy * 100).toFixed(1)}%
            </span>
          )}
        </div>
      </button>

      {/* Expanded content */}
      {isSelected && (
        <div className="px-6 pb-5 flex items-center gap-2">
          {/* Version dropdown */}
          {hasMultipleVersions && onVersionChange && (
            <select
              value={selectedVersionIndex ?? 0}
              onChange={(e) => onVersionChange(Number(e.target.value))}
              className="h-8 px-3 bg-white/[0.05] border border-white/[0.08] rounded-lg text-xs text-white/70 focus:outline-none focus:border-white/20 transition-colors appearance-none cursor-pointer"
              style={{
                backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 24 24' fill='none' stroke='%23888' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E")`,
                backgroundRepeat: 'no-repeat',
                backgroundPosition: 'right 8px center',
                paddingRight: '28px',
              }}
            >
              {versions!.map((v, i) => (
                <option key={i} value={i}>
                  {v.semver || `v${i + 1}`}{i === versions!.length - 1 ? ' (latest)' : ''}
                </option>
              ))}
            </select>
          )}

          {/* Spacer */}
          <div className="flex-1" />

          {/* View / Manage button */}
          {tokenId != null && (
            <Link
              href={isOwner ? `/my-models/${tokenId}` : `/models/${tokenId}`}
              onClick={(e) => e.stopPropagation()}
              className="h-8 px-4 inline-flex items-center justify-center rounded-lg bg-white text-black text-xs font-semibold hover:bg-white/90 transition-colors shrink-0"
            >
              {isOwner ? 'Manage' : 'View'}
            </Link>
          )}
        </div>
      )}
    </div>
  );
}
