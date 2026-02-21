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
  /** Inference fee in basis points (e.g. 500 = 5%) */
  inferenceFee?: number;
  /** Model architecture label */
  architecture?: string;
  /** Total inference count */
  inferenceCount?: number;
  /** Average user rating (0-5) */
  averageRating?: number;
  /** Number of user ratings */
  ratingCount?: number;
  /** Whether to show fee badge (default: true) */
  showFee?: boolean;
}

// ─── Trained Session Card ────────────────────────────────────────────────────

export interface TrainedSessionCardProps {
  sessionId: string;
  name: string;
  accuracy: number | null;
  isSelected: boolean;
  onSelect: () => void;
  /** Whether to show fee badge (default: true) */
  showFee?: boolean;
}

export function TrainedSessionCard({
  name, accuracy, isSelected, onSelect, showFee = true,
}: TrainedSessionCardProps) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        'w-full text-left rounded-2xl transition-all duration-200 relative overflow-hidden group',
        isSelected
          ? 'bg-white/[0.06] backdrop-blur-xl ring-1 ring-white/20 shadow-lg shadow-white/[0.03] px-5 py-4'
          : 'bg-white/[0.02] backdrop-blur-md border border-white/[0.06] hover:bg-white/[0.04] hover:border-white/[0.12] px-5 py-3.5',
      )}
    >
      {/* Top edge gradient */}
      <div className={cn(
        'absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent transition-opacity duration-200',
        isSelected ? 'opacity-100' : 'opacity-0 group-hover:opacity-60',
      )} />

      {/* Subtle inner glow when selected */}
      {isSelected && (
        <div className="absolute inset-0 bg-gradient-to-b from-white/[0.04] to-transparent pointer-events-none" />
      )}

      <div className="relative">
        {/* Row 1: Name + accuracy */}
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0 flex-1">
            <h3 className={cn(
              'font-semibold text-white truncate tracking-tight',
              isSelected ? 'text-2xl' : 'text-xl',
            )}>
              {name || 'Trained Model'}
            </h3>
          </div>
          {accuracy != null && (
            <div className="text-right shrink-0">
              <span className={cn(
                'font-mono tabular-nums text-white block font-semibold',
                isSelected ? 'text-2xl' : 'text-xl text-white/80',
              )}>
                {(accuracy * 100).toFixed(1)}%
              </span>
              <span className="text-sm text-white/50">accuracy</span>
            </div>
          )}
        </div>

        {/* Row 2: Fee badge (conditional) */}
        {showFee && (
          <div className="flex items-center gap-2 mt-3">
            <span className="inline-flex items-center gap-1.5 text-sm font-semibold px-3.5 py-1.5 rounded-full bg-emerald-500/10 text-emerald-400/90 border border-emerald-500/10">
              0% fee
            </span>
            <span className="text-sm text-white/40 ml-auto">Your model</span>
          </div>
        )}
      </div>
    </button>
  );
}

// ─── Version Dropdown (full-width, centered in card) ─────────────────────────

function VersionDropdown({
  versions,
  selectedIndex,
  onChange,
}: {
  versions: ModelCardVersion[];
  selectedIndex: number;
  onChange: (index: number) => void;
}) {
  if (versions.length === 0) return null;

  // Sort by accuracy descending, preserving original indices
  const sorted = versions
    .map((v, i) => ({ ...v, idx: i }))
    .sort((a, b) => b.accuracy - a.accuracy);

  return (
    <div className="w-full mt-3">
      <select
        value={selectedIndex}
        onChange={(e) => { e.stopPropagation(); onChange(Number(e.target.value)); }}
        onClick={(e) => e.stopPropagation()}
        className="w-full bg-white/[0.06] border border-white/[0.10] rounded-xl px-4 py-2.5 text-sm text-white/90 font-medium cursor-pointer focus:outline-none focus:ring-1 focus:ring-white/20 transition-all hover:bg-white/[0.09]"
        style={{ colorScheme: 'dark' }}
      >
        {sorted.map((v) => (
          <option key={v.idx} value={v.idx}>
            {v.semver || `v${v.idx + 1}`}  ·  {(v.accuracy * 100).toFixed(1)}% accuracy
          </option>
        ))}
      </select>
    </div>
  );
}

// ─── Model Card ──────────────────────────────────────────────────────────────

export function ModelCard({
  name, accuracy, isSelected, onSelect,
  versions, selectedVersionIndex, onVersionChange,
  ownerAddress, userAddress, tokenId,
  inferenceFee, architecture, inferenceCount,
  averageRating, ratingCount,
  showFee = true,
}: ModelCardProps) {
  const isOwner = ownerAddress && userAddress
    ? ownerAddress.toLowerCase() === userAddress.toLowerCase()
    : false;

  const sv = versions && selectedVersionIndex != null
    ? versions[selectedVersionIndex] ?? null
    : null;
  const displayAccuracy = isSelected && sv ? sv.accuracy : (accuracy ?? 0);

  // Fee display
  const feeBps = inferenceFee ?? 0;
  const feePercent = isOwner ? 0 : feeBps / 100;

  return (
    <div
      className={cn(
        'w-full rounded-2xl transition-all duration-200 overflow-hidden relative group',
        isSelected
          ? 'bg-white/[0.06] backdrop-blur-xl ring-1 ring-white/20 shadow-lg shadow-white/[0.03]'
          : 'bg-white/[0.02] backdrop-blur-md border border-white/[0.06] hover:bg-white/[0.04] hover:border-white/[0.12]',
      )}
    >
      {/* Top edge gradient */}
      <div className={cn(
        'absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent transition-opacity duration-200 z-10',
        isSelected ? 'opacity-100' : 'opacity-0 group-hover:opacity-60',
      )} />

      {/* Subtle inner glow when selected */}
      {isSelected && (
        <div className="absolute inset-0 bg-gradient-to-b from-white/[0.04] to-transparent pointer-events-none" />
      )}

      <button
        type="button"
        onClick={onSelect}
        className={cn(
          'w-full text-left relative',
          isSelected ? 'px-5 py-4' : 'px-5 py-3.5',
        )}
      >
        {/* Row 1: Name + accuracy */}
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0 flex-1">
            <h3 className={cn(
              'font-semibold text-white truncate tracking-tight',
              isSelected ? 'text-2xl' : 'text-xl',
            )}>
              {name}
            </h3>
            {architecture && (
              <p className="text-sm text-white/40 mt-1 font-medium truncate">{architecture}</p>
            )}
          </div>
          {displayAccuracy > 0 && (
            <div className="text-right shrink-0">
              <span className={cn(
                'font-mono tabular-nums text-white block font-semibold',
                isSelected ? 'text-2xl' : 'text-xl text-white/80',
              )}>
                {(displayAccuracy * 100).toFixed(1)}%
              </span>
              <span className="text-sm text-white/50">accuracy</span>
            </div>
          )}
        </div>

        {/* Version dropdown — shown in the middle when selected */}
        {isSelected && versions && versions.length > 0 && onVersionChange && (
          <VersionDropdown
            versions={versions}
            selectedIndex={selectedVersionIndex ?? versions.length - 1}
            onChange={onVersionChange}
          />
        )}

        {/* Row 2: Fee + metadata badges */}
        <div className="flex items-center gap-2 mt-3 flex-wrap">
          {showFee && (
            isOwner || feePercent === 0 ? (
              <span className="inline-flex items-center gap-1.5 text-sm font-semibold px-3.5 py-1.5 rounded-full bg-emerald-500/10 text-emerald-400/90 border border-emerald-500/10">
                0% fee
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 text-sm font-semibold px-3.5 py-1.5 rounded-full bg-amber-500/10 text-amber-400/80 border border-amber-500/10">
                {feePercent % 1 === 0 ? feePercent.toFixed(0) : feePercent.toFixed(1)}% fee
              </span>
            )
          )}

          {versions && versions.length > 0 && (
            <span className="inline-flex items-center text-sm font-medium px-3 py-1.5 rounded-full bg-white/[0.04] text-white/30 border border-white/[0.05]">
              {versions.length} ver{versions.length !== 1 ? 's' : ''}
            </span>
          )}

          {inferenceCount != null && inferenceCount > 0 && (
            <span className="inline-flex items-center text-sm font-medium px-3 py-1.5 rounded-full bg-white/[0.04] text-white/30 border border-white/[0.05]">
              {inferenceCount} run{inferenceCount !== 1 ? 's' : ''}
            </span>
          )}

          {averageRating != null && averageRating > 0 && (
            <span className="inline-flex items-center gap-1 text-sm font-medium px-3 py-1.5 rounded-full bg-yellow-500/[0.06] text-yellow-400/80 border border-yellow-500/[0.08]">
              ★ {averageRating.toFixed(1)}
            </span>
          )}

          <span className="text-sm text-white/40 ml-auto">
            {isOwner ? 'Your model' : 'Public'}
          </span>
        </div>
      </button>

      {/* Expanded content — actions */}
      {isSelected && tokenId != null && (
        <div className="px-5 pb-4 relative">
          <div className="flex items-center justify-end">
            <Link
              href={`/models/${tokenId}`}
              onClick={(e) => e.stopPropagation()}
              className="h-10 px-5 inline-flex items-center justify-center rounded-lg bg-white text-black text-sm font-semibold hover:bg-white/90 transition-colors shrink-0"
            >
              View
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
