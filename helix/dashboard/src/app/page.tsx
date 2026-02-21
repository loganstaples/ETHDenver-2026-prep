'use client';

import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Boxes,
  Search,
  Sparkles,
  Layers,
  User,
  Loader2,
  Globe,
  Lock,
  ArrowUpDown,
  X,
  ShoppingCart,
  DollarSign,
  Store,
  Star,
  Copy,
  CheckCircle,
  ChevronDown,
} from 'lucide-react';
import { useAccount } from 'wagmi';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { EmptyState } from '@/components/ui/EmptyState';
import { cn } from '@/lib/utils';
import { usePublicModels, type PublicModel, type ModelFilter } from '@/hooks/usePublicModels';
import { useModelRegistry } from '@/hooks/useModelRegistry';

// ============================================================================
// Helpers
// ============================================================================

function truncateAddress(addr: string): string {
  if (addr.length <= 12) return addr;
  return `${addr.slice(0, 6)}...${addr.slice(-4)}`;
}

function formatDate(timestamp: number): string {
  if (!timestamp) return '--';
  return new Date(timestamp * 1000).toLocaleDateString();
}

function formatFee(bps: number): string {
  if (bps === 0) return 'Free';
  return `${(bps / 100).toFixed(bps % 100 === 0 ? 0 : 1)}%`;
}

function formatAdiPrice(adi: number): string {
  if (adi === 0) return 'Free';
  if (adi < 0.001) return `${adi.toExponential(2)} ADI`;
  if (adi < 1) return `${adi.toFixed(4)} ADI`;
  return `${adi.toFixed(2)} ADI`;
}

// ============================================================================
// Mode & Sort & Category
// ============================================================================

type PageMode = 'inference' | 'marketplace';

type SortOption = 'newest' | 'oldest' | 'accuracy' | 'versions' | 'name' | 'price-low' | 'price-high';

const INFERENCE_SORT_OPTIONS: { id: SortOption; label: string }[] = [
  { id: 'newest', label: 'Newest' },
  { id: 'oldest', label: 'Oldest' },
  { id: 'accuracy', label: 'Best Accuracy' },
  { id: 'versions', label: 'Most Versions' },
  { id: 'name', label: 'Name A-Z' },
];

const MARKETPLACE_SORT_OPTIONS: { id: SortOption; label: string }[] = [
  { id: 'newest', label: 'Newest' },
  { id: 'oldest', label: 'Oldest' },
  { id: 'price-low', label: 'Price Low-High' },
  { id: 'price-high', label: 'Price High-Low' },
  { id: 'accuracy', label: 'Best Accuracy' },
  { id: 'versions', label: 'Most Versions' },
  { id: 'name', label: 'Name A-Z' },
];

function sortModels<T extends { createdAt: number; bestAccuracy: number; versionCount: number; name: string; salePrice: number }>(
  models: T[],
  sort: SortOption,
): T[] {
  const sorted = [...models];
  switch (sort) {
    case 'newest': return sorted.sort((a, b) => b.createdAt - a.createdAt);
    case 'oldest': return sorted.sort((a, b) => a.createdAt - b.createdAt);
    case 'accuracy': return sorted.sort((a, b) => b.bestAccuracy - a.bestAccuracy);
    case 'versions': return sorted.sort((a, b) => b.versionCount - a.versionCount);
    case 'name': return sorted.sort((a, b) => a.name.localeCompare(b.name));
    case 'price-low': return sorted.sort((a, b) => a.salePrice - b.salePrice);
    case 'price-high': return sorted.sort((a, b) => b.salePrice - a.salePrice);
    default: return sorted;
  }
}

function getModelTags(model: { name: string; description: string; slug: string }): string[] {
  const text = `${model.name} ${model.description} ${model.slug}`.toLowerCase();
  if (/mnist|digit|784/.test(text)) return ['MNIST'];
  return [];
}

function RatingStars({ rating, count }: { rating: number; count: number }) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex items-center gap-0.5">
        {Array.from({ length: 5 }, (_, i) => {
          const fill = Math.min(1, Math.max(0, rating - i));
          return (
            <div key={i} className="relative w-4 h-4">
              <Star size={16} className="text-white/10 absolute inset-0" />
              {fill > 0 && (
                <div className="absolute inset-0 overflow-hidden" style={{ width: `${fill * 100}%` }}>
                  <Star size={16} className="text-white fill-white" />
                </div>
              )}
            </div>
          );
        })}
      </div>
      <span className="text-sm text-helix-dim font-mono tabular-nums">
        {count > 0 ? `${rating.toFixed(1)}` : 'N/A'}
      </span>
    </div>
  );
}

// ============================================================================
// Mode Toggle
// ============================================================================

function ModeToggle({
  mode,
  onChange,
}: {
  mode: PageMode;
  onChange: (m: PageMode) => void;
}) {
  return (
    <div className="flex items-center gap-1 p-1.5 bg-helix-bg rounded-xl border border-helix-border">
      <button
        type="button"
        onClick={() => onChange('inference')}
        className={cn(
          'relative flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all',
          mode === 'inference'
            ? 'text-white'
            : 'text-helix-muted hover:text-helix-text2',
        )}
      >
        {mode === 'inference' && (
          <motion.div
            layoutId="mode-toggle-bg"
            className="absolute inset-0 bg-helix-surface border border-helix-border rounded-lg"
            transition={{ type: 'spring', stiffness: 500, damping: 35 }}
          />
        )}
        <Sparkles size={16} className="relative z-10" />
        <span className="relative z-10">Inference</span>
      </button>
      <button
        type="button"
        onClick={() => onChange('marketplace')}
        className={cn(
          'relative flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all',
          mode === 'marketplace'
            ? 'text-white'
            : 'text-helix-muted hover:text-helix-text2',
        )}
      >
        {mode === 'marketplace' && (
          <motion.div
            layoutId="mode-toggle-bg"
            className="absolute inset-0 bg-helix-surface border border-helix-border rounded-lg"
            transition={{ type: 'spring', stiffness: 500, damping: 35 }}
          />
        )}
        <Store size={16} className="relative z-10" />
        <span className="relative z-10">Marketplace</span>
      </button>
    </div>
  );
}

// ============================================================================
// Filter Tabs
// ============================================================================

const FILTERS: { id: ModelFilter; label: string; description: string }[] = [
  { id: 'all', label: 'All Public', description: 'All public models' },
  { id: 'others', label: 'By Others', description: 'Models you don\'t own' },
  { id: 'mine', label: 'My Public', description: 'Your public models' },
];

const MARKETPLACE_FILTERS: { id: ModelFilter; label: string; description: string }[] = [
  { id: 'all', label: 'All For Sale', description: 'All models for sale' },
  { id: 'others', label: 'By Others', description: 'Sale models you don\'t own' },
  { id: 'mine', label: 'Mine', description: 'Your models for sale' },
];

function FilterTabs({
  active,
  onChange,
  counts,
  filters,
}: {
  active: ModelFilter;
  onChange: (f: ModelFilter) => void;
  counts: Record<ModelFilter, number>;
  filters: { id: ModelFilter; label: string; description: string }[];
}) {
  return (
    <div className="flex items-center gap-1 p-1.5 bg-helix-bg rounded-xl border border-helix-border">
      {filters.map((f) => (
        <button
          key={f.id}
          type="button"
          onClick={() => onChange(f.id)}
          className={cn(
            'relative flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all',
            active === f.id
              ? 'text-white'
              : 'text-helix-muted hover:text-helix-text2',
          )}
        >
          {active === f.id && (
            <motion.div
              layoutId="filter-tab-bg"
              className="absolute inset-0 bg-helix-surface border border-helix-border rounded-lg"
              transition={{ type: 'spring', stiffness: 500, damping: 35 }}
            />
          )}
          <span className="relative z-10">{f.label}</span>
          <span className={cn(
            'relative z-10 text-sm font-mono px-2 py-0.5 rounded-full',
            active === f.id ? 'bg-white/10 text-white' : 'bg-helix-border text-helix-muted',
          )}>
            {counts[f.id]}
          </span>
        </button>
      ))}
    </div>
  );
}

// ============================================================================
// Buy Confirmation Modal
// ============================================================================

function BuyConfirmationModal({
  model,
  onConfirm,
  onCancel,
  isPending,
  isConfirming,
}: {
  model: PublicModel;
  onConfirm: () => void;
  onCancel: () => void;
  isPending: boolean;
  isConfirming: boolean;
}) {
  const isBusy = isPending || isConfirming;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget && !isBusy) onCancel(); }}
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        className="bg-helix-surface border border-helix-border rounded-2xl p-7 max-w-md w-full mx-4 shadow-2xl"
      >
        <div className="flex items-center gap-4 mb-5">
          <div className="w-12 h-12 rounded-xl bg-green-500/10 flex items-center justify-center">
            <ShoppingCart size={22} className="text-green-400" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-white">Confirm Purchase</h3>
            <p className="text-sm text-helix-muted">This action is irreversible</p>
          </div>
        </div>

        <div className="bg-helix-bg rounded-xl p-5 mb-5 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-helix-muted">Model</span>
            <span className="text-sm text-white font-medium">{model.name}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-helix-muted">Token ID</span>
            <span className="text-sm text-white font-mono">#{model.tokenId}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-helix-muted">Owner</span>
            <span className="text-sm text-white font-mono">{truncateAddress(model.owner)}</span>
          </div>
          <div className="border-t border-helix-border my-2" />
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-helix-muted">Price</span>
            <span className="text-lg font-mono font-medium text-green-400">{formatAdiPrice(model.salePrice)}</span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onCancel}
            disabled={isBusy}
            className="flex-1 px-4 py-2.5 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text hover:text-white transition-colors disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={isBusy}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-green-600 hover:bg-green-500 text-white text-sm font-medium transition-colors disabled:opacity-50"
          >
            {isBusy ? (
              <>
                <Loader2 size={14} className="animate-spin" />
                {isConfirming ? 'Confirming...' : 'Sending...'}
              </>
            ) : (
              <>
                <ShoppingCart size={14} />
                Buy for {formatAdiPrice(model.salePrice)}
              </>
            )}
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
}

// ============================================================================
// Rating Modal
// ============================================================================

function RatingModal({
  model,
  onSubmit,
  onCancel,
}: {
  model: PublicModel;
  onSubmit: (rating: number) => void;
  onCancel: () => void;
}) {
  const [hoveredStar, setHoveredStar] = useState(0);
  const [selectedStar, setSelectedStar] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async () => {
    if (selectedStar === 0) return;
    setIsSubmitting(true);
    onSubmit(selectedStar);
  };

  const displayRating = hoveredStar || selectedStar;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget && !isSubmitting) onCancel(); }}
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        className="bg-helix-surface border border-helix-border rounded-2xl p-7 max-w-sm w-full mx-4 shadow-2xl"
      >
        <div className="flex items-center gap-4 mb-5">
          <div className="w-12 h-12 rounded-xl bg-white/[0.06] flex items-center justify-center">
            <Star size={22} className="text-white" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-white">Rate Model</h3>
            <p className="text-sm text-helix-muted">{model.name}</p>
          </div>
        </div>

        <div className="flex items-center justify-center gap-2 py-6">
          {Array.from({ length: 5 }, (_, i) => {
            const starNum = i + 1;
            const isFilled = starNum <= displayRating;
            return (
              <button
                key={i}
                type="button"
                onMouseEnter={() => setHoveredStar(starNum)}
                onMouseLeave={() => setHoveredStar(0)}
                onClick={() => setSelectedStar(starNum)}
                className="transition-transform hover:scale-110"
              >
                <Star
                  size={32}
                  className={cn(
                    'transition-colors',
                    isFilled ? 'text-white fill-white' : 'text-white/20',
                  )}
                />
              </button>
            );
          })}
        </div>

        <p className="text-center text-sm text-helix-muted mb-4">
          {displayRating === 0 ? 'Select a rating' : `${displayRating} / 5`}
        </p>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onCancel}
            disabled={isSubmitting}
            className="flex-1 px-4 py-2.5 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text hover:text-white transition-colors disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleSubmit}
            disabled={selectedStar === 0 || isSubmitting}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-white hover:bg-white/90 text-black text-sm font-medium transition-colors disabled:opacity-50"
          >
            {isSubmitting ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <Star size={14} />
            )}
            Submit Rating
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
}

// ============================================================================
// Public Model Card (Inference Mode)
// ============================================================================

function InferenceModelCard({ model, isOwner, onRate }: { model: PublicModel; isOwner: boolean; onRate: (model: PublicModel) => void }) {
  const tags = useMemo(() => getModelTags(model), [model]);

  return (
    <Link href={`/models/${model.tokenId}`} className="block h-full">
      <Card variant="glass" hover className="cursor-pointer flex flex-col h-full">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-white/[0.08] to-white/[0.02] flex items-center justify-center shrink-0 border border-white/[0.06]">
              <Layers size={22} className="text-white/80" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white leading-tight">{model.name}</h3>
              <p className="text-sm font-mono text-helix-muted mt-0.5">{model.slug}</p>
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {model.inferenceFee > 0 && (
              <Badge variant="default" className="text-sm">
                {formatFee(model.inferenceFee)} fee
              </Badge>
            )}
            {isOwner && (
              <Badge variant="outline" className="text-sm text-green-400 border-green-500/30">
                Yours
              </Badge>
            )}
          </div>
        </div>

        {/* Tags */}
        {tags.length > 0 && (
          <div className="flex items-center gap-2 mb-4">
            {tags.map((tag) => (
              <span key={tag} className="px-3 py-1 text-sm rounded-full bg-white/[0.05] text-helix-text2 border border-white/[0.08]">
                {tag}
              </span>
            ))}
          </div>
        )}

        {/* Description */}
        {model.description && (
          <p className="text-sm text-helix-muted mb-4 line-clamp-2 leading-relaxed">{model.description}</p>
        )}

        {/* Meta row */}
        <div className="flex items-center gap-4 text-sm text-helix-dim mb-5">
          <span className="flex items-center gap-1.5">
            <User size={12} />
            {truncateAddress(model.creator)}
          </span>
          <span>{formatDate(model.createdAt)}</span>
          {model.latestVersion && (
            <span className="font-mono">v{model.latestVersion.semver}</span>
          )}
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
          <div className="bg-white/[0.03] rounded-lg px-4 py-3 text-center border border-white/[0.04]">
            <p className="text-sm text-helix-muted mb-1">Versions</p>
            <p className="text-xl font-mono font-medium text-white tabular-nums">{model.versionCount}</p>
          </div>
          <div className="bg-white/[0.03] rounded-lg px-4 py-3 text-center border border-white/[0.04]">
            <p className="text-sm text-helix-muted mb-1">Accuracy</p>
            <p className="text-xl font-mono font-medium text-white tabular-nums">
              {model.bestAccuracy > 0 ? `${(model.bestAccuracy * 100).toFixed(1)}%` : '--'}
            </p>
          </div>
          <div className="bg-white/[0.03] rounded-lg px-4 py-3 text-center border border-white/[0.04]">
            <p className="text-sm text-helix-muted mb-1">Fee</p>
            <p className="text-xl font-mono font-medium text-white tabular-nums">{formatFee(model.inferenceFee)}</p>
          </div>
          <div
            className="bg-white/[0.03] rounded-lg px-4 py-3 text-center border border-white/[0.04] cursor-pointer hover:bg-white/[0.06] hover:border-white/[0.08] transition-all"
            onClick={(e) => { e.preventDefault(); e.stopPropagation(); onRate(model); }}
          >
            <p className="text-sm text-helix-muted mb-1">Rating</p>
            <div className="flex justify-center mt-1">
              <RatingStars rating={model.averageRating} count={model.ratingCount} />
            </div>
          </div>
        </div>

        {/* Actions - pinned to bottom */}
        <div className="flex items-center gap-3 pt-4 border-t border-white/[0.06] mt-auto">
          <span className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-white text-black text-sm font-semibold">
            <Sparkles size={15} />
            View Model
          </span>
          {isOwner && (
            <span className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text2">
              <Layers size={15} />
              Yours
            </span>
          )}
        </div>
      </Card>
    </Link>
  );
}

// ============================================================================
// Contact Owner Button (copies address to clipboard)
// ============================================================================

function ContactOwnerButton({ ownerAddress }: { ownerAddress: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(ownerAddress);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button
      type="button"
      onClick={handleCopy}
      className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-helix-surface border border-helix-border text-sm font-medium text-helix-text hover:text-white hover:border-helix-border2 transition-all"
    >
      {copied ? (
        <>
          <CheckCircle size={14} className="text-green-400" />
          <span className="text-green-400">Copied!</span>
        </>
      ) : (
        <>
          <Copy size={14} />
          Copy Owner Address
        </>
      )}
    </button>
  );
}

// ============================================================================
// Marketplace Model Card
// ============================================================================

function MarketplaceModelCard({
  model,
  isOwner,
  onBuy,
  onRate,
}: {
  model: PublicModel;
  isOwner: boolean;
  onBuy: (model: PublicModel) => void;
  onRate: (model: PublicModel) => void;
}) {
  const tags = useMemo(() => getModelTags(model), [model]);
  const isContactOwner = model.forSale && model.salePrice === 0;

  return (
    <Card variant="glass" hover className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-green-500/15 to-green-500/[0.03] flex items-center justify-center shrink-0 border border-green-500/20">
            <Store size={22} className="text-green-400" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white leading-tight">{model.name}</h3>
            <p className="text-sm font-mono text-helix-muted mt-0.5">{model.slug}</p>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Badge variant="outline" className="text-sm text-green-400 border-green-500/30">
            For Sale
          </Badge>
          {isOwner && (
            <Badge variant="outline" className="text-sm text-blue-400 border-blue-500/30">
              Yours
            </Badge>
          )}
        </div>
      </div>

      {/* Sale Price - prominent display */}
      <div className="bg-gradient-to-r from-green-500/[0.08] to-green-500/[0.02] border border-green-500/20 rounded-xl px-5 py-4 mb-4">
        <div className="flex items-center justify-between">
          <span className="text-sm text-green-300/70 flex items-center gap-2">
            <DollarSign size={16} />
            Sale Price
          </span>
          <span className="text-2xl font-mono font-semibold text-green-400">
            {isContactOwner ? 'Contact Owner' : formatAdiPrice(model.salePrice)}
          </span>
        </div>
      </div>

      {/* Tags */}
      {tags.length > 0 && (
        <div className="flex items-center gap-2 mb-4">
          {tags.map((tag) => (
            <span key={tag} className="px-3 py-1 text-sm rounded-full bg-white/[0.05] text-helix-text2 border border-white/[0.08]">
              {tag}
            </span>
          ))}
        </div>
      )}

      {/* Description */}
      {model.description && (
        <p className="text-sm text-helix-muted mb-4 line-clamp-2 leading-relaxed">{model.description}</p>
      )}

      {/* Meta row */}
      <div className="flex items-center gap-4 text-sm text-helix-dim mb-5">
        <span className="flex items-center gap-1.5">
          <User size={12} />
          {truncateAddress(model.creator)}
        </span>
        <span>{formatDate(model.createdAt)}</span>
        {model.latestVersion && (
          <span className="font-mono">v{model.latestVersion.semver}</span>
        )}
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
        <div className="bg-white/[0.03] rounded-lg px-4 py-3 text-center border border-white/[0.04]">
          <p className="text-sm text-helix-muted mb-1">Versions</p>
          <p className="text-xl font-mono font-medium text-white tabular-nums">{model.versionCount}</p>
        </div>
        <div className="bg-white/[0.03] rounded-lg px-4 py-3 text-center border border-white/[0.04]">
          <p className="text-sm text-helix-muted mb-1">Accuracy</p>
          <p className="text-xl font-mono font-medium text-white tabular-nums">
            {model.bestAccuracy > 0 ? `${(model.bestAccuracy * 100).toFixed(1)}%` : '--'}
          </p>
        </div>
        <div className="bg-white/[0.03] rounded-lg px-4 py-3 text-center border border-white/[0.04]">
          <p className="text-sm text-helix-muted mb-1">Inferences</p>
          <p className="text-xl font-mono font-medium text-white tabular-nums">
            {model.inferenceCount > 0 ? model.inferenceCount : '--'}
          </p>
        </div>
        <div
          className="bg-white/[0.03] rounded-lg px-4 py-3 text-center border border-white/[0.04] cursor-pointer hover:bg-white/[0.06] hover:border-white/[0.08] transition-all"
          onClick={() => onRate(model)}
        >
          <p className="text-sm text-helix-muted mb-1">Rating</p>
          <div className="flex justify-center mt-1">
            <RatingStars rating={model.averageRating} count={model.ratingCount} />
          </div>
        </div>
      </div>

      {/* Actions - pinned to bottom */}
      <div className="flex items-center gap-3 pt-4 border-t border-white/[0.06] mt-auto">
        {!isOwner && !isContactOwner && (
          <button
            type="button"
            onClick={(e) => {
              e.preventDefault();
              onBuy(model);
            }}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-green-600 hover:bg-green-500 text-white text-sm font-semibold transition-colors"
          >
            <ShoppingCart size={15} />
            Buy
          </button>
        )}
        {!isOwner && isContactOwner && (
          <ContactOwnerButton ownerAddress={model.owner} />
        )}
        <Link
          href={`/models/${model.tokenId}`}
          className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-white text-black text-sm font-semibold hover:bg-white/90 transition-colors"
        >
          <Sparkles size={15} />
          View Model
        </Link>
        {isOwner && (
          <span className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text2">
            <Layers size={15} />
            Yours
          </span>
        )}
      </div>
    </Card>
  );
}

// ============================================================================
// Main Page
// ============================================================================

export default function ModelsPage() {
  const [mode, setMode] = useState<PageMode>('inference');
  const [filter, setFilter] = useState<ModelFilter>('all');
  const [search, setSearch] = useState('');
  const [sort, setSort] = useState<SortOption>('newest');
  const [categoryFilter, setCategoryFilter] = useState<string | null>(null);
  const [buyingModel, setBuyingModel] = useState<PublicModel | null>(null);
  const [ratingModel, setRatingModel] = useState<PublicModel | null>(null);
  const [sortOpen, setSortOpen] = useState(false);
  const sortRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (sortRef.current && !sortRef.current.contains(e.target as Node)) {
        setSortOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const { address: walletAddress, isConnected: isWalletConnected } = useAccount();

  const {
    address,
    isConnected,
    isContractDeployed,
    allModels,
    isLoading,
    refetch,
  } = usePublicModels();

  const {
    buyModel,
    isWritePending,
    isConfirming,
  } = useModelRegistry();

  // Reset sort if switching modes and current sort is mode-specific
  const handleModeChange = useCallback((newMode: PageMode) => {
    setMode(newMode);
    if (newMode === 'inference' && (sort === 'price-low' || sort === 'price-high')) {
      setSort('newest');
    }
  }, [sort]);

  // Only show public models on this page
  const publicModels = useMemo(
    () => allModels.filter((m) => m.isPublic),
    [allModels],
  );

  // For marketplace mode, filter to forSale models
  const baseModels = useMemo(() => {
    if (mode === 'marketplace') {
      return publicModels.filter((m) => m.forSale);
    }
    return publicModels;
  }, [publicModels, mode]);

  // Derive available categories from base models
  const availableCategories = useMemo(() => {
    const tagCounts = new Map<string, number>();
    for (const m of baseModels) {
      for (const tag of getModelTags(m)) {
        tagCounts.set(tag, (tagCounts.get(tag) ?? 0) + 1);
      }
    }
    return Array.from(tagCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([tag, count]) => ({ tag, count }));
  }, [baseModels]);

  // Apply filter + search + category + sort
  const filteredModels = useMemo(() => {
    let models = baseModels;

    // Filter by ownership
    if (filter === 'others') {
      models = models.filter((m) => !address || m.owner.toLowerCase() !== address.toLowerCase());
    } else if (filter === 'mine') {
      models = models.filter((m) => address && m.owner.toLowerCase() === address.toLowerCase());
    }

    // Filter by category
    if (categoryFilter) {
      models = models.filter((m) => getModelTags(m).includes(categoryFilter));
    }

    // Search
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      models = models.filter(
        (m) =>
          m.name.toLowerCase().includes(q) ||
          m.slug.toLowerCase().includes(q) ||
          m.description.toLowerCase().includes(q) ||
          m.creator.toLowerCase().includes(q),
      );
    }

    // Sort
    return sortModels(models, sort);
  }, [baseModels, filter, search, address, sort, categoryFilter]);

  // Counts for filter tabs
  const counts = useMemo(() => ({
    all: baseModels.length,
    others: baseModels.filter((m) => !address || m.owner.toLowerCase() !== address.toLowerCase()).length,
    mine: baseModels.filter((m) => address && m.owner.toLowerCase() === address.toLowerCase()).length,
  }), [baseModels, address]);

  // Sort options based on mode
  const sortOptions = mode === 'marketplace' ? MARKETPLACE_SORT_OPTIONS : INFERENCE_SORT_OPTIONS;
  const filterLabels = mode === 'marketplace' ? MARKETPLACE_FILTERS : FILTERS;

  // Handle buy action
  const handleBuyClick = useCallback((model: PublicModel) => {
    setBuyingModel(model);
  }, []);

  const handleBuyConfirm = useCallback(() => {
    if (!buyingModel) return;
    buyModel({ tokenId: buyingModel.tokenId, priceEth: buyingModel.salePrice });
  }, [buyingModel, buyModel]);

  const handleBuyCancel = useCallback(() => {
    if (!isWritePending && !isConfirming) {
      setBuyingModel(null);
    }
  }, [isWritePending, isConfirming]);

  // Handle rating
  const handleRateClick = useCallback((model: PublicModel) => {
    if (!isWalletConnected) return;
    setRatingModel(model);
  }, [isWalletConnected]);

  const handleRateSubmit = useCallback(async (rating: number) => {
    if (!ratingModel || !walletAddress) return;
    try {
      const res = await fetch(`/api/models/${ratingModel.tokenId}/rate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ wallet: walletAddress, rating }),
      });
      if (res.ok) {
        refetch();
      }
    } catch {
      // best-effort
    }
    setRatingModel(null);
  }, [ratingModel, walletAddress, refetch]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <h1 className="text-3xl font-bold text-helix-text tracking-tight">Marketplace</h1>
            <Badge variant="default" className="flex items-center gap-2 text-sm">
              <Globe size={12} />
              Public Registry
            </Badge>
          </div>
          <p className="text-sm text-helix-muted">Browse, discover, and acquire verified ML models</p>
        </div>
        {isConnected && (
          <Link
            href="/dashboard"
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-helix-surface border border-helix-border text-sm font-medium text-helix-text hover:border-helix-border2 hover:text-white transition-all"
          >
            <Layers size={16} />
            Dashboard
          </Link>
        )}
      </div>

      {/* Mode Toggle + Filter + Search + Sort */}
      <div className="space-y-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
          <FilterTabs active={filter} onChange={setFilter} counts={counts} filters={filterLabels} />
          <ModeToggle mode={mode} onChange={handleModeChange} />
          <div className="relative flex-1 max-w-sm">
            <Search size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-helix-muted" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search models..."
              className="w-full pl-10 pr-4 py-2.5 bg-helix-bg border border-helix-border rounded-xl text-sm text-helix-text placeholder:text-helix-dim focus:outline-none focus:border-helix-border2 focus:ring-1 focus:ring-helix-border2 transition-all"
            />
          </div>
          <div className="relative" ref={sortRef}>
            <button
              type="button"
              onClick={() => setSortOpen((v) => !v)}
              className={cn(
                'flex items-center gap-2 pl-3.5 pr-3 py-2.5 bg-helix-bg border rounded-xl text-sm text-helix-text transition-all cursor-pointer',
                sortOpen ? 'border-helix-border2 ring-1 ring-helix-border2' : 'border-helix-border hover:border-helix-border2',
              )}
            >
              <ArrowUpDown size={14} className="text-helix-muted" />
              <span>{sortOptions.find((o) => o.id === sort)?.label ?? 'Sort'}</span>
              <ChevronDown size={14} className={cn('text-helix-muted transition-transform', sortOpen && 'rotate-180')} />
            </button>
            <AnimatePresence>
              {sortOpen && (
                <motion.div
                  initial={{ opacity: 0, y: -4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -4 }}
                  transition={{ duration: 0.15 }}
                  className="absolute right-0 top-full mt-1.5 z-50 min-w-[180px] bg-helix-surface border border-helix-border rounded-xl shadow-xl overflow-hidden"
                >
                  {sortOptions.map((o) => (
                    <button
                      key={o.id}
                      type="button"
                      onClick={() => { setSort(o.id); setSortOpen(false); }}
                      className={cn(
                        'w-full text-left px-4 py-2.5 text-sm transition-colors',
                        sort === o.id
                          ? 'text-white bg-white/[0.06]'
                          : 'text-helix-text2 hover:bg-white/[0.04] hover:text-white',
                      )}
                    >
                      {o.label}
                    </button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Category tags */}
        {availableCategories.length > 0 && (
          <div className="flex items-center gap-2.5 flex-wrap">
            <span className="text-sm text-helix-dim font-medium">Type:</span>
            {availableCategories.map(({ tag, count }) => (
              <button
                key={tag}
                type="button"
                onClick={() => setCategoryFilter(categoryFilter === tag ? null : tag)}
                className={cn(
                  'flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs transition-all',
                  categoryFilter === tag
                    ? 'bg-white text-black font-medium shadow-sm'
                    : 'bg-white/[0.04] text-helix-text2 border border-white/[0.06] hover:border-white/[0.15] hover:bg-white/[0.06]',
                )}
              >
                {tag}
                <span className={cn(
                  'font-mono',
                  categoryFilter === tag ? 'text-black/50' : 'text-helix-dim',
                )}>
                  {count}
                </span>
                {categoryFilter === tag && <X size={12} />}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Content */}
      {isLoading ? (
        <div className="flex flex-col items-center py-24 gap-5">
          <Loader2 size={32} className="animate-spin text-helix-muted" />
          <p className="text-sm text-helix-muted">Loading models from chain...</p>
        </div>
      ) : !isContractDeployed ? (
        <EmptyState
          icon={<Lock size={32} />}
          title="Contract not deployed"
          description="The HelixModelStore contract is not deployed on this chain. Deploy it to see public models."
        />
      ) : filteredModels.length === 0 ? (
        <EmptyState
          icon={mode === 'marketplace' ? <Store size={32} /> : <Boxes size={32} />}
          title={
            search || categoryFilter
              ? 'No models match your filters'
              : mode === 'marketplace'
                ? filter === 'mine'
                  ? 'None of your models are for sale'
                  : 'No models for sale yet'
                : filter === 'mine'
                  ? 'None of your models are public'
                  : 'No public models yet'
          }
          description={
            search || categoryFilter
              ? 'Try a different search term or clear the filters.'
              : mode === 'marketplace'
                ? filter === 'mine'
                  ? 'Go to your Dashboard to list a model for sale.'
                  : 'No models are currently listed for sale on the marketplace.'
                : filter === 'mine'
                  ? 'Go to your Dashboard to make a model public.'
                  : 'Be the first to create and publish a model.'
          }
          action={
            search || categoryFilter ? (
              <button
                type="button"
                onClick={() => { setSearch(''); setCategoryFilter(null); }}
                className="px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text hover:text-white transition-colors"
              >
                Clear Filters
              </button>
            ) : !search && filter !== 'all' ? (
              <button
                type="button"
                onClick={() => setFilter('all')}
                className="px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text hover:text-white transition-colors"
              >
                Show All
              </button>
            ) : isConnected ? (
              <Link
                href="/dashboard"
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white text-black text-sm font-medium hover:bg-white/90 transition-colors"
              >
                <Layers size={14} />
                Create a Model
              </Link>
            ) : undefined
          }
        />
      ) : (
        <AnimatePresence mode="wait">
          <motion.div
            key={`${mode}-${filter}-${search}-${sort}-${categoryFilter}`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="grid grid-cols-1 lg:grid-cols-2 gap-5"
          >
            {filteredModels.map((model) => {
              const isOwner = !!address && model.owner.toLowerCase() === address.toLowerCase();
              return mode === 'marketplace' ? (
                <MarketplaceModelCard
                  key={model.tokenId}
                  model={model}
                  isOwner={isOwner}
                  onBuy={handleBuyClick}
                  onRate={handleRateClick}
                />
              ) : (
                <InferenceModelCard
                  key={model.tokenId}
                  model={model}
                  isOwner={isOwner}
                  onRate={handleRateClick}
                />
              );
            })}
          </motion.div>
        </AnimatePresence>
      )}

      {/* Buy Confirmation Modal */}
      <AnimatePresence>
        {buyingModel && (
          <BuyConfirmationModal
            model={buyingModel}
            onConfirm={handleBuyConfirm}
            onCancel={handleBuyCancel}
            isPending={isWritePending}
            isConfirming={isConfirming}
          />
        )}
      </AnimatePresence>

      {/* Rating Modal */}
      <AnimatePresence>
        {ratingModel && (
          <RatingModal
            model={ratingModel}
            onSubmit={handleRateSubmit}
            onCancel={() => setRatingModel(null)}
          />
        )}
      </AnimatePresence>
    </motion.div>
  );
}
