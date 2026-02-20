'use client';

import { useState, useMemo, useCallback } from 'react';
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
} from 'lucide-react';
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

type SortOption = 'newest' | 'oldest' | 'accuracy' | 'versions' | 'name' | 'price-low' | 'price-high' | 'quality';

const INFERENCE_SORT_OPTIONS: { id: SortOption; label: string }[] = [
  { id: 'newest', label: 'Newest' },
  { id: 'oldest', label: 'Oldest' },
  { id: 'quality', label: 'Quality' },
  { id: 'accuracy', label: 'Best Accuracy' },
  { id: 'versions', label: 'Most Versions' },
  { id: 'name', label: 'Name A-Z' },
];

const MARKETPLACE_SORT_OPTIONS: { id: SortOption; label: string }[] = [
  { id: 'newest', label: 'Newest' },
  { id: 'oldest', label: 'Oldest' },
  { id: 'quality', label: 'Quality' },
  { id: 'price-low', label: 'Price Low-High' },
  { id: 'price-high', label: 'Price High-Low' },
  { id: 'accuracy', label: 'Best Accuracy' },
  { id: 'versions', label: 'Most Versions' },
  { id: 'name', label: 'Name A-Z' },
];

function sortModels<T extends { createdAt: number; bestAccuracy: number; versionCount: number; name: string; salePrice: number; latestVersion?: { weightsStored: boolean } | null }>(
  models: T[],
  sort: SortOption,
): T[] {
  const sorted = [...models];
  switch (sort) {
    case 'newest': return sorted.sort((a, b) => b.createdAt - a.createdAt);
    case 'oldest': return sorted.sort((a, b) => a.createdAt - b.createdAt);
    case 'quality': return sorted.sort((a, b) => computeQualityScore(b) - computeQualityScore(a));
    case 'accuracy': return sorted.sort((a, b) => b.bestAccuracy - a.bestAccuracy);
    case 'versions': return sorted.sort((a, b) => b.versionCount - a.versionCount);
    case 'name': return sorted.sort((a, b) => a.name.localeCompare(b.name));
    case 'price-low': return sorted.sort((a, b) => a.salePrice - b.salePrice);
    case 'price-high': return sorted.sort((a, b) => b.salePrice - a.salePrice);
    default: return sorted;
  }
}

const CATEGORY_RULES: { tag: string; patterns: RegExp }[] = [
  { tag: 'Classifier', patterns: /classif|detector|detection/i },
  { tag: 'Image', patterns: /image|mnist|cifar|resnet|vision|x-ray|imaging/i },
  { tag: 'NLP', patterns: /sentiment|bert|text|language|nlp|embedding/i },
  { tag: 'Autoencoder', patterns: /autoencoder|denoising|vae/i },
  { tag: 'Generative', patterns: /generative|gan|diffusion/i },
  { tag: 'Finance', patterns: /fraud|finance|transaction|trading/i },
];

function getModelTags(model: { name: string; description: string; slug: string }): string[] {
  const text = `${model.name} ${model.description} ${model.slug}`;
  const tags: string[] = [];
  for (const rule of CATEGORY_RULES) {
    if (rule.patterns.test(text)) tags.push(rule.tag);
  }
  return tags;
}

/** Quality score from real on-chain data: accuracy (60%), versions (25%, capped 5), weights (15%). */
function computeQualityScore(model: { bestAccuracy: number; versionCount: number; latestVersion?: { weightsStored: boolean } | null }): number {
  const accScore = Math.min(1, model.bestAccuracy) * 60;
  const versionScore = Math.min(model.versionCount, 5) / 5 * 25;
  const weightsScore = model.latestVersion?.weightsStored ? 15 : 0;
  return Math.round(accScore + versionScore + weightsScore);
}

function QualityBar({ score }: { score: number }) {
  const color = score >= 70 ? 'bg-green-400' : score >= 40 ? 'bg-yellow-400' : 'bg-white/40';
  const textColor = score >= 70 ? 'text-green-400' : score >= 40 ? 'text-yellow-400' : 'text-white/60';
  return (
    <div className="mb-4">
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-2xs text-helix-muted">Quality</span>
        <span className={cn('text-2xs font-mono font-medium tabular-nums', textColor)}>{score}/100</span>
      </div>
      <div className="h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
        <div className={cn('h-full rounded-full transition-all', color)} style={{ width: `${score}%` }} />
      </div>
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
    <div className="flex items-center gap-1 p-1 bg-helix-bg rounded-lg border border-helix-border">
      <button
        type="button"
        onClick={() => onChange('inference')}
        className={cn(
          'relative flex items-center gap-2 px-3 py-1.5 rounded-md text-sm transition-all',
          mode === 'inference'
            ? 'text-white'
            : 'text-helix-muted hover:text-helix-text2',
        )}
      >
        {mode === 'inference' && (
          <motion.div
            layoutId="mode-toggle-bg"
            className="absolute inset-0 bg-helix-surface border border-helix-border rounded-md"
            transition={{ type: 'spring', stiffness: 500, damping: 35 }}
          />
        )}
        <Sparkles size={14} className="relative z-10" />
        <span className="relative z-10">Inference</span>
      </button>
      <button
        type="button"
        onClick={() => onChange('marketplace')}
        className={cn(
          'relative flex items-center gap-2 px-3 py-1.5 rounded-md text-sm transition-all',
          mode === 'marketplace'
            ? 'text-white'
            : 'text-helix-muted hover:text-helix-text2',
        )}
      >
        {mode === 'marketplace' && (
          <motion.div
            layoutId="mode-toggle-bg"
            className="absolute inset-0 bg-helix-surface border border-helix-border rounded-md"
            transition={{ type: 'spring', stiffness: 500, damping: 35 }}
          />
        )}
        <Store size={14} className="relative z-10" />
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
    <div className="flex items-center gap-1 p-1 bg-helix-bg rounded-lg border border-helix-border">
      {filters.map((f) => (
        <button
          key={f.id}
          type="button"
          onClick={() => onChange(f.id)}
          className={cn(
            'relative flex items-center gap-2 px-3 py-1.5 rounded-md text-sm transition-all',
            active === f.id
              ? 'text-white'
              : 'text-helix-muted hover:text-helix-text2',
          )}
        >
          {active === f.id && (
            <motion.div
              layoutId="filter-tab-bg"
              className="absolute inset-0 bg-helix-surface border border-helix-border rounded-md"
              transition={{ type: 'spring', stiffness: 500, damping: 35 }}
            />
          )}
          <span className="relative z-10">{f.label}</span>
          <span className={cn(
            'relative z-10 text-2xs font-mono px-1.5 py-0.5 rounded-full',
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
        className="bg-helix-card border border-helix-border rounded-xl p-6 max-w-md w-full mx-4 shadow-2xl"
      >
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-lg bg-green-500/10 flex items-center justify-center">
            <ShoppingCart size={20} className="text-green-400" />
          </div>
          <div>
            <h3 className="text-lg font-medium text-white">Confirm Purchase</h3>
            <p className="text-2xs text-helix-muted">This action is irreversible</p>
          </div>
        </div>

        <div className="bg-helix-bg rounded-lg p-4 mb-4 space-y-2">
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
// Public Model Card (Inference Mode)
// ============================================================================

function InferenceModelCard({ model, isOwner }: { model: PublicModel; isOwner: boolean }) {
  const tags = useMemo(() => getModelTags(model), [model]);
  const qualityScore = useMemo(() => computeQualityScore(model), [model]);

  return (
    <Link href={`/models/${model.tokenId}`} className="block h-full">
      <Card variant="glass" hover className="cursor-pointer flex flex-col h-full">
        {/* Header */}
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-white/[0.06] flex items-center justify-center shrink-0">
              <Layers size={20} className="text-white" />
            </div>
            <div>
              <h3 className="text-base font-medium text-white">{model.name}</h3>
              <p className="text-2xs font-mono text-helix-muted">{model.slug}</p>
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {model.inferenceFee > 0 && (
              <Badge variant="default" className="text-2xs">
                {formatFee(model.inferenceFee)} fee
              </Badge>
            )}
            {isOwner && (
              <Badge variant="outline" className="text-2xs text-green-400 border-green-500/30">
                Yours
              </Badge>
            )}
          </div>
        </div>

        {/* Tags */}
        {tags.length > 0 && (
          <div className="flex items-center gap-1.5 mb-3">
            {tags.map((tag) => (
              <span key={tag} className="px-2 py-0.5 text-2xs rounded-full bg-white/[0.04] text-helix-text2 border border-white/[0.06]">
                {tag}
              </span>
            ))}
          </div>
        )}

        {/* Description */}
        {model.description && (
          <p className="text-2xs text-helix-muted mb-3 line-clamp-2">{model.description}</p>
        )}

        {/* Meta row */}
        <div className="flex items-center gap-4 text-2xs text-helix-dim mb-4">
          <span className="flex items-center gap-1">
            <User size={10} />
            {truncateAddress(model.creator)}
          </span>
          <span>{formatDate(model.createdAt)}</span>
          {model.latestVersion && (
            <span className="font-mono">v{model.latestVersion.semver}</span>
          )}
        </div>

        {/* Quality Score */}
        <QualityBar score={qualityScore} />

        {/* Stats */}
        <div className="grid grid-cols-3 gap-3 mb-4">
          <div className="bg-helix-bg rounded-md px-3 py-2 text-center">
            <p className="text-2xs text-helix-muted">Versions</p>
            <p className="text-lg font-mono font-light text-white">{model.versionCount}</p>
          </div>
          <div className="bg-helix-bg rounded-md px-3 py-2 text-center">
            <p className="text-2xs text-helix-muted">Best Accuracy</p>
            <p className="text-lg font-mono font-light text-white">
              {model.bestAccuracy > 0 ? `${(model.bestAccuracy * 100).toFixed(1)}%` : '--'}
            </p>
          </div>
          <div className="bg-helix-bg rounded-md px-3 py-2 text-center">
            <p className="text-2xs text-helix-muted">Inference Fee</p>
            <p className="text-lg font-mono font-light text-white">{formatFee(model.inferenceFee)}</p>
          </div>
        </div>

        {/* Actions - pinned to bottom */}
        <div className="flex items-center gap-3 pt-3 border-t border-helix-border/50 mt-auto">
          <span className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white text-black text-sm font-medium">
            <Sparkles size={14} />
            View Model
          </span>
          {isOwner && (
            <span className="flex items-center gap-2 px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text">
              <Layers size={14} />
              Yours
            </span>
          )}
        </div>
      </Card>
    </Link>
  );
}

// ============================================================================
// Marketplace Model Card
// ============================================================================

function MarketplaceModelCard({
  model,
  isOwner,
  onBuy,
}: {
  model: PublicModel;
  isOwner: boolean;
  onBuy: (model: PublicModel) => void;
}) {
  const tags = useMemo(() => getModelTags(model), [model]);
  const qualityScore = useMemo(() => computeQualityScore(model), [model]);
  const isContactOwner = model.forSale && model.salePrice === 0;

  return (
    <Card variant="glass" hover className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-green-500/10 flex items-center justify-center shrink-0">
            <Store size={20} className="text-green-400" />
          </div>
          <div>
            <h3 className="text-base font-medium text-white">{model.name}</h3>
            <p className="text-2xs font-mono text-helix-muted">{model.slug}</p>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Badge variant="outline" className="text-2xs text-green-400 border-green-500/30">
            For Sale
          </Badge>
          {isOwner && (
            <Badge variant="outline" className="text-2xs text-blue-400 border-blue-500/30">
              Yours
            </Badge>
          )}
        </div>
      </div>

      {/* Sale Price - prominent display */}
      <div className="bg-green-500/5 border border-green-500/20 rounded-lg px-4 py-3 mb-3">
        <div className="flex items-center justify-between">
          <span className="text-sm text-green-300/70 flex items-center gap-1.5">
            <DollarSign size={14} />
            Sale Price
          </span>
          <span className="text-xl font-mono font-medium text-green-400">
            {isContactOwner ? 'Contact Owner' : formatAdiPrice(model.salePrice)}
          </span>
        </div>
      </div>

      {/* Tags */}
      {tags.length > 0 && (
        <div className="flex items-center gap-1.5 mb-3">
          {tags.map((tag) => (
            <span key={tag} className="px-2 py-0.5 text-2xs rounded-full bg-white/[0.04] text-helix-text2 border border-white/[0.06]">
              {tag}
            </span>
          ))}
        </div>
      )}

      {/* Description */}
      {model.description && (
        <p className="text-2xs text-helix-muted mb-3 line-clamp-2">{model.description}</p>
      )}

      {/* Meta row */}
      <div className="flex items-center gap-4 text-2xs text-helix-dim mb-4">
        <span className="flex items-center gap-1">
          <User size={10} />
          {truncateAddress(model.creator)}
        </span>
        <span>{formatDate(model.createdAt)}</span>
        {model.latestVersion && (
          <span className="font-mono">v{model.latestVersion.semver}</span>
        )}
      </div>

      {/* Quality Score */}
      <QualityBar score={qualityScore} />

      {/* Stats */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="bg-helix-bg rounded-md px-3 py-2 text-center">
          <p className="text-2xs text-helix-muted">Versions</p>
          <p className="text-lg font-mono font-light text-white">{model.versionCount}</p>
        </div>
        <div className="bg-helix-bg rounded-md px-3 py-2 text-center">
          <p className="text-2xs text-helix-muted">Best Accuracy</p>
          <p className="text-lg font-mono font-light text-white">
            {model.bestAccuracy > 0 ? `${(model.bestAccuracy * 100).toFixed(1)}%` : '--'}
          </p>
        </div>
        <div className="bg-helix-bg rounded-md px-3 py-2 text-center">
          <p className="text-2xs text-helix-muted">Sale Price</p>
          <p className="text-lg font-mono font-light text-green-400">
            {isContactOwner ? 'N/A' : formatAdiPrice(model.salePrice)}
          </p>
        </div>
      </div>

      {/* Actions - pinned to bottom */}
      <div className="flex items-center gap-3 pt-3 border-t border-helix-border/50 mt-auto">
        {!isOwner && !isContactOwner && (
          <button
            type="button"
            onClick={(e) => {
              e.preventDefault();
              onBuy(model);
            }}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-green-600 hover:bg-green-500 text-white text-sm font-medium transition-colors"
          >
            <ShoppingCart size={14} />
            Buy
          </button>
        )}
        {!isOwner && isContactOwner && (
          <span className="flex items-center gap-2 px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text">
            <User size={14} />
            Contact Owner
          </span>
        )}
        <Link
          href={`/models/${model.tokenId}`}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white text-black text-sm font-medium hover:bg-white/90 transition-colors"
        >
          <Sparkles size={14} />
          View Model
        </Link>
        {isOwner && (
          <span className="flex items-center gap-2 px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text">
            <Layers size={14} />
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

  const {
    address,
    isConnected,
    isContractDeployed,
    allModels,
    isLoading,
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

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="page-title">Models</h1>
          <Badge variant="default" className="flex items-center gap-1">
            <Globe size={10} />
            Public Registry
          </Badge>
        </div>
        {isConnected && (
          <Link
            href="/my-models"
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text hover:border-helix-border2 hover:text-white transition-colors"
          >
            <Layers size={14} />
            My Models
          </Link>
        )}
      </div>

      {/* Mode Toggle + Filter + Search + Sort */}
      <div className="space-y-3">
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
          <FilterTabs active={filter} onChange={setFilter} counts={counts} filters={filterLabels} />
          <ModeToggle mode={mode} onChange={handleModeChange} />
          <div className="relative flex-1 max-w-xs">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-helix-muted" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search models..."
              className="w-full pl-9 pr-3 py-2 bg-helix-bg border border-helix-border rounded-lg text-sm text-helix-text placeholder:text-helix-dim focus:outline-none focus:border-helix-border2 transition-colors"
            />
          </div>
          <div className="relative">
            <ArrowUpDown size={12} className="absolute left-3 top-1/2 -translate-y-1/2 text-helix-muted pointer-events-none" />
            <select
              value={sort}
              onChange={(e) => setSort(e.target.value as SortOption)}
              className="appearance-none pl-8 pr-8 py-2 bg-helix-bg border border-helix-border rounded-lg text-sm text-helix-text focus:outline-none focus:border-helix-border2 transition-colors cursor-pointer"
            >
              {sortOptions.map((o) => (
                <option key={o.id} value={o.id}>{o.label}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Category tags */}
        {availableCategories.length > 0 && (
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-2xs text-helix-dim">Type:</span>
            {availableCategories.map(({ tag, count }) => (
              <button
                key={tag}
                type="button"
                onClick={() => setCategoryFilter(categoryFilter === tag ? null : tag)}
                className={cn(
                  'flex items-center gap-1.5 px-2.5 py-1 rounded-full text-2xs transition-colors',
                  categoryFilter === tag
                    ? 'bg-white text-black'
                    : 'bg-white/[0.04] text-helix-text2 border border-white/[0.06] hover:border-white/[0.12]',
                )}
              >
                {tag}
                <span className={cn(
                  'font-mono',
                  categoryFilter === tag ? 'text-black/50' : 'text-helix-dim',
                )}>
                  {count}
                </span>
                {categoryFilter === tag && <X size={10} />}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Content */}
      {isLoading ? (
        <div className="flex flex-col items-center py-16 gap-4">
          <Loader2 size={24} className="animate-spin text-helix-muted" />
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
                  ? 'Go to My Models to list a model for sale.'
                  : 'No models are currently listed for sale on the marketplace.'
                : filter === 'mine'
                  ? 'Go to My Models to make a model public.'
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
                href="/my-models"
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
            className="grid grid-cols-1 lg:grid-cols-2 gap-4"
          >
            {filteredModels.map((model) => {
              const isOwner = !!address && model.owner.toLowerCase() === address.toLowerCase();
              return mode === 'marketplace' ? (
                <MarketplaceModelCard
                  key={model.tokenId}
                  model={model}
                  isOwner={isOwner}
                  onBuy={handleBuyClick}
                />
              ) : (
                <InferenceModelCard
                  key={model.tokenId}
                  model={model}
                  isOwner={isOwner}
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
    </motion.div>
  );
}
