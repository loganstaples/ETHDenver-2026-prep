'use client';

import { useState, useMemo } from 'react';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Boxes,
  Search,
  Sparkles,
  ShoppingCart,
  Layers,
  Trophy,
  User,
  Loader2,
  Globe,
  Lock,
  Tag,
} from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { StatCard } from '@/components/ui/StatCard';
import { EmptyState } from '@/components/ui/EmptyState';
import { cn } from '@/lib/utils';
import { usePublicModels, type PublicModel, type ModelFilter } from '@/hooks/usePublicModels';

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

// ============================================================================
// Filter Tabs
// ============================================================================

const FILTERS: { id: ModelFilter; label: string; description: string }[] = [
  { id: 'all', label: 'All Public', description: 'All public models' },
  { id: 'others', label: 'By Others', description: 'Models you don\'t own' },
  { id: 'mine', label: 'My Public', description: 'Your public models' },
];

function FilterTabs({
  active,
  onChange,
  counts,
}: {
  active: ModelFilter;
  onChange: (f: ModelFilter) => void;
  counts: Record<ModelFilter, number>;
}) {
  return (
    <div className="flex items-center gap-1 p-1 bg-helix-bg rounded-lg border border-helix-border">
      {FILTERS.map((f) => (
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
// Public Model Card
// ============================================================================

function PublicModelCard({ model, isOwner }: { model: PublicModel; isOwner: boolean }) {
  return (
    <Card variant="glass" hover>
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-white/[0.06] flex items-center justify-center">
            <Layers size={20} className="text-white" />
          </div>
          <div>
            <h3 className="text-base font-medium text-white">{model.name}</h3>
            <p className="text-2xs font-mono text-helix-muted">{model.slug}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
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

      {/* Actions */}
      <div className="flex items-center gap-3 pt-3 border-t border-helix-border/50">
        {model.latestVersion?.weightsStored && model.latestVersion.rootHash && (
          <Link
            href={`/inference?hash=${model.latestVersion.rootHash}&version=${model.latestVersion.semver}&model=${model.slug}`}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white text-black text-sm font-medium hover:bg-white/90 transition-colors"
          >
            <Sparkles size={14} />
            Run Inference
          </Link>
        )}
        {!isOwner && (
          <button
            type="button"
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text hover:border-helix-border2 hover:text-white transition-colors"
            title="Coming soon"
          >
            <ShoppingCart size={14} />
            Make Offer
          </button>
        )}
        {isOwner && (
          <Link
            href="/my-models"
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text hover:border-helix-border2 hover:text-white transition-colors"
          >
            <Layers size={14} />
            Manage
          </Link>
        )}
      </div>
    </Card>
  );
}

// ============================================================================
// Main Page
// ============================================================================

export default function ModelsPage() {
  const [filter, setFilter] = useState<ModelFilter>('all');
  const [search, setSearch] = useState('');

  const {
    address,
    isConnected,
    isContractDeployed,
    allModels,
    isLoading,
  } = usePublicModels();

  // Only show public models on this page
  const publicModels = useMemo(
    () => allModels.filter((m) => m.isPublic),
    [allModels],
  );

  // Apply filter + search
  const filteredModels = useMemo(() => {
    let models = publicModels;

    // Filter by ownership
    if (filter === 'others') {
      models = models.filter((m) => !address || m.owner.toLowerCase() !== address.toLowerCase());
    } else if (filter === 'mine') {
      models = models.filter((m) => address && m.owner.toLowerCase() === address.toLowerCase());
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

    return models;
  }, [publicModels, filter, search, address]);

  // Counts for filter tabs
  const counts = useMemo(() => ({
    all: publicModels.length,
    others: publicModels.filter((m) => !address || m.owner.toLowerCase() !== address.toLowerCase()).length,
    mine: publicModels.filter((m) => address && m.owner.toLowerCase() === address.toLowerCase()).length,
  }), [publicModels, address]);

  // Aggregate stats
  const totalVersions = publicModels.reduce((sum, m) => sum + m.versionCount, 0);
  const bestAccuracy = publicModels.reduce(
    (best, m) => (m.bestAccuracy > best ? m.bestAccuracy : best),
    0,
  );
  const uniqueCreators = new Set(publicModels.map((m) => m.creator.toLowerCase())).size;

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

      {/* Stats */}
      {publicModels.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard
            label="Public Models"
            value={publicModels.length}
            icon={<Boxes size={14} />}
          />
          <StatCard
            label="Total Versions"
            value={totalVersions}
            icon={<Tag size={14} />}
          />
          <StatCard
            label="Best Accuracy"
            value={bestAccuracy > 0 ? `${(bestAccuracy * 100).toFixed(1)}%` : '--'}
            icon={<Trophy size={14} />}
          />
          <StatCard
            label="Creators"
            value={uniqueCreators}
            icon={<User size={14} />}
          />
        </div>
      )}

      {/* Filter + Search */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
        <FilterTabs active={filter} onChange={setFilter} counts={counts} />
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
          icon={<Boxes size={32} />}
          title={search ? 'No models match your search' : filter === 'mine' ? 'None of your models are public' : 'No public models yet'}
          description={
            search
              ? 'Try a different search term or clear the filter.'
              : filter === 'mine'
                ? 'Go to My Models to make a model public.'
                : 'Be the first to create and publish a model.'
          }
          action={
            search ? (
              <button
                type="button"
                onClick={() => setSearch('')}
                className="px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text hover:text-white transition-colors"
              >
                Clear Search
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
            key={`${filter}-${search}`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="grid grid-cols-1 lg:grid-cols-2 gap-4"
          >
            {filteredModels.map((model) => (
              <PublicModelCard
                key={model.tokenId}
                model={model}
                isOwner={!!address && model.owner.toLowerCase() === address.toLowerCase()}
              />
            ))}
          </motion.div>
        </AnimatePresence>
      )}
    </motion.div>
  );
}
