'use client';

import { useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { motion } from 'framer-motion';
import {
  ArrowLeft,
  Layers,
  Globe,
  Lock,
  Sparkles,
  ShoppingCart,
  HardDrive,
  Trophy,
  Tag,
  Copy,
  CheckCircle,
  ExternalLink,
  DollarSign,
  Shield,
} from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Card } from '@/components/ui/Card';
import { Skeleton } from '@/components/ui/Skeleton';
import { Tabs } from '@/components/ui/Tabs';
import { cn } from '@/lib/utils';
import { useModelDetail } from '@/hooks/useModelDetail';
import type { OnChainVersion } from '@/lib/contracts';

// ============================================================================
// Helpers
// ============================================================================

function truncateAddress(addr: string): string {
  if (addr.length <= 12) return addr;
  return `${addr.slice(0, 6)}...${addr.slice(-4)}`;
}

function formatDate(timestamp: number): string {
  if (!timestamp) return '--';
  return new Date(timestamp * 1000).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

function formatFee(bps: number): string {
  if (bps === 0) return 'Free';
  return `${(bps / 100).toFixed(bps % 100 === 0 ? 0 : 1)}%`;
}

function truncateHash(hash: string): string {
  if (!hash || hash.length <= 16) return hash || '--';
  return `${hash.slice(0, 10)}...${hash.slice(-6)}`;
}

// ============================================================================
// Tabs
// ============================================================================

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'versions', label: 'Versions' },
  { id: 'inference', label: 'Inference' },
] as const;

type TabId = (typeof TABS)[number]['id'];

// ============================================================================
// Version Row (detailed, for version tab)
// ============================================================================

function VersionRow({ version, index, modelSlug }: { version: OnChainVersion; index: number; modelSlug: string }) {
  const [copied, setCopied] = useState(false);

  const copyHash = (hash: string) => {
    navigator.clipboard.writeText(hash);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Card variant="default" className="!p-4">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-md bg-white/[0.04] flex items-center justify-center text-sm font-mono text-helix-muted">
            {index + 1}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-white font-mono">v{version.semver}</span>
              {version.accuracy > 0 && (
                <Badge variant="default" className="text-2xs">
                  {(version.accuracy * 100).toFixed(1)}% accuracy
                </Badge>
              )}
            </div>
            <p className="text-2xs text-helix-dim mt-0.5">
              {formatDate(version.timestamp)}
              {version.sessionId && <span className="ml-2 font-mono">{version.sessionId}</span>}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {version.weightsStored ? (
            <Badge variant="default" className="text-green-400 text-2xs flex items-center gap-1">
              <HardDrive size={10} />
              On 0G
            </Badge>
          ) : (
            <Badge variant="default" className="text-helix-dim text-2xs">
              No weights
            </Badge>
          )}
        </div>
      </div>

      {/* Hash + actions */}
      {version.rootHash && (
        <div className="flex items-center gap-2 mt-3 pl-11">
          <code className="text-2xs font-mono text-helix-muted bg-helix-bg px-2 py-1 rounded border border-helix-border/50">
            {truncateHash(version.rootHash)}
          </code>
          <button
            type="button"
            onClick={() => copyHash(version.rootHash)}
            className="p-1 rounded text-helix-dim hover:text-white transition-colors"
          >
            {copied ? <CheckCircle size={12} /> : <Copy size={12} />}
          </button>
          <a
            href={`https://storagescan-galileo.0g.ai/file/${version.rootHash}`}
            target="_blank"
            rel="noopener noreferrer"
            className="p-1 rounded text-helix-dim hover:text-white transition-colors"
          >
            <ExternalLink size={12} />
          </a>
          {version.weightsStored && (
            <Link
              href={`/inference?hash=${version.rootHash}&version=${version.semver}&model=${modelSlug}`}
              className="ml-auto flex items-center gap-1.5 px-3 py-1 rounded-md bg-white text-black text-2xs font-medium hover:bg-white/90 transition-colors"
            >
              <Sparkles size={10} />
              Run Inference
            </Link>
          )}
        </div>
      )}
    </Card>
  );
}

// ============================================================================
// Overview Tab
// ============================================================================

function OverviewTab({ model, isOwner }: { model: NonNullable<ReturnType<typeof useModelDetail>['model']>; isOwner: boolean }) {
  const bestAccuracy = model.versions.reduce((best, v) => (v.accuracy > best ? v.accuracy : best), 0);
  const weightsCount = model.versions.filter((v) => v.weightsStored).length;

  return (
    <div className="space-y-6">
      {/* Description */}
      {model.description && (
        <Card variant="default">
          <h3 className="text-sm font-medium text-white mb-2">About</h3>
          <p className="text-sm text-helix-text2 leading-relaxed">{model.description}</p>
        </Card>
      )}

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Card variant="default" className="!p-4 text-center">
          <Tag size={14} className="text-helix-muted mx-auto mb-2" />
          <p className="text-2xs text-helix-muted">Versions</p>
          <p className="text-xl font-mono font-light text-white">{model.versions.length}</p>
        </Card>
        <Card variant="default" className="!p-4 text-center">
          <Trophy size={14} className="text-helix-muted mx-auto mb-2" />
          <p className="text-2xs text-helix-muted">Best Accuracy</p>
          <p className="text-xl font-mono font-light text-white">
            {bestAccuracy > 0 ? `${(bestAccuracy * 100).toFixed(1)}%` : '--'}
          </p>
        </Card>
        <Card variant="default" className="!p-4 text-center">
          <HardDrive size={14} className="text-helix-muted mx-auto mb-2" />
          <p className="text-2xs text-helix-muted">Weights on 0G</p>
          <p className="text-xl font-mono font-light text-white">{weightsCount}</p>
        </Card>
        <Card variant="default" className="!p-4 text-center">
          <DollarSign size={14} className="text-helix-muted mx-auto mb-2" />
          <p className="text-2xs text-helix-muted">Inference Fee</p>
          <p className="text-xl font-mono font-light text-white">{formatFee(model.inferenceFee)}</p>
        </Card>
      </div>

      {/* Owner Info */}
      <Card variant="default">
        <h3 className="text-sm font-medium text-white mb-3">Ownership</h3>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-2xs text-helix-muted">Creator</span>
            <code className="text-2xs font-mono text-helix-text">{model.creator}</code>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-2xs text-helix-muted">Current Owner</span>
            <div className="flex items-center gap-2">
              <code className="text-2xs font-mono text-helix-text">{model.owner}</code>
              {isOwner && (
                <Badge variant="outline" className="text-green-400 border-green-500/30 text-2xs">You</Badge>
              )}
            </div>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-2xs text-helix-muted">Token ID</span>
            <span className="text-2xs font-mono text-helix-text">#{model.tokenId}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-2xs text-helix-muted">Created</span>
            <span className="text-2xs text-helix-text">{formatDate(model.createdAt)}</span>
          </div>
        </div>
      </Card>

      {/* Latest Version Quick View */}
      {model.versions.length > 0 && (
        <Card variant="default">
          <h3 className="text-sm font-medium text-white mb-3">Latest Version</h3>
          {(() => {
            const latest = model.versions[model.versions.length - 1];
            return (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-2xs text-helix-muted">Version</span>
                  <span className="text-sm font-mono text-white">v{latest.semver}</span>
                </div>
                {latest.accuracy > 0 && (
                  <div className="flex items-center justify-between">
                    <span className="text-2xs text-helix-muted">Accuracy</span>
                    <span className="text-sm font-mono text-white">{(latest.accuracy * 100).toFixed(1)}%</span>
                  </div>
                )}
                <div className="flex items-center justify-between">
                  <span className="text-2xs text-helix-muted">Weights</span>
                  <span className="text-2xs text-helix-text">
                    {latest.weightsStored ? 'Stored on 0G' : 'Not stored'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-2xs text-helix-muted">Published</span>
                  <span className="text-2xs text-helix-text">{formatDate(latest.timestamp)}</span>
                </div>
              </div>
            );
          })()}
        </Card>
      )}
    </div>
  );
}

// ============================================================================
// Versions Tab
// ============================================================================

function VersionsTab({ model }: { model: NonNullable<ReturnType<typeof useModelDetail>['model']> }) {
  if (model.versions.length === 0) {
    return (
      <div className="flex flex-col items-center py-12 gap-3">
        <Tag size={32} className="text-helix-dim" />
        <p className="text-sm text-helix-text2">No versions yet</p>
        <p className="text-2xs text-helix-muted">The model owner hasn&apos;t published any versions.</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-sm text-helix-text2">
          {model.versions.length} version{model.versions.length !== 1 ? 's' : ''}
        </p>
      </div>
      {model.versions.slice().reverse().map((v, i) => (
        <VersionRow
          key={`${v.semver}-${i}`}
          version={v}
          index={model.versions.length - 1 - i}
          modelSlug={model.slug}
        />
      ))}
    </div>
  );
}

// ============================================================================
// Inference Tab
// ============================================================================

function InferenceTab({ model, isOwner }: { model: NonNullable<ReturnType<typeof useModelDetail>['model']>; isOwner: boolean }) {
  const versionsWithWeights = model.versions.filter((v) => v.weightsStored && v.rootHash);

  return (
    <div className="space-y-6">
      {/* Fee info */}
      <Card variant="default">
        <h3 className="text-sm font-medium text-white mb-3">Inference Pricing</h3>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-2xs text-helix-muted">Model Fee</span>
            <span className="text-sm font-mono text-white">{formatFee(model.inferenceFee)}</span>
          </div>
          <p className="text-2xs text-helix-dim">
            {model.inferenceFee > 0
              ? `A ${formatFee(model.inferenceFee)} surcharge is added on top of base MPC compute costs. This fee goes to the model owner.`
              : 'No model surcharge. You only pay the base MPC network compute costs.'}
          </p>
          {isOwner && (
            <div className="flex items-center gap-2 mt-2 pt-2 border-t border-helix-border/50">
              <Shield size={12} className="text-green-400" />
              <span className="text-2xs text-green-400">As the owner, you run inference for free.</span>
            </div>
          )}
        </div>
      </Card>

      {/* Available versions */}
      <div>
        <h3 className="text-sm font-medium text-white mb-3">Available for Inference</h3>
        {versionsWithWeights.length === 0 ? (
          <Card variant="default" className="text-center !py-8">
            <HardDrive size={24} className="text-helix-dim mx-auto mb-2" />
            <p className="text-sm text-helix-text2">No versions with stored weights</p>
            <p className="text-2xs text-helix-muted mt-1">
              Inference requires model weights to be stored on 0G.
            </p>
          </Card>
        ) : (
          <div className="space-y-2">
            {versionsWithWeights.slice().reverse().map((v, i) => (
              <Card key={`${v.semver}-${i}`} variant="default" className="!p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className="text-sm font-mono font-medium text-white">v{v.semver}</span>
                    {v.accuracy > 0 && (
                      <span className="text-2xs text-helix-muted">{(v.accuracy * 100).toFixed(1)}% accuracy</span>
                    )}
                    <span className="text-2xs text-helix-dim">{formatDate(v.timestamp)}</span>
                  </div>
                  <Link
                    href={`/inference?hash=${v.rootHash}&version=${v.semver}&model=${model.slug}`}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white text-black text-sm font-medium hover:bg-white/90 transition-colors"
                  >
                    <Sparkles size={14} />
                    Run
                  </Link>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>

      {/* Buy / Offer */}
      {!isOwner && (
        <Card variant="glass">
          <h3 className="text-sm font-medium text-white mb-2">Acquire This Model</h3>
          <p className="text-2xs text-helix-muted mb-4">
            Buying a model transfers the NFT to you, giving you full ownership, access to encrypted weights,
            and the right to set inference fees.
          </p>
          <button
            type="button"
            className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text hover:border-helix-border2 hover:text-white transition-colors"
            title="Coming soon"
          >
            <ShoppingCart size={14} />
            Make Offer (Coming Soon)
          </button>
        </Card>
      )}
    </div>
  );
}

// ============================================================================
// Main Page
// ============================================================================

export default function ModelDetailPage() {
  const params = useParams<{ id: string }>();
  const tokenId = Number(params.id);
  const [activeTab, setActiveTab] = useState<TabId>('overview');

  const {
    model,
    isLoading,
    isError,
    isOwner,
    isContractDeployed,
  } = useModelDetail(tokenId);

  // Loading
  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <Skeleton width="24px" height="24px" />
          <Skeleton width="240px" height="1.5rem" />
        </div>
        <Skeleton width="100%" height="2.5rem" />
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Skeleton width="100%" height="100px" />
          <Skeleton width="100%" height="100px" />
          <Skeleton width="100%" height="100px" />
          <Skeleton width="100%" height="100px" />
        </div>
        <Skeleton width="100%" height="200px" />
      </div>
    );
  }

  // Not found
  if (!model || isError || !isContractDeployed) {
    return (
      <div className="flex flex-col items-center py-20 gap-4">
        <Layers size={40} className="text-helix-dim" />
        <p className="text-lg font-medium text-helix-text2">Model Not Found</p>
        <p className="text-sm text-helix-muted">
          {!isContractDeployed
            ? 'The model registry contract is not deployed on this chain.'
            : `Token #${tokenId} does not exist or has been burned.`}
        </p>
        <Link
          href="/"
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text hover:text-white transition-colors"
        >
          <ArrowLeft size={14} />
          Back to Models
        </Link>
      </div>
    );
  }

  // Not public and not owner
  if (!model.isPublic && !isOwner) {
    return (
      <div className="flex flex-col items-center py-20 gap-4">
        <Lock size={40} className="text-helix-dim" />
        <p className="text-lg font-medium text-helix-text2">Private Model</p>
        <p className="text-sm text-helix-muted">
          This model is private. Only the owner can view it.
        </p>
        <Link
          href="/"
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text hover:text-white transition-colors"
        >
          <ArrowLeft size={14} />
          Back to Models
        </Link>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.2 }}
      className="space-y-6"
    >
      {/* Back link */}
      <Link
        href="/"
        className="inline-flex items-center gap-1.5 text-2xs text-helix-muted hover:text-white transition-colors"
      >
        <ArrowLeft size={12} />
        Back to Models
      </Link>

      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-white/[0.06] flex items-center justify-center">
            <Layers size={24} className="text-white" />
          </div>
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-lg font-medium text-white tracking-tight">{model.name}</h1>
              {model.isPublic ? (
                <Badge variant="default" className="text-green-400 flex items-center gap-1">
                  <Globe size={10} />
                  Public
                </Badge>
              ) : (
                <Badge variant="default" className="flex items-center gap-1">
                  <Lock size={10} />
                  Private
                </Badge>
              )}
              <Badge variant="default">#{model.tokenId}</Badge>
            </div>
            <div className="flex items-center gap-3 mt-1">
              <span className="text-2xs font-mono text-helix-muted">{model.slug}</span>
              <span className="text-2xs text-helix-dim">by {truncateAddress(model.creator)}</span>
            </div>
          </div>
        </div>

        {/* Header actions */}
        <div className="flex items-center gap-2">
          {isOwner && (
            <Link
              href="/my-models"
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-helix-surface border border-helix-border text-2xs text-helix-text hover:text-white transition-colors"
            >
              Manage
            </Link>
          )}
        </div>
      </div>

      {/* Tabs */}
      <Tabs
        tabs={[...TABS]}
        activeTab={activeTab}
        onChange={(id) => setActiveTab(id as TabId)}
        className="border-b border-helix-border pb-px"
      />

      {/* Tab content */}
      <div className="min-h-[400px]">
        {activeTab === 'overview' && <OverviewTab model={model} isOwner={isOwner} />}
        {activeTab === 'versions' && <VersionsTab model={model} />}
        {activeTab === 'inference' && <InferenceTab model={model} isOwner={isOwner} />}
      </div>
    </motion.div>
  );
}
