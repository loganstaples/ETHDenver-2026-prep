'use client';

import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Link from 'next/link';
import {
  Layers,
  Tag,
  CheckCircle,
  XCircle,
  HardDrive,
  Copy,
  ExternalLink,
  ChevronDown,
  ChevronRight,
  Clock,
  Trophy,
  Upload,
  Download,
  Loader2,
  Wallet,
  Link2,
  Plus,
  AlertTriangle,
  Shield,
  Globe,
  Lock,
  Settings,
  Play,
  ArrowUpDown,
  Search,
  X,
} from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { StatCard } from '@/components/ui/StatCard';
import { EmptyState } from '@/components/ui/EmptyState';
import { Modal } from '@/components/ui/Modal';
import { cn } from '@/lib/utils';
import { useModelRegistry, type ModelWithVersions } from '@/hooks/useModelRegistry';
import { useSignMessage } from 'wagmi';
import { deriveModelKey, encryptWeights, decryptWeights } from '@/lib/model-encryption';
import type { OnChainVersion } from '@/lib/contracts';
import type { TrainingSessionState } from '@/hooks/useMpcTraining';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

// ============================================================================
// Constants
// ============================================================================

const SEMVER_REGEX = /^\d+\.\d+\.\d+$/;
const SLUG_REGEX = /^[a-z0-9]+(-[a-z0-9]+)*$/;
const LEGACY_HISTORY_KEY = 'helix-training-history';

// ============================================================================
// Sort & Category
// ============================================================================

type SortOption = 'newest' | 'oldest' | 'accuracy' | 'versions' | 'name';

const SORT_OPTIONS: { id: SortOption; label: string }[] = [
  { id: 'newest', label: 'Newest' },
  { id: 'oldest', label: 'Oldest' },
  { id: 'accuracy', label: 'Best Accuracy' },
  { id: 'versions', label: 'Most Versions' },
  { id: 'name', label: 'Name A-Z' },
];

function sortMyModels(models: ModelWithVersions[], sort: SortOption): ModelWithVersions[] {
  const sorted = [...models];
  switch (sort) {
    case 'newest': return sorted.sort((a, b) => b.createdAt - a.createdAt);
    case 'oldest': return sorted.sort((a, b) => a.createdAt - b.createdAt);
    case 'accuracy': return sorted.sort((a, b) => {
      const aBest = a.versions.reduce((best, v) => (v.accuracy > best ? v.accuracy : best), 0);
      const bBest = b.versions.reduce((best, v) => (v.accuracy > best ? v.accuracy : best), 0);
      return bBest - aBest;
    });
    case 'versions': return sorted.sort((a, b) => b.versions.length - a.versions.length);
    case 'name': return sorted.sort((a, b) => a.name.localeCompare(b.name));
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

// ============================================================================
// Helpers
// ============================================================================

function truncateAddress(addr: string): string {
  if (addr.length <= 12) return addr;
  return `${addr.slice(0, 6)}...${addr.slice(-4)}`;
}

function truncateHash(hash: string): string {
  if (!hash || hash.length <= 16) return hash || '--';
  return `${hash.slice(0, 10)}...${hash.slice(-6)}`;
}

function formatDate(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleDateString();
}

function hasLegacyHistory(): boolean {
  if (typeof window === 'undefined') return false;
  try {
    const raw = localStorage.getItem(LEGACY_HISTORY_KEY);
    if (!raw) return false;
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) && parsed.length > 0;
  } catch {
    return false;
  }
}

// ============================================================================
// Create Model Modal
// ============================================================================

interface CreateModelModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCreate: (params: { slug: string; name: string; description: string }) => void;
  isPending: boolean;
}

function CreateModelModal({ isOpen, onClose, onCreate, isPending }: CreateModelModalProps) {
  const [slug, setSlug] = useState('');
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [slugError, setSlugError] = useState<string | null>(null);

  const reset = () => {
    setSlug('');
    setName('');
    setDescription('');
    setSlugError(null);
  };

  const handleClose = () => {
    reset();
    onClose();
  };

  const handleSlugChange = (v: string) => {
    const lower = v.toLowerCase().replace(/\s+/g, '-');
    setSlug(lower);
    if (lower && !SLUG_REGEX.test(lower)) {
      setSlugError('Must be kebab-case (e.g. my-mnist-model)');
    } else {
      setSlugError(null);
    }
  };

  const canSubmit = slug.trim() && name.trim() && !slugError && !isPending;

  const handleSubmit = () => {
    if (!canSubmit) return;
    onCreate({ slug: slug.trim(), name: name.trim(), description: description.trim() });
  };

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="Create Model">
      <div className="space-y-4">
        {isPending ? (
          <div className="flex flex-col items-center py-8 gap-4">
            <Loader2 size={24} className="animate-spin text-white" />
            <p className="text-sm text-helix-text2">Creating model (confirm in wallet)...</p>
          </div>
        ) : (
          <>
            {/* Slug */}
            <div>
              <label className="label-text block mb-1.5">Slug</label>
              <input
                type="text"
                value={slug}
                onChange={(e) => handleSlugChange(e.target.value)}
                placeholder="my-mnist-model"
                className={cn(
                  'w-full px-3 py-2 bg-helix-bg border rounded-md text-sm text-helix-text font-mono focus:outline-none transition-colors',
                  slugError ? 'border-red-500/50' : 'border-helix-border focus:border-helix-border2',
                )}
              />
              {slugError && <p className="text-2xs text-red-400 mt-1">{slugError}</p>}
              <p className="text-2xs text-helix-dim mt-1">Unique identifier, kebab-case</p>
            </div>

            {/* Name */}
            <div>
              <label className="label-text block mb-1.5">Name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="MNIST Classifier"
                className="w-full px-3 py-2 bg-helix-bg border border-helix-border rounded-md text-sm text-helix-text focus:outline-none focus:border-helix-border2 transition-colors"
              />
            </div>

            {/* Description */}
            <div>
              <label className="label-text block mb-1.5">Description (optional)</label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="784 to 128 to 10 neural network for handwritten digit classification"
                rows={3}
                className="w-full px-3 py-2 bg-helix-bg border border-helix-border rounded-md text-sm text-helix-text focus:outline-none focus:border-helix-border2 transition-colors resize-none"
              />
            </div>

            <button
              type="button"
              onClick={handleSubmit}
              disabled={!canSubmit}
              className={cn(
                'w-full flex items-center justify-center gap-2 py-3 rounded-lg font-medium text-sm transition-all',
                !canSubmit
                  ? 'bg-helix-border text-helix-muted cursor-not-allowed'
                  : 'bg-white text-black hover:bg-white/90',
              )}
            >
              <Plus size={16} />
              Create Model
            </button>
          </>
        )}
      </div>
    </Modal>
  );
}

// ============================================================================
// Add Version Modal
// ============================================================================

type VersionPhase = 'idle' | 'validating' | 'encrypting' | 'uploading' | 'registering' | 'done' | 'error';

interface AddVersionModalProps {
  isOpen: boolean;
  onClose: () => void;
  targetModel: ModelWithVersions | null;
  addVersion: (params: {
    tokenId: number;
    semver: string;
    rootHash: string;
    accuracy: number;
    sessionId: string;
    weightsStored: boolean;
  }) => void;
  isPending: boolean;
  isConnected: boolean;
}

function AddVersionModal({
  isOpen,
  onClose,
  targetModel,
  addVersion,
  isPending,
  isConnected,
}: AddVersionModalProps) {
  const [version, setVersion] = useState('1.0.0');
  const [accuracy, setAccuracy] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [versionError, setVersionError] = useState<string | null>(null);
  const [phase, setPhase] = useState<VersionPhase>('idle');
  const [errorMsg, setErrorMsg] = useState('');
  const [resultHash, setResultHash] = useState('');
  const fileRef = useRef<HTMLInputElement>(null);

  const { signMessageAsync } = useSignMessage();

  const reset = () => {
    setVersion('1.0.0');
    setAccuracy('');
    setFile(null);
    setVersionError(null);
    setPhase('idle');
    setErrorMsg('');
    setResultHash('');
  };

  const handleClose = () => {
    reset();
    onClose();
  };

  const handleVersionChange = (v: string) => {
    setVersion(v);
    if (v.trim() && !SEMVER_REGEX.test(v.trim())) {
      setVersionError('Must be semver (e.g. 1.0.0)');
    } else {
      setVersionError(null);
    }
  };

  const handleSubmit = async () => {
    if (!targetModel) return;
    const v = version.trim() || '1.0.0';
    if (!SEMVER_REGEX.test(v)) return;

    const acc = accuracy ? parseFloat(accuracy) : 0;
    const sessionId = `v${v}-${Date.now()}`;

    try {
      if (file) {
        // --- With weights file: validate -> encrypt -> upload -> register ---
        setPhase('validating');
        const text = await file.text();
        const data = JSON.parse(text);
        const weights = data.weights || data;
        if (!weights.w1 || !weights.b1 || !weights.w2 || !weights.b2) {
          throw new Error('Invalid model format. Expected { w1, b1, w2, b2 } weight arrays.');
        }

        let rootHash = '';

        if (isConnected) {
          // Encrypt weights using wallet-derived key
          setPhase('encrypting');
          const key = await deriveModelKey(
            (message: string) => signMessageAsync({ message }),
            targetModel.tokenId,
          );
          const encrypted = await encryptWeights(key, JSON.stringify(weights));
          const encryptedBase64 = btoa(String.fromCharCode(...encrypted));

          // Upload encrypted to 0G
          setPhase('uploading');
          const res = await fetch('/api/store-on-0g', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              session_id: sessionId,
              encrypted: true,
              encryptedPayload: encryptedBase64,
              accuracy: acc > 0 ? acc / 100 : null,
              version: v,
            }),
          });

          if (!res.ok) {
            const errBody = await res.json().catch(() => ({}));
            throw new Error(errBody.error || `Upload failed: HTTP ${res.status}`);
          }

          const result = await res.json();
          rootHash = result.root_hash;
          setResultHash(rootHash);
        } else {
          // Not connected: upload unencrypted
          setPhase('uploading');
          const res = await fetch('/api/store-on-0g', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              session_id: sessionId,
              weights,
              accuracy: acc > 0 ? acc / 100 : null,
              version: v,
            }),
          });

          if (!res.ok) {
            const errBody = await res.json().catch(() => ({}));
            throw new Error(errBody.error || `Upload failed: HTTP ${res.status}`);
          }

          const result = await res.json();
          rootHash = result.root_hash;
          setResultHash(rootHash);
        }

        // Register on-chain
        setPhase('registering');
        addVersion({
          tokenId: targetModel.tokenId,
          semver: v,
          rootHash,
          accuracy: acc,
          sessionId,
          weightsStored: true,
        });

        setPhase('done');
      } else {
        // --- No weights file: just register version on-chain ---
        setPhase('registering');
        addVersion({
          tokenId: targetModel.tokenId,
          semver: v,
          rootHash: '',
          accuracy: acc,
          sessionId,
          weightsStored: false,
        });

        setPhase('done');
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Failed to add version');
      setPhase('error');
    }
  };

  const canSubmit = !versionError && version.trim() && !isPending;

  return (
    <Modal
      isOpen={isOpen}
      onClose={handleClose}
      title={targetModel ? `Add Version to ${targetModel.name}` : 'Add Version'}
    >
      <div className="space-y-4">
        {phase === 'idle' && (
          <>
            {/* Version */}
            <div>
              <label className="label-text block mb-1.5">Version (semver)</label>
              <input
                type="text"
                value={version}
                onChange={(e) => handleVersionChange(e.target.value)}
                placeholder="1.0.0"
                className={cn(
                  'w-full px-3 py-2 bg-helix-bg border rounded-md text-sm text-helix-text font-mono focus:outline-none transition-colors',
                  versionError ? 'border-red-500/50' : 'border-helix-border focus:border-helix-border2',
                )}
              />
              {versionError && <p className="text-2xs text-red-400 mt-1">{versionError}</p>}
            </div>

            {/* Weights File */}
            <div>
              <label className="label-text block mb-1.5">Weights File (optional, JSON)</label>
              <div
                onClick={() => fileRef.current?.click()}
                className={cn(
                  'flex flex-col items-center justify-center gap-2 p-6 rounded-lg border-2 border-dashed cursor-pointer transition-colors',
                  file
                    ? 'border-green-500/30 bg-green-500/5'
                    : 'border-helix-border hover:border-helix-border2 bg-helix-bg',
                )}
              >
                {file ? (
                  <>
                    <CheckCircle size={20} className="text-green-400" />
                    <span className="text-sm text-green-300">{file.name}</span>
                    <span className="text-2xs text-helix-muted">
                      {(file.size / 1024).toFixed(0)} KB
                    </span>
                  </>
                ) : (
                  <>
                    <Upload size={20} className="text-helix-muted" />
                    <span className="text-sm text-helix-text2">Click to select weights file</span>
                    <span className="text-2xs text-helix-muted">
                      JSON with w1, b1, w2, b2 weight arrays
                    </span>
                  </>
                )}
                <input
                  ref={fileRef}
                  type="file"
                  accept=".json"
                  className="hidden"
                  onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                />
              </div>
              {file && isConnected && (
                <div className="flex items-center gap-2 mt-2 text-2xs text-helix-muted">
                  <Shield size={12} className="text-green-400 shrink-0" />
                  Weights will be encrypted with your wallet key before upload
                </div>
              )}
            </div>

            {/* Accuracy */}
            <div>
              <label className="label-text block mb-1.5">Accuracy % (optional)</label>
              <input
                type="number"
                value={accuracy}
                onChange={(e) => setAccuracy(e.target.value)}
                placeholder="e.g. 95.5"
                min={0}
                max={100}
                step={0.1}
                className="w-full px-3 py-2 bg-helix-bg border border-helix-border rounded-md text-sm text-helix-text font-mono focus:outline-none focus:border-helix-border2 transition-colors"
              />
            </div>

            <button
              type="button"
              onClick={handleSubmit}
              disabled={!canSubmit}
              className={cn(
                'w-full flex items-center justify-center gap-2 py-3 rounded-lg font-medium text-sm transition-all',
                !canSubmit
                  ? 'bg-helix-border text-helix-muted cursor-not-allowed'
                  : 'bg-white text-black hover:bg-white/90',
              )}
            >
              <Plus size={16} />
              {file ? 'Upload & Register Version' : 'Register Version'}
            </button>
          </>
        )}

        {/* Progress states */}
        {(phase === 'validating' || phase === 'encrypting' || phase === 'uploading' || phase === 'registering') && (
          <div className="flex flex-col items-center py-8 gap-4">
            <Loader2 size={24} className="animate-spin text-white" />
            <p className="text-sm text-helix-text2">
              {phase === 'validating' && 'Validating model weights...'}
              {phase === 'encrypting' && 'Encrypting weights (sign in wallet)...'}
              {phase === 'uploading' && 'Uploading to 0G decentralized storage...'}
              {phase === 'registering' && 'Registering on-chain (confirm in wallet)...'}
            </p>
            {/* Step indicator */}
            <div className="flex items-center gap-2">
              {(['validating', 'encrypting', 'uploading', 'registering'] as const).map((s, i) => (
                <div
                  key={s}
                  className={cn(
                    'w-2 h-2 rounded-full transition-colors',
                    phase === s ? 'bg-white' :
                    (['validating', 'encrypting', 'uploading', 'registering'].indexOf(phase) > i)
                      ? 'bg-green-400' : 'bg-helix-border',
                  )}
                />
              ))}
            </div>
          </div>
        )}

        {phase === 'done' && (
          <div className="flex flex-col items-center py-6 gap-3">
            <CheckCircle size={32} className="text-green-400" />
            <p className="text-sm font-medium text-white">Version Registered</p>
            {resultHash && (
              <div className="w-full">
                <p className="label-text mb-1">0G Root Hash</p>
                <code className="block text-xs font-mono text-helix-text bg-helix-bg px-3 py-2 rounded-md border border-helix-border truncate">
                  {resultHash}
                </code>
              </div>
            )}
            <button
              type="button"
              onClick={handleClose}
              className="mt-2 px-6 py-2 rounded-lg bg-white text-black text-sm font-medium hover:bg-white/90 transition-colors"
            >
              Done
            </button>
          </div>
        )}

        {phase === 'error' && (
          <div className="flex flex-col items-center py-6 gap-3">
            <XCircle size={32} className="text-red-400" />
            <p className="text-sm text-red-300">{errorMsg}</p>
            <button
              type="button"
              onClick={() => setPhase('idle')}
              className="mt-2 px-6 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text hover:text-white transition-colors"
            >
              Try Again
            </button>
          </div>
        )}
      </div>
    </Modal>
  );
}

// ============================================================================
// Version Row
// ============================================================================

function VersionRow({ version }: { version: OnChainVersion }) {
  const [copied, setCopied] = useState(false);

  const copyHash = (hash: string) => {
    navigator.clipboard.writeText(hash);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex items-center gap-3 px-3 py-2.5 bg-helix-bg rounded-md border border-helix-border/50">
      <Badge variant="default" className="font-mono text-2xs shrink-0">v{version.semver}</Badge>

      {version.accuracy > 0 && (
        <span className="text-sm font-mono text-white">
          {(version.accuracy * 100).toFixed(1)}%
        </span>
      )}

      {version.weightsStored ? (
        <Badge variant="default" className="text-green-400">
          <HardDrive size={10} />
          0G
        </Badge>
      ) : (
        <span className="text-2xs text-helix-dim">No weights</span>
      )}

      <span className="text-2xs text-helix-dim ml-auto shrink-0">
        {formatDate(version.timestamp)}
      </span>

      {version.rootHash && (
        <div className="flex items-center gap-1 shrink-0">
          <code className="text-2xs font-mono text-helix-muted">
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
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Model Detail Modal
// ============================================================================

interface ModelDetailModalProps {
  model: ModelWithVersions | null;
  onClose: () => void;
  onTogglePublic: (model: ModelWithVersions) => void;
  onDownloadWeights: (model: ModelWithVersions) => void;
  onAddVersion: (model: ModelWithVersions) => void;
  isToggling: boolean;
  isDownloading: number | null;
}

function ModelDetailModal({
  model,
  onClose,
  onTogglePublic,
  onDownloadWeights,
  onAddVersion,
  isToggling,
  isDownloading,
}: ModelDetailModalProps) {
  const [copiedHash, setCopiedHash] = useState<string | null>(null);

  const handleEscape = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') onClose();
  }, [onClose]);

  useEffect(() => {
    if (model) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }
    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = '';
    };
  }, [model, handleEscape]);

  const bestAccuracy = model
    ? model.versions.reduce((best, v) => (v.accuracy > best ? v.accuracy : best), 0)
    : 0;
  const tags = model ? getModelTags(model) : [];
  const hasWeightsStored = model ? model.versions.some((v) => v.weightsStored) : false;
  const latestWithWeights = model
    ? [...model.versions].reverse().find((v) => v.weightsStored && v.rootHash)
    : null;
  const isThisDownloading = model ? isDownloading === model.tokenId : false;
  const reversedVersions = model ? [...model.versions].reverse() : [];

  const copyHash = (hash: string) => {
    navigator.clipboard.writeText(hash);
    setCopiedHash(hash);
    setTimeout(() => setCopiedHash(null), 2000);
  };

  return (
    <AnimatePresence>
      {model && (
        <motion.div
          key="detail-backdrop"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-[6px]"
          onClick={onClose}
        >
          <motion.div
            key="detail-panel"
            initial={{ opacity: 0, scale: 0.96, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 10 }}
            transition={{ type: 'spring', damping: 26, stiffness: 300 }}
            className="bg-[#111113] border border-white/[0.08] rounded-2xl max-w-[700px] w-full mx-4 max-h-[88vh] overflow-hidden flex flex-col relative shadow-2xl shadow-black/40"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Top gradient line */}
            <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/15 to-transparent rounded-t-2xl z-10" />

            {/* Close */}
            <button
              type="button"
              onClick={onClose}
              className="absolute top-5 right-5 z-10 p-1.5 rounded-full bg-white/[0.06] hover:bg-white/[0.1] text-helix-muted hover:text-white transition-all"
            >
              <X size={16} />
            </button>

            {/* Scrollable content */}
            <div className="overflow-y-auto flex-1">
              {/* Header */}
              <div className="px-8 pt-8 pb-2">
                <div className="flex items-start gap-4">
                  <div className="w-12 h-12 rounded-xl bg-white/[0.06] flex items-center justify-center shrink-0">
                    <Layers size={22} className="text-white/80" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <h2 className="text-xl font-semibold text-white tracking-tight">{model.name}</h2>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-sm font-mono text-helix-muted">{model.slug}</span>
                      <span className="text-helix-dim">&middot;</span>
                      <span className="text-sm text-helix-muted">#{model.tokenId}</span>
                    </div>
                  </div>
                </div>

                {model.description && (
                  <p className="text-sm text-helix-text2 mt-3 leading-relaxed">{model.description}</p>
                )}

                <div className="flex items-center gap-2 mt-3 flex-wrap">
                  {tags.map((tag) => (
                    <span key={tag} className="px-2.5 py-0.5 text-2xs rounded-full bg-white/[0.04] text-helix-text2 border border-white/[0.06]">
                      {tag}
                    </span>
                  ))}
                  <span className="text-2xs text-helix-dim">
                    Created {formatDate(model.createdAt)}
                  </span>
                </div>
              </div>

              {/* Hero Accuracy */}
              <div className="text-center py-8 px-8">
                <div className="inline-flex flex-col items-center">
                  {bestAccuracy > 0 ? (
                    <>
                      <span className="text-[56px] font-semibold tracking-tighter text-white font-mono leading-none">
                        {(bestAccuracy * 100).toFixed(1)}
                        <span className="text-[28px] text-helix-text2 font-normal ml-0.5">%</span>
                      </span>
                      <span className="text-sm text-helix-muted mt-2">Best Accuracy</span>
                    </>
                  ) : (
                    <>
                      <span className="text-[56px] font-semibold tracking-tighter text-helix-dim font-mono leading-none">--</span>
                      <span className="text-sm text-helix-muted mt-2">No accuracy data yet</span>
                    </>
                  )}
                </div>
              </div>

              {/* Stats Row */}
              <div className="px-8 pb-6">
                <div className="grid grid-cols-3 gap-3">
                  <div className="bg-white/[0.03] rounded-xl px-4 py-3.5 text-center">
                    <p className="text-lg font-mono font-medium text-white tracking-tight">
                      {model.versions.length}
                    </p>
                    <p className="text-2xs text-helix-muted mt-1 uppercase tracking-wider">Versions</p>
                  </div>
                  <div className="bg-white/[0.03] rounded-xl px-4 py-3.5 text-center">
                    <p className="text-lg font-mono font-medium text-white tracking-tight">
                      {hasWeightsStored ? 'Yes' : 'No'}
                    </p>
                    <p className="text-2xs text-helix-muted mt-1 uppercase tracking-wider">Weights Stored</p>
                  </div>
                  <div className="bg-white/[0.03] rounded-xl px-4 py-3.5 text-center">
                    <p className={cn('text-lg font-mono font-medium tracking-tight', model.isPublic ? 'text-green-400' : 'text-helix-text2')}>
                      {model.isPublic ? 'Public' : 'Private'}
                    </p>
                    <p className="text-2xs text-helix-muted mt-1 uppercase tracking-wider">Visibility</p>
                  </div>
                </div>
              </div>

              {/* Divider */}
              <div className="mx-8 h-px bg-white/[0.06]" />

              {/* Version History */}
              <div className="px-8 py-6">
                <h3 className="text-sm font-medium text-white mb-4">
                  Version History
                  <span className="text-helix-dim ml-2 font-normal">{model.versions.length}</span>
                </h3>

                {reversedVersions.length === 0 ? (
                  <p className="text-sm text-helix-dim py-3">No versions registered yet</p>
                ) : (
                  <div className="space-y-1">
                    {reversedVersions.map((v, i) => {
                      const barWidth = bestAccuracy > 0 ? (v.accuracy / bestAccuracy) * 100 : 0;
                      const isLatest = i === 0;
                      return (
                        <div
                          key={`${v.semver}-${i}`}
                          className={cn(
                            'flex items-center gap-3 py-2.5 px-3 rounded-lg transition-colors',
                            isLatest ? 'bg-white/[0.03]' : 'hover:bg-white/[0.02]',
                          )}
                        >
                          <span className={cn(
                            'text-xs font-mono w-14 shrink-0',
                            isLatest ? 'text-white' : 'text-helix-text2',
                          )}>
                            v{v.semver}
                          </span>

                          <div className="flex-1 h-1 bg-white/[0.04] rounded-full overflow-hidden">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${barWidth}%` }}
                              transition={{ duration: 0.6, delay: i * 0.08, ease: [0.25, 0.1, 0.25, 1] }}
                              className={cn(
                                'h-full rounded-full',
                                isLatest ? 'bg-white/40' : 'bg-white/20',
                              )}
                            />
                          </div>

                          <span className={cn(
                            'text-sm font-mono w-14 text-right shrink-0',
                            isLatest ? 'text-white' : 'text-helix-text2',
                          )}>
                            {v.accuracy > 0 ? `${(v.accuracy * 100).toFixed(1)}%` : '--'}
                          </span>

                          {v.weightsStored && (
                            <span className="flex items-center gap-1 text-green-400/70 text-2xs shrink-0">
                              <HardDrive size={9} />
                              0G
                            </span>
                          )}

                          <span className="text-2xs text-helix-dim w-20 text-right shrink-0">
                            {formatDate(v.timestamp)}
                          </span>

                          {v.rootHash && (
                            <button
                              type="button"
                              onClick={() => copyHash(v.rootHash)}
                              className="p-1 rounded text-helix-dim hover:text-white transition-colors shrink-0"
                              title={copiedHash === v.rootHash ? 'Copied!' : 'Copy hash'}
                            >
                              {copiedHash === v.rootHash ? <CheckCircle size={12} className="text-green-400" /> : <Copy size={12} />}
                            </button>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Divider */}
              <div className="mx-8 h-px bg-white/[0.06]" />

              {/* Settings */}
              <div className="px-8 py-6">
                <h3 className="text-sm font-medium text-white mb-4">Settings</h3>

                <div className="space-y-5">
                  {/* Visibility toggle */}
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-white">Visibility</p>
                      <p className="text-2xs text-helix-muted mt-0.5">
                        {model.isPublic ? 'Anyone can use this model for inference' : 'Only you can access this model'}
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={() => onTogglePublic(model)}
                      disabled={isToggling}
                      className={cn(
                        'relative inline-flex h-7 w-12 items-center rounded-full transition-colors duration-300 focus:outline-none',
                        model.isPublic ? 'bg-green-500' : 'bg-white/[0.1]',
                        isToggling && 'opacity-50 cursor-not-allowed',
                      )}
                    >
                      <span
                        className={cn(
                          'inline-block h-5 w-5 rounded-full bg-white shadow-sm transition-transform duration-200',
                          model.isPublic ? 'translate-x-6' : 'translate-x-1',
                        )}
                      />
                    </button>
                  </div>

                  {/* Inference Fee */}
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-white">Inference Fee</p>
                      <p className="text-2xs text-helix-muted mt-0.5">Commission earned per inference</p>
                    </div>
                    <span className="text-sm font-mono text-helix-text2">
                      {(model.inferenceFee / 100).toFixed(1)}%
                    </span>
                  </div>

                  {/* Token ID */}
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-white">Token ID</p>
                      <p className="text-2xs text-helix-muted mt-0.5">ERC-721 on-chain identifier</p>
                    </div>
                    <span className="text-sm font-mono text-helix-text2">#{model.tokenId}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Sticky Action Bar */}
            <div className="px-8 py-5 border-t border-white/[0.06] bg-[#111113] flex items-center gap-3">
              <Link
                href={`/train?model=${model.tokenId}`}
                className="flex-1 flex items-center justify-center gap-2 py-3 rounded-xl bg-white text-black text-sm font-medium hover:bg-white/90 transition-colors"
              >
                <Play size={15} />
                Train
              </Link>

              <button
                type="button"
                onClick={() => { onClose(); onAddVersion(model); }}
                className="flex-1 flex items-center justify-center gap-2 py-3 rounded-xl bg-white/[0.06] border border-white/[0.08] text-sm text-white hover:bg-white/[0.1] transition-colors"
              >
                <Plus size={15} />
                Add Version
              </button>

              {latestWithWeights && (
                <button
                  type="button"
                  onClick={() => onDownloadWeights(model)}
                  disabled={isThisDownloading}
                  className={cn(
                    'flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-white/[0.06] border border-white/[0.08] text-sm text-white hover:bg-white/[0.1] transition-colors',
                    isThisDownloading && 'opacity-50 cursor-not-allowed',
                  )}
                >
                  {isThisDownloading ? <Loader2 size={15} className="animate-spin" /> : <Download size={15} />}
                </button>
              )}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// ============================================================================
// Trained Model Detail Modal
// ============================================================================

function formatDuration(secs: number): string {
  if (secs < 60) return `${Math.round(secs)}s`;
  const mins = Math.floor(secs / 60);
  const rem = Math.round(secs % 60);
  return rem > 0 ? `${mins}m ${rem}s` : `${mins}m`;
}

interface TrainedModelDetailModalProps {
  session: TrainingSessionState | null;
  onClose: () => void;
  onDownload: (sessionId: string) => void;
}

function TrainedModelDetailModal({ session, onClose, onDownload }: TrainedModelDetailModalProps) {
  const handleEscape = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') onClose();
  }, [onClose]);

  useEffect(() => {
    if (session) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }
    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = '';
    };
  }, [session, handleEscape]);

  if (!session) return null;

  const name = session.model_name || 'Unnamed Model';
  const slug = session.model_slug || session.session_id.slice(0, 8);
  const accuracy = session.accuracy;
  const latestLoss = session.losses.length > 0 ? session.losses[session.losses.length - 1] : session.current_loss;
  const steps = session.losses.length || session.current_step;
  const startedAt = session.started_at > 0
    ? new Date(session.started_at * 1000).toLocaleDateString()
    : '--';

  return (
    <AnimatePresence>
      {session && (
        <motion.div
          key="trained-detail-backdrop"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-[6px]"
          onClick={onClose}
        >
          <motion.div
            key="trained-detail-panel"
            initial={{ opacity: 0, scale: 0.96, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 10 }}
            transition={{ type: 'spring', damping: 26, stiffness: 300 }}
            className="bg-[#111113] border border-white/[0.08] rounded-2xl max-w-[700px] w-full mx-4 max-h-[88vh] overflow-hidden flex flex-col relative shadow-2xl shadow-black/40"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Top gradient line */}
            <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-green-400/20 to-transparent rounded-t-2xl z-10" />

            {/* Close */}
            <button
              type="button"
              onClick={onClose}
              className="absolute top-5 right-5 z-10 p-1.5 rounded-full bg-white/[0.06] hover:bg-white/[0.1] text-helix-muted hover:text-white transition-all"
            >
              <X size={16} />
            </button>

            {/* Scrollable content */}
            <div className="overflow-y-auto flex-1">
              {/* Header */}
              <div className="px-8 pt-8 pb-2">
                <div className="flex items-start gap-4">
                  <div className="w-12 h-12 rounded-xl bg-green-500/[0.08] flex items-center justify-center shrink-0">
                    <Layers size={22} className="text-green-400" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <h2 className="text-xl font-semibold text-white tracking-tight">{name}</h2>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-sm font-mono text-helix-muted">{slug}</span>
                      <span className="text-helix-dim">&middot;</span>
                      <span className="text-2xs px-2 py-0.5 rounded-full bg-green-500/10 text-green-400 border border-green-500/20">Trained</span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2 mt-3 flex-wrap">
                  <span className="text-2xs text-helix-dim">
                    Trained {startedAt}
                  </span>
                  {session.elapsed_secs > 0 && (
                    <>
                      <span className="text-helix-dim">&middot;</span>
                      <span className="text-2xs text-helix-dim">
                        {formatDuration(session.elapsed_secs)} elapsed
                      </span>
                    </>
                  )}
                </div>
              </div>

              {/* Hero Accuracy */}
              <div className="text-center py-8 px-8">
                <div className="inline-flex flex-col items-center">
                  {accuracy != null && accuracy > 0 ? (
                    <>
                      <span className="text-[56px] font-semibold tracking-tighter text-white font-mono leading-none">
                        {(accuracy * 100).toFixed(1)}
                        <span className="text-[28px] text-helix-text2 font-normal ml-0.5">%</span>
                      </span>
                      <span className="text-sm text-helix-muted mt-2">Accuracy</span>
                    </>
                  ) : (
                    <>
                      <span className="text-[56px] font-semibold tracking-tighter text-helix-dim font-mono leading-none">--</span>
                      <span className="text-sm text-helix-muted mt-2">No accuracy data yet</span>
                    </>
                  )}
                </div>
              </div>

              {/* Stats Row */}
              <div className="px-8 pb-6">
                <div className="grid grid-cols-3 gap-3">
                  <div className="bg-white/[0.03] rounded-xl px-4 py-3.5 text-center">
                    <p className="text-lg font-mono font-medium text-white tracking-tight">
                      {latestLoss != null ? latestLoss.toFixed(4) : '--'}
                    </p>
                    <p className="text-2xs text-helix-muted mt-1 uppercase tracking-wider">Latest Loss</p>
                  </div>
                  <div className="bg-white/[0.03] rounded-xl px-4 py-3.5 text-center">
                    <p className="text-lg font-mono font-medium text-white tracking-tight">
                      {steps}
                    </p>
                    <p className="text-2xs text-helix-muted mt-1 uppercase tracking-wider">Steps Trained</p>
                  </div>
                  <div className="bg-white/[0.03] rounded-xl px-4 py-3.5 text-center">
                    <p className="text-lg font-mono font-medium text-white tracking-tight">
                      {session.checkpoints_submitted}
                    </p>
                    <p className="text-2xs text-helix-muted mt-1 uppercase tracking-wider">Checkpoints</p>
                  </div>
                </div>
              </div>

              {/* Divider */}
              <div className="mx-8 h-px bg-white/[0.06]" />

              {/* Training Details */}
              <div className="px-8 py-6">
                <h3 className="text-sm font-medium text-white mb-4">Training Details</h3>

                <div className="space-y-3.5">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-helix-text2">Steps Completed</span>
                    <span className="text-sm font-mono text-white">{steps}</span>
                  </div>

                  {session.workers_active != null && session.workers_active > 0 && (
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-helix-text2">Workers</span>
                      <span className="text-sm font-mono text-white">{session.workers_active}</span>
                    </div>
                  )}

                  {session.mac_checks_passed > 0 && (
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-helix-text2">MAC Checks Passed</span>
                      <span className="text-sm font-mono text-green-400">{session.mac_checks_passed}</span>
                    </div>
                  )}

                  {session.checkpoints_submitted > 0 && (
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-helix-text2">Checkpoints</span>
                      <span className="text-sm font-mono text-white">{session.checkpoints_submitted}</span>
                    </div>
                  )}

                  {session.zk_proofs_generated > 0 && (
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-helix-text2">ZK Proofs</span>
                      <span className="text-sm font-mono text-white">{session.zk_proofs_generated}</span>
                    </div>
                  )}

                  <div className="flex items-center justify-between">
                    <span className="text-sm text-helix-text2">Session</span>
                    <span className="text-xs font-mono text-helix-muted">{session.session_id.slice(0, 16)}...</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Sticky Action Bar */}
            <div className="px-8 py-5 border-t border-white/[0.06] bg-[#111113] flex items-center gap-3">
              <button
                type="button"
                onClick={() => onDownload(session.session_id)}
                className="flex-1 flex items-center justify-center gap-2 py-3 rounded-xl bg-white text-black text-sm font-medium hover:bg-white/90 transition-colors"
              >
                <Download size={15} />
                Download Weights
              </button>

              <Link
                href="/train"
                className="flex-1 flex items-center justify-center gap-2 py-3 rounded-xl bg-white/[0.06] border border-white/[0.08] text-sm text-white hover:bg-white/[0.1] transition-colors"
              >
                <Play size={15} />
                Train More
              </Link>

              <Link
                href="/inference"
                className="flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-white/[0.06] border border-white/[0.08] text-sm text-white hover:bg-white/[0.1] transition-colors"
              >
                <Shield size={15} />
              </Link>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// ============================================================================
// Model Card
// ============================================================================

interface ModelCardProps {
  model: ModelWithVersions;
  onClickCard: (model: ModelWithVersions) => void;
  onAddVersion: (model: ModelWithVersions) => void;
  onTogglePublic: (model: ModelWithVersions) => void;
  onDownloadWeights: (model: ModelWithVersions) => void;
  isToggling: boolean;
  isDownloading: number | null;
}

function ModelCard({ model, onClickCard, onAddVersion, onTogglePublic, onDownloadWeights, isToggling, isDownloading }: ModelCardProps) {
  const [expanded, setExpanded] = useState(false);
  const tags = useMemo(() => getModelTags(model), [model]);

  const versionCount = model.versions.length;
  const bestAccuracy = model.versions.reduce(
    (best, v) => (v.accuracy > best ? v.accuracy : best),
    0,
  );
  const weightsCount = model.versions.filter((v) => v.weightsStored).length;
  const latestWithWeights = [...model.versions].reverse().find((v) => v.weightsStored && v.rootHash);
  const isThisDownloading = isDownloading === model.tokenId;

  return (
    <Card variant="glass" hover className="flex flex-col h-full cursor-pointer" onClick={() => onClickCard(model)}>
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
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
          {model.isPublic ? (
            <Badge variant="default" className="text-green-400 text-2xs flex items-center gap-1">
              <Globe size={10} />
              Public
            </Badge>
          ) : (
            <Badge variant="default" className="text-helix-muted text-2xs flex items-center gap-1">
              <Lock size={10} />
              Private
            </Badge>
          )}
          {model.versions.length > 0 && (
            <Badge variant="default" className="font-mono">
              v{model.versions[model.versions.length - 1]?.semver}
            </Badge>
          )}
          <Badge variant="default">#{model.tokenId}</Badge>
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

      {/* Description - clamped for consistent height */}
      {model.description && (
        <p className="text-2xs text-helix-muted mb-4 line-clamp-2">{model.description}</p>
      )}

      {/* Creator / Date */}
      <div className="flex items-center gap-4 text-2xs text-helix-dim mb-4">
        <span>Creator: {truncateAddress(model.creator)}</span>
        <span>Created: {formatDate(model.createdAt)}</span>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="bg-helix-bg rounded-md px-3 py-2 text-center">
          <p className="text-2xs text-helix-muted">Versions</p>
          <p className="text-lg font-mono font-light text-white">{versionCount}</p>
        </div>
        <div className="bg-helix-bg rounded-md px-3 py-2 text-center">
          <p className="text-2xs text-helix-muted">Best Accuracy</p>
          <p className="text-lg font-mono font-light text-white">
            {bestAccuracy > 0 ? `${(bestAccuracy * 100).toFixed(1)}%` : '--'}
          </p>
        </div>
        <div className="bg-helix-bg rounded-md px-3 py-2 text-center">
          <p className="text-2xs text-helix-muted">On 0G</p>
          <p className="text-lg font-mono font-light text-white">{weightsCount}</p>
        </div>
      </div>

      {/* Version History Toggle */}
      {versionCount > 0 && (
        <div className="mb-3" onClick={(e) => e.stopPropagation()}>
          <button
            type="button"
            onClick={() => setExpanded(!expanded)}
            className="flex items-center gap-2 text-sm text-helix-text2 hover:text-white transition-colors"
          >
            {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            <Clock size={14} />
            Version History ({versionCount})
          </button>

          <AnimatePresence>
            {expanded && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="space-y-2 mt-3">
                  {model.versions.slice().reverse().map((v, i) => (
                    <VersionRow key={`${v.semver}-${i}`} version={v} />
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* Actions - pinned to bottom */}
      <div className="flex items-center gap-2 flex-wrap pt-3 border-t border-helix-border/50 mt-auto" onClick={(e) => e.stopPropagation()}>
        <Link
          href={`/my-models/${model.tokenId}`}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white text-black text-sm font-medium hover:bg-white/90 transition-colors"
        >
          <Settings size={14} />
          Manage
        </Link>

        <Link
          href={`/train?model=${model.tokenId}`}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text hover:border-helix-border2 hover:text-white transition-colors"
        >
          <Play size={14} />
          Train
        </Link>

        {latestWithWeights && (
          <button
            type="button"
            onClick={() => onDownloadWeights(model)}
            disabled={isThisDownloading}
            className={cn(
              'flex items-center gap-2 px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text hover:border-helix-border2 hover:text-white transition-colors',
              isThisDownloading && 'opacity-50 cursor-not-allowed',
            )}
          >
            {isThisDownloading ? <Loader2 size={14} className="animate-spin" /> : <Download size={14} />}
            Download
          </button>
        )}

        <button
          type="button"
          onClick={() => onAddVersion(model)}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text hover:border-helix-border2 hover:text-white transition-colors"
        >
          <Plus size={14} />
          Add Version
        </button>

        <button
          type="button"
          onClick={() => onTogglePublic(model)}
          disabled={isToggling}
          className={cn(
            'flex items-center gap-2 px-4 py-2 rounded-lg border text-sm transition-colors',
            model.isPublic
              ? 'bg-helix-surface border-helix-border text-helix-text hover:border-helix-border2 hover:text-white'
              : 'bg-green-500/10 border-green-500/30 text-green-400 hover:bg-green-500/20',
            isToggling && 'opacity-50 cursor-not-allowed',
          )}
        >
          {isToggling ? (
            <Loader2 size={14} className="animate-spin" />
          ) : model.isPublic ? (
            <Lock size={14} />
          ) : (
            <Globe size={14} />
          )}
          {model.isPublic ? 'Make Private' : 'Make Public'}
        </button>
      </div>
    </Card>
  );
}

// ============================================================================
// Trained Model Card (from backend sessions)
// ============================================================================

function TrainedModelCard({ session, onDownload, onClickCard }: {
  session: TrainingSessionState;
  onDownload: (sessionId: string) => void;
  onClickCard: (session: TrainingSessionState) => void;
}) {
  const name = session.model_name || 'Unnamed Model';
  const slug = session.model_slug || session.session_id.slice(0, 8);
  const accuracy = session.accuracy;
  const steps = session.losses.length || session.current_step;
  const startedAt = session.started_at > 0
    ? new Date(session.started_at * 1000).toLocaleDateString()
    : '--';

  return (
    <Card variant="glass" hover className="flex flex-col h-full cursor-pointer" onClick={() => onClickCard(session)}>
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-green-500/[0.08] flex items-center justify-center shrink-0">
            <Layers size={20} className="text-green-400" />
          </div>
          <div>
            <h3 className="text-base font-medium text-white">{name}</h3>
            <p className="text-2xs font-mono text-helix-muted">{slug}</p>
          </div>
        </div>
        <Badge variant="default" className="text-green-400 text-2xs flex items-center gap-1">
          <CheckCircle size={10} />
          Trained
        </Badge>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="bg-helix-bg rounded-md px-3 py-2 text-center">
          <p className="text-2xs text-helix-muted">Accuracy</p>
          <p className="text-lg font-mono font-light text-white">
            {accuracy != null ? `${(accuracy * 100).toFixed(1)}%` : '--'}
          </p>
        </div>
        <div className="bg-helix-bg rounded-md px-3 py-2 text-center">
          <p className="text-2xs text-helix-muted">Steps</p>
          <p className="text-lg font-mono font-light text-white">{steps}</p>
        </div>
        <div className="bg-helix-bg rounded-md px-3 py-2 text-center">
          <p className="text-2xs text-helix-muted">Trained</p>
          <p className="text-sm font-mono font-light text-white">{startedAt}</p>
        </div>
      </div>

      {/* Session ID */}
      <div className="text-2xs text-helix-dim mb-4">
        Session: <code className="font-mono">{session.session_id.slice(0, 12)}...</code>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2 flex-wrap pt-3 border-t border-helix-border/50 mt-auto" onClick={(e) => e.stopPropagation()}>
        <button
          type="button"
          onClick={() => onDownload(session.session_id)}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white text-black text-sm font-medium hover:bg-white/90 transition-colors"
        >
          <Download size={14} />
          Download Weights
        </button>

        <Link
          href={`/train`}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text hover:border-helix-border2 hover:text-white transition-colors"
        >
          <Play size={14} />
          Train More
        </Link>

        <Link
          href={`/inference`}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text hover:border-helix-border2 hover:text-white transition-colors"
        >
          <Shield size={14} />
          Inference
        </Link>
      </div>
    </Card>
  );
}

// ============================================================================
// Main Page
// ============================================================================

export default function MyModelsPage() {
  const [createOpen, setCreateOpen] = useState(false);
  const [versionTarget, setVersionTarget] = useState<ModelWithVersions | null>(null);
  const [showLegacyNotice, setShowLegacyNotice] = useState(false);
  const [search, setSearch] = useState('');
  const [sort, setSort] = useState<SortOption>('newest');
  const [categoryFilter, setCategoryFilter] = useState<string | null>(null);
  const [downloadingTokenId, setDownloadingTokenId] = useState<number | null>(null);
  const [detailTokenId, setDetailTokenId] = useState<number | null>(null);
  const [detailSession, setDetailSession] = useState<TrainingSessionState | null>(null);
  const [trainedModels, setTrainedModels] = useState<TrainingSessionState[]>([]);
  const [isLoadingTrained, setIsLoadingTrained] = useState(true);

  const { signMessageAsync } = useSignMessage();

  const {
    isConnected,
    isContractDeployed,
    models,
    isLoading,
    createModel,
    addVersion,
    setPublic,
    isWritePending,
    isConfirming,
    writeError,
    isSuccess,
    refetch,
  } = useModelRegistry();

  const detailModel = detailTokenId !== null
    ? models.find(m => m.tokenId === detailTokenId) ?? null
    : null;

  // Fetch completed training sessions from backend
  const fetchTrainedModels = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/training/sessions`);
      if (!res.ok) return;
      const sessions: TrainingSessionState[] = await res.json();
      setTrainedModels(sessions.filter(s => s.status === 'complete'));
    } catch {
      // Non-fatal
    } finally {
      setIsLoadingTrained(false);
    }
  }, []);

  useEffect(() => {
    fetchTrainedModels();
    const interval = setInterval(fetchTrainedModels, 10000);
    return () => clearInterval(interval);
  }, [fetchTrainedModels]);

  // Download weights from a trained session
  const handleDownloadTrainedWeights = useCallback(async (sessionId: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/training/sessions/${sessionId}/model`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `Download failed: HTTP ${res.status}`);
      }
      const weights = await res.json();
      const blob = new Blob([JSON.stringify(weights, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `helix-model-${sessionId.slice(0, 8)}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Download failed:', err);
      alert(`Download failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  }, []);

  // Download weights from 0G with decryption
  const handleDownloadWeights = useCallback(async (model: ModelWithVersions) => {
    const latest = [...model.versions].reverse().find((v) => v.weightsStored && v.rootHash);
    if (!latest) return;

    setDownloadingTokenId(model.tokenId);
    try {
      const res = await fetch('/api/fetch-from-0g', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rootHash: latest.rootHash }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `Fetch failed: HTTP ${res.status}`);
      }
      const result = await res.json();

      let weightsJson: string;
      if (result.encoding === 'base64') {
        // Encrypted — decrypt with wallet key
        const key = await deriveModelKey(
          (message: string) => signMessageAsync({ message }),
          model.tokenId,
        );
        const binary = Uint8Array.from(atob(result.data), (c) => c.charCodeAt(0));
        weightsJson = await decryptWeights(key, binary);
      } else {
        weightsJson = JSON.stringify(result.data?.weights || result.data, null, 2);
      }

      // Trigger browser download
      const blob = new Blob([weightsJson], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${model.slug}-v${latest.semver}-weights.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Download failed:', err);
      alert(`Download failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setDownloadingTokenId(null);
    }
  }, [signMessageAsync]);

  // Check for legacy localStorage data
  useEffect(() => {
    setShowLegacyNotice(hasLegacyHistory());
  }, []);

  // Refetch models after successful writes
  useEffect(() => {
    if (isSuccess) {
      refetch();
      // Close modals after success
      setCreateOpen(false);
      setVersionTarget(null);
    }
  }, [isSuccess, refetch]);

  // Derive categories from user's models
  const availableCategories = useMemo(() => {
    const tagCounts = new Map<string, number>();
    for (const m of models) {
      for (const tag of getModelTags(m)) {
        tagCounts.set(tag, (tagCounts.get(tag) ?? 0) + 1);
      }
    }
    return Array.from(tagCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([tag, count]) => ({ tag, count }));
  }, [models]);

  // Apply search + category + sort
  const filteredModels = useMemo(() => {
    let result = models;

    if (categoryFilter) {
      result = result.filter((m) => getModelTags(m).includes(categoryFilter));
    }

    if (search.trim()) {
      const q = search.trim().toLowerCase();
      result = result.filter(
        (m) =>
          m.name.toLowerCase().includes(q) ||
          m.slug.toLowerCase().includes(q) ||
          m.description.toLowerCase().includes(q),
      );
    }

    return sortMyModels(result, sort);
  }, [models, search, sort, categoryFilter]);

  // Aggregate stats (on-chain + trained)
  const totalOnChainVersions = models.reduce((sum, m) => sum + m.versions.length, 0);
  const totalModels = models.length + trainedModels.length;
  const bestOnChainAccuracy = models.reduce((best, m) => {
    const modelBest = m.versions.reduce((b, v) => (v.accuracy > b ? v.accuracy : b), 0);
    return modelBest > best ? modelBest : best;
  }, 0);
  const bestTrainedAccuracy = trainedModels.reduce((best, s) => {
    return (s.accuracy ?? 0) > best ? (s.accuracy ?? 0) : best;
  }, 0);
  const bestAccuracy = Math.max(bestOnChainAccuracy, bestTrainedAccuracy);
  const totalWeights = models.reduce(
    (sum, m) => sum + m.versions.filter((v) => v.weightsStored).length,
    0,
  ) + trainedModels.length;

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
          <h1 className="page-title">My Models</h1>
          {totalModels > 0 && (
            <Badge variant="default">{totalModels} model{totalModels !== 1 ? 's' : ''}</Badge>
          )}
        </div>
        <div className="flex items-center gap-3">
          {/* Wallet status */}
          <div className={cn(
            'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-2xs font-mono',
            isConnected
              ? 'bg-green-500/10 text-green-400 border border-green-500/20'
              : 'bg-helix-bg text-helix-muted border border-helix-border',
          )}>
            <Wallet size={12} />
            {isConnected ? 'Wallet Connected' : 'No Wallet'}
          </div>

          <button
            type="button"
            onClick={() => setCreateOpen(true)}
            disabled={!isConnected || !isContractDeployed}
            className={cn(
              'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all',
              (!isConnected || !isContractDeployed)
                ? 'bg-helix-border text-helix-muted cursor-not-allowed'
                : 'bg-white text-black hover:bg-white/90',
            )}
          >
            <Plus size={14} />
            Create Model
          </button>
        </div>
      </div>

      {/* Legacy localStorage notice */}
      {showLegacyNotice && (
        <div className="flex items-start gap-3 px-4 py-3 bg-yellow-500/5 border border-yellow-500/20 rounded-lg">
          <AlertTriangle size={16} className="text-yellow-400 shrink-0 mt-0.5" />
          <div>
            <p className="text-sm text-yellow-300/90 font-medium">Legacy training data found</p>
            <p className="text-2xs text-yellow-300/60 mt-1">
              You have old training entries stored in your browser. These are no longer used.
              To preserve them, create a new model on-chain and add versions manually.
            </p>
            <button
              type="button"
              onClick={() => setShowLegacyNotice(false)}
              className="text-2xs text-yellow-400 hover:text-yellow-300 mt-2 underline transition-colors"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {/* Wallet/contract notices */}
      {!isConnected && (
        <div className="flex items-center gap-3 px-4 py-3 bg-helix-bg border border-helix-border rounded-lg">
          <Wallet size={16} className="text-helix-muted shrink-0" />
          <p className="text-2xs text-helix-muted">
            Connect your wallet to manage models. All data is stored on-chain via the ERC-721 model registry.
          </p>
        </div>
      )}
      {isConnected && !isContractDeployed && (
        <div className="flex items-center gap-3 px-4 py-3 bg-yellow-500/5 border border-yellow-500/20 rounded-lg">
          <Link2 size={16} className="text-yellow-400 shrink-0" />
          <p className="text-2xs text-yellow-300/70">
            HelixModelStore contract not deployed on this chain. Deploy it with: <code className="font-mono">forge script script/DeployModelStore.s.sol --broadcast</code>
          </p>
        </div>
      )}

      {/* Write error notice */}
      {writeError && (
        <div className="flex items-center gap-3 px-4 py-3 bg-red-500/5 border border-red-500/20 rounded-lg">
          <XCircle size={16} className="text-red-400 shrink-0" />
          <p className="text-2xs text-red-300/70">
            Transaction failed: {writeError.message.slice(0, 120)}
          </p>
        </div>
      )}

      {/* Modals */}
      <CreateModelModal
        isOpen={createOpen}
        onClose={() => setCreateOpen(false)}
        onCreate={createModel}
        isPending={isWritePending || isConfirming}
      />

      <AddVersionModal
        isOpen={!!versionTarget}
        onClose={() => setVersionTarget(null)}
        targetModel={versionTarget}
        addVersion={addVersion}
        isPending={isWritePending || isConfirming}
        isConnected={isConnected}
      />

      <ModelDetailModal
        model={detailModel}
        onClose={() => setDetailTokenId(null)}
        onTogglePublic={(m) => setPublic({ tokenId: m.tokenId, isPublic: !m.isPublic })}
        onDownloadWeights={handleDownloadWeights}
        onAddVersion={(m) => { setDetailTokenId(null); setVersionTarget(m); }}
        isToggling={isWritePending || isConfirming}
        isDownloading={downloadingTokenId}
      />

      <TrainedModelDetailModal
        session={detailSession}
        onClose={() => setDetailSession(null)}
        onDownload={handleDownloadTrainedWeights}
      />

      {/* Content */}
      {isLoading && isLoadingTrained ? (
        <div className="flex flex-col items-center py-16 gap-4">
          <Loader2 size={24} className="animate-spin text-helix-muted" />
          <p className="text-sm text-helix-muted">Loading models...</p>
        </div>
      ) : totalModels === 0 ? (
        <EmptyState
          icon={<Layers size={32} />}
          title="No models yet"
          description="Train a model to get started, or create one on-chain via the ERC-721 registry."
          action={
            <Link
              href="/train"
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white text-black text-sm font-medium hover:bg-white/90 transition-colors"
            >
              <Play size={14} />
              Start Training
            </Link>
          }
        />
      ) : (
        <>
          {/* Aggregate Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <StatCard
              label="Models"
              value={totalModels}
              icon={<Layers size={14} />}
            />
            <StatCard
              label="Trained"
              value={trainedModels.length}
              icon={<CheckCircle size={14} />}
            />
            <StatCard
              label="Best Accuracy"
              value={bestAccuracy > 0 ? `${(bestAccuracy * 100).toFixed(1)}%` : '--'}
              icon={<Trophy size={14} />}
            />
            <StatCard
              label="Weights Available"
              value={totalWeights}
              icon={<HardDrive size={14} />}
            />
          </div>

          {/* Trained Models Section */}
          {trainedModels.length > 0 && (
            <div className="space-y-3">
              <h2 className="text-lg font-medium text-white">Trained Models</h2>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {trainedModels.map((session) => (
                  <TrainedModelCard
                    key={session.session_id}
                    session={session}
                    onDownload={handleDownloadTrainedWeights}
                    onClickCard={(s) => setDetailSession(s)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* On-Chain Models Section */}
          {models.length > 0 && (
            <div className="space-y-3">
              <h2 className="text-lg font-medium text-white flex items-center gap-2">
                On-Chain Registry
                <Badge variant="default" className="text-2xs">{models.length}</Badge>
              </h2>

              {/* Search + Sort + Category */}
              {models.length > 1 && (
                <div className="space-y-3">
                  <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
                    <div className="relative flex-1 max-w-xs">
                      <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-helix-muted" />
                      <input
                        type="text"
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        placeholder="Search your models..."
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
                        {SORT_OPTIONS.map((o) => (
                          <option key={o.id} value={o.id}>{o.label}</option>
                        ))}
                      </select>
                    </div>
                  </div>

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
              )}

              {/* Model Cards Grid */}
              {filteredModels.length === 0 ? (
                <EmptyState
                  icon={<Search size={32} />}
                  title="No models match your filters"
                  description="Try a different search term or clear the filters."
                  action={
                    <button
                      type="button"
                      onClick={() => { setSearch(''); setCategoryFilter(null); }}
                      className="px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text hover:text-white transition-colors"
                    >
                      Clear Filters
                    </button>
                  }
                />
              ) : (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {filteredModels.map((model) => (
                    <ModelCard
                      key={model.tokenId}
                      model={model}
                      onClickCard={(m) => setDetailTokenId(m.tokenId)}
                      onAddVersion={(m) => setVersionTarget(m)}
                      onTogglePublic={(m) => setPublic({ tokenId: m.tokenId, isPublic: !m.isPublic })}
                      onDownloadWeights={handleDownloadWeights}
                      isToggling={isWritePending || isConfirming}
                      isDownloading={downloadingTokenId}
                    />
                  ))}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </motion.div>
  );
}
