'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Link from 'next/link';
import {
  Layers,
  Tag,
  CheckCircle,
  XCircle,
  HardDrive,
  Sparkles,
  Copy,
  ExternalLink,
  ChevronDown,
  ChevronRight,
  Clock,
  Trophy,
  Upload,
  Loader2,
  Wallet,
  Link2,
  Plus,
  AlertTriangle,
  Shield,
  Globe,
  Lock,
} from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { StatCard } from '@/components/ui/StatCard';
import { EmptyState } from '@/components/ui/EmptyState';
import { Modal } from '@/components/ui/Modal';
import { cn } from '@/lib/utils';
import { useModelRegistry, type ModelWithVersions } from '@/hooks/useModelRegistry';
import { useSignMessage } from 'wagmi';
import { deriveModelKey, encryptWeights } from '@/lib/model-encryption';
import type { OnChainVersion } from '@/lib/contracts';

// ============================================================================
// Constants
// ============================================================================

const SEMVER_REGEX = /^\d+\.\d+\.\d+$/;
const SLUG_REGEX = /^[a-z0-9]+(-[a-z0-9]+)*$/;
const LEGACY_HISTORY_KEY = 'helix-training-history';

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
                placeholder="784 to 32 to 10 neural network for handwritten digit classification"
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
// Model Card
// ============================================================================

interface ModelCardProps {
  model: ModelWithVersions;
  onAddVersion: (model: ModelWithVersions) => void;
  onTogglePublic: (model: ModelWithVersions) => void;
  isToggling: boolean;
}

function ModelCard({ model, onAddVersion, onTogglePublic, isToggling }: ModelCardProps) {
  const [expanded, setExpanded] = useState(false);

  const versionCount = model.versions.length;
  const bestAccuracy = model.versions.reduce(
    (best, v) => (v.accuracy > best ? v.accuracy : best),
    0,
  );
  const weightsCount = model.versions.filter((v) => v.weightsStored).length;

  return (
    <Card variant="glass" hover>
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
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

      {/* Description */}
      {model.description && (
        <p className="text-2xs text-helix-muted mb-4">{model.description}</p>
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
        <div className="mb-3">
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

      {/* Actions */}
      <div className="flex items-center gap-3 pt-2 border-t border-helix-border/50">
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

        {/* Quick inference link for best version with weights */}
        {(() => {
          const best = model.versions
            .filter((v) => v.weightsStored && v.accuracy > 0)
            .sort((a, b) => b.accuracy - a.accuracy)[0];
          if (!best) return null;
          return (
            <Link
              href={`/inference?hash=${best.rootHash}&version=${best.semver}`}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white text-black text-sm font-medium hover:bg-white/90 transition-colors"
            >
              <Sparkles size={14} />
              Inference
            </Link>
          );
        })()}
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

  const {
    address,
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

  // Aggregate stats
  const totalVersions = models.reduce((sum, m) => sum + m.versions.length, 0);
  const bestAccuracy = models.reduce((best, m) => {
    const modelBest = m.versions.reduce((b, v) => (v.accuracy > b ? v.accuracy : b), 0);
    return modelBest > best ? modelBest : best;
  }, 0);
  const totalWeights = models.reduce(
    (sum, m) => sum + m.versions.filter((v) => v.weightsStored).length,
    0,
  );

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
          {models.length > 0 && (
            <Badge variant="default">{models.length} model{models.length !== 1 ? 's' : ''}</Badge>
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

      {/* Content */}
      {isLoading ? (
        <div className="flex flex-col items-center py-16 gap-4">
          <Loader2 size={24} className="animate-spin text-helix-muted" />
          <p className="text-sm text-helix-muted">Loading models from chain...</p>
        </div>
      ) : models.length === 0 ? (
        <EmptyState
          icon={<Layers size={32} />}
          title="No models yet"
          description={
            isConnected && isContractDeployed
              ? 'Create your first model to get started. Each model is an ERC-721 NFT with versioned weights.'
              : 'Connect your wallet and ensure the contract is deployed to manage models.'
          }
          action={
            isConnected && isContractDeployed ? (
              <button
                type="button"
                onClick={() => setCreateOpen(true)}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white text-black text-sm font-medium hover:bg-white/90 transition-colors"
              >
                <Plus size={14} />
                Create Model
              </button>
            ) : undefined
          }
        />
      ) : (
        <>
          {/* Aggregate Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <StatCard
              label="Models"
              value={models.length}
              icon={<Layers size={14} />}
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
              label="Weights on 0G"
              value={totalWeights}
              icon={<HardDrive size={14} />}
            />
          </div>

          {/* Model Cards Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {models.map((model) => (
              <ModelCard
                key={model.tokenId}
                model={model}
                onAddVersion={(m) => setVersionTarget(m)}
                onTogglePublic={(m) => setPublic({ tokenId: m.tokenId, isPublic: !m.isPublic })}
                isToggling={isWritePending || isConfirming}
              />
            ))}
          </div>
        </>
      )}
    </motion.div>
  );
}
