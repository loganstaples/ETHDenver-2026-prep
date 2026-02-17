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
  Download,
  Sparkles,
  Copy,
  ExternalLink,
  ChevronDown,
  ChevronRight,
  Trash2,
  BarChart3,
  Clock,
  Trophy,
  Upload,
  Loader2,
  Wallet,
  Link2,
  Plus,
} from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { StatCard } from '@/components/ui/StatCard';
import { EmptyState } from '@/components/ui/EmptyState';
import { Modal } from '@/components/ui/Modal';
import { cn } from '@/lib/utils';
import { useModelRegistry } from '@/hooks/useModelRegistry';

// ============================================================================
// Types — mirrors TrainingHistoryEntry from train/page.tsx
// ============================================================================

interface TrainingHistoryEntry {
  sessionId: string;
  version: string;
  accuracy: number | null;
  steps: number;
  totalSteps: number;
  date: string;
  storedOn0G: boolean;
  rootHash?: string;
  status: 'complete' | 'failed';
  onChain?: boolean;
}

const HISTORY_KEY = 'helix-training-history';

function getTrainingHistory(): TrainingHistoryEntry[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveTrainingHistory(entries: TrainingHistoryEntry[]): void {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(entries));
  } catch {
    // noop
  }
}

const VERSION_REGEX = /^\d+(\.\d+){0,2}$/;

// ============================================================================
// Upload Modal
// ============================================================================

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUploaded: (entry: TrainingHistoryEntry) => void;
  isWalletConnected: boolean;
  isContractDeployed: boolean;
  onRegisterOnChain: (params: {
    version: string;
    rootHash: string;
    accuracy: number;
    sessionId: string;
  }) => void;
  isRegistering: boolean;
}

function UploadModal({
  isOpen,
  onClose,
  onUploaded,
  isWalletConnected,
  isContractDeployed,
  onRegisterOnChain,
  isRegistering,
}: UploadModalProps) {
  const [file, setFile] = useState<File | null>(null);
  const [version, setVersion] = useState('1.0.0');
  const [accuracy, setAccuracy] = useState('');
  const [versionError, setVersionError] = useState<string | null>(null);
  const [phase, setPhase] = useState<'idle' | 'validating' | 'storing' | 'registering' | 'done' | 'error'>('idle');
  const [errorMsg, setErrorMsg] = useState('');
  const [resultHash, setResultHash] = useState('');
  const fileRef = useRef<HTMLInputElement>(null);

  const reset = () => {
    setFile(null);
    setVersion('1.0.0');
    setAccuracy('');
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
    if (v.trim() && !VERSION_REGEX.test(v.trim())) {
      setVersionError('Must be x, x.y, or x.y.z');
    } else {
      setVersionError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    const v = version.trim() || '1.0.0';
    if (!VERSION_REGEX.test(v)) return;

    try {
      // Phase 1: Validate
      setPhase('validating');
      const text = await file.text();
      const data = JSON.parse(text);

      // Accept either { w1, b1, w2, b2 } or { weights: { w1, b1, w2, b2 } }
      const weights = data.weights || data;
      if (!weights.w1 || !weights.b1 || !weights.w2 || !weights.b2) {
        throw new Error('Invalid model format. Expected { w1, b1, w2, b2 } weight arrays.');
      }

      // Phase 2: Store on 0G
      setPhase('storing');
      const sessionId = `upload-${crypto.randomUUID()}`;
      const acc = accuracy ? parseFloat(accuracy) / 100 : null;

      const res = await fetch('/api/store-on-0g', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          weights,
          accuracy: acc,
          version: v,
        }),
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.error || `Upload failed: HTTP ${res.status}`);
      }

      const result = await res.json();
      const rootHash = result.root_hash;
      setResultHash(rootHash);

      // Create history entry
      const entry: TrainingHistoryEntry = {
        sessionId,
        version: v,
        accuracy: acc,
        steps: 0,
        totalSteps: 0,
        date: new Date().toISOString(),
        storedOn0G: true,
        rootHash,
        status: 'complete',
      };

      // Phase 3: Register on-chain if wallet connected
      if (isWalletConnected && isContractDeployed) {
        setPhase('registering');
        onRegisterOnChain({
          version: v,
          rootHash,
          accuracy: acc !== null ? acc * 100 : 0,
          sessionId,
        });
        entry.onChain = true;
      }

      onUploaded(entry);
      setPhase('done');
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Upload failed');
      setPhase('error');
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="Upload Model">
      <div className="space-y-4">
        {phase === 'idle' && (
          <>
            {/* File Upload */}
            <div>
              <label className="label-text block mb-1.5">Model Weights (JSON)</label>
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
                    <span className="text-sm text-helix-text2">Click to select model file</span>
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
            </div>

            {/* Version */}
            <div>
              <label className="label-text block mb-1.5">Version</label>
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

            {/* On-chain note */}
            {isWalletConnected && isContractDeployed && (
              <div className="flex items-center gap-2 text-2xs text-helix-muted bg-helix-bg border border-helix-border rounded-md px-3 py-2">
                <Link2 size={12} className="text-green-400 shrink-0" />
                Model will be registered on-chain after 0G upload
              </div>
            )}

            <button
              type="button"
              onClick={handleUpload}
              disabled={!file || !!versionError}
              className={cn(
                'w-full flex items-center justify-center gap-2 py-3 rounded-lg font-medium text-sm transition-all',
                (!file || !!versionError)
                  ? 'bg-helix-border text-helix-muted cursor-not-allowed'
                  : 'bg-white text-black hover:bg-white/90',
              )}
            >
              <Upload size={16} />
              Store on 0G{isWalletConnected && isContractDeployed ? ' & Register On-Chain' : ''}
            </button>
          </>
        )}

        {/* Progress states */}
        {(phase === 'validating' || phase === 'storing' || phase === 'registering') && (
          <div className="flex flex-col items-center py-8 gap-4">
            <Loader2 size={24} className="animate-spin text-white" />
            <p className="text-sm text-helix-text2">
              {phase === 'validating' && 'Validating model weights...'}
              {phase === 'storing' && 'Uploading to 0G decentralized storage...'}
              {phase === 'registering' && 'Registering on-chain (confirm in wallet)...'}
            </p>
            {isRegistering && phase === 'registering' && (
              <p className="text-2xs text-helix-muted">Waiting for transaction confirmation...</p>
            )}
          </div>
        )}

        {phase === 'done' && (
          <div className="flex flex-col items-center py-6 gap-3">
            <CheckCircle size={32} className="text-green-400" />
            <p className="text-sm font-medium text-white">Model Stored Successfully</p>
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
// Version Detail Expansion
// ============================================================================

interface VersionRowProps {
  entry: TrainingHistoryEntry;
  isExpanded: boolean;
  onToggle: () => void;
  onDelete: (sessionId: string) => void;
  onRegisterOnChain?: (entry: TrainingHistoryEntry) => void;
  isWalletConnected: boolean;
  isContractDeployed: boolean;
}

function VersionRow({ entry, isExpanded, onToggle, onDelete, onRegisterOnChain, isWalletConnected, isContractDeployed }: VersionRowProps) {
  const [copied, setCopied] = useState(false);

  const copyHash = (hash: string) => {
    navigator.clipboard.writeText(hash);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isComplete = entry.status === 'complete';

  return (
    <div className="border border-helix-border rounded-lg overflow-hidden">
      {/* Row Header */}
      <button
        type="button"
        onClick={onToggle}
        className="w-full flex items-center gap-3 px-4 py-3 bg-helix-surface hover:bg-helix-surface2 transition-colors text-left"
      >
        {isExpanded ? (
          <ChevronDown size={14} className="text-helix-muted shrink-0" />
        ) : (
          <ChevronRight size={14} className="text-helix-muted shrink-0" />
        )}

        {isComplete ? (
          <CheckCircle size={14} className="text-green-400 shrink-0" />
        ) : (
          <XCircle size={14} className="text-red-400 shrink-0" />
        )}

        <Badge variant="default" className="font-mono text-2xs shrink-0">v{entry.version}</Badge>

        <span className="text-xs font-mono text-helix-muted truncate">
          {entry.sessionId.slice(0, 12)}...
        </span>

        <div className="flex items-center gap-3 ml-auto shrink-0">
          {entry.accuracy !== null && (
            <span className="text-sm font-mono text-white">
              {(entry.accuracy * 100).toFixed(1)}%
            </span>
          )}

          {entry.steps > 0 && (
            <span className="text-xs font-mono text-helix-muted">
              {entry.steps}/{entry.totalSteps}
            </span>
          )}

          {entry.onChain && (
            <Badge variant="default" className="text-blue-400">
              <Link2 size={10} />
              On-Chain
            </Badge>
          )}

          {entry.storedOn0G ? (
            <Badge variant="default" className="text-green-400">
              <HardDrive size={10} />
              0G
            </Badge>
          ) : (
            <span className="text-2xs text-helix-dim">Local</span>
          )}

          <span className="text-2xs text-helix-dim">
            {new Date(entry.date).toLocaleDateString()}
          </span>
        </div>
      </button>

      {/* Expanded Detail */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-4 py-4 bg-helix-bg border-t border-helix-border space-y-4">
              {/* Stats Grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="bg-helix-surface rounded-lg p-3">
                  <p className="label-text mb-1">Accuracy</p>
                  <p className="text-lg font-mono font-light text-white">
                    {entry.accuracy !== null ? `${(entry.accuracy * 100).toFixed(2)}%` : '--'}
                  </p>
                </div>
                <div className="bg-helix-surface rounded-lg p-3">
                  <p className="label-text mb-1">Steps</p>
                  <p className="text-lg font-mono font-light text-white">
                    {entry.steps > 0 ? `${entry.steps} / ${entry.totalSteps}` : 'Uploaded'}
                  </p>
                </div>
                <div className="bg-helix-surface rounded-lg p-3">
                  <p className="label-text mb-1">Storage</p>
                  <p className={cn(
                    'text-lg font-mono font-light',
                    entry.storedOn0G ? 'text-green-400' : 'text-helix-muted',
                  )}>
                    {entry.storedOn0G ? '0G Storage' : 'Local'}
                  </p>
                </div>
                <div className="bg-helix-surface rounded-lg p-3">
                  <p className="label-text mb-1">On-Chain</p>
                  <p className={cn(
                    'text-lg font-mono font-light',
                    entry.onChain ? 'text-blue-400' : 'text-helix-muted',
                  )}>
                    {entry.onChain ? 'Registered' : 'No'}
                  </p>
                </div>
              </div>

              {/* Session ID */}
              <div>
                <p className="label-text mb-1.5">Session ID</p>
                <div className="flex items-center gap-2">
                  <code className="flex-1 text-xs font-mono text-helix-text bg-helix-surface px-3 py-2 rounded-md border border-helix-border truncate">
                    {entry.sessionId}
                  </code>
                  <button
                    type="button"
                    onClick={() => copyHash(entry.sessionId)}
                    className="shrink-0 p-2 rounded-md bg-helix-surface border border-helix-border text-helix-muted hover:text-white transition-colors"
                  >
                    {copied ? <CheckCircle size={14} /> : <Copy size={14} />}
                  </button>
                </div>
              </div>

              {/* 0G Storage Info */}
              {entry.storedOn0G && entry.rootHash && (
                <div>
                  <p className="label-text mb-1.5">0G Storage Root Hash</p>
                  <div className="flex items-center gap-2">
                    <code className="flex-1 text-xs font-mono text-helix-text bg-helix-surface px-3 py-2 rounded-md border border-helix-border truncate">
                      {entry.rootHash}
                    </code>
                    <button
                      type="button"
                      onClick={() => copyHash(entry.rootHash!)}
                      className="shrink-0 p-2 rounded-md bg-helix-surface border border-helix-border text-helix-muted hover:text-white transition-colors"
                    >
                      {copied ? <CheckCircle size={14} /> : <Copy size={14} />}
                    </button>
                    <a
                      href={`https://storagescan-galileo.0g.ai/file/${entry.rootHash}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="shrink-0 p-2 rounded-md bg-helix-surface border border-helix-border text-helix-muted hover:text-white transition-colors"
                    >
                      <ExternalLink size={14} />
                    </a>
                  </div>
                </div>
              )}

              {/* Actions */}
              <div className="flex items-center gap-3 pt-1 flex-wrap">
                {isComplete && ((entry.storedOn0G && entry.rootHash) || !entry.sessionId.startsWith('upload-')) && (
                  <Link
                    href={`/inference?session=${entry.sessionId}${entry.rootHash ? `&hash=${entry.rootHash}` : ''}&version=${entry.version}`}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white text-black text-sm font-medium hover:bg-white/90 transition-colors"
                  >
                    <Sparkles size={14} />
                    Run Inference
                  </Link>
                )}
                {isComplete && !entry.sessionId.startsWith('upload-') && (
                  <a
                    href={`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001'}/api/training/sessions/${entry.sessionId}/model`}
                    download={`helix-model-v${entry.version}-${entry.sessionId.slice(0, 8)}.json`}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text hover:border-helix-border2 hover:text-white transition-colors"
                  >
                    <Download size={14} />
                    Download
                  </a>
                )}
                {/* Register on-chain if not already */}
                {!entry.onChain && isWalletConnected && isContractDeployed && onRegisterOnChain && (
                  <button
                    type="button"
                    onClick={() => onRegisterOnChain(entry)}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-blue-400 hover:border-blue-500/30 transition-colors"
                  >
                    <Link2 size={14} />
                    Register On-Chain
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => onDelete(entry.sessionId)}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-muted hover:text-red-400 hover:border-red-500/30 transition-colors ml-auto"
                >
                  <Trash2 size={14} />
                  Remove
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ============================================================================
// Main Page
// ============================================================================

export default function MyModelsPage() {
  const [history, setHistory] = useState<TrainingHistoryEntry[]>([]);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [uploadOpen, setUploadOpen] = useState(false);

  const {
    isConnected,
    isContractDeployed,
    onChainModels,
    isLoading: isLoadingOnChain,
    registerModel,
    isRegistering,
    isConfirming,
    isRegistered,
    refetch,
  } = useModelRegistry();

  // Load history from localStorage
  useEffect(() => {
    setHistory(getTrainingHistory());
  }, []);

  // Merge on-chain data: mark local entries as on-chain if they exist in the contract
  useEffect(() => {
    if (onChainModels.length === 0) return;

    setHistory((prev) => {
      let changed = false;
      const updated = prev.map((entry) => {
        const onChain = onChainModels.find(
          (m) => m.sessionId === entry.sessionId || m.rootHash === entry.rootHash,
        );
        if (onChain && !entry.onChain) {
          changed = true;
          return { ...entry, onChain: true };
        }
        return entry;
      });

      // Also add on-chain entries not in localStorage
      for (const m of onChainModels) {
        const exists = updated.some(
          (e) => e.sessionId === m.sessionId || (m.rootHash && e.rootHash === m.rootHash),
        );
        if (!exists) {
          changed = true;
          updated.push({
            sessionId: m.sessionId,
            version: m.version,
            accuracy: m.accuracy > 0 ? m.accuracy / 100 : null,
            steps: 0,
            totalSteps: 0,
            date: new Date(m.timestamp * 1000).toISOString(),
            storedOn0G: m.rootHash !== '',
            rootHash: m.rootHash || undefined,
            status: 'complete',
            onChain: true,
          });
        }
      }

      if (changed) saveTrainingHistory(updated);
      return changed ? updated : prev;
    });
  }, [onChainModels]);

  // Refetch on-chain data after successful registration
  useEffect(() => {
    if (isRegistered) refetch();
  }, [isRegistered, refetch]);

  const handleDelete = useCallback((sessionId: string) => {
    setHistory((prev) => {
      const updated = prev.filter((e) => e.sessionId !== sessionId);
      saveTrainingHistory(updated);
      return updated;
    });
    if (expandedId === sessionId) setExpandedId(null);
  }, [expandedId]);

  const handleUploaded = useCallback((entry: TrainingHistoryEntry) => {
    setHistory((prev) => {
      const updated = [...prev, entry];
      saveTrainingHistory(updated);
      return updated;
    });
  }, []);

  const handleRegisterOnChain = useCallback((entry: TrainingHistoryEntry) => {
    registerModel({
      version: entry.version,
      rootHash: entry.rootHash || '',
      accuracy: entry.accuracy !== null ? entry.accuracy * 100 : 0,
      sessionId: entry.sessionId,
    });

    // Optimistically mark as on-chain
    setHistory((prev) => {
      const updated = prev.map((e) =>
        e.sessionId === entry.sessionId ? { ...e, onChain: true } : e,
      );
      saveTrainingHistory(updated);
      return updated;
    });
  }, [registerModel]);

  // Compute aggregate stats
  const completedVersions = history.filter((e) => e.status === 'complete');
  const storedOn0G = history.filter((e) => e.storedOn0G);
  const onChainCount = history.filter((e) => e.onChain).length;
  const bestAccuracy = completedVersions.reduce(
    (best, e) => (e.accuracy !== null && e.accuracy > best ? e.accuracy : best),
    0,
  );
  const latestVersion = history.length > 0
    ? history[history.length - 1].version
    : '--';

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
          {history.length > 0 && (
            <Badge variant="default">{history.length} version{history.length !== 1 ? 's' : ''}</Badge>
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
            onClick={() => setUploadOpen(true)}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text hover:border-helix-border2 hover:text-white transition-colors"
          >
            <Plus size={14} />
            Upload Model
          </button>

          <Link
            href="/train"
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white text-black text-sm font-medium hover:bg-white/90 transition-colors"
          >
            Train New
          </Link>
        </div>
      </div>

      {/* Upload Modal */}
      <UploadModal
        isOpen={uploadOpen}
        onClose={() => setUploadOpen(false)}
        onUploaded={handleUploaded}
        isWalletConnected={isConnected}
        isContractDeployed={isContractDeployed}
        onRegisterOnChain={(params) => registerModel(params)}
        isRegistering={isRegistering || isConfirming}
      />

      {history.length === 0 && !isLoadingOnChain ? (
        <EmptyState
          icon={<Layers size={32} />}
          title="No models yet"
          description="Upload a model or train one to see it here. Connect your wallet to store the model list on-chain."
          action={
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => setUploadOpen(true)}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text hover:text-white transition-colors"
              >
                <Upload size={14} />
                Upload Model
              </button>
              <Link
                href="/train"
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white text-black text-sm font-medium hover:bg-white/90 transition-colors"
              >
                Start Training
              </Link>
            </div>
          }
        />
      ) : (
        <>
          {/* Model Overview */}
          <Card variant="glass">
            <div className="flex items-center gap-3 mb-5">
              <div className="w-10 h-10 rounded-lg bg-white/[0.06] flex items-center justify-center">
                <Layers size={20} className="text-white" />
              </div>
              <div>
                <h2 className="text-base font-medium text-white">HELIX MNIST Classifier</h2>
                <p className="text-2xs text-helix-muted font-mono">784 → 128 → 10 · ~102K params</p>
              </div>
              <Badge variant="default" className="ml-auto font-mono">v{latestVersion}</Badge>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              <StatCard
                label="Versions"
                value={history.length}
                icon={<Tag size={14} />}
              />
              <StatCard
                label="Best Accuracy"
                value={bestAccuracy > 0 ? `${(bestAccuracy * 100).toFixed(1)}%` : '--'}
                icon={<Trophy size={14} />}
              />
              <StatCard
                label="On 0G Storage"
                value={storedOn0G.length}
                icon={<HardDrive size={14} />}
              />
              <StatCard
                label="On-Chain"
                value={onChainCount}
                icon={<Link2 size={14} />}
              />
              <StatCard
                label="Completed"
                value={`${completedVersions.length}/${history.length}`}
                icon={<BarChart3 size={14} />}
              />
            </div>
          </Card>

          {/* On-chain notice */}
          {!isConnected && (
            <div className="flex items-center gap-3 px-4 py-3 bg-helix-bg border border-helix-border rounded-lg">
              <Wallet size={16} className="text-helix-muted shrink-0" />
              <p className="text-2xs text-helix-muted">
                Connect your wallet to register models on-chain and persist your model list across devices.
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

          {/* Version History */}
          <div>
            <div className="flex items-center gap-2 mb-4">
              <Clock size={14} className="text-helix-text2" />
              <h3 className="text-sm font-medium text-white">Version History</h3>
            </div>

            <div className="space-y-2">
              {history.slice().reverse().map((entry) => (
                <VersionRow
                  key={entry.sessionId}
                  entry={entry}
                  isExpanded={expandedId === entry.sessionId}
                  onToggle={() =>
                    setExpandedId(expandedId === entry.sessionId ? null : entry.sessionId)
                  }
                  onDelete={handleDelete}
                  onRegisterOnChain={handleRegisterOnChain}
                  isWalletConnected={isConnected}
                  isContractDeployed={isContractDeployed}
                />
              ))}
            </div>
          </div>

          {/* Quick Actions */}
          {storedOn0G.length > 0 && (
            <Card variant="default">
              <div className="flex items-center gap-2 mb-3">
                <Sparkles size={14} className="text-helix-text2" />
                <h3 className="text-sm font-medium text-white">Quick Inference</h3>
              </div>
              <p className="text-2xs text-helix-muted mb-3">
                Run inference on your best model stored on 0G decentralized storage.
              </p>
              {(() => {
                const best = storedOn0G
                  .filter((e) => e.status === 'complete' && e.accuracy !== null)
                  .sort((a, b) => (b.accuracy ?? 0) - (a.accuracy ?? 0))[0];
                if (!best) return null;
                return (
                  <Link
                    href={`/inference?session=${best.sessionId}${best.rootHash ? `&hash=${best.rootHash}` : ''}&version=${best.version}`}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-white text-black text-sm font-medium hover:bg-white/90 transition-colors"
                  >
                    <Sparkles size={14} />
                    Run Inference (v{best.version} · {((best.accuracy ?? 0) * 100).toFixed(1)}%)
                  </Link>
                );
              })()}
            </Card>
          )}
        </>
      )}
    </motion.div>
  );
}
