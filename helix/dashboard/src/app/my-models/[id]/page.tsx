'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ArrowLeft,
  Layers,
  Tag,
  CheckCircle,
  XCircle,
  HardDrive,
  Copy,
  ExternalLink,
  Trophy,
  Upload,
  Loader2,
  Plus,
  Shield,
  Globe,
  Lock,
  Play,
  DollarSign,
  Save,
  Download,
} from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { cn } from '@/lib/utils';
import { useModelDetail, type ModelDetail } from '@/hooks/useModelDetail';
import { useModelRegistry } from '@/hooks/useModelRegistry';
import { useSignMessage } from 'wagmi';
import { deriveModelKey, encryptWeights, decryptWeights } from '@/lib/model-encryption';
import type { OnChainVersion } from '@/lib/contracts';

// ============================================================================
// Constants & Helpers
// ============================================================================

const SEMVER_REGEX = /^\d+\.\d+\.\d+$/;

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

function formatFee(bps: number): string {
  return `${(bps / 100).toFixed(1)}%`;
}

// ============================================================================
// Download weights helper
// ============================================================================

async function downloadWeightsFromVersion(
  version: OnChainVersion,
  tokenId: number,
  signMessageAsync: (args: { message: string }) => Promise<string>,
): Promise<void> {
  // Fetch from 0G
  const res = await fetch('/api/fetch-from-0g', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rootHash: version.rootHash }),
  });

  if (!res.ok) {
    const errBody = await res.json().catch(() => ({}));
    throw new Error(errBody.error || `Fetch failed: HTTP ${res.status}`);
  }

  const result = await res.json();
  let weightsJson: string;

  if (result.encoding === 'base64') {
    // Encrypted data - need to decrypt
    const key = await deriveModelKey(
      (message: string) => signMessageAsync({ message }),
      tokenId,
    );
    const binaryStr = atob(result.data);
    const bytes = new Uint8Array(binaryStr.length);
    for (let i = 0; i < binaryStr.length; i++) {
      bytes[i] = binaryStr.charCodeAt(i);
    }
    weightsJson = await decryptWeights(key, bytes);
  } else {
    // Plain JSON
    weightsJson = JSON.stringify(result.data, null, 2);
  }

  // Trigger browser download
  const blob = new Blob([weightsJson], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `weights-v${version.semver}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ============================================================================
// Tab types
// ============================================================================

type TabId = 'overview' | 'settings' | 'versions';

const TABS: { id: TabId; label: string }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'settings', label: 'Settings' },
  { id: 'versions', label: 'Versions' },
];

// ============================================================================
// Version Row
// ============================================================================

function VersionRow({
  version,
  tokenId,
  signMessageAsync,
}: {
  version: OnChainVersion;
  tokenId: number;
  signMessageAsync: (args: { message: string }) => Promise<string>;
}) {
  const [copied, setCopied] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  const copyHash = (hash: string) => {
    navigator.clipboard.writeText(hash);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = async () => {
    setIsDownloading(true);
    setDownloadError(null);
    try {
      await downloadWeightsFromVersion(version, tokenId, signMessageAsync);
    } catch (err) {
      setDownloadError(err instanceof Error ? err.message : 'Download failed');
      setTimeout(() => setDownloadError(null), 5000);
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div className="flex items-center gap-3 px-4 py-3 bg-helix-bg rounded-md border border-helix-border/50">
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

      {/* Download button for versions with stored weights */}
      {version.weightsStored && version.rootHash && (
        <button
          type="button"
          onClick={handleDownload}
          disabled={isDownloading}
          title={downloadError || 'Download weights'}
          className={cn(
            'flex items-center gap-1.5 px-2.5 py-1 rounded-md text-2xs font-medium transition-colors',
            downloadError
              ? 'bg-red-500/10 border border-red-500/30 text-red-400'
              : isDownloading
                ? 'bg-helix-surface border border-helix-border text-helix-muted cursor-wait'
                : 'bg-helix-surface border border-helix-border text-helix-text hover:border-helix-border2 hover:text-white',
          )}
        >
          {isDownloading ? (
            <Loader2 size={10} className="animate-spin" />
          ) : (
            <Download size={10} />
          )}
          {downloadError ? 'Error' : isDownloading ? 'Downloading...' : 'Download'}
        </button>
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
// Add Version Modal (inline)
// ============================================================================

type VersionPhase = 'idle' | 'validating' | 'encrypting' | 'uploading' | 'registering' | 'done' | 'error';

function AddVersionSection({
  model,
  addVersion,
  isPending,
  isConnected,
}: {
  model: ModelDetail;
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
}) {
  const [isOpen, setIsOpen] = useState(false);
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

  const handleVersionChange = (v: string) => {
    setVersion(v);
    if (v.trim() && !SEMVER_REGEX.test(v.trim())) {
      setVersionError('Must be semver (e.g. 1.0.0)');
    } else {
      setVersionError(null);
    }
  };

  const canSubmit = !versionError && version.trim() && !isPending && phase === 'idle';

  const handleSubmit = async () => {
    const v = version.trim() || '1.0.0';
    if (!SEMVER_REGEX.test(v)) return;

    const acc = accuracy ? parseFloat(accuracy) : 0;
    const sessionId = `v${v}-${Date.now()}`;

    try {
      if (file) {
        setPhase('validating');
        const text = await file.text();
        const data = JSON.parse(text);
        const weights = data.weights || data;
        if (!weights.w1 || !weights.b1 || !weights.w2 || !weights.b2) {
          throw new Error('Invalid model format. Expected { w1, b1, w2, b2 } weight arrays.');
        }

        let rootHash = '';

        if (isConnected) {
          setPhase('encrypting');
          const key = await deriveModelKey(
            (message: string) => signMessageAsync({ message }),
            model.tokenId,
          );
          const encrypted = await encryptWeights(key, JSON.stringify(weights));
          const encryptedBase64 = btoa(String.fromCharCode(...encrypted));

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

        setPhase('registering');
        addVersion({
          tokenId: model.tokenId,
          semver: v,
          rootHash,
          accuracy: acc,
          sessionId,
          weightsStored: true,
        });
        setPhase('done');
      } else {
        setPhase('registering');
        addVersion({
          tokenId: model.tokenId,
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

  if (!isOpen) {
    return (
      <button
        type="button"
        onClick={() => setIsOpen(true)}
        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text hover:border-helix-border2 hover:text-white transition-colors"
      >
        <Plus size={14} />
        Add Version
      </button>
    );
  }

  return (
    <Card variant="default" className="space-y-4">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-white">Add Version</h4>
        <button
          type="button"
          onClick={() => { reset(); setIsOpen(false); }}
          className="text-2xs text-helix-muted hover:text-white transition-colors"
        >
          Cancel
        </button>
      </div>

      {phase === 'idle' && (
        <>
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
                  <span className="text-2xs text-helix-muted">{(file.size / 1024).toFixed(0)} KB</span>
                </>
              ) : (
                <>
                  <Upload size={20} className="text-helix-muted" />
                  <span className="text-sm text-helix-text2">Click to select weights file</span>
                  <span className="text-2xs text-helix-muted">JSON with w1, b1, w2, b2 weight arrays</span>
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

      {(phase === 'validating' || phase === 'encrypting' || phase === 'uploading' || phase === 'registering') && (
        <div className="flex flex-col items-center py-8 gap-4">
          <Loader2 size={24} className="animate-spin text-white" />
          <p className="text-sm text-helix-text2">
            {phase === 'validating' && 'Validating model weights...'}
            {phase === 'encrypting' && 'Encrypting weights (sign in wallet)...'}
            {phase === 'uploading' && 'Uploading to 0G decentralized storage...'}
            {phase === 'registering' && 'Registering on-chain (confirm in wallet)...'}
          </p>
        </div>
      )}

      {phase === 'done' && (
        <div className="flex flex-col items-center py-6 gap-3">
          <CheckCircle size={32} className="text-green-400" />
          <p className="text-sm font-medium text-white">Version Registered</p>
          {resultHash && (
            <code className="block text-xs font-mono text-helix-text bg-helix-bg px-3 py-2 rounded-md border border-helix-border truncate w-full">
              {resultHash}
            </code>
          )}
          <button
            type="button"
            onClick={() => { reset(); setIsOpen(false); }}
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
    </Card>
  );
}

// ============================================================================
// Overview Tab
// ============================================================================

function OverviewTab({ model }: { model: ModelDetail }) {
  const versionCount = model.versions.length;
  const bestAccuracy = model.versions.reduce(
    (best, v) => (v.accuracy > best ? v.accuracy : best),
    0,
  );
  const weightsCount = model.versions.filter((v) => v.weightsStored).length;
  const latestVersion = versionCount > 0 ? model.versions[versionCount - 1] : null;

  return (
    <div className="space-y-4">
      {/* Description */}
      {model.description && (
        <Card variant="default">
          <h4 className="text-sm font-medium text-white mb-2">Description</h4>
          <p className="text-sm text-helix-text2 leading-relaxed">{model.description}</p>
        </Card>
      )}

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-helix-surface border border-helix-border rounded-lg p-4">
          <div className="flex items-center gap-2 mb-1.5">
            <Tag size={12} className="text-helix-dim" />
            <span className="label-text">Versions</span>
          </div>
          <p className="text-xl font-mono font-light text-white">{versionCount}</p>
        </div>
        <div className="bg-helix-surface border border-helix-border rounded-lg p-4">
          <div className="flex items-center gap-2 mb-1.5">
            <Trophy size={12} className="text-helix-dim" />
            <span className="label-text">Best Accuracy</span>
          </div>
          <p className="text-xl font-mono font-light text-white">
            {bestAccuracy > 0 ? `${(bestAccuracy * 100).toFixed(1)}%` : '--'}
          </p>
        </div>
        <div className="bg-helix-surface border border-helix-border rounded-lg p-4">
          <div className="flex items-center gap-2 mb-1.5">
            <HardDrive size={12} className="text-helix-dim" />
            <span className="label-text">On 0G</span>
          </div>
          <p className="text-xl font-mono font-light text-white">{weightsCount}</p>
        </div>
        <div className="bg-helix-surface border border-helix-border rounded-lg p-4">
          <div className="flex items-center gap-2 mb-1.5">
            <DollarSign size={12} className="text-helix-dim" />
            <span className="label-text">Inference Fee</span>
          </div>
          <p className="text-xl font-mono font-light text-white">{formatFee(model.inferenceFee)}</p>
        </div>
      </div>

      {/* Model Info */}
      <Card variant="default">
        <h4 className="text-sm font-medium text-white mb-3">Model Info</h4>
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-helix-muted">Token ID</span>
            <span className="font-mono text-helix-text">#{model.tokenId}</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-helix-muted">Slug</span>
            <span className="font-mono text-helix-text">{model.slug}</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-helix-muted">Creator</span>
            <span className="font-mono text-helix-text">{truncateAddress(model.creator)}</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-helix-muted">Owner</span>
            <span className="font-mono text-helix-text">{truncateAddress(model.owner)}</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-helix-muted">Created</span>
            <span className="text-helix-text">{formatDate(model.createdAt)}</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-helix-muted">Visibility</span>
            <span className={cn('flex items-center gap-1.5', model.isPublic ? 'text-green-400' : 'text-helix-muted')}>
              {model.isPublic ? <Globe size={12} /> : <Lock size={12} />}
              {model.isPublic ? 'Public' : 'Private'}
            </span>
          </div>
        </div>
      </Card>

      {/* Latest Version */}
      {latestVersion && (
        <Card variant="default">
          <h4 className="text-sm font-medium text-white mb-3">Latest Version</h4>
          <VersionRow version={latestVersion} tokenId={model.tokenId} signMessageAsync={() => { throw new Error('Connect wallet to download'); }} />
        </Card>
      )}
    </div>
  );
}

// ============================================================================
// Settings Tab
// ============================================================================

function SettingsTab({
  model,
  setPublic,
  setInferenceFee,
  setForSale,
  setSalePrice,
  isWritePending,
}: {
  model: ModelDetail;
  setPublic: (params: { tokenId: number; isPublic: boolean }) => void;
  setInferenceFee: (params: { tokenId: number; feeBps: number }) => void;
  setForSale: (params: { tokenId: number; forSale: boolean }) => void;
  setSalePrice: (params: { tokenId: number; priceEth: number }) => void;
  isWritePending: boolean;
}) {
  const [feeBps, setFeeBps] = useState(model.inferenceFee);
  const [feeInput, setFeeInput] = useState((model.inferenceFee / 100).toFixed(1));
  const feeChanged = feeBps !== model.inferenceFee;

  const [priceInput, setPriceInput] = useState(model.salePrice > 0 ? model.salePrice.toString() : '');
  const parsedPrice = parseFloat(priceInput);
  const priceValid = !isNaN(parsedPrice) && parsedPrice > 0;
  const priceChanged = priceValid && parsedPrice !== model.salePrice;

  const handleFeeInput = (v: string) => {
    setFeeInput(v);
    const parsed = parseFloat(v);
    if (!isNaN(parsed) && parsed >= 0 && parsed <= 50) {
      setFeeBps(Math.round(parsed * 100));
    }
  };

  const handleSaveFee = () => {
    if (!feeChanged || isWritePending) return;
    setInferenceFee({ tokenId: model.tokenId, feeBps });
  };

  const handleToggleForSale = () => {
    if (isWritePending) return;
    setForSale({ tokenId: model.tokenId, forSale: !model.forSale });
  };

  const handleSavePrice = () => {
    if (!priceChanged || isWritePending) return;
    setSalePrice({ tokenId: model.tokenId, priceEth: parsedPrice });
  };

  return (
    <div className="space-y-4">
      {/* Visibility */}
      <Card variant="default">
        <h4 className="text-sm font-medium text-white mb-3">Visibility</h4>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-helix-text">
              {model.isPublic ? 'Public' : 'Private'} Model
            </p>
            <p className="text-2xs text-helix-muted mt-0.5">
              {model.isPublic
                ? 'Visible to everyone on the marketplace. Others can request inference.'
                : 'Only you can see and use this model.'}
            </p>
          </div>
          <button
            type="button"
            onClick={() => setPublic({ tokenId: model.tokenId, isPublic: !model.isPublic })}
            disabled={isWritePending}
            className={cn(
              'flex items-center gap-2 px-4 py-2 rounded-lg border text-sm transition-colors',
              model.isPublic
                ? 'bg-helix-surface border-helix-border text-helix-text hover:border-helix-border2 hover:text-white'
                : 'bg-green-500/10 border-green-500/30 text-green-400 hover:bg-green-500/20',
              isWritePending && 'opacity-50 cursor-not-allowed',
            )}
          >
            {isWritePending ? (
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

      {/* Inference Fee */}
      <Card variant="default">
        <h4 className="text-sm font-medium text-white mb-3">Inference Fee</h4>
        <p className="text-2xs text-helix-muted mb-4">
          Fee charged to users who run inference on your model. Set as a percentage (0-50%). Stored as basis points on-chain.
        </p>
        <div className="flex items-end gap-3">
          <div className="flex-1">
            <label className="label-text block mb-1.5">Fee (%)</label>
            <input
              type="number"
              value={feeInput}
              onChange={(e) => handleFeeInput(e.target.value)}
              min={0}
              max={50}
              step={0.1}
              className="w-full px-3 py-2 bg-helix-bg border border-helix-border rounded-md text-sm text-helix-text font-mono focus:outline-none focus:border-helix-border2 transition-colors"
            />
          </div>
          <div className="text-sm text-helix-muted font-mono pb-2">
            = {feeBps} bps
          </div>
          <button
            type="button"
            onClick={handleSaveFee}
            disabled={!feeChanged || isWritePending}
            className={cn(
              'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all',
              (!feeChanged || isWritePending)
                ? 'bg-helix-border text-helix-muted cursor-not-allowed'
                : 'bg-white text-black hover:bg-white/90',
            )}
          >
            {isWritePending ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <Save size={14} />
            )}
            Save
          </button>
        </div>
        {/* Fee preview */}
        <div className="mt-3 px-3 py-2 bg-helix-bg rounded-md border border-helix-border/50">
          <div className="flex items-center justify-between text-2xs text-helix-muted">
            <span>Current on-chain</span>
            <span className="font-mono">{formatFee(model.inferenceFee)} ({model.inferenceFee} bps)</span>
          </div>
          {feeChanged && (
            <div className="flex items-center justify-between text-2xs text-white mt-1">
              <span>New fee</span>
              <span className="font-mono">{formatFee(feeBps)} ({feeBps} bps)</span>
            </div>
          )}
        </div>
      </Card>

      {/* Marketplace / Sale Settings */}
      <Card variant="default">
        <h4 className="text-sm font-medium text-white mb-3">Marketplace</h4>
        <p className="text-2xs text-helix-muted mb-4">
          Control whether your model NFT is listed for sale on the marketplace.
        </p>

        {/* For Sale toggle */}
        <div className="flex items-center justify-between mb-4">
          <div>
            <p className="text-sm text-helix-text">
              {model.forSale ? 'Listed for Sale' : 'Not for Sale'}
            </p>
            <p className="text-2xs text-helix-muted mt-0.5">
              {model.forSale
                ? 'Your model is visible on the marketplace and can be purchased.'
                : 'Your model is not listed. Toggle to make it available for purchase.'}
            </p>
          </div>
          <button
            type="button"
            onClick={handleToggleForSale}
            disabled={isWritePending}
            className={cn(
              'flex items-center gap-2 px-4 py-2 rounded-lg border text-sm transition-colors',
              model.forSale
                ? 'bg-helix-surface border-helix-border text-helix-text hover:border-helix-border2 hover:text-white'
                : 'bg-green-500/10 border-green-500/30 text-green-400 hover:bg-green-500/20',
              isWritePending && 'opacity-50 cursor-not-allowed',
            )}
          >
            {isWritePending ? (
              <Loader2 size={14} className="animate-spin" />
            ) : model.forSale ? (
              <XCircle size={14} />
            ) : (
              <DollarSign size={14} />
            )}
            {model.forSale ? 'Remove Listing' : 'List for Sale'}
          </button>
        </div>

        {/* Sale price input (only when for sale) */}
        {model.forSale && (
          <>
            <div className="flex items-end gap-3">
              <div className="flex-1">
                <label className="label-text block mb-1.5">Sale Price (ETH)</label>
                <input
                  type="number"
                  value={priceInput}
                  onChange={(e) => setPriceInput(e.target.value)}
                  placeholder="e.g. 0.5"
                  min={0}
                  step={0.01}
                  className="w-full px-3 py-2 bg-helix-bg border border-helix-border rounded-md text-sm text-helix-text font-mono focus:outline-none focus:border-helix-border2 transition-colors"
                />
              </div>
              <button
                type="button"
                onClick={handleSavePrice}
                disabled={!priceChanged || isWritePending}
                className={cn(
                  'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all',
                  (!priceChanged || isWritePending)
                    ? 'bg-helix-border text-helix-muted cursor-not-allowed'
                    : 'bg-white text-black hover:bg-white/90',
                )}
              >
                {isWritePending ? (
                  <Loader2 size={14} className="animate-spin" />
                ) : (
                  <Save size={14} />
                )}
                Save
              </button>
            </div>

            {/* Price preview */}
            <div className="mt-3 px-3 py-2 bg-helix-bg rounded-md border border-helix-border/50">
              <div className="flex items-center justify-between text-2xs text-helix-muted">
                <span>Current on-chain price</span>
                <span className="font-mono">{model.salePrice > 0 ? `${model.salePrice} ETH` : 'Not set'}</span>
              </div>
              {priceChanged && (
                <div className="flex items-center justify-between text-2xs text-white mt-1">
                  <span>New price</span>
                  <span className="font-mono">{parsedPrice} ETH</span>
                </div>
              )}
            </div>

            <p className="text-2xs text-helix-dim mt-3">
              When listed for sale, your model NFT can be purchased by anyone on the marketplace.
            </p>
          </>
        )}
      </Card>

      {/* Danger Zone (placeholder) */}
      <Card variant="default" className="border-red-500/20">
        <h4 className="text-sm font-medium text-red-400 mb-3">Danger Zone</h4>
        <p className="text-2xs text-helix-muted">
          Model ownership can be transferred via ERC-721 transfer. This is irreversible.
          Use your wallet or a marketplace to transfer the NFT.
        </p>
      </Card>
    </div>
  );
}

// ============================================================================
// Versions Tab
// ============================================================================

function VersionsTab({
  model,
  addVersion,
  isPending,
  isConnected,
  signMessageAsync,
}: {
  model: ModelDetail;
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
  signMessageAsync: (args: { message: string }) => Promise<string>;
}) {
  const versions = [...model.versions].reverse();

  return (
    <div className="space-y-4">
      <AddVersionSection
        model={model}
        addVersion={addVersion}
        isPending={isPending}
        isConnected={isConnected}
      />

      {versions.length === 0 ? (
        <Card variant="default" className="text-center py-8">
          <Tag size={24} className="text-helix-muted mx-auto mb-2" />
          <p className="text-sm text-helix-muted">No versions yet</p>
          <p className="text-2xs text-helix-dim mt-1">Add a version to get started, or train your model first.</p>
        </Card>
      ) : (
        <div className="space-y-2">
          {versions.map((v, i) => (
            <VersionRow
              key={`${v.semver}-${i}`}
              version={v}
              tokenId={model.tokenId}
              signMessageAsync={signMessageAsync}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Main Page
// ============================================================================

export default function MyModelDetailPage() {
  const params = useParams();
  const tokenId = Number(params.id);
  const [activeTab, setActiveTab] = useState<TabId>('overview');
  const [isDownloadingLatest, setIsDownloadingLatest] = useState(false);

  const {
    model,
    isLoading,
    isError,
    isOwner,
    isConnected,
    refetch: refetchDetail,
  } = useModelDetail(tokenId);

  const {
    addVersion,
    setPublic,
    setInferenceFee,
    setForSale,
    setSalePrice,
    isWritePending,
    isConfirming,
    writeError,
    isSuccess,
  } = useModelRegistry();

  const { signMessageAsync } = useSignMessage();

  // Refetch after successful write
  useEffect(() => {
    if (isSuccess) {
      refetchDetail();
    }
  }, [isSuccess, refetchDetail]);

  // Determine if latest version has downloadable weights
  const latestVersion = model && model.versions.length > 0
    ? model.versions[model.versions.length - 1]
    : null;
  const latestHasWeights = latestVersion?.weightsStored && !!latestVersion?.rootHash;

  const handleDownloadLatest = useCallback(async () => {
    if (!latestVersion || !latestHasWeights || !model) return;
    setIsDownloadingLatest(true);
    try {
      await downloadWeightsFromVersion(latestVersion, model.tokenId, signMessageAsync);
    } catch {
      // Error is handled silently for header button; user can use per-version button for details
    } finally {
      setIsDownloadingLatest(false);
    }
  }, [latestVersion, latestHasWeights, model, signMessageAsync]);

  // Handle invalid tokenId
  if (isNaN(tokenId)) {
    return (
      <div className="flex flex-col items-center py-16 gap-4">
        <XCircle size={32} className="text-red-400" />
        <p className="text-sm text-helix-muted">Invalid model ID</p>
        <Link href="/my-models" className="text-sm text-helix-text2 hover:text-white transition-colors">
          Back to My Models
        </Link>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex flex-col items-center py-16 gap-4">
        <Loader2 size={24} className="animate-spin text-helix-muted" />
        <p className="text-sm text-helix-muted">Loading model...</p>
      </div>
    );
  }

  if (isError || !model) {
    return (
      <div className="flex flex-col items-center py-16 gap-4">
        <XCircle size={32} className="text-red-400" />
        <p className="text-sm text-helix-muted">Model not found</p>
        <Link href="/my-models" className="text-sm text-helix-text2 hover:text-white transition-colors">
          Back to My Models
        </Link>
      </div>
    );
  }

  if (!isOwner) {
    return (
      <div className="flex flex-col items-center py-16 gap-4">
        <Lock size={32} className="text-helix-muted" />
        <p className="text-sm text-helix-muted">You don&#39;t own this model</p>
        <Link href="/my-models" className="text-sm text-helix-text2 hover:text-white transition-colors">
          Back to My Models
        </Link>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <Link
              href="/my-models"
              className="p-1.5 rounded-md text-helix-muted hover:text-white hover:bg-helix-surface transition-colors"
            >
              <ArrowLeft size={16} />
            </Link>
            <div className="w-10 h-10 rounded-lg bg-white/[0.06] flex items-center justify-center">
              <Layers size={20} className="text-white" />
            </div>
            <div>
              <h1 className="text-xl font-medium text-white">{model.name}</h1>
              <p className="text-2xs font-mono text-helix-muted">{model.slug} &middot; #{model.tokenId}</p>
            </div>
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
          {latestHasWeights && (
            <button
              type="button"
              onClick={handleDownloadLatest}
              disabled={isDownloadingLatest}
              className={cn(
                'flex items-center gap-2 px-4 py-2 rounded-lg border text-sm font-medium transition-colors',
                isDownloadingLatest
                  ? 'bg-helix-surface border-helix-border text-helix-muted cursor-wait'
                  : 'bg-helix-surface border-helix-border text-helix-text hover:border-helix-border2 hover:text-white',
              )}
            >
              {isDownloadingLatest ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                <Download size={14} />
              )}
              Download Weights
            </button>
          )}
          <Link
            href={`/train?model=${model.tokenId}`}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white text-black text-sm font-medium hover:bg-white/90 transition-colors"
          >
            <Play size={14} />
            Train
          </Link>
        </div>
      </div>

      {/* Write error notice */}
      {writeError && (
        <div className="flex items-center gap-3 px-4 py-3 bg-red-500/5 border border-red-500/20 rounded-lg">
          <XCircle size={16} className="text-red-400 shrink-0" />
          <p className="text-2xs text-red-300/70">
            Transaction failed: {writeError.message.slice(0, 120)}
          </p>
        </div>
      )}

      {/* Tabs */}
      <div className="border-b border-helix-border">
        <div className="flex gap-0">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                'relative px-5 py-3 text-sm font-medium transition-colors',
                activeTab === tab.id
                  ? 'text-white'
                  : 'text-helix-muted hover:text-helix-text2',
              )}
            >
              {tab.label}
              {activeTab === tab.id && (
                <motion.div
                  layoutId="mymodel-tab-indicator"
                  className="absolute bottom-0 inset-x-0 h-0.5 bg-white"
                  transition={{ type: 'spring', stiffness: 500, damping: 35 }}
                />
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <AnimatePresence mode="wait">
        {activeTab === 'overview' && (
          <motion.div
            key="overview"
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.15 }}
          >
            <OverviewTab model={model} />
          </motion.div>
        )}

        {activeTab === 'settings' && (
          <motion.div
            key="settings"
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.15 }}
          >
            <SettingsTab
              model={model}
              setPublic={setPublic}
              setInferenceFee={setInferenceFee}
              setForSale={setForSale}
              setSalePrice={setSalePrice}
              isWritePending={isWritePending || isConfirming}
            />
          </motion.div>
        )}

        {activeTab === 'versions' && (
          <motion.div
            key="versions"
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.15 }}
          >
            <VersionsTab
              model={model}
              addVersion={addVersion}
              isPending={isWritePending || isConfirming}
              isConnected={isConnected}
              signMessageAsync={signMessageAsync}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
