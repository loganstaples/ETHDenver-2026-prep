'use client';

import { useState, useMemo, useEffect, useCallback, useRef } from 'react';
import { useParams, useRouter } from 'next/navigation';
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
  Loader2,
  XCircle,
  ArrowRightLeft,
  Download,
  Plus,
  Upload,
  Save,
  Play,
  Trash2,
  AlertTriangle,
  GitBranch,
} from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Card } from '@/components/ui/Card';
import { Skeleton } from '@/components/ui/Skeleton';
import { Tabs } from '@/components/ui/Tabs';
import { cn } from '@/lib/utils';
import { useModelDetail } from '@/hooks/useModelDetail';
import { useModelRegistry } from '@/hooks/useModelRegistry';
import { useWeightStorage } from '@/hooks/useWeightStorage';
import {
  useAccount,
  useChainId,
  useReadContract,
  useWriteContract,
  useSignMessage,
  useWaitForTransactionReceipt,
} from 'wagmi';
import { deriveModelKey, encryptWeights, decryptWeights } from '@/lib/model-encryption';
import { fetchFrom0G, encryptAndStoreWeights } from '@/lib/0g-client';
import { HELIX_MODEL_STORE_ABI, getContractAddress, getExplorerAddressUrl, getExplorerTokenUrl, CHAIN_NAMES } from '@/lib/contracts';
import type { OnChainVersion } from '@/lib/contracts';

// ============================================================================
// Helpers
// ============================================================================

const SEMVER_REGEX = /^\d+\.\d+\.\d+$/;

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
// Download weights helper
// ============================================================================

/** Decompress gzip data in the browser using DecompressionStream. */
async function decompressGzip(data: Uint8Array<ArrayBuffer>): Promise<Uint8Array<ArrayBuffer>> {
  const ds = new DecompressionStream('gzip');
  const writer = ds.writable.getWriter();
  writer.write(data as unknown as BufferSource);
  writer.close();
  const reader = ds.readable.getReader();
  const chunks: Uint8Array<ArrayBuffer>[] = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
  }
  let totalLength = 0;
  for (const chunk of chunks) totalLength += chunk.length;
  const result = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}

/** If bytes start with gzip magic (0x1f 0x8b), decompress; otherwise return as-is. */
async function ensureDecompressed(bytes: Uint8Array<ArrayBuffer>): Promise<Uint8Array<ArrayBuffer>> {
  if (bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b) {
    try {
      return await decompressGzip(bytes);
    } catch {
      // Not actually gzip or corrupt — return raw
    }
  }
  return bytes;
}

async function downloadWeightsFromVersion(
  version: OnChainVersion,
  tokenId: number,
  signMessageAsync: (args: { message: string }) => Promise<string>,
): Promise<void> {
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
    const key = await deriveModelKey(
      (message: string) => signMessageAsync({ message }),
      tokenId,
    );
    const binaryStr = atob(result.data);
    let bytes = new Uint8Array(binaryStr.length);
    for (let i = 0; i < binaryStr.length; i++) {
      bytes[i] = binaryStr.charCodeAt(i);
    }
    // Safety net: if server-side decompression failed, the bytes are still
    // gzip-compressed. Decompress before decrypting.
    bytes = await ensureDecompressed(bytes);
    weightsJson = await decryptWeights(key, bytes);
  } else {
    weightsJson = JSON.stringify(result.data, null, 2);
  }

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
// Tabs — dynamically includes Settings when owner
// ============================================================================

type TabId = 'overview' | 'settings' | 'versions' | 'inference';

function getTabList(isOwner: boolean): { id: TabId; label: string }[] {
  const tabs: { id: TabId; label: string }[] = [
    { id: 'overview', label: 'Overview' },
  ];
  if (isOwner) {
    tabs.push({ id: 'settings', label: 'Settings' });
  }
  tabs.push({ id: 'versions', label: 'Versions' });
  tabs.push({ id: 'inference', label: 'Inference' });
  return tabs;
}

// ============================================================================
// Version Row (detailed, for version tab)
// ============================================================================

function VersionRow({
  version,
  index,
  modelSlug,
  isOwner,
  tokenId,
  signMessageAsync,
  hasLocalWeights,
  onDownloadLocal,
}: {
  version: OnChainVersion;
  index: number;
  modelSlug: string;
  isOwner: boolean;
  tokenId: number;
  signMessageAsync: (args: { message: string }) => Promise<string>;
  hasLocalWeights?: boolean;
  onDownloadLocal?: () => void;
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
    <Card variant="default" className="!p-5">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-white/[0.04] flex items-center justify-center text-sm font-mono text-helix-muted">
            {index + 1}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-white font-mono">v{version.semver}</span>
              {version.accuracy > 0 && (
                <Badge variant="default" className="text-sm">
                  {(version.accuracy * 100).toFixed(1)}% accuracy
                </Badge>
              )}
            </div>
            <p className="text-sm text-helix-dim mt-0.5">
              {formatDate(version.timestamp)}
              {version.sessionId && <span className="ml-2 font-mono">{version.sessionId}</span>}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {version.weightsStored ? (
            <Badge variant="default" className="text-green-400 text-sm flex items-center gap-1">
              <HardDrive size={14} />
              On 0G
            </Badge>
          ) : hasLocalWeights ? (
            <Badge variant="default" className="text-blue-400 text-sm flex items-center gap-1">
              <Save size={14} />
              Local
            </Badge>
          ) : (
            <Badge variant="default" className="text-helix-dim text-sm">
              No weights
            </Badge>
          )}
          {/* Owner download from local when not on 0G */}
          {isOwner && !version.weightsStored && hasLocalWeights && onDownloadLocal && (
            <button
              type="button"
              onClick={onDownloadLocal}
              className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-base font-medium transition-colors bg-helix-surface border border-helix-border text-helix-text hover:border-helix-border2 hover:text-white"
            >
              <Download size={14} />
              Download
            </button>
          )}
          {/* Owner download button (from 0G) */}
          {isOwner && version.weightsStored && version.rootHash && (
            <button
              type="button"
              onClick={handleDownload}
              disabled={isDownloading}
              title={downloadError || 'Download & decrypt weights'}
              className={cn(
                'flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-base font-medium transition-colors',
                downloadError
                  ? 'bg-red-500/10 border border-red-500/30 text-red-400'
                  : isDownloading
                    ? 'bg-helix-surface border border-helix-border text-helix-muted cursor-wait'
                    : 'bg-helix-surface border border-helix-border text-helix-text hover:border-helix-border2 hover:text-white',
              )}
            >
              {isDownloading ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                <Download size={14} />
              )}
              {downloadError ? 'Error' : isDownloading ? 'Downloading...' : 'Download'}
            </button>
          )}
        </div>
      </div>

      {/* Hash + actions */}
      {version.rootHash && (
        <div className="flex items-center gap-2 mt-3 pl-11">
          <code className="text-sm font-mono text-helix-muted bg-helix-bg px-2 py-1 rounded border border-helix-border/50">
            {truncateHash(version.rootHash)}
          </code>
          <button
            type="button"
            onClick={() => copyHash(version.rootHash)}
            className="p-1 rounded text-helix-dim hover:text-white transition-colors"
          >
            {copied ? <CheckCircle size={16} /> : <Copy size={16} />}
          </button>
          <a
            href={`https://storagescan-galileo.0g.ai/file/${version.rootHash}`}
            target="_blank"
            rel="noopener noreferrer"
            className="p-1 rounded text-helix-dim hover:text-white transition-colors"
          >
            <ExternalLink size={16} />
          </a>
          {version.weightsStored && (
            <Link
              href={`/inference?hash=${version.rootHash}&version=${version.semver}&model=${modelSlug}`}
              className="ml-auto flex items-center gap-2 px-4 py-2 rounded-lg bg-white text-black text-base font-medium hover:bg-white/90 transition-colors"
            >
              <Sparkles size={14} />
              Run Inference
            </Link>
          )}
        </div>
      )}
    </Card>
  );
}

// ============================================================================
// Add Version Section (owner-only, inline in Versions tab)
// ============================================================================

type VersionPhase = 'idle' | 'validating' | 'encrypting' | 'uploading' | 'registering' | 'done' | 'error';

function AddVersionSection({
  model,
  addVersion,
  isPending,
  isConnected,
}: {
  model: NonNullable<ReturnType<typeof useModelDetail>['model']>;
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
        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-base text-helix-text hover:border-helix-border2 hover:text-white transition-colors"
      >
        <Plus size={14} />
        Add Version
      </button>
    );
  }

  return (
    <Card variant="default" className="space-y-4">
      <div className="flex items-center justify-between">
        <h4 className="text-lg font-semibold text-white">Add Version</h4>
        <button
          type="button"
          onClick={() => { reset(); setIsOpen(false); }}
          className="text-base text-helix-muted hover:text-white transition-colors"
        >
          Cancel
        </button>
      </div>

      {phase === 'idle' && (
        <>
          <div>
            <label className="text-base font-medium text-helix-muted block mb-1.5">Version (semver)</label>
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
            {versionError && <p className="text-sm text-red-400 mt-1">{versionError}</p>}
          </div>

          <div>
            <label className="text-base font-medium text-helix-muted block mb-1.5">Weights File (optional, JSON)</label>
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
                  <span className="text-base text-helix-muted">{(file.size / 1024).toFixed(0)} KB</span>
                </>
              ) : (
                <>
                  <Upload size={20} className="text-helix-muted" />
                  <span className="text-base text-helix-text2">Click to select weights file</span>
                  <span className="text-base text-helix-muted">JSON with w1, b1, w2, b2 weight arrays</span>
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
              <div className="flex items-center gap-2 mt-2 text-sm text-helix-muted">
                <Shield size={16} className="text-green-400 shrink-0" />
                Weights will be encrypted with your wallet key before upload
              </div>
            )}
          </div>

          <div>
            <label className="text-base font-medium text-helix-muted block mb-1.5">Accuracy % (optional)</label>
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
          <p className="text-base text-helix-text2">
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
            className="mt-2 px-6 py-2 rounded-lg bg-white text-black text-base font-medium hover:bg-white/90 transition-colors"
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
// Version Branch Graph — horizontal tree of version lineage (semver-inferred)
// ============================================================================

interface BranchNode {
  version: OnChainVersion;
  children: BranchNode[];
}

function parseSemver(s: string): [number, number, number] {
  const parts = s.split('.').map(Number);
  return [parts[0] || 0, parts[1] || 0, parts[2] || 0];
}

function buildVersionTree(versions: OnChainVersion[]): BranchNode | null {
  if (versions.length === 0) return null;

  const sorted = [...versions].sort((a, b) => {
    const [aMaj, aMin, aPat] = parseSemver(a.semver);
    const [bMaj, bMin, bPat] = parseSemver(b.semver);
    return aMaj - bMaj || aMin - bMin || aPat - bPat;
  });

  // Group by major.minor
  const branches = new Map<string, OnChainVersion[]>();
  for (const v of sorted) {
    const [maj, min] = parseSemver(v.semver);
    const key = `${maj}.${min}`;
    if (!branches.has(key)) branches.set(key, []);
    branches.get(key)!.push(v);
  }

  // Build nodes for each branch (minor chain). First version is the head.
  const branchNodes = new Map<string, BranchNode>();
  for (const [key, versionList] of branches) {
    // Build a linear chain for patch versions
    let head: BranchNode | null = null;
    let tail: BranchNode | null = null;
    for (const v of versionList) {
      const node: BranchNode = { version: v, children: [] };
      if (!head) {
        head = node;
        tail = node;
      } else {
        tail!.children.push(node);
        tail = node;
      }
    }
    branchNodes.set(key, head!);
  }

  // Connect branches: each minor branch attaches to the previous minor's tail (or prev major's best)
  const branchKeys = Array.from(branches.keys()).sort((a, b) => {
    const [aMaj, aMin] = a.split('.').map(Number);
    const [bMaj, bMin] = b.split('.').map(Number);
    return aMaj - bMaj || aMin - bMin;
  });

  const root = branchNodes.get(branchKeys[0])!;

  // Get the tail node of a branch chain
  function getTail(node: BranchNode): BranchNode {
    if (node.children.length === 0) return node;
    // Follow the first child (linear patch chain)
    return getTail(node.children[0]);
  }

  // Find the best version node in a branch (highest accuracy)
  function getBest(key: string): BranchNode {
    const versions = branches.get(key)!;
    let bestV = versions[0];
    for (const v of versions) {
      if (v.accuracy > bestV.accuracy) bestV = v;
    }
    // Find corresponding node in the chain
    let node = branchNodes.get(key)!;
    while (node.version.semver !== bestV.semver && node.children.length > 0) {
      node = node.children[0];
    }
    return node;
  }

  for (let i = 1; i < branchKeys.length; i++) {
    const key = branchKeys[i];
    const [maj, min] = key.split('.').map(Number);
    const branchHead = branchNodes.get(key)!;

    // Find parent: same major + previous minor, or previous major's best
    let parentNode: BranchNode | null = null;
    const prevMinorKey = `${maj}.${min - 1}`;
    if (min > 0 && branchNodes.has(prevMinorKey)) {
      parentNode = getTail(branchNodes.get(prevMinorKey)!);
    } else {
      // Different major — attach to best of previous major
      for (let j = i - 1; j >= 0; j--) {
        const [pMaj] = branchKeys[j].split('.').map(Number);
        if (pMaj < maj) {
          parentNode = getBest(branchKeys[j]);
          break;
        }
      }
    }

    if (parentNode) {
      parentNode.children.push(branchHead);
    }
  }

  return root;
}

/** Find the path from root to the node with highest accuracy. */
function findMainLine(root: BranchNode): Set<string> {
  const path = new Set<string>();

  function dfs(node: BranchNode): { accuracy: number; trail: string[] } {
    let best = { accuracy: node.version.accuracy, trail: [node.version.semver] };
    for (const child of node.children) {
      const result = dfs(child);
      if (result.accuracy > best.accuracy) {
        best = { accuracy: result.accuracy, trail: [node.version.semver, ...result.trail] };
      }
    }
    return best;
  }

  const result = dfs(root);
  for (const s of result.trail) path.add(s);
  return path;
}

/** Check if the tree has any actual branching (any node with >1 child). */
function hasBranching(node: BranchNode): boolean {
  if (node.children.length > 1) return true;
  return node.children.some(hasBranching);
}

function BranchNodeView({
  node,
  mainLine,
  isLast = true,
}: {
  node: BranchNode;
  mainLine: Set<string>;
  isLast?: boolean;
}) {
  const isMain = mainLine.has(node.version.semver);
  const acc = node.version.accuracy;

  return (
    <div className="flex items-center">
      {/* Node */}
      <div
        className={cn(
          'flex flex-col items-center px-3 py-1.5 rounded-lg border shrink-0',
          isMain
            ? 'border-white/25 bg-white/[0.08] text-white'
            : 'border-helix-border bg-helix-surface/60 text-helix-muted',
        )}
      >
        <span className={cn('font-mono text-xs font-medium', isMain ? 'text-white' : 'text-helix-text')}>
          v{node.version.semver}
        </span>
        {acc > 0 && (
          <span className={cn('text-sm font-mono', isMain ? 'text-green-400' : 'text-helix-dim')}>
            {(acc * 100).toFixed(1)}%
          </span>
        )}
      </div>

      {/* Connector + children */}
      {node.children.length > 0 && (
        <>
          {/* Horizontal line to children */}
          <div className={cn('w-5 h-px shrink-0', isMain ? 'bg-white/30' : 'bg-helix-border')} />

          {node.children.length === 1 ? (
            <BranchNodeView node={node.children[0]} mainLine={mainLine} />
          ) : (
            <div className="flex flex-col gap-1.5">
              {node.children.map((child, idx) => {
                const childIsMain = mainLine.has(child.version.semver);
                const isFirst = idx === 0;
                return (
                  <div key={child.version.semver} className="flex items-center relative">
                    {/* Vertical bar connecting siblings */}
                    {!isFirst && (
                      <div
                        className="absolute left-0 w-px bg-helix-border/60"
                        style={{ top: '-6px', height: '6px' }}
                      />
                    )}
                    {idx < node.children.length - 1 && (
                      <div
                        className="absolute left-0 w-px bg-helix-border/60"
                        style={{ bottom: '-6px', height: '6px' }}
                      />
                    )}
                    {/* Horizontal stub */}
                    <div className={cn('w-4 h-px shrink-0', childIsMain ? 'bg-white/30' : 'bg-helix-border/60')} />
                    <BranchNodeView
                      node={child}
                      mainLine={mainLine}
                      isLast={idx === node.children.length - 1}
                    />
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function VersionBranchGraph({ versions }: { versions: OnChainVersion[] }) {
  const tree = useMemo(() => buildVersionTree(versions), [versions]);

  if (!tree || !hasBranching(tree)) return null;

  const mainLine = useMemo(() => findMainLine(tree), [tree]);

  return (
    <Card variant="default">
      <div className="flex items-center gap-2 mb-3">
        <GitBranch size={14} className="text-helix-muted" />
        <h3 className="text-lg font-semibold text-white">Version Lineage</h3>
        <span className="text-sm text-helix-dim ml-auto">best path highlighted</span>
      </div>
      <div className="overflow-x-auto py-3 -mx-1 px-1">
        <div className="flex items-center gap-0 min-w-fit">
          <BranchNodeView node={tree} mainLine={mainLine} />
        </div>
      </div>
    </Card>
  );
}

// ============================================================================
// Overview Tab
// ============================================================================

function OverviewTab({ model, isOwner, chainId, contractAddress }: { model: NonNullable<ReturnType<typeof useModelDetail>['model']>; isOwner: boolean; chainId: number; contractAddress: string }) {
  const bestAccuracy = model.versions.reduce((best, v) => (v.accuracy > best ? v.accuracy : best), 0);
  const weightsCount = model.versions.filter((v) => v.weightsStored).length;

  return (
    <div className="space-y-6">
      {/* Encryption notice for non-owners */}
      {!isOwner && weightsCount > 0 && (
        <div className="flex items-center gap-2.5 px-4 py-3 bg-white/[0.02] border border-white/[0.06] rounded-xl">
          <Shield size={14} className="text-green-400 shrink-0" />
          <p className="text-base text-helix-muted">
            Encrypted on 0G — owner-only access. Model weights are encrypted with the owner&apos;s wallet key.
          </p>
        </div>
      )}

      {/* Description */}
      {model.description && (
        <Card variant="default">
          <h3 className="text-lg font-semibold text-white mb-2">About</h3>
          <p className="text-base text-helix-text2 leading-relaxed">{model.description}</p>
        </Card>
      )}

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card variant="default" className="!p-6 text-center">
          <Tag size={24} className="text-helix-muted mx-auto mb-3" />
          <p className="text-sm text-helix-muted mb-1">Versions</p>
          <p className="text-3xl font-mono font-light text-white">{model.versions.length}</p>
        </Card>
        <Card variant="default" className="!p-6 text-center">
          <Trophy size={24} className="text-helix-muted mx-auto mb-3" />
          <p className="text-sm text-helix-muted mb-1">Best Accuracy</p>
          <p className="text-3xl font-mono font-light text-white">
            {bestAccuracy > 0 ? `${(bestAccuracy * 100).toFixed(1)}%` : '--'}
          </p>
        </Card>
        <Card variant="default" className="!p-6 text-center">
          <HardDrive size={24} className="text-helix-muted mx-auto mb-3" />
          <p className="text-sm text-helix-muted mb-1">Weights on 0G</p>
          <p className="text-3xl font-mono font-light text-white">{weightsCount}</p>
        </Card>
        <Card variant="default" className="!p-6 text-center">
          <DollarSign size={24} className="text-helix-muted mx-auto mb-3" />
          <p className="text-sm text-helix-muted mb-1">Inference Fee</p>
          <p className="text-3xl font-mono font-light text-white">{formatFee(model.inferenceFee)}</p>
        </Card>
      </div>

      {/* Version Branch Graph (only shows when there's actual branching) */}
      <VersionBranchGraph versions={model.versions} />

      {/* Owner Info */}
      <Card variant="default">
        <h3 className="text-lg font-semibold text-white mb-4">Ownership</h3>
        <div className="space-y-3">
          <div className="flex items-center justify-between py-1">
            <span className="text-base text-helix-muted">Creator</span>
            <code className="text-base font-mono text-helix-text">{model.creator}</code>
          </div>
          <div className="flex items-center justify-between py-1">
            <span className="text-base text-helix-muted">Current Owner</span>
            <div className="flex items-center gap-2">
              <code className="text-base font-mono text-helix-text">{model.owner}</code>
              {isOwner && (
                <Badge variant="outline" className="text-green-400 border-green-500/30">You</Badge>
              )}
            </div>
          </div>
          <div className="flex items-center justify-between py-1">
            <span className="text-base text-helix-muted">Token ID</span>
            <span className="text-base font-mono text-helix-text">#{model.tokenId}</span>
          </div>
          <div className="flex items-center justify-between py-1">
            <span className="text-base text-helix-muted">Created</span>
            <span className="text-base text-helix-text">{formatDate(model.createdAt)}</span>
          </div>
        </div>
      </Card>

      {/* On-Chain Verification */}
      {(() => {
        const contractUrl = getExplorerAddressUrl(chainId, contractAddress);
        const tokenUrl = getExplorerTokenUrl(chainId, contractAddress, model.tokenId);
        const chainName = CHAIN_NAMES[chainId] || `Chain ${chainId}`;
        const hasExplorer = !!contractUrl;

        return (
          <Card variant="default">
            <div className="flex items-center gap-2.5 mb-4">
              <Shield size={18} className="text-green-400" />
              <h3 className="text-lg font-semibold text-white">On-Chain Verification</h3>
              {hasExplorer && (
                <Badge variant="outline" className="text-green-400 border-green-500/30 ml-auto">
                  Verified
                </Badge>
              )}
            </div>
            <p className="text-base text-helix-muted mb-4">
              This model is minted as an ERC-721 NFT on {chainName}. All ownership, versions, and metadata are verifiable on-chain.
            </p>
            <div className="space-y-3">
              <div className="flex items-center justify-between py-1">
                <span className="text-base text-helix-muted">Network</span>
                <div className="flex items-center gap-2">
                  <span className="inline-block w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                  <span className="text-base font-medium text-helix-text">{chainName}</span>
                </div>
              </div>
              <div className="flex items-center justify-between py-1">
                <span className="text-base text-helix-muted">Contract</span>
                <div className="flex items-center gap-2">
                  <code className="text-base font-mono text-helix-text">{truncateAddress(contractAddress)}</code>
                  {contractUrl && (
                    <a href={contractUrl} target="_blank" rel="noopener noreferrer" className="p-1 rounded text-helix-dim hover:text-white transition-colors">
                      <ExternalLink size={16} />
                    </a>
                  )}
                </div>
              </div>
              <div className="flex items-center justify-between py-1">
                <span className="text-base text-helix-muted">Token ID</span>
                <div className="flex items-center gap-2">
                  <span className="text-base font-mono text-helix-text">#{model.tokenId}</span>
                  {tokenUrl && (
                    <a href={tokenUrl} target="_blank" rel="noopener noreferrer" className="p-1 rounded text-helix-dim hover:text-white transition-colors">
                      <ExternalLink size={16} />
                    </a>
                  )}
                </div>
              </div>
              <div className="flex items-center justify-between py-1">
                <span className="text-base text-helix-muted">Standard</span>
                <span className="text-base font-mono text-helix-text">ERC-721</span>
              </div>
            </div>
            {hasExplorer && (
              <a
                href={tokenUrl || contractUrl!}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-center gap-2 mt-5 px-5 py-3 rounded-lg border border-green-500/20 bg-green-500/[0.06] text-green-400 text-base font-medium hover:bg-green-500/10 transition-colors"
              >
                <ExternalLink size={16} />
                View on Block Explorer
              </a>
            )}
          </Card>
        );
      })()}

      {/* Latest Version Quick View */}
      {model.versions.length > 0 && (
        <Card variant="default">
          <h3 className="text-lg font-semibold text-white mb-3">Latest Version</h3>
          {(() => {
            const latest = model.versions[model.versions.length - 1];
            return (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-base text-helix-muted">Version</span>
                  <span className="text-base font-mono text-white">v{latest.semver}</span>
                </div>
                {latest.accuracy > 0 && (
                  <div className="flex items-center justify-between">
                    <span className="text-base text-helix-muted">Accuracy</span>
                    <span className="text-base font-mono text-white">{(latest.accuracy * 100).toFixed(1)}%</span>
                  </div>
                )}
                <div className="flex items-center justify-between">
                  <span className="text-base text-helix-muted">Weights</span>
                  <span className="text-base text-helix-text">
                    {latest.weightsStored ? 'Stored on 0G' : 'Not stored'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-base text-helix-muted">Published</span>
                  <span className="text-base text-helix-text">{formatDate(latest.timestamp)}</span>
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
// Delete Model Section (with confirmation)
// ============================================================================

function DeleteModelSection({
  tokenId,
  modelName,
  deleteModel,
  isDeletePending,
  onDeleteSuccess,
}: {
  tokenId: number;
  modelName: string;
  deleteModel: (params: { tokenId: number }) => void;
  isDeletePending: boolean;
  onDeleteSuccess?: () => void;
}) {
  const [showConfirm, setShowConfirm] = useState(false);
  const [confirmText, setConfirmText] = useState('');

  const confirmed = confirmText === modelName;

  const handleDelete = () => {
    if (!confirmed || isDeletePending) return;
    deleteModel({ tokenId });
    onDeleteSuccess?.();
  };

  if (!showConfirm) {
    return (
      <div className="flex items-center justify-between">
        <div>
          <p className="text-base text-helix-text">Delete Model</p>
          <p className="text-base text-helix-muted mt-1">
            Permanently burn this model NFT and release the slug.
          </p>
        </div>
        <button
          type="button"
          onClick={() => setShowConfirm(true)}
          className="flex items-center gap-2 px-4 py-2 rounded-lg border border-red-500/30 bg-red-500/10 text-red-400 text-sm hover:bg-red-500/20 transition-colors"
        >
          <Trash2 size={14} />
          Delete Model
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-start gap-2.5 px-4 py-3 bg-red-500/[0.06] border border-red-500/20 rounded-xl">
        <AlertTriangle size={16} className="text-red-400 shrink-0 mt-0.5" />
        <div className="text-sm text-red-300/80">
          <p className="font-medium text-red-400 mb-1">This action is irreversible</p>
          <p>The model NFT will be burned, all version history deleted, and any accrued inference fees withdrawn to your wallet. The slug will be released for reuse.</p>
        </div>
      </div>

      <div>
        <label className="text-base font-medium text-helix-muted block mb-1.5">
          Type <span className="font-mono text-red-400">{modelName}</span> to confirm
        </label>
        <input
          type="text"
          value={confirmText}
          onChange={(e) => setConfirmText(e.target.value)}
          placeholder={modelName}
          className="w-full px-3 py-2 bg-helix-bg border border-red-500/30 rounded-md text-sm text-helix-text font-mono focus:outline-none focus:border-red-500/60 transition-colors"
        />
      </div>

      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={handleDelete}
          disabled={!confirmed || isDeletePending}
          className={cn(
            'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all',
            (!confirmed || isDeletePending)
              ? 'bg-helix-border text-helix-muted cursor-not-allowed'
              : 'bg-red-500 text-white hover:bg-red-600',
          )}
        >
          {isDeletePending ? (
            <Loader2 size={14} className="animate-spin" />
          ) : (
            <Trash2 size={14} />
          )}
          {isDeletePending ? 'Deleting...' : 'Permanently Delete'}
        </button>
        <button
          type="button"
          onClick={() => { setShowConfirm(false); setConfirmText(''); }}
          disabled={isDeletePending}
          className="px-4 py-2 rounded-lg text-base text-helix-muted hover:text-helix-text transition-colors"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}

// ============================================================================
// Settings Tab (owner-only)
// ============================================================================

function SettingsTab({
  model,
  setPublic,
  setInferenceFee,
  setForSale,
  setSalePrice,
  deleteModel,
  isSettingsWritePending,
  isDeletePending,
  onDeleteSuccess,
}: {
  model: NonNullable<ReturnType<typeof useModelDetail>['model']>;
  setPublic: (params: { tokenId: number; isPublic: boolean }) => void;
  setInferenceFee: (params: { tokenId: number; feeBps: number }) => void;
  setForSale: (params: { tokenId: number; forSale: boolean }) => void;
  setSalePrice: (params: { tokenId: number; priceEth: number }) => void;
  deleteModel: (params: { tokenId: number }) => void;
  isSettingsWritePending: boolean;
  isDeletePending: boolean;
  onDeleteSuccess?: () => void;
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
    if (!feeChanged || isSettingsWritePending) return;
    setInferenceFee({ tokenId: model.tokenId, feeBps });
  };

  const handleToggleForSale = () => {
    if (isSettingsWritePending) return;
    setForSale({ tokenId: model.tokenId, forSale: !model.forSale });
  };

  const handleSavePrice = () => {
    if (!priceChanged || isSettingsWritePending) return;
    setSalePrice({ tokenId: model.tokenId, priceEth: parsedPrice });
  };

  return (
    <div className="space-y-4">
      {/* Visibility */}
      <Card variant="default">
        <h4 className="text-lg font-semibold text-white mb-3">Visibility</h4>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-base text-helix-text">
              {model.isPublic ? 'Public' : 'Private'} Model
            </p>
            <p className="text-base text-helix-muted mt-1">
              {model.isPublic
                ? 'Visible to everyone on the marketplace. Others can request inference.'
                : 'Only you can see and use this model.'}
            </p>
          </div>
          <button
            type="button"
            onClick={() => setPublic({ tokenId: model.tokenId, isPublic: !model.isPublic })}
            disabled={isSettingsWritePending}
            className={cn(
              'flex items-center gap-2 px-4 py-2 rounded-lg border text-sm transition-colors',
              model.isPublic
                ? 'bg-helix-surface border-helix-border text-helix-text hover:border-helix-border2 hover:text-white'
                : 'bg-green-500/10 border-green-500/30 text-green-400 hover:bg-green-500/20',
              isSettingsWritePending && 'opacity-50 cursor-not-allowed',
            )}
          >
            {isSettingsWritePending ? (
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
        <h4 className="text-lg font-semibold text-white mb-3">Inference Fee</h4>
        <p className="text-sm text-helix-muted mb-4">
          Fee charged to users who run inference on your model. Set as a percentage (0-50%). Stored as basis points on-chain.
        </p>
        <div className="flex items-end gap-3">
          <div className="flex-1">
            <label className="text-base font-medium text-helix-muted block mb-1.5">Fee (%)</label>
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
            disabled={!feeChanged || isSettingsWritePending}
            className={cn(
              'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all',
              (!feeChanged || isSettingsWritePending)
                ? 'bg-helix-border text-helix-muted cursor-not-allowed'
                : 'bg-white text-black hover:bg-white/90',
            )}
          >
            {isSettingsWritePending ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <Save size={14} />
            )}
            Save
          </button>
        </div>
        {/* Fee preview */}
        <div className="mt-3 px-3 py-2 bg-helix-bg rounded-md border border-helix-border/50">
          <div className="flex items-center justify-between text-sm text-helix-muted">
            <span>Current on-chain</span>
            <span className="font-mono">{formatFee(model.inferenceFee)} ({model.inferenceFee} bps)</span>
          </div>
          {feeChanged && (
            <div className="flex items-center justify-between text-sm text-white mt-1">
              <span>New fee</span>
              <span className="font-mono">{formatFee(feeBps)} ({feeBps} bps)</span>
            </div>
          )}
        </div>
      </Card>

      {/* Marketplace / Sale Settings */}
      <Card variant="default">
        <h4 className="text-lg font-semibold text-white mb-3">Marketplace</h4>
        <p className="text-sm text-helix-muted mb-4">
          Control whether your model NFT is listed for sale on the marketplace.
        </p>

        {/* For Sale toggle */}
        <div className="flex items-center justify-between mb-4">
          <div>
            <p className="text-base text-helix-text">
              {model.forSale ? 'Listed for Sale' : 'Not for Sale'}
            </p>
            <p className="text-base text-helix-muted mt-1">
              {model.forSale
                ? 'Your model is visible on the marketplace and can be purchased.'
                : 'Your model is not listed. Toggle to make it available for purchase.'}
            </p>
          </div>
          <button
            type="button"
            onClick={handleToggleForSale}
            disabled={isSettingsWritePending}
            className={cn(
              'flex items-center gap-2 px-4 py-2 rounded-lg border text-sm transition-colors',
              model.forSale
                ? 'bg-helix-surface border-helix-border text-helix-text hover:border-helix-border2 hover:text-white'
                : 'bg-green-500/10 border-green-500/30 text-green-400 hover:bg-green-500/20',
              isSettingsWritePending && 'opacity-50 cursor-not-allowed',
            )}
          >
            {isSettingsWritePending ? (
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
                <label className="text-base font-medium text-helix-muted block mb-1.5">Sale Price (ADI)</label>
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
                disabled={!priceChanged || isSettingsWritePending}
                className={cn(
                  'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all',
                  (!priceChanged || isSettingsWritePending)
                    ? 'bg-helix-border text-helix-muted cursor-not-allowed'
                    : 'bg-white text-black hover:bg-white/90',
                )}
              >
                {isSettingsWritePending ? (
                  <Loader2 size={14} className="animate-spin" />
                ) : (
                  <Save size={14} />
                )}
                Save
              </button>
            </div>

            {/* Price preview */}
            <div className="mt-3 px-3 py-2 bg-helix-bg rounded-md border border-helix-border/50">
              <div className="flex items-center justify-between text-sm text-helix-muted">
                <span>Current on-chain price</span>
                <span className="font-mono">{model.salePrice > 0 ? `${model.salePrice} ADI` : 'Not set'}</span>
              </div>
              {priceChanged && (
                <div className="flex items-center justify-between text-sm text-white mt-1">
                  <span>New price</span>
                  <span className="font-mono">{parsedPrice} ADI</span>
                </div>
              )}
            </div>

            <p className="text-sm text-helix-dim mt-3">
              When listed for sale, your model NFT can be purchased by anyone on the marketplace.
            </p>
          </>
        )}
      </Card>

      {/* Danger Zone */}
      <Card variant="default" className="border-red-500/20">
        <h4 className="text-sm font-medium text-red-400 mb-3">Danger Zone</h4>
        <p className="text-sm text-helix-muted mb-4">
          These actions are permanent and cannot be undone.
        </p>

        {/* Delete Model */}
        <DeleteModelSection
          tokenId={model.tokenId}
          modelName={model.name}
          deleteModel={deleteModel}
          isDeletePending={isDeletePending}
          onDeleteSuccess={onDeleteSuccess}
        />

        <div className="mt-4 pt-4 border-t border-helix-border/30">
          <p className="text-base text-helix-muted">
            Model ownership can also be transferred via ERC-721 transfer.
            Use your wallet or a marketplace to transfer the NFT.
          </p>
        </div>
      </Card>
    </div>
  );
}

// ============================================================================
// Versions Tab — enhanced with owner features
// ============================================================================

function VersionsTab({
  model,
  isOwner,
  isConnected,
  addVersion,
  isVersionWritePending,
  signMessageAsync,
  hasLocalWeights,
  onDownloadLocal,
}: {
  model: NonNullable<ReturnType<typeof useModelDetail>['model']>;
  isOwner: boolean;
  isConnected: boolean;
  addVersion: (params: {
    tokenId: number;
    semver: string;
    rootHash: string;
    accuracy: number;
    sessionId: string;
    weightsStored: boolean;
  }) => void;
  isVersionWritePending: boolean;
  signMessageAsync: (args: { message: string }) => Promise<string>;
  hasLocalWeights?: boolean;
  onDownloadLocal?: () => void;
}) {
  const hasWeights = model.versions.some((v) => v.weightsStored);

  return (
    <div className="space-y-3">
      {/* Owner encryption notice */}
      {isOwner && hasWeights && (
        <div className="flex items-center gap-2.5 px-4 py-3 bg-green-500/[0.04] border border-green-500/20 rounded-xl">
          <Shield size={14} className="text-green-400 shrink-0" />
          <p className="text-sm text-green-300/70">
            Only you (the owner) can decrypt and download these weights
          </p>
        </div>
      )}

      {/* Add version (owner-only) */}
      {isOwner && (
        <AddVersionSection
          model={model}
          addVersion={addVersion}
          isPending={isVersionWritePending}
          isConnected={isConnected}
        />
      )}

      {/* Version count header */}
      <div className="flex items-center justify-between">
        <p className="text-base text-helix-text2">
          {model.versions.length} version{model.versions.length !== 1 ? 's' : ''}
        </p>
      </div>

      {model.versions.length === 0 ? (
        <div className="flex flex-col items-center py-12 gap-3">
          <Tag size={32} className="text-helix-dim" />
          <p className="text-base text-helix-text2">No versions yet</p>
          <p className="text-base text-helix-muted">
            {isOwner
              ? 'Add a version to get started, or train your model first.'
              : 'The model owner hasn\'t published any versions.'}
          </p>
        </div>
      ) : (
        model.versions.slice().reverse().map((v, i) => (
          <VersionRow
            key={`${v.semver}-${i}`}
            version={v}
            index={model.versions.length - 1 - i}
            modelSlug={model.slug}
            isOwner={isOwner}
            tokenId={model.tokenId}
            signMessageAsync={signMessageAsync}
            hasLocalWeights={hasLocalWeights}
            onDownloadLocal={onDownloadLocal}
          />
        ))
      )}
    </div>
  );
}

// ============================================================================
// Inference Tab
// ============================================================================

function InferenceTab({
  model,
  isOwner,
  inferenceEnabled,
  isEnablingInference,
  isDisablingInference,
  onEnableInference,
  onDisableInference,
}: {
  model: NonNullable<ReturnType<typeof useModelDetail>['model']>;
  isOwner: boolean;
  inferenceEnabled: boolean;
  isEnablingInference: boolean;
  isDisablingInference: boolean;
  onEnableInference: (versionIndex: number) => void;
  onDisableInference: () => void;
}) {
  const versionsWithWeights = model.versions.filter((v) => v.weightsStored && v.rootHash);

  return (
    <div className="space-y-6">
      {/* Public Inference Control (owner only, public models) */}
      {isOwner && model.isPublic && (
        <div className="rounded-2xl border border-helix-border bg-helix-surface/50 p-6">
          <h3 className="text-lg font-semibold text-white mb-3">Public Inference</h3>
          <p className="text-sm text-helix-muted mb-4">
            Enable inference so anyone can run predictions on your public model.
            Your weights are cached in memory on the server — never stored or exposed.
          </p>
          {inferenceEnabled ? (
            <button
              type="button"
              onClick={onDisableInference}
              disabled={isDisablingInference}
              className="px-4 py-2 rounded-lg bg-red-500/20 text-red-400 text-sm hover:bg-red-500/30 transition-colors"
            >
              {isDisablingInference ? 'Disabling...' : 'Disable Public Inference'}
            </button>
          ) : (
            <button
              type="button"
              onClick={() => onEnableInference(model.versions.length - 1)}
              disabled={isEnablingInference || model.versions.length === 0}
              className="px-4 py-2 rounded-lg bg-emerald-500/20 text-emerald-400 text-sm hover:bg-emerald-500/30 transition-colors"
            >
              {isEnablingInference ? 'Decrypting & Caching...' : 'Enable Public Inference'}
            </button>
          )}
        </div>
      )}

      {/* Fee info */}
      <Card variant="default">
        <h3 className="text-lg font-semibold text-white mb-3">Inference Pricing</h3>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-base text-helix-muted">Model Fee</span>
            <span className="text-base font-mono text-white">{formatFee(model.inferenceFee)}</span>
          </div>
          <p className="text-base text-helix-dim">
            {model.inferenceFee > 0
              ? `A ${formatFee(model.inferenceFee)} surcharge is added on top of base MPC compute costs. This fee goes to the model owner.`
              : 'No model surcharge. You only pay the base MPC network compute costs.'}
          </p>
          {isOwner && (
            <div className="flex items-center gap-2 mt-2 pt-2 border-t border-helix-border/50">
              <Shield size={16} className="text-green-400" />
              <span className="text-sm text-green-400">As the owner, you run inference for free.</span>
            </div>
          )}
        </div>
      </Card>

      {/* Available versions */}
      <div>
        <h3 className="text-lg font-semibold text-white mb-3">Available for Inference</h3>
        {versionsWithWeights.length === 0 ? (
          <Card variant="default" className="text-center !py-8">
            <HardDrive size={24} className="text-helix-dim mx-auto mb-2" />
            <p className="text-base text-helix-text2">No versions with stored weights</p>
            <p className="text-sm text-helix-muted mt-1">
              Inference requires model weights to be stored on 0G.
            </p>
          </Card>
        ) : (
          <div className="space-y-2">
            {versionsWithWeights.slice().reverse().map((v, i) => (
              <Card key={`${v.semver}-${i}`} variant="default" className="!p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className="text-base font-mono font-medium text-white">v{v.semver}</span>
                    {v.accuracy > 0 && (
                      <span className="text-base text-helix-muted">{(v.accuracy * 100).toFixed(1)}% accuracy</span>
                    )}
                    <span className="text-base text-helix-dim">{formatDate(v.timestamp)}</span>
                  </div>
                  <Link
                    href={`/inference?hash=${v.rootHash}&version=${v.semver}&model=${model.slug}`}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white text-black text-base font-medium hover:bg-white/90 transition-colors"
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
          <h3 className="text-lg font-semibold text-white mb-2">Acquire This Model</h3>
          <p className="text-sm text-helix-muted mb-4">
            Buying a model transfers the NFT to you, giving you full ownership, access to encrypted weights,
            and the right to set inference fees.
          </p>
          <button
            type="button"
            className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-helix-surface border border-helix-border text-base text-helix-text hover:border-helix-border2 hover:text-white transition-colors"
            title="Offer functionality will be available in a future release"
          >
            <ShoppingCart size={14} />
            Make Offer
          </button>
        </Card>
      )}
    </div>
  );
}

// ============================================================================
// Pending Transfer Card
// ============================================================================

const ZERO_ADDR = '0x0000000000000000000000000000000000000000';

function PendingTransferCard({
  model: _model,
  isOwner,
  pendingTransfer,
  isCompletingTransfer,
  isCancellingSale,
  onCompleteTransfer,
  onCancelSale,
}: {
  model: NonNullable<ReturnType<typeof useModelDetail>['model']>;
  isOwner: boolean;
  pendingTransfer: { buyer: string; payment: bigint; deadline: number } | null;
  isCompletingTransfer: boolean;
  isCancellingSale: boolean;
  onCompleteTransfer: () => void;
  onCancelSale: () => void;
}) {
  if (!pendingTransfer || pendingTransfer.buyer === ZERO_ADDR) return null;

  const paymentEth = Number(pendingTransfer.payment) / 1e18;
  const deadlineDate = new Date(pendingTransfer.deadline * 1000);
  const isExpired = deadlineDate.getTime() < Date.now();

  return (
    <Card variant="default" className="border-yellow-500/20">
      <div className="flex items-center gap-2 mb-3">
        <ArrowRightLeft size={14} className="text-yellow-400" />
        <h3 className="text-sm font-semibold text-yellow-400">Pending Transfer</h3>
      </div>

      <div className="space-y-2 mb-4">
        <div className="flex items-center justify-between">
          <span className="text-base text-helix-muted">Buyer</span>
          <code className="text-base font-mono text-helix-text">{truncateAddress(pendingTransfer.buyer)}</code>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-base text-helix-muted">Escrowed</span>
          <span className="text-base font-mono text-white">{paymentEth.toFixed(4)} ADI</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-base text-helix-muted">Deadline</span>
          <span className={cn('text-sm', isExpired ? 'text-red-400' : 'text-helix-text')}>
            {isExpired ? 'Expired' : deadlineDate.toLocaleString()}
          </span>
        </div>
      </div>

      {isOwner ? (
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onCompleteTransfer}
            disabled={isCompletingTransfer || isExpired}
            className={cn(
              'flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-base font-medium transition-colors',
              isCompletingTransfer || isExpired
                ? 'bg-helix-border text-helix-muted cursor-not-allowed'
                : 'bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30',
            )}
          >
            {isCompletingTransfer ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <CheckCircle size={14} />
            )}
            {isCompletingTransfer ? 'Transferring...' : 'Complete Transfer'}
          </button>
          <button
            type="button"
            onClick={onCancelSale}
            disabled={isCancellingSale}
            className={cn(
              'flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm transition-colors',
              isCancellingSale
                ? 'bg-helix-border text-helix-muted cursor-not-allowed'
                : 'bg-red-500/10 border border-red-500/30 text-red-400 hover:bg-red-500/20',
            )}
          >
            {isCancellingSale ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <XCircle size={14} />
            )}
            Cancel
          </button>
        </div>
      ) : (
        <div className="flex items-center gap-2 px-3 py-2 bg-yellow-500/5 border border-yellow-500/20 rounded-lg">
          <Loader2 size={16} className="animate-spin text-yellow-400" />
          <p className="text-sm text-yellow-300/70">
            Transfer in progress. The owner must complete or cancel the transfer.
          </p>
        </div>
      )}
    </Card>
  );
}

// ============================================================================
// Claim Weights Card (buyer re-encrypts after transfer)
// ============================================================================

const SENTINEL_HASH = 'pending-buyer-reencrypt';

function ClaimWeightsCard({
  model,
  onClaim,
  isClaiming,
}: {
  model: NonNullable<ReturnType<typeof useModelDetail>['model']>;
  onClaim: () => void;
  isClaiming: boolean;
}) {
  const latestVersion = model.versions[model.versions.length - 1];
  const needsClaim = latestVersion?.rootHash === SENTINEL_HASH;

  if (!needsClaim) return null;

  return (
    <Card variant="default" className="border-blue-500/20">
      <div className="flex items-center gap-2 mb-3">
        <Download size={14} className="text-blue-400" />
        <h3 className="text-sm font-semibold text-blue-400">Claim Purchased Weights</h3>
      </div>

      <p className="text-sm text-helix-muted mb-4">
        The seller has transferred this model to you. Claim the weights to re-encrypt them
        with your wallet key and upload to 0G. Until you claim, the weights are temporarily
        held on the server relay.
      </p>

      <button
        type="button"
        onClick={onClaim}
        disabled={isClaiming}
        className={cn(
          'flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-base font-medium transition-colors w-full',
          isClaiming
            ? 'bg-helix-border text-helix-muted cursor-not-allowed'
            : 'bg-blue-500/20 text-blue-400 hover:bg-blue-500/30',
        )}
      >
        {isClaiming ? (
          <>
            <Loader2 size={14} className="animate-spin" />
            Claiming & Re-encrypting...
          </>
        ) : (
          <>
            <Download size={14} />
            Claim & Re-encrypt Weights
          </>
        )}
      </button>
    </Card>
  );
}

// ============================================================================
// Main Page
// ============================================================================

export default function ModelDetailPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const tokenId = Number(params.id);
  const [activeTab, setActiveTab] = useState<TabId>('overview');

  const { address } = useAccount();
  const chainId = useChainId();
  const { signMessageAsync } = useSignMessage();

  const {
    model,
    isLoading,
    isError,
    isOwner,
    isConnected,
    isContractDeployed,
    refetch: refetchDetail,
  } = useModelDetail(tokenId);

  // ── Model registry (owner write operations) ────────────────────────
  const {
    addVersion,
    setPublic,
    setInferenceFee,
    setForSale,
    setSalePrice,
    deleteModel,
    isWritePending: isRegistryWritePending,
    isConfirming: isRegistryConfirming,
    writeError: registryWriteError,
    isSuccess: isRegistrySuccess,
  } = useModelRegistry();

  // ── Weight storage (local weights download) ─────────────────────────
  const { downloadWeights, loadWeights } = useWeightStorage();

  // ── Contract address ────────────────────────────────────────────────
  const contractAddress = useMemo(() => {
    return getContractAddress(chainId, 'helixModelStore');
  }, [chainId]);
  const addr = contractAddress as `0x${string}`;

  // ── Write contract for completeTransfer / cancelSale ────────────────
  const {
    writeContract,
    data: txHash,
    isPending: isWritePending,
    error: writeError,
  } = useWriteContract();

  const { isLoading: isConfirming, isSuccess } = useWaitForTransactionReceipt({
    hash: txHash,
  });

  // Refetch after successful write (transfer/cancel)
  useEffect(() => {
    if (isSuccess) {
      refetchDetail();
    }
  }, [isSuccess, refetchDetail]);

  // Refetch after successful registry write (settings/version)
  useEffect(() => {
    if (isRegistrySuccess) {
      refetchDetail();
    }
  }, [isRegistrySuccess, refetchDetail]);

  // ── Pending transfer read ───────────────────────────────────────────
  const {
    data: rawPendingTransfer,
    refetch: refetchPending,
  } = useReadContract({
    address: addr,
    abi: HELIX_MODEL_STORE_ABI,
    functionName: 'pendingTransfers',
    args: [BigInt(tokenId)],
    query: { enabled: isContractDeployed && !isNaN(tokenId) },
  });

  // Refetch pending transfer after tx success
  useEffect(() => {
    if (isSuccess) {
      refetchPending();
    }
  }, [isSuccess, refetchPending]);

  const pendingTransfer = useMemo(() => {
    if (!rawPendingTransfer) return null;
    const [buyer, payment, deadline] = rawPendingTransfer as unknown as [string, bigint, bigint];
    return {
      buyer,
      payment,
      deadline: Number(deadline),
    };
  }, [rawPendingTransfer]);

  const hasPendingTransfer = !!pendingTransfer && pendingTransfer.buyer !== ZERO_ADDR;

  // ── Inference state ─────────────────────────────────────────────────
  const [inferenceEnabled, setInferenceEnabled] = useState(false);
  const [isEnablingInference, setIsEnablingInference] = useState(false);
  const [isDisablingInference, setIsDisablingInference] = useState(false);
  const [isCompletingTransfer, setIsCompletingTransfer] = useState(false);
  const [isClaimingWeights, setIsClaimingWeights] = useState(false);
  const [inferenceError, setInferenceError] = useState<string | null>(null);
  const [isDownloadingLatest, setIsDownloadingLatest] = useState(false);
  const [hasLocalWeights, setHasLocalWeights] = useState(false);

  // ── Determine if latest version has downloadable weights ────────────
  const latestVersion = model && model.versions.length > 0
    ? model.versions[model.versions.length - 1]
    : null;
  const latestHasWeights = latestVersion?.weightsStored && !!latestVersion?.rootHash;

  // Check if local weights exist in IndexedDB for this model
  useEffect(() => {
    if (!model) return;
    loadWeights(model.tokenId).then((data) => {
      setHasLocalWeights(!!data);
    }).catch(() => setHasLocalWeights(false));
  }, [model, loadWeights]);

  // ── Handlers ────────────────────────────────────────────────────────

  const handleDownloadLatest = useCallback(async () => {
    if (!model) return;
    setIsDownloadingLatest(true);
    try {
      if (latestHasWeights && latestVersion) {
        // Download from 0G
        await downloadWeightsFromVersion(latestVersion, model.tokenId, signMessageAsync);
      } else {
        // Download from local IndexedDB
        await downloadWeights(model.tokenId);
      }
    } catch {
      // Error handled silently for header button; per-version button shows details
    } finally {
      setIsDownloadingLatest(false);
    }
  }, [latestVersion, latestHasWeights, model, signMessageAsync, downloadWeights]);

  const handleEnableInference = useCallback(async (versionIndex: number) => {
    if (!model || !signMessageAsync) return;
    setIsEnablingInference(true);
    setInferenceError(null);
    try {
      const version = model.versions[versionIndex];
      if (!version?.rootHash) throw new Error('No root hash for this version');

      // Fetch encrypted weights from 0G
      const result = await fetchFrom0G(version.rootHash);

      // Derive key
      const key = await deriveModelKey(
        (message: string) => signMessageAsync({ message }),
        model.tokenId,
      );

      let weightsJson: string;
      if (result.encoding === 'base64') {
        const binaryStr = atob(result.data as string);
        let bytes = new Uint8Array(binaryStr.length);
        for (let i = 0; i < binaryStr.length; i++) {
          bytes[i] = binaryStr.charCodeAt(i);
        }
        // Safety net: decompress if server-side gunzip didn't fire
        bytes = await ensureDecompressed(bytes);
        weightsJson = await decryptWeights(key, bytes);
      } else {
        weightsJson = JSON.stringify(result.data);
      }

      // POST decrypted weights to enable inference
      const res = await fetch(`/api/models/${model.tokenId}/enable-inference`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ weights: JSON.parse(weightsJson), version: versionIndex, ownerAddress: address }),
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.error || `Enable inference failed: HTTP ${res.status}`);
      }

      setInferenceEnabled(true);
    } catch (err) {
      setInferenceError(err instanceof Error ? err.message : 'Failed to enable inference');
      setTimeout(() => setInferenceError(null), 5000);
    } finally {
      setIsEnablingInference(false);
    }
  }, [model, signMessageAsync]);

  const handleDisableInference = useCallback(async () => {
    if (!model) return;
    setIsDisablingInference(true);
    setInferenceError(null);
    try {
      const res = await fetch(`/api/models/${model.tokenId}/disable-inference`, {
        method: 'DELETE',
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.error || `Disable inference failed: HTTP ${res.status}`);
      }

      setInferenceEnabled(false);
    } catch (err) {
      setInferenceError(err instanceof Error ? err.message : 'Failed to disable inference');
      setTimeout(() => setInferenceError(null), 5000);
    } finally {
      setIsDisablingInference(false);
    }
  }, [model]);

  const handleCompleteTransfer = useCallback(async () => {
    if (!model || !pendingTransfer || !signMessageAsync || !address) return;
    setIsCompletingTransfer(true);
    setInferenceError(null);
    try {
      // Step 1: Fetch encrypted weights from 0G
      const transferLatest = model.versions[model.versions.length - 1];
      if (!transferLatest?.rootHash) throw new Error('No weights stored for this model');

      const result = await fetchFrom0G(transferLatest.rootHash);

      // Step 2: Decrypt with seller's key
      const key = await deriveModelKey(
        (message: string) => signMessageAsync({ message }),
        model.tokenId,
      );

      let weightsJson: string;
      if (result.encoding === 'base64') {
        const binaryStr = atob(result.data as string);
        let bytes = new Uint8Array(binaryStr.length);
        for (let i = 0; i < binaryStr.length; i++) {
          bytes[i] = binaryStr.charCodeAt(i);
        }
        bytes = await ensureDecompressed(bytes);
        weightsJson = await decryptWeights(key, bytes);
      } else {
        weightsJson = JSON.stringify(result.data);
      }

      // Step 3: Send plaintext weights to transfer relay for buyer pickup
      const relayRes = await fetch(`/api/models/${model.tokenId}/transfer-relay`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          weights: JSON.parse(weightsJson),
          sellerAddress: address,
          buyerAddress: pendingTransfer.buyer,
        }),
      });

      if (!relayRes.ok) {
        const errBody = await relayRes.json().catch(() => ({}));
        throw new Error(errBody.error || `Transfer relay failed: HTTP ${relayRes.status}`);
      }

      // Step 4: Complete transfer on-chain with sentinel hash
      const sentinelHash = 'pending-buyer-reencrypt';
      writeContract({
        address: addr,
        abi: HELIX_MODEL_STORE_ABI,
        functionName: 'completeTransfer',
        args: [BigInt(model.tokenId), sentinelHash],
      });
    } catch (err) {
      setInferenceError(err instanceof Error ? err.message : 'Transfer failed');
      setTimeout(() => setInferenceError(null), 5000);
    } finally {
      setIsCompletingTransfer(false);
    }
  }, [model, pendingTransfer, signMessageAsync, address, addr, writeContract]);

  const handleCancelSale = useCallback(() => {
    if (!model) return;
    writeContract({
      address: addr,
      abi: HELIX_MODEL_STORE_ABI,
      functionName: 'cancelSale',
      args: [BigInt(model.tokenId)],
    });
  }, [model, addr, writeContract]);

  // ── Claim weights (buyer re-encrypts after purchase) ──────────────
  const handleClaimWeights = useCallback(async () => {
    if (!model || !signMessageAsync || !address) return;
    setIsClaimingWeights(true);
    setInferenceError(null);
    try {
      // Step 1: Fetch plaintext weights from relay
      const relayRes = await fetch(
        `/api/models/${model.tokenId}/transfer-relay?buyer=${address.toLowerCase()}`,
      );
      if (!relayRes.ok) {
        const errBody = await relayRes.json().catch(() => ({}));
        throw new Error(errBody.error || `Relay fetch failed: HTTP ${relayRes.status}`);
      }
      const relayData = await relayRes.json();
      const weightsJson = JSON.stringify(relayData.weights);

      // Step 2: Derive buyer's own key
      const buyerKey = await deriveModelKey(
        (message: string) => signMessageAsync({ message }),
        model.tokenId,
      );

      // Step 3: Encrypt with buyer's key and upload to 0G
      const sessionId = `claim-${model.tokenId}-${Date.now()}`;
      const claimLatest = model.versions[model.versions.length - 1];
      const storeResult = await encryptAndStoreWeights(
        sessionId,
        weightsJson,
        async (data: string) => encryptWeights(buyerKey, data),
        claimLatest?.semver,
      );

      // Step 4: Update root hash on-chain
      const versionIndex = model.versions.length - 1;
      writeContract({
        address: addr,
        abi: HELIX_MODEL_STORE_ABI,
        functionName: 'updateVersionRootHash',
        args: [BigInt(model.tokenId), BigInt(versionIndex), storeResult.root_hash],
      });

      // Step 5: Clear relay entry
      await fetch(
        `/api/models/${model.tokenId}/transfer-relay?buyer=${address.toLowerCase()}`,
        { method: 'DELETE' },
      );
    } catch (err) {
      setInferenceError(err instanceof Error ? err.message : 'Claim failed');
      setTimeout(() => setInferenceError(null), 5000);
    } finally {
      setIsClaimingWeights(false);
    }
  }, [model, signMessageAsync, address, addr, writeContract]);

  // ── Computed values ─────────────────────────────────────────────────
  const tabs = useMemo(() => getTabList(isOwner), [isOwner]);
  const allErrors = writeError || registryWriteError || inferenceError;
  const errorMessage = inferenceError
    || (writeError ? writeError.message.slice(0, 120) : '')
    || (registryWriteError ? registryWriteError.message.slice(0, 120) : '');

  // Loading
  if (isLoading) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
        className="space-y-6"
      >
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
      </motion.div>
    );
  }

  // Not found
  if (!model || isError || !isContractDeployed) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
        className="flex flex-col items-center py-20 gap-4"
      >
        <Layers size={40} className="text-helix-dim" />
        <p className="text-lg font-medium text-helix-text2">Model Not Found</p>
        <p className="text-base text-helix-muted">
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
      </motion.div>
    );
  }

  // Not public and not owner
  if (!model.isPublic && !isOwner) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
        className="flex flex-col items-center py-20 gap-4"
      >
        <Lock size={40} className="text-helix-dim" />
        <p className="text-lg font-medium text-helix-text2">Private Model</p>
        <p className="text-base text-helix-muted">
          This model is private. Only the owner can view it.
        </p>
        <Link
          href="/"
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text hover:text-white transition-colors"
        >
          <ArrowLeft size={14} />
          Back to Models
        </Link>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
      className="space-y-6"
    >
      {/* Back link */}
      <Link
        href="/"
        className="inline-flex items-center gap-2 text-base text-helix-muted hover:text-white transition-colors"
      >
        <ArrowLeft size={18} />
        Back to Models
      </Link>

      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-xl bg-white/[0.06] flex items-center justify-center">
            <Layers size={28} className="text-white" />
          </div>
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-3xl font-semibold text-white tracking-tight">{model.name}</h1>
              {model.isPublic ? (
                <Badge variant="default" className="text-green-400 flex items-center gap-1.5">
                  <Globe size={16} />
                  Public
                </Badge>
              ) : (
                <Badge variant="default" className="flex items-center gap-1.5">
                  <Lock size={16} />
                  Private
                </Badge>
              )}
              <Badge variant="default">#{model.tokenId}</Badge>
              {model.versions.length === 0 && (
                <span className="px-3 py-1 rounded-full text-sm font-medium bg-zinc-500/20 text-zinc-400 border border-zinc-500/30">
                  Stopped
                </span>
              )}
            </div>
            <div className="flex items-center gap-3 mt-1">
              <span className="text-base font-mono text-helix-muted">{model.slug}</span>
              <span className="text-base text-helix-dim">by {truncateAddress(model.creator)}</span>
            </div>
          </div>
        </div>

        {/* Header actions — owner gets download and train buttons */}
        <div className="flex items-center gap-3">
          {isOwner && (latestHasWeights || hasLocalWeights) && (
            <button
              type="button"
              onClick={handleDownloadLatest}
              disabled={isDownloadingLatest}
              className={cn(
                'flex items-center gap-2 px-5 py-2.5 rounded-lg border text-base font-medium transition-colors',
                isDownloadingLatest
                  ? 'bg-helix-surface border-helix-border text-helix-muted cursor-wait'
                  : 'bg-helix-surface border-helix-border text-helix-text hover:border-helix-border2 hover:text-white',
              )}
            >
              {isDownloadingLatest ? (
                <Loader2 size={18} className="animate-spin" />
              ) : (
                <Download size={18} />
              )}
              {latestHasWeights ? 'Download Weights' : 'Download Local'}
            </button>
          )}
          {isOwner && (
            <Link
              href={`/train?model=${model.tokenId}`}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-white text-black text-base font-medium hover:bg-white/90 transition-colors"
            >
              <Play size={18} />
              Train
            </Link>
          )}
        </div>
      </div>

      {/* Error notice */}
      {allErrors && (
        <div className="flex items-center gap-3 px-4 py-3 bg-red-500/5 border border-red-500/20 rounded-lg">
          <XCircle size={16} className="text-red-400 shrink-0" />
          <p className="text-sm text-red-300/70">{errorMessage}</p>
        </div>
      )}

      {/* Pending Transfer Card (shown above tabs when active) */}
      {hasPendingTransfer && (
        <PendingTransferCard
          model={model}
          isOwner={isOwner}
          pendingTransfer={pendingTransfer}
          isCompletingTransfer={isCompletingTransfer || isWritePending || isConfirming}
          isCancellingSale={isWritePending || isConfirming}
          onCompleteTransfer={handleCompleteTransfer}
          onCancelSale={handleCancelSale}
        />
      )}

      {/* Claim Weights Card (buyer re-encrypts after purchase) */}
      {isOwner && (
        <ClaimWeightsCard
          model={model}
          onClaim={handleClaimWeights}
          isClaiming={isClaimingWeights || isWritePending || isConfirming}
        />
      )}

      {/* Tabs */}
      <Tabs
        tabs={[...tabs]}
        activeTab={activeTab}
        onChange={(id) => setActiveTab(id as TabId)}
        className="border-b border-helix-border pb-px"
      />

      {/* Tab content */}
      <div className="min-h-[400px]">
        {activeTab === 'overview' && <OverviewTab model={model} isOwner={isOwner} chainId={chainId} contractAddress={contractAddress} />}
        {activeTab === 'settings' && isOwner && (
          <SettingsTab
            model={model}
            setPublic={setPublic}
            setInferenceFee={setInferenceFee}
            setForSale={setForSale}
            setSalePrice={setSalePrice}
            deleteModel={deleteModel}
            isSettingsWritePending={isRegistryWritePending || isRegistryConfirming}
            isDeletePending={isRegistryWritePending || isRegistryConfirming}
            onDeleteSuccess={() => router.push('/dashboard')}
          />
        )}
        {activeTab === 'versions' && (
          <VersionsTab
            model={model}
            isOwner={isOwner}
            isConnected={isConnected}
            addVersion={addVersion}
            isVersionWritePending={isRegistryWritePending || isRegistryConfirming}
            signMessageAsync={signMessageAsync}
            hasLocalWeights={hasLocalWeights}
            onDownloadLocal={() => downloadWeights(model.tokenId)}
          />
        )}
        {activeTab === 'inference' && (
          <InferenceTab
            model={model}
            isOwner={isOwner}
            inferenceEnabled={inferenceEnabled}
            isEnablingInference={isEnablingInference}
            isDisablingInference={isDisablingInference}
            onEnableInference={handleEnableInference}
            onDisableInference={handleDisableInference}
          />
        )}
      </div>
    </motion.div>
  );
}
