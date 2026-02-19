'use client';

import { useState, useMemo, useEffect, useCallback, useRef, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Loader2,
  Eraser,
  CheckCircle,
  AlertTriangle,
  Upload,
  Image as ImageIcon,
  Pencil,
  Shield,
  Clock,
  Users,
  Search,
  HardDrive,
  X,
  Coins,
  ExternalLink,
} from 'lucide-react';
import { useAccount, useSignMessage, useWriteContract, useWaitForTransactionReceipt, useChainId } from 'wagmi';
import { parseEther } from 'viem';
import { Badge } from '@/components/ui/Badge';
import { cn } from '@/lib/utils';
import { usePublicModels, type PublicModel } from '@/hooks/usePublicModels';
import { useModelRegistry } from '@/hooks/useModelRegistry';
import { deriveModelKey, decryptWeights } from '@/lib/model-encryption';
import { HELIX_MODEL_STORE_ABI, getContractAddress } from '@/lib/contracts';

// ============================================================================
// Types & Constants
// ============================================================================

interface MPCInferenceResult {
  prediction: number;
  confidence: number;
  probabilities: number[];
  num_parties: number;
  distributed: boolean;
  attestation?: {
    input_hash: string;
    output_hash: string;
    worker_signatures: string[];
    chain_tx_hash?: string;
    inference_id?: number;
  };
  timing: {
    share_generation_ms: number;
    forward_pass_ms: number;
    signing_ms: number;
    total_ms: number;
  };
}

type InferencePhase = 'idle' | 'submitting' | 'done' | 'error';
type InputMode = 'draw' | 'upload';
type ModelFilter = 'all' | 'mine' | 'others';
type WeightFetchStatus = 'idle' | 'fetching' | 'decrypting' | 'uploading' | 'done' | 'error';

const CANVAS_SIZE = 280;
const GRID_SIZE = 28;
const BRUSH_RADIUS = 12;
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';
const BASE_INFERENCE_COST = 0.001; // ADI per inference

// ============================================================================
// Drawing Canvas
// ============================================================================

interface DrawingCanvasProps {
  onPixelsReady: (pixels: number[]) => void;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
}

function DrawingCanvas({ onPixelsReady, canvasRef }: DrawingCanvasProps) {
  const isDrawing = useRef(false);
  const lastPos = useRef<{ x: number; y: number } | null>(null);

  const getCanvasPos = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return { x: 0, y: 0 };
      const rect = canvas.getBoundingClientRect();
      const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
      const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
      return {
        x: (clientX - rect.left) * (canvas.width / rect.width),
        y: (clientY - rect.top) * (canvas.height / rect.height),
      };
    },
    [canvasRef],
  );

  const draw = useCallback(
    (x: number, y: number) => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext('2d');
      if (!ctx) return;
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(x, y, BRUSH_RADIUS, 0, Math.PI * 2);
      ctx.fill();
      if (lastPos.current) {
        const dx = x - lastPos.current.x;
        const dy = y - lastPos.current.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const steps = Math.ceil(dist / 4);
        for (let i = 1; i < steps; i++) {
          const t = i / steps;
          ctx.beginPath();
          ctx.arc(lastPos.current.x + dx * t, lastPos.current.y + dy * t, BRUSH_RADIUS, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      lastPos.current = { x, y };
    },
    [canvasRef],
  );

  const handleStart = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      e.preventDefault();
      isDrawing.current = true;
      lastPos.current = null;
      const pos = getCanvasPos(e);
      draw(pos.x, pos.y);
    },
    [getCanvasPos, draw],
  );

  const handleMove = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      e.preventDefault();
      if (!isDrawing.current) return;
      const pos = getCanvasPos(e);
      draw(pos.x, pos.y);
    },
    [getCanvasPos, draw],
  );

  const extractPixels = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = GRID_SIZE;
    tempCanvas.height = GRID_SIZE;
    const tempCtx = tempCanvas.getContext('2d')!;
    tempCtx.imageSmoothingEnabled = true;
    tempCtx.imageSmoothingQuality = 'high';
    tempCtx.drawImage(canvas, 0, 0, GRID_SIZE, GRID_SIZE);
    const imageData = tempCtx.getImageData(0, 0, GRID_SIZE, GRID_SIZE);
    const pixels: number[] = [];
    for (let i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
      pixels.push(imageData.data[i * 4] / 255);
    }
    onPixelsReady(pixels);
  }, [canvasRef, onPixelsReady]);

  const handleEnd = useCallback(() => {
    isDrawing.current = false;
    lastPos.current = null;
    extractPixels();
  }, [extractPixels]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  }, [canvasRef]);

  return (
    <canvas
      ref={canvasRef as React.RefObject<HTMLCanvasElement>}
      width={CANVAS_SIZE}
      height={CANVAS_SIZE}
      className="rounded-xl border border-helix-border cursor-crosshair touch-none"
      style={{ width: CANVAS_SIZE, height: CANVAS_SIZE }}
      onMouseDown={handleStart}
      onMouseMove={handleMove}
      onMouseUp={handleEnd}
      onMouseLeave={handleEnd}
      onTouchStart={handleStart}
      onTouchMove={handleMove}
      onTouchEnd={handleEnd}
    />
  );
}

// ============================================================================
// Image Upload
// ============================================================================

interface ImageUploadProps {
  onPixelsReady: (pixels: number[]) => void;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
}

function ImageUpload({ onPixelsReady, canvasRef }: ImageUploadProps) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  const handleFile = useCallback(
    (file: File) => {
      const reader = new FileReader();
      reader.onload = () => {
        const img = new window.Image();
        img.onload = () => {
          const canvas = canvasRef.current;
          if (!canvas) return;
          const ctx = canvas.getContext('2d');
          if (!ctx) return;
          ctx.fillStyle = '#000000';
          ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
          const scale = Math.min(CANVAS_SIZE / img.width, CANVAS_SIZE / img.height);
          const w = img.width * scale;
          const h = img.height * scale;
          ctx.drawImage(img, (CANVAS_SIZE - w) / 2, (CANVAS_SIZE - h) / 2, w, h);
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = GRID_SIZE;
          tempCanvas.height = GRID_SIZE;
          const tempCtx = tempCanvas.getContext('2d')!;
          tempCtx.imageSmoothingEnabled = true;
          tempCtx.imageSmoothingQuality = 'high';
          tempCtx.drawImage(canvas, 0, 0, GRID_SIZE, GRID_SIZE);
          const imageData = tempCtx.getImageData(0, 0, GRID_SIZE, GRID_SIZE);
          const pixels: number[] = [];
          for (let i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
            const r = imageData.data[i * 4];
            const g = imageData.data[i * 4 + 1];
            const b = imageData.data[i * 4 + 2];
            pixels.push((0.299 * r + 0.587 * g + 0.114 * b) / 255);
          }
          onPixelsReady(pixels);
          setFileName(file.name);
        };
        img.src = reader.result as string;
      };
      reader.readAsDataURL(file);
    },
    [canvasRef, onPixelsReady],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) handleFile(file);
    },
    [handleFile],
  );

  return (
    <div className="space-y-3">
      <canvas
        ref={canvasRef as React.RefObject<HTMLCanvasElement>}
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        className="rounded-xl border border-helix-border"
        style={{ width: CANVAS_SIZE, height: CANVAS_SIZE, display: fileName ? 'block' : 'none' }}
      />
      {!fileName && (
        <div
          onClick={() => fileRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          className="flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed border-helix-border hover:border-helix-border2 bg-helix-bg cursor-pointer transition-colors"
          style={{ width: CANVAS_SIZE, height: CANVAS_SIZE }}
        >
          <Upload size={24} className="text-helix-muted" />
          <p className="text-sm text-helix-text2">Drop an image or click to upload</p>
          <p className="text-xs text-helix-muted">PNG, JPG, or any image of a digit</p>
        </div>
      )}
      {fileName && (
        <div className="flex items-center gap-2">
          <ImageIcon size={12} className="text-green-400" />
          <span className="text-xs text-helix-text truncate">{fileName}</span>
          <button
            type="button"
            onClick={() => {
              setFileName(null);
              onPixelsReady([]);
              const canvas = canvasRef.current;
              if (canvas) {
                const ctx = canvas.getContext('2d');
                if (ctx) { ctx.fillStyle = '#000000'; ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE); }
              }
            }}
            className="text-xs text-helix-muted hover:text-white ml-auto"
          >
            Remove
          </button>
        </div>
      )}
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => { const file = e.target.files?.[0]; if (file) handleFile(file); }}
      />
    </div>
  );
}

// ============================================================================
// Probability Bars
// ============================================================================

function ProbabilityBars({ probabilities, prediction }: { probabilities: number[]; prediction: number }) {
  return (
    <div className="space-y-1.5">
      {probabilities.map((prob, digit) => (
        <div key={digit} className="flex items-center gap-3">
          <span className={cn('w-5 text-right text-sm font-mono', digit === prediction ? 'text-white font-semibold' : 'text-helix-muted')}>
            {digit}
          </span>
          <div className="flex-1 h-5 bg-helix-border rounded-sm overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${prob * 100}%` }}
              transition={{ duration: 0.4, ease: 'easeOut' }}
              className={cn('h-full rounded-sm', digit === prediction ? 'bg-white' : 'bg-white/20')}
            />
          </div>
          <span className={cn('w-14 text-right text-xs font-mono', digit === prediction ? 'text-white' : 'text-helix-muted')}>
            {(prob * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}

// ============================================================================
// MPC Timing Breakdown
// ============================================================================

function TimingBreakdown({ timing }: { timing: MPCInferenceResult['timing'] }) {
  const phases = [
    { label: 'Secret sharing', ms: timing.share_generation_ms, color: 'bg-blue-400' },
    { label: 'Distributed forward pass', ms: timing.forward_pass_ms, color: 'bg-cyan-400' },
    { label: 'Worker attestation signing', ms: timing.signing_ms, color: 'bg-green-400' },
  ];
  return (
    <div className="space-y-1.5">
      {phases.map((p) => (
        <div key={p.label} className="flex items-center gap-3">
          <div className={cn('w-2 h-2 rounded-full shrink-0', p.color)} />
          <span className="text-xs text-helix-text flex-1">{p.label}</span>
          <span className="text-xs text-helix-muted font-mono">{p.ms}ms</span>
        </div>
      ))}
      <div className="flex items-center gap-3 pt-1 border-t border-helix-border">
        <Clock size={10} className="text-white shrink-0" />
        <span className="text-xs text-white font-medium flex-1">Total</span>
        <span className="text-xs text-white font-mono font-medium">{timing.total_ms}ms</span>
      </div>
    </div>
  );
}

// ============================================================================
// MPC Result Card
// ============================================================================

function MPCResultCard({ result }: { result: MPCInferenceResult }) {
  const [showDetails, setShowDetails] = useState(false);

  return (
    <div className="rounded-2xl bg-helix-surface border border-helix-border overflow-hidden">
      <div className="p-5">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Shield size={14} className="text-green-400" />
            <h3 className="text-base font-medium text-white">MPC Result</h3>
          </div>
          <div className="flex items-center gap-2">
            {result.distributed && (
              <Badge variant="default" className="text-green-400">
                <Shield size={10} /> Distributed
              </Badge>
            )}
            <Badge variant="default" className="text-blue-400">
              <Users size={10} /> {result.num_parties} workers
            </Badge>
            <Badge variant="default" className="text-helix-text2">
              <Clock size={10} /> {result.timing.total_ms}ms
            </Badge>
          </div>
        </div>

        {/* Prediction */}
        <div className="flex items-center gap-6 mb-5">
          <div className="w-20 h-20 rounded-xl bg-white/[0.06] flex items-center justify-center">
            <span className="text-4xl font-light text-white font-mono">{result.prediction}</span>
          </div>
          <div>
            <p className="text-2xl font-mono font-light text-white">
              {(result.confidence * 100).toFixed(1)}%
            </p>
            <p className="text-xs text-helix-muted mt-0.5">confidence</p>
          </div>
          <div className="ml-auto">
            <CheckCircle size={24} className="text-green-400" />
          </div>
        </div>

        <ProbabilityBars probabilities={result.probabilities} prediction={result.prediction} />

        {/* Attestation */}
        {result.attestation && (
          <div className="mt-4 pt-3 border-t border-helix-border">
            <div className="flex items-center gap-2 mb-2">
              <Shield size={12} className="text-green-400" />
              <span className="text-xs font-medium text-white">On-chain Attestation</span>
            </div>
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <span className="text-xs text-helix-muted w-20">Input hash</span>
                <span className="text-xs text-helix-text font-mono truncate">
                  {result.attestation.input_hash.slice(0, 16)}...
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-helix-muted w-20">Output hash</span>
                <span className="text-xs text-helix-text font-mono truncate">
                  {result.attestation.output_hash.slice(0, 16)}...
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-helix-muted w-20">Signatures</span>
                <div className="flex items-center gap-1">
                  {result.attestation.worker_signatures.map((sig, i) => (
                    <div
                      key={i}
                      className="w-5 h-5 rounded-full bg-green-400/20 flex items-center justify-center"
                      title={`Worker ${i}: ${sig.slice(0, 16)}...`}
                    >
                      <CheckCircle size={10} className="text-green-400" />
                    </div>
                  ))}
                  <span className="text-xs text-green-400 ml-1">
                    {result.attestation.worker_signatures.length}/{result.num_parties} signed
                  </span>
                </div>
              </div>
              {result.attestation.chain_tx_hash && (
                <div className="flex items-center gap-2 mt-1">
                  <span className="text-xs text-helix-muted w-20">On-chain TX</span>
                  <span className="text-xs text-green-400 font-mono truncate">
                    {result.attestation.chain_tx_hash.slice(0, 18)}...
                  </span>
                  {result.attestation.inference_id != null && (
                    <span className="text-xs text-helix-muted">(ID: {result.attestation.inference_id})</span>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Timing */}
        <div className="mt-4 pt-3 border-t border-helix-border">
          <button
            type="button"
            onClick={() => setShowDetails(!showDetails)}
            className="text-xs text-helix-muted hover:text-white transition-colors flex items-center gap-1"
          >
            <Clock size={10} />
            {showDetails ? 'Hide' : 'Show'} timing breakdown
          </button>
          <AnimatePresence>
            {showDetails && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden"
              >
                <div className="mt-3">
                  <TimingBreakdown timing={result.timing} />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Inner Page
// ============================================================================

function InferencePageInner() {
  const searchParams = useSearchParams();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { address } = useAccount();
  const { signMessageAsync } = useSignMessage();

  const chainId = useChainId();

  // Model discovery
  const { allModels, isLoading: isLoadingPublic } = usePublicModels();
  const { models: myModels, isLoading: isLoadingMine } = useModelRegistry();

  const [filter, setFilter] = useState<ModelFilter>('all');
  const [search, setSearch] = useState('');
  const [selectedModel, setSelectedModel] = useState<PublicModel | null>(null);
  const [selectedVersionIndex, setSelectedVersionIndex] = useState<number>(0);

  // Selected version (derived from model + index)
  const selectedVersion = useMemo(() => {
    if (!selectedModel || selectedModel.versions.length === 0) return selectedModel?.latestVersion ?? null;
    const idx = Math.min(selectedVersionIndex, selectedModel.versions.length - 1);
    return selectedModel.versions[idx] ?? null;
  }, [selectedModel, selectedVersionIndex]);

  // Weights
  const [weightFetchStatus, setWeightFetchStatus] = useState<WeightFetchStatus>('idle');
  const [weightFetchError, setWeightFetchError] = useState<string | null>(null);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const weightsFileRef = useRef<HTMLInputElement>(null);

  // Owner detection
  const isOwnerOfSelected = selectedModel && address
    ? selectedModel.owner.toLowerCase() === address.toLowerCase()
    : false;
  const [backendWeightsReady, setBackendWeightsReady] = useState(false);

  // Inference payment (non-owner flow)
  const { writeContract, data: inferenceTxHash, isPending: isPaymentPending } = useWriteContract();
  const { isSuccess: paymentConfirmed, isLoading: isPaymentConfirming } = useWaitForTransactionReceipt({ hash: inferenceTxHash });

  // Inference
  const [pixels, setPixels] = useState<number[]>([]);
  const [result, setResult] = useState<MPCInferenceResult | null>(null);
  const [phase, setPhase] = useState<InferencePhase>('idle');
  const [error, setError] = useState<string | null>(null);
  const [inputMode, setInputMode] = useState<InputMode>('draw');

  // Auto-discover latest completed training session when no on-chain models
  useEffect(() => {
    if (activeSessionId || allModels.length > 0) return;
    const discover = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/training/sessions`);
        if (!res.ok) return;
        const sessions = await res.json();
        const completed = (Array.isArray(sessions) ? sessions : [])
          .filter((s: { status: string }) => s.status === 'complete')
          .pop();
        if (completed) {
          setActiveSessionId(completed.session_id);
          setWeightFetchStatus('done');
        }
      } catch { /* non-fatal */ }
    };
    discover();
  }, [activeSessionId, allModels.length]);

  // Filtered models
  const filteredModels = useMemo(() => {
    // Hide private models from non-owners
    let models = allModels.filter((m) =>
      m.isPublic || m.owner.toLowerCase() === address?.toLowerCase()
    );
    if (filter === 'mine') {
      models = models.filter((m) => m.owner.toLowerCase() === address?.toLowerCase());
    } else if (filter === 'others') {
      models = models.filter((m) => m.owner.toLowerCase() !== address?.toLowerCase());
    }
    if (search.trim()) {
      const q = search.toLowerCase();
      models = models.filter(
        (m) =>
          m.name.toLowerCase().includes(q) ||
          m.slug.toLowerCase().includes(q) ||
          m.description.toLowerCase().includes(q),
      );
    }
    return models;
  }, [allModels, filter, search, address]);

  // Fee calculation
  const ownerFeeBps = selectedModel?.inferenceFee ?? 0;
  const ownerCommission = BASE_INFERENCE_COST * (ownerFeeBps / 10000);
  const totalFee = BASE_INFERENCE_COST + ownerCommission;

  // Auto-select from query param
  useEffect(() => {
    const tokenIdParam = searchParams.get('model');
    if (!tokenIdParam || allModels.length === 0) return;
    const tokenId = Number(tokenIdParam);
    if (isNaN(tokenId)) return;
    const found = allModels.find((m) => m.tokenId === tokenId);
    if (found && !selectedModel) setSelectedModel(found);
  }, [searchParams, allModels, selectedModel]);

  // Upload weights to MPC backend
  const uploadWeightsToBackend = useCallback(async (weightsData: unknown): Promise<string | null> => {
    const res = await fetch(`${API_BASE}/api/training/weights`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(weightsData),
    });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.message || errBody.error || `HTTP ${res.status}`);
    }
    const result = await res.json();
    return result.session_id ?? null;
  }, []);

  // Fetch weights from 0G (owner) or check backend cache (non-owner)
  // Re-triggers when model or version selection changes
  const fetchKeyRef = useRef<string | null>(null);
  useEffect(() => {
    if (!selectedModel) {
      fetchKeyRef.current = null;
      setWeightFetchStatus('idle');
      setWeightFetchError(null);
      setBackendWeightsReady(false);
      return;
    }

    const versionKey = `${selectedModel.tokenId}:${selectedVersionIndex}`;
    if (fetchKeyRef.current === versionKey) return;
    fetchKeyRef.current = versionKey;

    const isOwner = address && selectedModel.owner.toLowerCase() === address.toLowerCase();

    // Non-owner: check if backend has cached weights for this version
    if (!isOwner) {
      const checkBackend = async () => {
        try {
          const res = await fetch(`${API_BASE}/api/models/${selectedModel.tokenId}/inference-ready?version=${selectedVersionIndex}`);
          if (res.ok) {
            const data = await res.json();
            if (data.ready) {
              setBackendWeightsReady(true);
              setWeightFetchStatus('done');
              return;
            }
          }
        } catch { /* non-fatal */ }
        // Backend doesn't have weights cached — show message
        setWeightFetchError('Weights not available for this version. The model owner must run inference first.');
        setWeightFetchStatus('error');
        fetchKeyRef.current = null;
      };
      checkBackend();
      return;
    }

    // Owner: fetch from 0G using the selected version's rootHash, decrypt, upload to backend
    const sv = selectedVersion;
    if (!sv || !sv.weightsStored || !sv.rootHash) {
      setWeightFetchStatus('idle');
      setWeightFetchError(null);
      return;
    }

    const doFetch = async () => {
      setWeightFetchStatus('fetching');
      setWeightFetchError(null);

      try {
        const res = await fetch('/api/fetch-from-0g', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ rootHash: sv.rootHash }),
        });

        if (!res.ok) {
          const errBody = await res.json().catch(() => ({}));
          throw new Error(errBody.error || `Failed to fetch from 0G (HTTP ${res.status})`);
        }

        const result = await res.json();
        let weightsData: unknown;

        if (result.encoding === 'base64') {
          setWeightFetchStatus('decrypting');
          const rawBytes = Uint8Array.from(atob(result.data), (c) => c.charCodeAt(0));
          const signMsg = async (message: string): Promise<string> => signMessageAsync({ message });
          const key = await deriveModelKey(signMsg, selectedModel.tokenId);
          const decryptedJson = await decryptWeights(key, rawBytes);
          weightsData = JSON.parse(decryptedJson);
        } else {
          weightsData = result.data;
        }

        setWeightFetchStatus('uploading');
        const backendSessionId = await uploadWeightsToBackend(weightsData);
        setActiveSessionId(backendSessionId ?? sv.sessionId);
        setWeightFetchStatus('done');

        // Also cache weights by token ID + version for future non-owner inference
        try {
          await fetch(`${API_BASE}/api/models/${selectedModel.tokenId}/cache-weights?version=${selectedVersionIndex}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(weightsData),
          });
        } catch { /* non-fatal */ }
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to fetch weights';
        setWeightFetchError(message);
        setWeightFetchStatus('error');
        fetchKeyRef.current = null;
      }
    };

    doFetch();
  }, [selectedModel, selectedVersion, selectedVersionIndex, signMessageAsync, uploadWeightsToBackend, address]);

  // Manual weight upload
  const handleManualWeightUpload = useCallback(async (file: File) => {
    try {
      setWeightFetchStatus('uploading');
      setWeightFetchError(null);
      const text = await file.text();
      const data = JSON.parse(text);
      const backendSessionId = await uploadWeightsToBackend(data);
      setActiveSessionId(backendSessionId ?? `manual-${Date.now()}`);
      setWeightFetchStatus('done');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to upload weights';
      setWeightFetchError(message);
      setWeightFetchStatus('error');
    }
  }, [uploadWeightsToBackend]);

  // Clear canvas
  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    setPixels([]);
    setResult(null);
    setPhase('idle');
    setError(null);
  }, []);

  // Run inference (owner: direct, non-owner: after payment confirmation)
  const runInference = useCallback(async (paymentTxHash?: string) => {
    if (!pixels.length) return;
    setPhase('submitting');
    setError(null);
    setResult(null);

    try {
      // Model-based inference (non-owner with cached weights, or owner with model selected)
      const useModelEndpoint = selectedModel && (backendWeightsReady || isOwnerOfSelected);
      const res = await fetch('/api/inference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(useModelEndpoint ? {
          model_token_id: selectedModel.tokenId,
          model_version_index: selectedVersionIndex,
          pixels,
          num_parties: 3,
          wallet_address: isOwnerOfSelected ? address : undefined,
          payment_tx: paymentTxHash,
        } : {
          session_id: activeSessionId,
          pixels,
          num_parties: 3,
        }),
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.error || `Inference HTTP ${res.status}`);
      }
      const data: MPCInferenceResult = await res.json();
      setResult(data);
      setPhase('done');

      // Persist to localStorage for Dashboard inference tab
      try {
        const entry = {
          id: `inf-${Date.now()}`,
          model: selectedModel?.name ?? 'Unknown',
          status: 'completed' as const,
          progress: 100,
          phase: 'done',
          workers: data.num_parties,
          created: Date.now(),
          inputLabel: `Digit prediction`,
          result: String(data.prediction),
          confidence: data.confidence,
          duration: data.timing.total_ms,
        };
        const raw = localStorage.getItem('helix-inference-history');
        const history = raw ? JSON.parse(raw) : [];
        history.unshift(entry);
        localStorage.setItem('helix-inference-history', JSON.stringify(history.slice(0, 50)));
      } catch {
        // Non-critical
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Inference failed');
      setPhase('error');
    }
  }, [activeSessionId, pixels, selectedModel, selectedVersionIndex, backendWeightsReady, isOwnerOfSelected, address]);

  // Non-owner: initiate on-chain payment then run inference
  const handlePayAndRun = useCallback(() => {
    if (!selectedModel) return;
    const modelStoreAddress = getContractAddress(chainId, 'helixModelStore') as `0x${string}`;
    const totalFeeWei = parseEther(totalFee.toFixed(18));
    writeContract({
      address: modelStoreAddress,
      abi: HELIX_MODEL_STORE_ABI,
      functionName: 'payForInference',
      args: [BigInt(selectedModel.tokenId)],
      value: totalFeeWei,
    });
  }, [selectedModel, chainId, totalFee, writeContract]);

  // When payment confirms, trigger inference automatically
  useEffect(() => {
    if (paymentConfirmed && inferenceTxHash && phase === 'idle') {
      runInference(inferenceTxHash);
    }
  }, [paymentConfirmed, inferenceTxHash, phase, runInference]);

  const hasDrawing = pixels.length > 0 && pixels.some((p) => p > 0.01);
  const isRunning = phase === 'submitting';
  const weightsReady = weightFetchStatus === 'done';
  const canRunInference = isOwnerOfSelected
    ? weightsReady && hasDrawing && !isRunning && !!activeSessionId
    : backendWeightsReady && hasDrawing && !isRunning && !isPaymentPending && !isPaymentConfirming;

  // Select model handler
  const handleSelectModel = useCallback((model: PublicModel) => {
    fetchKeyRef.current = null;
    setSelectedModel(model);
    // Default to latest version
    setSelectedVersionIndex(model.versions.length > 0 ? model.versions.length - 1 : 0);
    setResult(null);
    setPhase('idle');
    setError(null);
    setWeightFetchStatus('idle');
    setWeightFetchError(null);
    setActiveSessionId(null);
    setBackendWeightsReady(false);
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-4xl font-semibold tracking-tight text-white">Inference</h1>
          <p className="text-base text-helix-muted mt-1">
            MNIST 784 → 128 → 10 · ~102K params · Secure MPC
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="default">MNIST</Badge>
          <Badge variant="default" className="text-green-400">
            <Shield size={10} /> MPC
          </Badge>
        </div>
      </div>

      {/* Two-column grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* ── LEFT COLUMN: Model Discovery ──────────────────────── */}
        <div className="flex flex-col gap-5">

          {/* Search bar */}
          <div className="relative">
            <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-helix-muted pointer-events-none" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search models..."
              className="w-full pl-11 pr-4 py-3.5 bg-helix-surface border border-helix-border rounded-2xl text-base text-white placeholder:text-helix-dim focus:outline-none focus:border-helix-border2 transition-colors"
            />
            {search && (
              <button
                type="button"
                onClick={() => setSearch('')}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-helix-muted hover:text-white"
              >
                <X size={14} />
              </button>
            )}
          </div>

          {/* Filter tabs */}
          <div className="grid grid-cols-3 gap-2">
            {(['all', 'mine', 'others'] as const).map((f) => (
              <button
                key={f}
                type="button"
                onClick={() => setFilter(f)}
                className={cn(
                  'py-2.5 rounded-xl text-base font-medium transition-all',
                  filter === f
                    ? 'bg-white text-black shadow-lg shadow-white/5'
                    : 'bg-helix-surface border border-helix-border text-helix-muted hover:text-white hover:border-helix-border2',
                )}
              >
                {f === 'all' ? 'All' : f === 'mine' ? 'Mine' : 'Others'}
              </button>
            ))}
          </div>

          {/* Model cards */}
          <div className="space-y-2 max-h-[420px] overflow-y-auto pr-1">
            {isLoadingPublic ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 size={20} className="animate-spin text-helix-muted" />
              </div>
            ) : filteredModels.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 rounded-2xl bg-helix-surface border border-helix-border">
                <Search size={20} className="text-helix-dim mb-2" />
                <p className="text-base text-helix-text2">
                  {filter === 'mine' && !address ? 'Connect wallet to see your models' : 'No models found'}
                </p>
                {search && (
                  <p className="text-sm text-helix-muted mt-1">Try a different search term</p>
                )}
              </div>
            ) : (
              filteredModels.map((model) => {
                const isSelected = selectedModel?.tokenId === model.tokenId;
                const lv = model.latestVersion;
                return (
                  <button
                    key={model.tokenId}
                    type="button"
                    onClick={() => handleSelectModel(model)}
                    className={cn(
                      'w-full text-left px-4 py-3.5 rounded-2xl border transition-all',
                      isSelected
                        ? 'bg-white/[0.06] border-white/30'
                        : 'bg-helix-surface border-helix-border hover:border-helix-border2',
                    )}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="text-base font-medium text-white truncate">{model.name}</span>
                        {lv?.weightsStored && (
                          <HardDrive size={12} className="text-green-400 shrink-0" />
                        )}
                      </div>
                      {model.bestAccuracy > 0 && (
                        <span className="text-sm font-mono text-green-400 shrink-0">
                          {(model.bestAccuracy * 100).toFixed(1)}%
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-3 text-sm text-helix-muted">
                      {model.inferenceFee > 0 && (
                        <span className="flex items-center gap-1">
                          <Coins size={10} />
                          {(model.inferenceFee / 100).toFixed(1)}% fee
                        </span>
                      )}
                      {lv && (
                        <span className="font-mono">v{lv.semver}</span>
                      )}
                      <span className="font-mono truncate ml-auto">
                        {model.owner.slice(0, 6)}...{model.owner.slice(-4)}
                      </span>
                    </div>
                  </button>
                );
              })
            )}
          </div>

          {/* Divider */}
          <div className="flex items-center gap-3 py-1">
            <div className="flex-1 border-t border-helix-border" />
            <span className="text-sm text-helix-dim">or</span>
            <div className="flex-1 border-t border-helix-border" />
          </div>

          {/* Manual weight upload */}
          <div
            className={cn(
              'flex items-center justify-between px-5 py-3.5 rounded-2xl border transition-colors',
              weightFetchStatus === 'done' && !selectedModel
                ? 'bg-green-500/[0.04] border-green-500/15'
                : 'bg-helix-surface border-helix-border',
            )}
          >
            <div className="flex items-center gap-3 min-w-0">
              {weightFetchStatus === 'done' && !selectedModel ? (
                <CheckCircle size={16} className="text-green-400 shrink-0" />
              ) : (
                <Upload size={16} className="text-helix-muted shrink-0" />
              )}
              <div className="min-w-0">
                <p className={cn(
                  'text-base truncate',
                  weightFetchStatus === 'done' && !selectedModel ? 'text-green-300' : 'text-helix-text2',
                )}>
                  Upload Weights
                  <span className="text-helix-dim ml-1.5">.json</span>
                </p>
              </div>
            </div>
            <label className="shrink-0 px-3.5 py-1.5 rounded-xl bg-white/[0.06] text-xs text-helix-text2 hover:text-white hover:bg-white/[0.1] transition-colors cursor-pointer">
              Upload
              <input
                ref={weightsFileRef}
                type="file"
                accept=".json"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) {
                    setSelectedModel(null);
                    fetchKeyRef.current = null;
                    handleManualWeightUpload(file);
                  }
                }}
              />
            </label>
          </div>

          {/* Weight fetch status */}
          <AnimatePresence mode="wait">
            {weightFetchStatus === 'fetching' && (
              <motion.div
                key="fetching"
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                className="flex items-center gap-3 px-4 py-3 rounded-2xl bg-white/[0.03] border border-white/[0.06]"
              >
                <Loader2 size={14} className="animate-spin text-white" />
                <span className="text-sm text-helix-text2">Fetching weights from 0G...</span>
              </motion.div>
            )}
            {weightFetchStatus === 'decrypting' && (
              <motion.div
                key="decrypting"
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                className="flex items-center gap-3 px-4 py-3 rounded-2xl bg-white/[0.03] border border-white/[0.06]"
              >
                <Loader2 size={14} className="animate-spin text-white" />
                <span className="text-sm text-helix-text2">Decrypting weights (sign wallet prompt)...</span>
              </motion.div>
            )}
            {weightFetchStatus === 'uploading' && (
              <motion.div
                key="uploading"
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                className="flex items-center gap-3 px-4 py-3 rounded-2xl bg-white/[0.03] border border-white/[0.06]"
              >
                <Loader2 size={14} className="animate-spin text-white" />
                <span className="text-sm text-helix-text2">Uploading to MPC workers...</span>
              </motion.div>
            )}
            {weightFetchStatus === 'done' && selectedModel && (
              <motion.div
                key="done"
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                className="flex items-center gap-3 px-4 py-3 rounded-2xl bg-green-500/[0.06] border border-green-500/20"
              >
                <CheckCircle size={14} className="text-green-400" />
                <span className="text-sm text-green-300">
                  <span className="font-medium">{selectedModel.name}</span>
                  {selectedVersion && <span className="text-green-400/60 ml-1"> {selectedVersion.semver || `v${selectedVersionIndex + 1}`}</span>}
                  <span className="text-green-400/60 ml-1.5">weights loaded</span>
                </span>
              </motion.div>
            )}
            {weightFetchStatus === 'error' && (
              <motion.div
                key="error"
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                className="flex items-center gap-3 px-4 py-3 rounded-2xl bg-red-500/[0.06] border border-red-500/20"
              >
                <AlertTriangle size={14} className="text-red-400" />
                <span className="text-sm text-red-300 truncate">{weightFetchError}</span>
              </motion.div>
            )}
            {selectedModel && selectedVersion && !selectedVersion.weightsStored && weightFetchStatus === 'idle' && (
              <motion.div
                key="no-weights"
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                className="flex items-center gap-3 px-4 py-3 rounded-2xl bg-yellow-500/[0.06] border border-yellow-500/20"
              >
                <AlertTriangle size={14} className="text-yellow-400" />
                <span className="text-sm text-yellow-300/80">No stored weights — upload manually below</span>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Version selector (only show when model has multiple versions) */}
          {selectedModel && selectedModel.versions.length > 1 && (
            <div className="rounded-2xl bg-helix-surface border border-helix-border p-4">
              <label className="text-xs text-helix-muted block mb-2">Version</label>
              <select
                value={selectedVersionIndex}
                onChange={(e) => {
                  const idx = Number(e.target.value);
                  setSelectedVersionIndex(idx);
                  // Reset weight fetch so it re-triggers for the new version
                  fetchKeyRef.current = null;
                  setWeightFetchStatus('idle');
                  setWeightFetchError(null);
                  setBackendWeightsReady(false);
                  setActiveSessionId(null);
                }}
                className="w-full px-3 py-2.5 bg-helix-bg border border-helix-border rounded-xl text-sm text-white focus:outline-none focus:border-helix-border2 transition-colors"
              >
                {selectedModel.versions.map((v, i) => (
                  <option key={i} value={i}>
                    {v.semver || `v${i + 1}`} — {(v.accuracy * 100).toFixed(1)}% accuracy
                    {i === selectedModel.versions.length - 1 ? ' (latest)' : ''}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        {/* ── RIGHT COLUMN: Fee, Input, Results ────────────────── */}
        <div className="flex flex-col gap-5">

          {/* Fee / Payment + Run button */}
          <div className="rounded-2xl bg-helix-surface border border-helix-border overflow-hidden">
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <span className="text-base font-medium text-helix-text2">Inference Fee</span>
                {isOwnerOfSelected ? (
                  <Badge variant="default" className="text-green-400">
                    <CheckCircle size={10} /> Free (you own this model)
                  </Badge>
                ) : selectedModel && ownerFeeBps > 0 ? (
                  <Badge variant="default" className="text-helix-text2">
                    <Coins size={10} />
                    {(ownerFeeBps / 100).toFixed(1)}% owner fee
                  </Badge>
                ) : null}
              </div>

              {/* Big fee number */}
              <div className="flex items-baseline justify-center gap-4 py-2">
                <span className="text-5xl font-bold tracking-tighter tabular-nums text-white">
                  {isOwnerOfSelected ? '0.000000' : totalFee.toFixed(6)}
                </span>
                <span className="text-xl font-semibold text-helix-text2">ADI</span>
              </div>

              {/* Fee breakdown */}
              <div className="flex items-center justify-center gap-4 mt-3 text-sm text-helix-dim">
                {isOwnerOfSelected ? (
                  <span>Owner inference is always free</span>
                ) : (
                  <>
                    <span>Workers: {BASE_INFERENCE_COST.toFixed(4)} ADI</span>
                    {ownerFeeBps > 0 && (
                      <>
                        <span>+</span>
                        <span>Owner: {(ownerFeeBps / 100).toFixed(1)}%</span>
                      </>
                    )}
                  </>
                )}
              </div>
            </div>

            {/* Run button */}
            <motion.button
              type="button"
              onClick={isOwnerOfSelected ? () => runInference() : handlePayAndRun}
              disabled={!canRunInference}
              whileTap={canRunInference ? { scale: 0.98 } : {}}
              className={cn(
                'w-full py-5 text-xl font-bold tracking-tight transition-all',
                'flex items-center justify-center gap-3 border-t',
                canRunInference
                  ? 'bg-zinc-200 text-black border-zinc-200 hover:bg-zinc-300 active:bg-zinc-400'
                  : 'bg-helix-border text-helix-muted border-helix-border cursor-not-allowed',
              )}
            >
              {isPaymentPending ? (
                <><Loader2 size={22} className="animate-spin" /> Confirm in Wallet...</>
              ) : isPaymentConfirming ? (
                <><Loader2 size={22} className="animate-spin" /> Confirming Payment...</>
              ) : isRunning ? (
                <><Loader2 size={22} className="animate-spin" /> Running MPC Inference...</>
              ) : isOwnerOfSelected ? (
                <><Shield size={22} /> Run Inference</>
              ) : (
                <><Coins size={22} /> Pay &amp; Run Inference</>
              )}
            </motion.button>
          </div>

          {/* Input card */}
          <div className="rounded-2xl bg-helix-surface border border-helix-border overflow-hidden p-5">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-1 p-0.5 rounded-xl bg-helix-bg border border-helix-border">
                <button
                  type="button"
                  onClick={() => { setInputMode('draw'); clearCanvas(); }}
                  className={cn(
                    'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all',
                    inputMode === 'draw'
                      ? 'bg-white/[0.08] text-white'
                      : 'text-helix-muted hover:text-helix-text',
                  )}
                >
                  <Pencil size={12} /> Draw
                </button>
                <button
                  type="button"
                  onClick={() => { setInputMode('upload'); clearCanvas(); }}
                  className={cn(
                    'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all',
                    inputMode === 'upload'
                      ? 'bg-white/[0.08] text-white'
                      : 'text-helix-muted hover:text-helix-text',
                  )}
                >
                  <Upload size={12} /> Upload
                </button>
              </div>
              {inputMode === 'draw' && (
                <button
                  type="button"
                  onClick={clearCanvas}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm text-helix-muted bg-helix-bg border border-helix-border hover:text-white hover:border-helix-border2 transition-colors"
                >
                  <Eraser size={12} /> Clear
                </button>
              )}
            </div>

            <div className="flex justify-center mb-3">
              {inputMode === 'draw' ? (
                <div className="relative">
                  <DrawingCanvas canvasRef={canvasRef} onPixelsReady={setPixels} />
                  <div
                    className="absolute inset-0 pointer-events-none rounded-xl opacity-[0.03]"
                    style={{
                      backgroundImage: `linear-gradient(rgba(255,255,255,0.5) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.5) 1px, transparent 1px)`,
                      backgroundSize: `${CANVAS_SIZE / GRID_SIZE}px ${CANVAS_SIZE / GRID_SIZE}px`,
                    }}
                  />
                </div>
              ) : (
                <ImageUpload canvasRef={canvasRef} onPixelsReady={setPixels} />
              )}
            </div>

            <p className="text-sm text-helix-dim text-center">
              {inputMode === 'draw' ? 'Draw a digit (0-9)' : 'Upload an image of a handwritten digit'}
            </p>
          </div>

          {/* Results */}
          <div className="space-y-4">
            {isRunning && (
              <div className="rounded-2xl bg-helix-surface border border-helix-border p-5">
                <div className="flex items-center gap-3">
                  <Loader2 size={16} className="animate-spin text-white" />
                  <div>
                    <p className="text-base text-white">Running distributed MPC inference...</p>
                    <p className="text-sm text-helix-muted mt-0.5">
                      Secret-sharing across 3 workers, running forward pass via transport
                    </p>
                  </div>
                </div>
              </div>
            )}

            {error && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="bg-red-500/10 border border-red-500/30 rounded-2xl px-5 py-4"
              >
                <div className="flex items-start gap-2">
                  <AlertTriangle size={14} className="text-red-400 shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm text-red-300">{error}</p>
                    <button
                      type="button"
                      onClick={() => { setError(null); setPhase('idle'); }}
                      className="text-xs text-red-400 hover:text-red-300 mt-1 underline"
                    >
                      Dismiss
                    </button>
                  </div>
                </div>
              </motion.div>
            )}

            {phase === 'done' && result && (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <MPCResultCard result={result} />
              </motion.div>
            )}

            {!isRunning && !error && phase === 'idle' && (
              <div className="rounded-2xl bg-helix-surface border border-helix-border p-5">
                <div className="flex flex-col items-center justify-center py-8">
                  <Shield size={24} className="text-helix-dim mb-3" />
                  <p className="text-base text-helix-text2">
                    {!weightsReady
                      ? 'Select a model to get started'
                      : !hasDrawing
                        ? 'Draw a digit and click Run Inference'
                        : 'Ready to classify'}
                  </p>
                  <p className="text-sm text-helix-muted mt-1">
                    Secure multi-party computation across independent workers
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
}

// ============================================================================
// Page (Suspense wrapper)
// ============================================================================

export default function InferencePage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center py-20">
          <Loader2 size={24} className="animate-spin text-helix-muted" />
        </div>
      }
    >
      <InferencePageInner />
    </Suspense>
  );
}
