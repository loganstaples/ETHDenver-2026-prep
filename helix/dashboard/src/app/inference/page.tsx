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
  X,
  Coins,
  History,
  Star,
} from 'lucide-react';
import { useAccount, useSignMessage, useWriteContract, useWaitForTransactionReceipt, useChainId } from 'wagmi';
import { parseEther } from 'viem';
import { Badge } from '@/components/ui/Badge';
import { ModelCard, TrainedSessionCard } from '@/components/ModelCard';
import { cn } from '@/lib/utils';
import { usePublicModels, type PublicModel } from '@/hooks/usePublicModels';
import { useModelRegistry } from '@/hooks/useModelRegistry';
import { deriveModelKey, decryptWeights } from '@/lib/model-encryption';
import { HELIX_MODEL_STORE_ABI, getContractAddress } from '@/lib/contracts';
import { getTrustedNodes, checkTrustedWorkersActive } from '@/hooks/useTrustedNodes';

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
const BRUSH_RADIUS = 10;
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

    // Get the raw canvas image data at full resolution
    const fullData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    // Find bounding box of the drawn content
    let minX = CANVAS_SIZE, minY = CANVAS_SIZE, maxX = 0, maxY = 0;
    for (let y = 0; y < CANVAS_SIZE; y++) {
      for (let x = 0; x < CANVAS_SIZE; x++) {
        if (fullData.data[(y * CANVAS_SIZE + x) * 4] > 10) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    if (maxX <= minX || maxY <= minY) {
      onPixelsReady([]);
      return;
    }

    // Step 1: Scale digit to fit in 20x20, preserving aspect ratio
    const bw = maxX - minX + 1;
    const bh = maxY - minY + 1;
    const targetSize = 20;
    const scale = targetSize / Math.max(bw, bh);
    const scaledW = Math.round(bw * scale);
    const scaledH = Math.round(bh * scale);

    // First render the scaled digit to a temp canvas (top-left)
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = scaledW;
    tmpCanvas.height = scaledH;
    const tmpCtx = tmpCanvas.getContext('2d')!;
    tmpCtx.fillStyle = '#000000';
    tmpCtx.fillRect(0, 0, scaledW, scaledH);
    tmpCtx.imageSmoothingEnabled = true;
    tmpCtx.imageSmoothingQuality = 'high';
    tmpCtx.drawImage(canvas, minX, minY, bw, bh, 0, 0, scaledW, scaledH);

    // Step 2: Compute center of mass of the scaled digit
    const tmpData = tmpCtx.getImageData(0, 0, scaledW, scaledH);
    let massX = 0, massY = 0, totalMass = 0;
    for (let y = 0; y < scaledH; y++) {
      for (let x = 0; x < scaledW; x++) {
        const val = tmpData.data[(y * scaledW + x) * 4];
        massX += x * val;
        massY += y * val;
        totalMass += val;
      }
    }

    // Step 3: Place digit so center of mass lands at (14, 14) — center of 28x28
    // This matches MNIST's preprocessing exactly
    const comX = totalMass > 0 ? massX / totalMass : scaledW / 2;
    const comY = totalMass > 0 ? massY / totalMass : scaledH / 2;
    const offsetX = Math.round(14 - comX);
    const offsetY = Math.round(14 - comY);

    // Step 4: Draw into final 28x28 canvas
    const outCanvas = document.createElement('canvas');
    outCanvas.width = GRID_SIZE;
    outCanvas.height = GRID_SIZE;
    const outCtx = outCanvas.getContext('2d')!;
    outCtx.fillStyle = '#000000';
    outCtx.fillRect(0, 0, GRID_SIZE, GRID_SIZE);
    outCtx.drawImage(tmpCanvas, offsetX, offsetY);

    const imageData = outCtx.getImageData(0, 0, GRID_SIZE, GRID_SIZE);
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
          const imgScale = Math.min(CANVAS_SIZE / img.width, CANVAS_SIZE / img.height);
          const w = img.width * imgScale;
          const h = img.height * imgScale;
          ctx.drawImage(img, (CANVAS_SIZE - w) / 2, (CANVAS_SIZE - h) / 2, w, h);

          // Convert to grayscale and find bounding box
          const fullData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
          let minX = CANVAS_SIZE, minY = CANVAS_SIZE, maxX = 0, maxY = 0;
          for (let y = 0; y < CANVAS_SIZE; y++) {
            for (let x = 0; x < CANVAS_SIZE; x++) {
              const idx = (y * CANVAS_SIZE + x) * 4;
              const gray = 0.299 * fullData.data[idx] + 0.587 * fullData.data[idx + 1] + 0.114 * fullData.data[idx + 2];
              if (gray > 10) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
              }
            }
          }

          let imageData: ImageData;
          if (maxX > minX && maxY > minY) {
            // Step 1: Scale digit to fit in 20x20, preserving aspect ratio
            const bw = maxX - minX + 1;
            const bh = maxY - minY + 1;
            const targetSize = 20;
            const fitScale = targetSize / Math.max(bw, bh);
            const scaledW = Math.round(bw * fitScale);
            const scaledH = Math.round(bh * fitScale);

            // Render scaled digit to temp canvas
            const tmpCanvas = document.createElement('canvas');
            tmpCanvas.width = scaledW;
            tmpCanvas.height = scaledH;
            const tmpCtx = tmpCanvas.getContext('2d')!;
            tmpCtx.fillStyle = '#000000';
            tmpCtx.fillRect(0, 0, scaledW, scaledH);
            tmpCtx.imageSmoothingEnabled = true;
            tmpCtx.imageSmoothingQuality = 'high';
            tmpCtx.drawImage(canvas, minX, minY, bw, bh, 0, 0, scaledW, scaledH);

            // Step 2: Compute center of mass
            const tmpData = tmpCtx.getImageData(0, 0, scaledW, scaledH);
            let massX = 0, massY = 0, totalMass = 0;
            for (let py = 0; py < scaledH; py++) {
              for (let px = 0; px < scaledW; px++) {
                const idx = (py * scaledW + px) * 4;
                const gray = 0.299 * tmpData.data[idx] + 0.587 * tmpData.data[idx + 1] + 0.114 * tmpData.data[idx + 2];
                massX += px * gray;
                massY += py * gray;
                totalMass += gray;
              }
            }

            // Step 3: Place so center of mass is at (14, 14)
            const comX = totalMass > 0 ? massX / totalMass : scaledW / 2;
            const comY = totalMass > 0 ? massY / totalMass : scaledH / 2;
            const offsetX = Math.round(14 - comX);
            const offsetY = Math.round(14 - comY);

            // Step 4: Draw into final 28x28
            const outCanvas = document.createElement('canvas');
            outCanvas.width = GRID_SIZE;
            outCanvas.height = GRID_SIZE;
            const outCtx = outCanvas.getContext('2d')!;
            outCtx.fillStyle = '#000000';
            outCtx.fillRect(0, 0, GRID_SIZE, GRID_SIZE);
            outCtx.drawImage(tmpCanvas, offsetX, offsetY);

            imageData = outCtx.getImageData(0, 0, GRID_SIZE, GRID_SIZE);
          } else {
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = GRID_SIZE;
            tempCanvas.height = GRID_SIZE;
            const tempCtx = tempCanvas.getContext('2d')!;
            tempCtx.fillStyle = '#000000';
            tempCtx.fillRect(0, 0, GRID_SIZE, GRID_SIZE);
            tempCtx.drawImage(canvas, 0, 0, GRID_SIZE, GRID_SIZE);
            imageData = tempCtx.getImageData(0, 0, GRID_SIZE, GRID_SIZE);
          }

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
          <p className="text-sm text-helix-muted">PNG, JPG, or any image of a digit</p>
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
            className="text-sm text-helix-muted hover:text-white ml-auto"
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
          <span className={cn('w-14 text-right text-sm font-mono', digit === prediction ? 'text-white' : 'text-helix-muted')}>
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
          <span className="text-sm text-helix-muted font-mono">{p.ms}ms</span>
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
            <p className="text-sm text-helix-muted mt-0.5">confidence</p>
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
              <span className="text-sm font-medium text-white">On-chain Attestation</span>
            </div>
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <span className="text-sm text-helix-muted w-20">Input hash</span>
                <span className="text-xs text-helix-text font-mono truncate">
                  {result.attestation.input_hash.slice(0, 16)}...
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm text-helix-muted w-20">Output hash</span>
                <span className="text-xs text-helix-text font-mono truncate">
                  {result.attestation.output_hash.slice(0, 16)}...
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm text-helix-muted w-20">Signatures</span>
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
                  <span className="text-sm text-helix-muted w-20">On-chain TX</span>
                  <span className="text-xs text-green-400 font-mono truncate">
                    {result.attestation.chain_tx_hash.slice(0, 18)}...
                  </span>
                  {result.attestation.inference_id != null && (
                    <span className="text-sm text-helix-muted">(ID: {result.attestation.inference_id})</span>
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
            className="text-sm text-helix-muted hover:text-white transition-colors flex items-center gap-1"
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
// Inference History Types & View
// ============================================================================

interface InferenceHistoryEntry {
  id: string;
  model: string;
  status: string;
  progress: number;
  phase: string;
  workers: number;
  created: number;
  inputLabel: string;
  result: string;
  confidence: number;
  duration: number;
}

function InferenceHistoryView() {
  const [entries, setEntries] = useState<InferenceHistoryEntry[]>([]);
  const [historySearch, setHistorySearch] = useState('');
  const [selectedEntry, setSelectedEntry] = useState<InferenceHistoryEntry | null>(null);

  useEffect(() => {
    try {
      const raw = localStorage.getItem('helix-inference-history');
      setEntries(raw ? JSON.parse(raw) : []);
    } catch {
      setEntries([]);
    }
  }, []);

  const filteredEntries = useMemo(() => {
    if (!historySearch.trim()) return entries;
    const q = historySearch.toLowerCase();
    return entries.filter((e) =>
      e.model.toLowerCase().includes(q) ||
      e.result.includes(q),
    );
  }, [entries, historySearch]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

      {/* ── LEFT COLUMN: Inference List ──────────────────────── */}
      <div className="flex flex-col gap-5">

        {/* Search bar */}
        <div className="relative">
          <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-helix-muted pointer-events-none" />
          <input
            type="text"
            value={historySearch}
            onChange={(e) => setHistorySearch(e.target.value)}
            placeholder="Search inferences..."
            className="w-full pl-11 pr-4 py-3.5 bg-helix-surface border border-helix-border rounded-2xl text-base text-white placeholder:text-helix-dim focus:outline-none focus:border-helix-border2 transition-colors"
          />
          {historySearch && (
            <button
              type="button"
              onClick={() => setHistorySearch('')}
              className="absolute right-4 top-1/2 -translate-y-1/2 text-helix-muted hover:text-white"
            >
              <X size={14} />
            </button>
          )}
        </div>

        {/* Entry list */}
        <div className="space-y-2 max-h-[520px] overflow-y-auto pr-1">
          {entries.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 rounded-2xl bg-helix-surface border border-helix-border">
              <History size={20} className="text-helix-dim mb-2" />
              <p className="text-sm text-helix-text2">No inference history yet</p>
              <p className="text-sm text-helix-muted mt-1">Run an inference to see it here</p>
            </div>
          ) : filteredEntries.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 rounded-2xl bg-helix-surface border border-helix-border">
              <Search size={20} className="text-helix-dim mb-2" />
              <p className="text-sm text-helix-text2">No inferences match your search</p>
            </div>
          ) : (
            filteredEntries.map((entry) => {
              const isSelected = selectedEntry?.id === entry.id;
              const date = new Date(entry.created);

              return (
                <button
                  key={entry.id}
                  type="button"
                  onClick={() => setSelectedEntry(isSelected ? null : entry)}
                  className={cn(
                    'w-full text-left rounded-2xl transition-all',
                    isSelected
                      ? 'bg-white/[0.07] ring-1 ring-white/20 px-6 py-5'
                      : 'bg-helix-surface border border-helix-border hover:border-helix-border2 px-5 py-4',
                  )}
                >
                  <div className="flex items-baseline justify-between gap-4">
                    <h3 className={cn(
                      'font-semibold text-white truncate tracking-tight',
                      isSelected ? 'text-xl' : 'text-[15px]',
                    )}>
                      {entry.model}
                    </h3>
                    <span className={cn(
                      'font-mono tabular-nums shrink-0',
                      isSelected ? 'text-xl font-semibold text-white' : 'text-sm font-medium text-white/60',
                    )}>
                      Predicted: {entry.result}
                    </span>
                  </div>

                  {/* Expanded details when selected */}
                  {isSelected && (
                    <div className="mt-3 flex items-center gap-3 flex-wrap">
                      <span className="inline-flex items-center gap-1.5 text-sm font-medium px-2.5 py-1 rounded-full bg-green-500/10 text-green-400/90">
                        <span className="w-1.5 h-1.5 rounded-full bg-green-400" />
                        {entry.workers} Worker{entry.workers !== 1 ? 's' : ''}
                      </span>
                      <span className="text-sm text-helix-dim">
                        {date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                        {' '}
                        {date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                      </span>
                      {entry.duration > 0 && (
                        <span className="text-sm text-helix-dim font-mono">{entry.duration}ms</span>
                      )}
                    </div>
                  )}

                  {/* Compact info when not selected */}
                  {!isSelected && (
                    <div className="flex items-center gap-3 mt-1.5">
                      <span className="text-sm text-helix-dim tabular-nums">
                        {date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                      </span>
                      <span className="text-sm text-helix-dim tabular-nums ml-auto">
                        {(entry.confidence * 100).toFixed(1)}% conf
                      </span>
                    </div>
                  )}
                </button>
              );
            })
          )}
        </div>
      </div>

      {/* ── RIGHT COLUMN: Entry Details ────────────────────────── */}
      <div className="flex flex-col gap-5">
        {selectedEntry ? (
          <InferenceHistoryDetailPanel entry={selectedEntry} />
        ) : (
          <div className="rounded-2xl bg-helix-surface border border-helix-border p-5">
            <div className="flex flex-col items-center justify-center py-16">
              <History size={24} className="text-helix-dim mb-3" />
              <p className="text-base text-helix-text2">Select an inference to view details</p>
              <p className="text-sm text-helix-muted mt-1">
                {entries.length} inference{entries.length !== 1 ? 's' : ''} recorded
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Inference History Detail Panel
// ============================================================================

function InferenceHistoryDetailPanel({ entry }: { entry: InferenceHistoryEntry }) {
  const date = new Date(entry.created);

  return (
    <motion.div
      key={entry.id}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="flex flex-col gap-5"
    >
      {/* Hero result card */}
      <div className="rounded-2xl bg-helix-surface border border-helix-border overflow-hidden">
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Shield size={14} className="text-green-400" />
              <h3 className="text-base font-medium text-white">{entry.model}</h3>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="default" className="text-green-400">
                <Shield size={10} /> MPC
              </Badge>
              <Badge variant="default" className="text-blue-400">
                <Users size={10} /> {entry.workers} worker{entry.workers !== 1 ? 's' : ''}
              </Badge>
              {entry.duration > 0 && (
                <Badge variant="default" className="text-helix-text2">
                  <Clock size={10} /> {entry.duration}ms
                </Badge>
              )}
            </div>
          </div>

          {/* Big prediction */}
          <div className="flex items-center gap-6 mb-2">
            <div className="w-20 h-20 rounded-xl bg-white/[0.06] flex items-center justify-center">
              <span className="text-4xl font-light text-white font-mono">{entry.result}</span>
            </div>
            <div>
              <p className="text-2xl font-mono font-light text-white">
                {(entry.confidence * 100).toFixed(1)}%
              </p>
              <p className="text-sm text-helix-muted mt-0.5">confidence</p>
            </div>
            <div className="ml-auto">
              <CheckCircle size={24} className="text-green-400" />
            </div>
          </div>
        </div>
      </div>

      {/* Details card */}
      <div className="p-4 rounded-xl bg-helix-surface border border-helix-border space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-helix-dim">Model</span>
          <span className="text-helix-text2">{entry.model}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-helix-dim">Prediction</span>
          <span className="text-helix-text2 font-mono">{entry.result}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-helix-dim">Confidence</span>
          <span className="text-helix-text2 font-mono">{(entry.confidence * 100).toFixed(1)}%</span>
        </div>
        <div className="flex justify-between">
          <span className="text-helix-dim">Workers</span>
          <span className="text-helix-text2">{entry.workers}</span>
        </div>
        {entry.duration > 0 && (
          <div className="flex justify-between">
            <span className="text-helix-dim">Duration</span>
            <span className="text-helix-text2 font-mono">{entry.duration}ms</span>
          </div>
        )}
        <div className="flex justify-between">
          <span className="text-helix-dim">Timestamp</span>
          <span className="text-helix-text2">
            {date.toLocaleString()}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-helix-dim">ID</span>
          <span className="text-helix-text2 font-mono">{entry.id}</span>
        </div>
      </div>
    </motion.div>
  );
}

// ============================================================================
// Inference Rating Prompt
// ============================================================================

function InferenceRatingPrompt({
  tokenId,
  walletAddress,
  onRated,
}: {
  tokenId: number;
  walletAddress: string;
  onRated: () => void;
}) {
  const [hoverRating, setHoverRating] = useState(0);
  const [selectedRating, setSelectedRating] = useState(0);
  const [avgRating, setAvgRating] = useState(0);
  const [count, setCount] = useState(0);
  const [submitted, setSubmitted] = useState(false);

  useEffect(() => {
    fetch(`/api/models/${tokenId}/rate?wallet=${encodeURIComponent(walletAddress)}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) {
          setAvgRating(data.averageRating ?? 0);
          setCount(data.ratingCount ?? 0);
          if (data.userRating) setSelectedRating(data.userRating);
        }
      })
      .catch(() => {});
  }, [tokenId, walletAddress]);

  const handleRate = async (rating: number) => {
    setSelectedRating(rating);
    try {
      const res = await fetch(`/api/models/${tokenId}/rate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ wallet: walletAddress, rating }),
      });
      if (res.ok) {
        const data = await res.json();
        setAvgRating(data.averageRating ?? avgRating);
        setCount(data.ratingCount ?? count);
        setSubmitted(true);
        onRated();
      }
    } catch { /* best-effort */ }
  };

  const displayRating = hoverRating || selectedRating;

  return (
    <div className="rounded-2xl bg-helix-surface border border-helix-border p-5">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-base font-semibold text-white">Rate this model</h3>
        {count > 0 && (
          <span className="text-sm text-helix-muted font-mono tabular-nums">
            {avgRating.toFixed(1)} avg · {count} rating{count !== 1 ? 's' : ''}
          </span>
        )}
      </div>
      <div className="flex items-center gap-0.5">
        {[1, 2, 3, 4, 5].map((star) => (
          <button
            key={star}
            type="button"
            onMouseEnter={() => setHoverRating(star)}
            onMouseLeave={() => setHoverRating(0)}
            onClick={() => handleRate(star)}
            className="p-1 transition-transform hover:scale-110"
          >
            <Star
              size={24}
              className={cn(
                'transition-colors',
                star <= displayRating
                  ? 'text-yellow-400 fill-yellow-400'
                  : 'text-white/[0.08]',
              )}
            />
          </button>
        ))}
        {submitted && (
          <span className="text-sm text-green-400 ml-3">Rated!</span>
        )}
        {selectedRating > 0 && !submitted && (
          <span className="text-sm text-helix-dim ml-3">Your rating: {selectedRating}/5</span>
        )}
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
  const { allModels, isLoading: isLoadingPublic, refetch: refetchPublicModels } = usePublicModels();
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

  // Non-owner public inference state
  const [inferenceReady, setInferenceReady] = useState(false);
  const [inferenceSessionId, setInferenceSessionId] = useState<string | null>(null);

  // Inference payment (non-owner flow)
  const { writeContract, data: inferenceTxHash, isPending: isPaymentPending } = useWriteContract();
  const { isSuccess: paymentConfirmed, isLoading: isPaymentConfirming } = useWaitForTransactionReceipt({ hash: inferenceTxHash });

  // Inference
  const [pixels, setPixels] = useState<number[]>([]);
  const [result, setResult] = useState<MPCInferenceResult | null>(null);
  const [phase, setPhase] = useState<InferencePhase>('idle');
  const [error, setError] = useState<string | null>(null);
  const [inputMode, setInputMode] = useState<InputMode>('draw');
  const [viewMode, setViewMode] = useState<'live' | 'history'>('live');

  // Trained sessions from backend
  const [trainedSessions, setTrainedSessions] = useState<{ session_id: string; model_name?: string; model_slug?: string; accuracy: number | null; status: string; losses: number[] }[]>([]);
  const [isLoadingTrained, setIsLoadingTrained] = useState(true);

  useEffect(() => {
    const fetchSessions = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/training/sessions`);
        if (!res.ok) return;
        const sessions = await res.json();
        const completed = (Array.isArray(sessions) ? sessions : [])
          .filter((s: { status: string }) => s.status === 'complete');
        setTrainedSessions(completed);
      } catch { /* non-fatal */ }
      finally { setIsLoadingTrained(false); }
    };
    fetchSessions();
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

  // Filtered trained sessions (apply search + filter)
  const filteredSessions = useMemo(() => {
    let sessions = trainedSessions;
    if (filter === 'others') return [];
    if (search.trim()) {
      const q = search.toLowerCase();
      sessions = sessions.filter((s) =>
        (s.model_name ?? '').toLowerCase().includes(q),
      );
    }
    return sessions;
  }, [trainedSessions, filter, search]);

  // Whether a model is currently selected
  const hasSelection = selectedModel !== null;

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

  // Check if a public model has inference enabled (non-owner flow)
  const checkInferenceReady = useCallback(async () => {
    if (!selectedModel) return;
    try {
      const res = await fetch(
        `${API_BASE}/api/models/${selectedModel.tokenId}/inference-ready?version=${selectedVersionIndex}`
      );
      if (!res.ok) {
        setInferenceReady(false);
        setInferenceSessionId(null);
        return;
      }
      const data = await res.json();
      setInferenceReady(data.ready);
      if (data.sessionId) setInferenceSessionId(data.sessionId);
      else setInferenceSessionId(null);
    } catch {
      setInferenceReady(false);
      setInferenceSessionId(null);
    }
  }, [selectedModel, selectedVersionIndex]);

  // Non-owner public inference: submit input to server-side forward pass (no MPC, no payment)
  const handlePublicInference = useCallback(async (pixelData: number[]) => {
    if (!selectedModel) return;
    setPhase('submitting');
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`/api/models/${selectedModel.tokenId}/infer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input: pixelData,
          version: selectedVersionIndex,
        }),
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.error || 'Inference failed');
      }
      const data = await res.json();
      // Map the public inference response into the MPCInferenceResult shape
      const mappedResult: MPCInferenceResult = {
        prediction: data.prediction,
        confidence: data.confidence,
        probabilities: data.probabilities,
        num_parties: 1,
        distributed: false,
        timing: {
          share_generation_ms: 0,
          forward_pass_ms: 0,
          signing_ms: 0,
          total_ms: 0,
        },
      };
      setResult(mappedResult);
      setPhase('done');

      // Persist to localStorage for Dashboard inference tab
      try {
        const entry = {
          id: `inf-${Date.now()}`,
          model: selectedModel.name,
          status: 'completed' as const,
          progress: 100,
          phase: 'done',
          workers: 1,
          created: Date.now(),
          inputLabel: 'Digit prediction',
          result: String(data.prediction),
          confidence: data.confidence,
          duration: 0,
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
  }, [selectedModel, selectedVersionIndex]);

  // Fetch weights from 0G (owner) or check backend cache (non-owner)
  // Re-triggers when model or version selection changes
  const fetchKeyRef = useRef<string | null>(null);
  useEffect(() => {
    if (!selectedModel) {
      fetchKeyRef.current = null;
      setWeightFetchStatus('idle');
      setWeightFetchError(null);
      setBackendWeightsReady(false);
      setInferenceReady(false);
      setInferenceSessionId(null);
      return;
    }

    const versionKey = `${selectedModel.tokenId}:${selectedVersionIndex}`;
    if (fetchKeyRef.current === versionKey) return;
    fetchKeyRef.current = versionKey;

    const isOwner = address && selectedModel.owner.toLowerCase() === address.toLowerCase();

    // For ALL users (owner and non-owner): first check if backend already has cached weights
    const doFetch = async () => {
      // Step 1: Check Rust backend cache (for owner MPC inference)
      try {
        const res = await fetch(`${API_BASE}/api/models/${selectedModel.tokenId}/inference-ready?version=${selectedVersionIndex}`);
        if (res.ok) {
          const data = await res.json();
          if (data.ready) {
            setBackendWeightsReady(true);
            if (data.sessionId) setActiveSessionId(data.sessionId);
            // Owner can use MPC inference directly via Rust backend
            if (isOwner) {
              setInferenceReady(true);
              if (data.sessionId) setInferenceSessionId(data.sessionId);
              setWeightFetchStatus('done');
              return;
            }
          }
        }
      } catch { /* non-fatal */ }

      // Step 2: For non-owner, check Next.js cache (used by public inference JS forward pass)
      if (!isOwner) {
        try {
          const res = await fetch(`/api/models/${selectedModel.tokenId}/inference-ready?version=${selectedVersionIndex}`);
          if (res.ok) {
            const data = await res.json();
            if (data.ready) {
              setInferenceReady(true);
              if (data.sessionId) setInferenceSessionId(data.sessionId);
              setWeightFetchStatus('done');
              return;
            }
          }
        } catch { /* non-fatal */ }
        setInferenceReady(false);
        setInferenceSessionId(null);
        setWeightFetchError('Model owner hasn\u2019t enabled inference yet');
        setWeightFetchStatus('error');
        fetchKeyRef.current = null;
        return;
      }

      // Step 3: Owner — fetch from 0G, decrypt, upload to backend
      const sv = selectedVersion;
      if (!sv || !sv.weightsStored || !sv.rootHash) {
        // No weights on 0G either — still allow model-based inference for owner
        setBackendWeightsReady(false);
        setWeightFetchStatus('done');
        setActiveSessionId(`owner-${selectedModel.tokenId}`);
        return;
      }
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
          await fetch(`${API_BASE}/api/models/${selectedModel.tokenId}/cache-weights?version=${selectedVersionIndex}&owner=${encodeURIComponent(address ?? '')}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(weightsData),
          });
        } catch { /* non-fatal */ }

        // Cache to Next.js for non-owner public inference (JS forward pass)
        try {
          await fetch(`/api/models/${selectedModel.tokenId}/enable-inference`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ weights: weightsData, version: selectedVersionIndex, ownerAddress: address }),
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
    console.log('[Inference] runInference called', {
      pixelsLength: pixels.length,
      hasDrawing: pixels.length > 0 && pixels.some(p => p > 0.01),
      activeSessionId,
      selectedModel: selectedModel?.tokenId ?? null,
      weightFetchStatus,
      backendWeightsReady,
      isOwnerOfSelected,
    });

    if (!pixels.length) {
      console.warn('[Inference] No pixels, aborting');
      setError('No drawing detected — please draw a digit first');
      setPhase('error');
      return;
    }

    // ── Trusted nodes gate ────────────────────────────────────────────
    const trustedNodes = getTrustedNodes();
    if (trustedNodes.length > 0) {
      const check = await checkTrustedWorkersActive(API_BASE, trustedNodes);
      if (!check.ok) {
        setError(check.error ?? 'Trusted node check failed');
        setPhase('error');
        return;
      }
      if (check.warning) {
        console.warn('[Inference] Trusted node check:', check.warning);
      }
    }

    setPhase('submitting');
    setError(null);
    setResult(null);

    try {
      if (!selectedModel) {
        setError('No model selected');
        setPhase('error');
        return;
      }
      const body = {
        model_token_id: selectedModel.tokenId,
        model_version_index: selectedVersionIndex,
        pixels,
        num_parties: 3,
        wallet_address: address || undefined,
        payment_tx: paymentTxHash,
        trusted_nodes: trustedNodes.length > 0 ? trustedNodes : undefined,
      };
      console.log('[Inference] Sending request', { modelId: selectedModel.tokenId, pixelsSample: pixels.slice(0, 5) });

      const res = await fetch('/api/inference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.error || `Inference HTTP ${res.status}`);
      }
      const data: MPCInferenceResult = await res.json();
      console.log('[Inference] Success', { prediction: data.prediction, confidence: data.confidence });
      setResult(data);
      setPhase('done');

      // Persist to localStorage for Dashboard inference tab
      try {
        const entry = {
          id: `inf-${Date.now()}`,
          model: selectedModel?.name ?? 'Unknown Model',
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
  }, [activeSessionId, pixels, selectedModel, selectedVersionIndex, backendWeightsReady, isOwnerOfSelected, address, weightFetchStatus]);

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
  // All inference (owner and non-owner) routes through MPC backend
  const canRunInference = !selectedModel
    ? weightsReady && hasDrawing && !isRunning && !!activeSessionId
    : (weightsReady || backendWeightsReady || inferenceReady) && hasDrawing && !isRunning;

  // Select model handler (toggle: click again to deselect)
  const handleSelectModel = useCallback((model: PublicModel) => {
    if (selectedModel?.tokenId === model.tokenId) {
      // Deselect
      fetchKeyRef.current = null;
      setSelectedModel(null);
      setSelectedVersionIndex(0);
      setWeightFetchStatus('idle');
      setWeightFetchError(null);
      setActiveSessionId(null);
      setBackendWeightsReady(false);
      setInferenceReady(false);
      setInferenceSessionId(null);
      return;
    }
    fetchKeyRef.current = null;
    setSelectedModel(model);
    setSelectedVersionIndex(model.versions.length > 0 ? model.versions.length - 1 : 0);
    setResult(null);
    setPhase('idle');
    setError(null);
    setWeightFetchStatus('idle');
    setWeightFetchError(null);
    setActiveSessionId(null);
    setBackendWeightsReady(false);
    setInferenceReady(false);
    setInferenceSessionId(null);
  }, [selectedModel]);

  // Select trained session handler (toggle: click again to deselect)
  const handleSelectSession = useCallback((session: { session_id: string; model_name?: string }) => {
    if (!selectedModel && activeSessionId === session.session_id) {
      // Deselect
      fetchKeyRef.current = null;
      setActiveSessionId(null);
      setWeightFetchStatus('idle');
      setWeightFetchError(null);
      return;
    }
    fetchKeyRef.current = null;
    setSelectedModel(null);
    setActiveSessionId(session.session_id);
    setWeightFetchStatus('done');
    setWeightFetchError(null);
    setBackendWeightsReady(false);
    setResult(null);
    setPhase('idle');
    setError(null);
  }, [selectedModel, activeSessionId]);

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
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Badge variant="default">MNIST</Badge>
            <Badge variant="default" className="text-green-400">
              <Shield size={10} /> MPC
            </Badge>
          </div>
          <div className="flex items-center gap-1 p-1 rounded-xl bg-helix-surface border border-helix-border">
            <button
              type="button"
              onClick={() => setViewMode('live')}
              className={cn(
                'px-4 py-1.5 rounded-lg text-sm font-medium transition-all',
                viewMode === 'live' ? 'bg-white text-black' : 'text-helix-dim hover:text-helix-text2'
              )}
            >
              Live
            </button>
            <button
              type="button"
              onClick={() => setViewMode('history')}
              className={cn(
                'px-4 py-1.5 rounded-lg text-sm font-medium transition-all',
                viewMode === 'history' ? 'bg-white text-black' : 'text-helix-dim hover:text-helix-text2'
              )}
            >
              History
            </button>
          </div>
        </div>
      </div>

      {viewMode === 'live' ? (
      <>
      {/* ── TOP ROW: Model Discovery + Fee Card ─────────────── */}
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
          <div className="space-y-3 max-h-[400px] overflow-y-auto pr-1">
            {isLoadingPublic && isLoadingTrained ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 size={20} className="animate-spin text-helix-muted" />
              </div>
            ) : filteredModels.length === 0 && filteredSessions.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 rounded-2xl bg-helix-surface border border-helix-border">
                <Search size={20} className="text-helix-dim mb-2" />
                <p className="text-sm text-helix-text2">
                  {filter === 'mine' && !address ? 'Connect wallet to see your models' : 'No models found'}
                </p>
              </div>
            ) : (
              <>
                {/* Trained sessions */}
                {filteredSessions.length > 0 && !selectedModel && (
                  filteredSessions.map((session) => {
                    const isSelected = !selectedModel && activeSessionId === session.session_id;
                    if (activeSessionId && !selectedModel && !isSelected) return null;
                    return (
                      <TrainedSessionCard
                        key={session.session_id}
                        sessionId={session.session_id}
                        name={session.model_name || 'Trained Model'}
                        accuracy={session.accuracy}
                        isSelected={isSelected}
                        onSelect={() => handleSelectSession(session)}
                        showFee={false}
                      />
                    );
                  })
                )}

                {/* On-chain models */}
                {filteredModels.length > 0 && !(activeSessionId && !selectedModel) && (
                  filteredModels.map((model) => {
                    const isSelected = selectedModel?.tokenId === model.tokenId;
                    if (selectedModel && !isSelected) return null;
                    return (
                      <ModelCard
                        key={model.tokenId}
                        id={String(model.tokenId)}
                        name={model.name}
                        accuracy={model.bestAccuracy}
                        isSelected={isSelected}
                        onSelect={() => handleSelectModel(model)}
                        versions={model.versions}
                        selectedVersionIndex={selectedVersionIndex}
                        onVersionChange={(idx) => {
                          setSelectedVersionIndex(idx);
                          fetchKeyRef.current = null;
                          setWeightFetchStatus('idle');
                          setWeightFetchError(null);
                          setBackendWeightsReady(false);
                          setActiveSessionId(null);
                        }}
                        ownerAddress={model.owner}
                        userAddress={address}
                        tokenId={model.tokenId}
                        inferenceFee={model.inferenceFee}
                        architecture={model.architecture}
                        inferenceCount={model.inferenceCount}
                        averageRating={model.averageRating}
                        ratingCount={model.ratingCount}
                      />
                    );
                  })
                )}
              </>
            )}
          </div>

          {/* Manual weight upload */}
          <div className="flex items-center justify-between px-5 py-3 rounded-2xl bg-helix-surface border border-helix-border">
            <div className="flex items-center gap-3 min-w-0">
              <Upload size={14} className="text-helix-muted shrink-0" />
              <span className="text-sm text-helix-text2">Upload weights</span>
            </div>
            <label className="shrink-0 h-7 px-3 inline-flex items-center rounded-lg bg-white/[0.06] text-xs text-helix-text2 hover:text-white hover:bg-white/[0.1] transition-colors cursor-pointer">
              Browse
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

          {/* Weight status — only errors and loading */}
          <AnimatePresence mode="wait">
            {(weightFetchStatus === 'fetching' || weightFetchStatus === 'decrypting' || weightFetchStatus === 'uploading') && (
              <motion.div
                key="loading"
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                className="flex items-center gap-3 px-4 py-2.5 rounded-2xl bg-white/[0.03] border border-white/[0.06]"
              >
                <Loader2 size={12} className="animate-spin text-white/60" />
                <span className="text-xs text-white/50">
                  {weightFetchStatus === 'fetching' ? 'Fetching weights...'
                    : weightFetchStatus === 'decrypting' ? 'Decrypting...'
                    : 'Uploading to workers...'}
                </span>
              </motion.div>
            )}
            {weightFetchStatus === 'error' && (
              <motion.div
                key="error"
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                className="flex items-center gap-3 px-4 py-2.5 rounded-2xl bg-red-500/[0.06] border border-red-500/20"
              >
                <AlertTriangle size={12} className="text-red-400" />
                <span className="text-xs text-red-300 truncate">{weightFetchError}</span>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* ── RIGHT COLUMN: Fee Card (stretches to match left) ── */}
        <div className="flex flex-col gap-5">
          {/* Security badge for non-owner public model */}
          {selectedModel && !isOwnerOfSelected && inferenceReady && (
            <div className="flex items-center gap-2.5 px-4 py-3 bg-green-500/[0.04] border border-green-500/20 rounded-xl">
              <Shield size={14} className="text-green-400 shrink-0" />
              <p className="text-2xs text-green-300/70">
                Public inference enabled — model weights are never exposed to you
              </p>
            </div>
          )}

          {/* Non-owner: model not inference-ready */}
          {selectedModel && !isOwnerOfSelected && !inferenceReady && weightFetchStatus === 'error' && (
            <div className="flex items-center gap-2.5 px-4 py-3 bg-amber-500/[0.06] border border-amber-500/20 rounded-xl">
              <AlertTriangle size={14} className="text-amber-400 shrink-0" />
              <p className="text-2xs text-amber-300/70">
                Model owner hasn&apos;t enabled inference yet. Check back later.
              </p>
            </div>
          )}

          {/* Fee / Payment + Run button */}
          <div className="rounded-2xl bg-helix-surface border border-helix-border overflow-hidden flex flex-col">
            <div className="px-5 pt-5 pb-4">
              {/* Title row — bold white, Apple/Cash App style */}
              <div className="flex items-start justify-between">
                <h3 className="text-2xl font-bold tracking-tight text-white">Inference Fee</h3>
                {isOwnerOfSelected ? (
                  <Badge variant="default" className="text-green-400">
                    <CheckCircle size={10} /> Free
                  </Badge>
                ) : selectedModel && !isOwnerOfSelected && inferenceReady ? (
                  <Badge variant="default" className="text-green-400">
                    <CheckCircle size={10} /> Free
                  </Badge>
                ) : selectedModel && ownerFeeBps > 0 ? (
                  <Badge variant="default" className="text-helix-text2">
                    <Coins size={10} />
                    {(ownerFeeBps / 100).toFixed(1)}% fee
                  </Badge>
                ) : null}
              </div>

              {/* Fee amount */}
              <div className="flex items-baseline gap-2 mt-3">
                <span className="text-4xl font-bold tracking-tighter tabular-nums text-white">
                  {(isOwnerOfSelected || (!isOwnerOfSelected && inferenceReady)) ? '0.000000' : totalFee.toFixed(6)}
                </span>
                <span className="text-lg font-semibold text-helix-text2">ADI</span>
              </div>

              {/* Fee breakdown — single line */}
              <p className="text-sm text-helix-dim mt-1.5">
                {isOwnerOfSelected ? (
                  'Owner inference is always free'
                ) : !isOwnerOfSelected && (inferenceReady || backendWeightsReady) ? (
                  'Public inference — MPC distributed'
                ) : ownerFeeBps > 0 ? (
                  `Workers ${BASE_INFERENCE_COST.toFixed(4)} + Owner ${(ownerFeeBps / 100).toFixed(1)}%`
                ) : (
                  `Workers ${BASE_INFERENCE_COST.toFixed(4)} ADI`
                )}
                {!isOwnerOfSelected && !inferenceReady && !backendWeightsReady && (
                  <span className="ml-2 text-helix-dim/60 border-b border-dotted border-helix-border cursor-help group relative">
                    · deposit refundable
                  </span>
                )}
              </p>
            </div>

            {/* Run button */}
            <motion.button
              type="button"
              onClick={
                isOwnerOfSelected || !selectedModel
                  ? () => runInference()
                  : (inferenceReady || backendWeightsReady)
                    ? () => runInference()
                    : handlePayAndRun
              }
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
                <><Loader2 size={22} className="animate-spin" /> Running Inference...</>
              ) : isOwnerOfSelected || !selectedModel ? (
                <><Shield size={22} /> Run Inference</>
              ) : (inferenceReady || backendWeightsReady) ? (
                <><Shield size={22} /> Run MPC Inference</>
              ) : (
                <><Coins size={22} /> Pay &amp; Run Inference</>
              )}
            </motion.button>
          </div>
        </div>
      </div>

      {/* ── BOTTOM ROW: Input + Results side by side ────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* ── LEFT: Drawing / Upload Input ──────────────────── */}
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

        {/* ── RIGHT: Prediction / Results ───────────────────── */}
        <div className="flex flex-col gap-4">
          {isRunning && (
            <div className="rounded-2xl bg-helix-surface border border-helix-border p-5 flex-1 flex items-center">
              <div className="flex items-center gap-3">
                <Loader2 size={16} className="animate-spin text-white" />
                <div>
                  <p className="text-base text-white">
                    {selectedModel && !isOwnerOfSelected && inferenceReady
                      ? 'Running public inference...'
                      : 'Running distributed MPC inference...'}
                  </p>
                  <p className="text-sm text-helix-muted mt-0.5">
                    {selectedModel && !isOwnerOfSelected && inferenceReady
                      ? 'Server-side forward pass on cached model weights'
                      : 'Secret-sharing across 3 workers, running forward pass via transport'}
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
            <>
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                className="flex-1"
              >
                <MPCResultCard result={result} />
              </motion.div>
              {selectedModel && address && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: 0.15 }}
                >
                  <InferenceRatingPrompt
                    tokenId={selectedModel.tokenId}
                    walletAddress={address}
                    onRated={refetchPublicModels}
                  />
                </motion.div>
              )}
            </>
          )}

          {!isRunning && !error && phase === 'idle' && (
            <div className="rounded-2xl bg-helix-surface border border-helix-border p-5 flex-1 flex items-center justify-center">
              <div className="flex flex-col items-center justify-center py-8">
                <Shield size={24} className="text-helix-dim mb-3" />
                <p className="text-base text-helix-text2">
                  {selectedModel && !isOwnerOfSelected && !inferenceReady
                    ? 'Inference not available for this model'
                    : !weightsReady && !inferenceReady
                      ? 'Select a model to get started'
                      : !hasDrawing
                        ? 'Draw a digit and click Run Inference'
                        : 'Ready to classify'}
                </p>
                <p className="text-sm text-helix-muted mt-1">
                  {selectedModel && !isOwnerOfSelected && inferenceReady
                    ? 'Public inference — prediction without weight access'
                    : 'Secure multi-party computation across independent workers'}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
      </>
      ) : (
        <InferenceHistoryView />
      )}
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
