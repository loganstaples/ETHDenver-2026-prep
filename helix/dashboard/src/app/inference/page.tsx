'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { useSearchParams } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Sparkles,
  Loader2,
  Eraser,
  HardDrive,
  Server,
  CheckCircle,
  AlertTriangle,
  Tag,
  ChevronDown,
  Upload,
  Image as ImageIcon,
  Pencil,
  Shield,
  Clock,
  Users,
} from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { cn } from '@/lib/utils';

// ============================================================================
// Types
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
}

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

// ============================================================================
// Drawing Canvas
// ============================================================================

const CANVAS_SIZE = 280;
const GRID_SIZE = 28;
const BRUSH_RADIUS = 12;

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
          const ix = lastPos.current.x + dx * t;
          const iy = lastPos.current.y + dy * t;
          ctx.beginPath();
          ctx.arc(ix, iy, BRUSH_RADIUS, 0, Math.PI * 2);
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
      className="rounded-lg border border-helix-border cursor-crosshair touch-none"
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
          const x = (CANVAS_SIZE - w) / 2;
          const y = (CANVAS_SIZE - h) / 2;
          ctx.drawImage(img, x, y, w, h);

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
        className="rounded-lg border border-helix-border"
        style={{
          width: CANVAS_SIZE,
          height: CANVAS_SIZE,
          display: fileName ? 'block' : 'none',
        }}
      />

      {!fileName && (
        <div
          onClick={() => fileRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          className="flex flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed border-helix-border hover:border-helix-border2 bg-helix-bg cursor-pointer transition-colors"
          style={{ width: CANVAS_SIZE, height: CANVAS_SIZE }}
        >
          <Upload size={24} className="text-helix-muted" />
          <p className="text-sm text-helix-text2">Drop an image or click to upload</p>
          <p className="text-2xs text-helix-muted">PNG, JPG, or any image of a digit</p>
        </div>
      )}

      {fileName && (
        <div className="flex items-center gap-2">
          <ImageIcon size={12} className="text-green-400" />
          <span className="text-2xs text-helix-text truncate">{fileName}</span>
          <button
            type="button"
            onClick={() => {
              setFileName(null);
              onPixelsReady([]);
              const canvas = canvasRef.current;
              if (canvas) {
                const ctx = canvas.getContext('2d');
                if (ctx) {
                  ctx.fillStyle = '#000000';
                  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
                }
              }
            }}
            className="text-2xs text-helix-muted hover:text-white ml-auto"
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
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
        }}
      />
    </div>
  );
}

// ============================================================================
// Probability Bars
// ============================================================================

function ProbabilityBars({
  probabilities,
  prediction,
}: {
  probabilities: number[];
  prediction: number;
}) {
  return (
    <div className="space-y-1.5">
      {probabilities.map((prob, digit) => (
        <div key={digit} className="flex items-center gap-3">
          <span
            className={cn(
              'w-5 text-right text-sm font-mono',
              digit === prediction ? 'text-white font-semibold' : 'text-helix-muted',
            )}
          >
            {digit}
          </span>
          <div className="flex-1 h-5 bg-helix-border rounded-sm overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${prob * 100}%` }}
              transition={{ duration: 0.4, ease: 'easeOut' }}
              className={cn(
                'h-full rounded-sm',
                digit === prediction ? 'bg-white' : 'bg-white/20',
              )}
            />
          </div>
          <span
            className={cn(
              'w-14 text-right text-xs font-mono',
              digit === prediction ? 'text-white' : 'text-helix-muted',
            )}
          >
            {(prob * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}

// ============================================================================
// Model Selector
// ============================================================================

function ModelSelector({
  models,
  selected,
  onSelect,
}: {
  models: TrainingHistoryEntry[];
  selected: TrainingHistoryEntry | null;
  onSelect: (entry: TrainingHistoryEntry) => void;
}) {
  const [open, setOpen] = useState(false);

  if (models.length === 0) return null;

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 bg-helix-bg border border-helix-border rounded-lg text-left hover:border-helix-border2 transition-colors"
      >
        {selected ? (
          <div className="flex items-center gap-3">
            {selected.storedOn0G ? (
              <HardDrive size={14} className="text-green-400" />
            ) : (
              <Server size={14} className="text-helix-text2" />
            )}
            <Badge variant="default" className="font-mono text-2xs">
              v{selected.version}
            </Badge>
            <span className="text-sm text-helix-text">
              {selected.accuracy !== null
                ? `${(selected.accuracy * 100).toFixed(1)}% accuracy`
                : 'No accuracy data'}
            </span>
            <span className="text-2xs text-helix-muted font-mono">
              {selected.sessionId.slice(0, 8)}
            </span>
          </div>
        ) : (
          <span className="text-sm text-helix-muted">Select a model version...</span>
        )}
        <ChevronDown
          size={14}
          className={cn('text-helix-muted transition-transform', open && 'rotate-180')}
        />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.15 }}
            className="absolute z-20 top-full mt-1 w-full bg-helix-surface border border-helix-border rounded-lg shadow-xl overflow-hidden max-h-64 overflow-y-auto"
          >
            {models.map((entry) => (
              <button
                key={entry.sessionId}
                type="button"
                onClick={() => {
                  onSelect(entry);
                  setOpen(false);
                }}
                className={cn(
                  'w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors',
                  selected?.sessionId === entry.sessionId
                    ? 'bg-white/[0.06]'
                    : 'hover:bg-white/[0.03]',
                )}
              >
                {entry.storedOn0G ? (
                  <HardDrive size={12} className="text-green-400 shrink-0" />
                ) : (
                  <Server size={12} className="text-helix-muted shrink-0" />
                )}
                <Badge variant="default" className="font-mono text-2xs shrink-0">
                  v{entry.version}
                </Badge>
                <span className="text-xs text-helix-text">
                  {entry.accuracy !== null ? `${(entry.accuracy * 100).toFixed(1)}%` : '--'}
                </span>
                <span className="text-2xs text-helix-dim font-mono ml-auto">
                  {new Date(entry.date).toLocaleDateString()}
                </span>
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
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
          <span className="text-2xs text-helix-text flex-1">{p.label}</span>
          <span className="text-2xs text-helix-muted font-mono">{p.ms}ms</span>
        </div>
      ))}
      <div className="flex items-center gap-3 pt-1 border-t border-helix-border">
        <Clock size={10} className="text-white shrink-0" />
        <span className="text-2xs text-white font-medium flex-1">Total</span>
        <span className="text-2xs text-white font-mono font-medium">{timing.total_ms}ms</span>
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
    <Card variant="glass">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Shield size={14} className="text-green-400" />
          <h3 className="text-sm font-medium text-white">Distributed MPC Result</h3>
        </div>
        <div className="flex items-center gap-2">
          {result.distributed && (
            <Badge variant="default" className="text-green-400">
              <Shield size={10} />
              Distributed
            </Badge>
          )}
          <Badge variant="default" className="text-blue-400">
            <Users size={10} />
            {result.num_parties} workers
          </Badge>
          <Badge variant="default" className="text-helix-text2">
            <Clock size={10} />
            {result.timing.total_ms}ms
          </Badge>
        </div>
      </div>

      <div className="flex items-center gap-6 mb-4">
        <div className="w-20 h-20 rounded-xl bg-white/[0.06] flex items-center justify-center">
          <span className="text-4xl font-light text-white font-mono">{result.prediction}</span>
        </div>
        <div>
          <p className="text-2xl font-mono font-light text-white">
            {(result.confidence * 100).toFixed(1)}%
          </p>
          <p className="text-2xs text-helix-muted mt-0.5">confidence</p>
        </div>
        <div className="ml-auto">
          <CheckCircle size={24} className="text-green-400" />
        </div>
      </div>

      <ProbabilityBars probabilities={result.probabilities} prediction={result.prediction} />

      {/* Attestation summary */}
      {result.attestation && (
        <div className="mt-4 pt-3 border-t border-helix-border">
          <div className="flex items-center gap-2 mb-2">
            <Shield size={12} className="text-green-400" />
            <span className="text-2xs font-medium text-white">On-chain Attestation</span>
          </div>
          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="text-2xs text-helix-muted w-20">Input hash</span>
              <span className="text-2xs text-helix-text font-mono truncate">
                {result.attestation.input_hash.slice(0, 16)}...
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-2xs text-helix-muted w-20">Output hash</span>
              <span className="text-2xs text-helix-text font-mono truncate">
                {result.attestation.output_hash.slice(0, 16)}...
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-2xs text-helix-muted w-20">Signatures</span>
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
                <span className="text-2xs text-green-400 ml-1">
                  {result.attestation.worker_signatures.length}/{result.num_parties} signed
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Expandable timing details */}
      <div className="mt-4 pt-3 border-t border-helix-border">
        <button
          type="button"
          onClick={() => setShowDetails(!showDetails)}
          className="text-2xs text-helix-muted hover:text-white transition-colors flex items-center gap-1"
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
    </Card>
  );
}

// ============================================================================
// Main Page
// ============================================================================

export default function InferencePage() {
  const searchParams = useSearchParams();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [models, setModels] = useState<TrainingHistoryEntry[]>([]);
  const [selected, setSelected] = useState<TrainingHistoryEntry | null>(null);
  const [pixels, setPixels] = useState<number[]>([]);
  const [result, setResult] = useState<MPCInferenceResult | null>(null);
  const [phase, setPhase] = useState<InferencePhase>('idle');
  const [error, setError] = useState<string | null>(null);
  const [inputMode, setInputMode] = useState<InputMode>('draw');

  // Load models that have retrievable weights
  useEffect(() => {
    const history = getTrainingHistory();
    const withWeights = history.filter(
      (e) =>
        e.status === 'complete' &&
        ((e.storedOn0G && e.rootHash) || !e.sessionId.startsWith('upload-')),
    );
    withWeights.sort((a, b) => {
      if (a.storedOn0G && !b.storedOn0G) return -1;
      if (!a.storedOn0G && b.storedOn0G) return 1;
      return (b.accuracy ?? 0) - (a.accuracy ?? 0);
    });
    setModels(withWeights);

    const sessionParam = searchParams.get('session');
    if (sessionParam) {
      const match = withWeights.find((e) => e.sessionId === sessionParam);
      if (match) setSelected(match);
    } else if (withWeights.length > 0) {
      setSelected(withWeights[0]);
    }
  }, [searchParams]);

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

  const runInference = useCallback(async () => {
    if (!selected || !pixels.length) return;

    setPhase('submitting');
    setError(null);
    setResult(null);

    try {
      const res = await fetch('/api/inference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: selected.sessionId,
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
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Inference failed');
      setPhase('error');
    }
  }, [selected, pixels]);

  const hasDrawing = pixels.length > 0 && pixels.some((p) => p > 0.01);
  const isRunning = phase === 'submitting';

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="flex items-center gap-3">
        <h1 className="page-title">Inference</h1>
        <Badge variant="default">MNIST</Badge>
        <Badge variant="default" className="text-green-400">
          <Shield size={10} />
          MPC
        </Badge>
      </div>

      {models.length === 0 ? (
        <Card variant="default">
          <div className="flex flex-col items-center justify-center py-12">
            <Sparkles size={32} className="text-helix-dim mb-3" />
            <p className="text-sm font-medium text-helix-text2">No models with stored weights</p>
            <p className="text-2xs text-helix-muted mt-1">
              Train a model and store it on 0G, or upload weights on the My Models page.
            </p>
          </div>
        </Card>
      ) : (
        <>
          {/* Model Selection */}
          <Card variant="default">
            <div className="flex items-center gap-2 mb-3">
              <Tag size={14} className="text-helix-text2" />
              <h3 className="text-sm font-medium text-white">Model</h3>
            </div>
            <ModelSelector
              models={models}
              selected={selected}
              onSelect={(entry) => {
                setSelected(entry);
                setResult(null);
                setPhase('idle');
                setError(null);
              }}
            />
            {selected && (
              <div className="flex items-center gap-4 mt-3 text-2xs text-helix-muted">
                <span className="font-mono">784 &rarr; 128 &rarr; 10</span>
                <span>&middot;</span>
                <span className="flex items-center gap-1">
                  {selected.storedOn0G ? (
                    <>
                      <HardDrive size={10} className="text-green-400" /> 0G Storage
                    </>
                  ) : (
                    <>
                      <Server size={10} /> Backend
                    </>
                  )}
                </span>
                <span>&middot;</span>
                <span>Trained {new Date(selected.date).toLocaleDateString()}</span>
              </div>
            )}
          </Card>

          {/* Input + Results */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Input Card */}
            <Card variant="default">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-1 p-0.5 rounded-lg bg-helix-bg border border-helix-border">
                  <button
                    type="button"
                    onClick={() => {
                      setInputMode('draw');
                      clearCanvas();
                    }}
                    className={cn(
                      'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-2xs font-medium transition-all',
                      inputMode === 'draw'
                        ? 'bg-white/[0.08] text-white'
                        : 'text-helix-muted hover:text-helix-text',
                    )}
                  >
                    <Pencil size={12} />
                    Draw
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setInputMode('upload');
                      clearCanvas();
                    }}
                    className={cn(
                      'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-2xs font-medium transition-all',
                      inputMode === 'upload'
                        ? 'bg-white/[0.08] text-white'
                        : 'text-helix-muted hover:text-helix-text',
                    )}
                  >
                    <Upload size={12} />
                    Upload Image
                  </button>
                </div>

                {inputMode === 'draw' && (
                  <button
                    type="button"
                    onClick={clearCanvas}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-2xs text-helix-muted bg-helix-bg border border-helix-border hover:text-white hover:border-helix-border2 transition-colors"
                  >
                    <Eraser size={12} />
                    Clear
                  </button>
                )}
              </div>

              <div className="flex justify-center mb-4">
                {inputMode === 'draw' ? (
                  <div className="relative">
                    <DrawingCanvas canvasRef={canvasRef} onPixelsReady={setPixels} />
                    <div
                      className="absolute inset-0 pointer-events-none rounded-lg opacity-[0.03]"
                      style={{
                        backgroundImage: `linear-gradient(rgba(255,255,255,0.5) 1px, transparent 1px),
                           linear-gradient(90deg, rgba(255,255,255,0.5) 1px, transparent 1px)`,
                        backgroundSize: `${CANVAS_SIZE / GRID_SIZE}px ${CANVAS_SIZE / GRID_SIZE}px`,
                      }}
                    />
                  </div>
                ) : (
                  <ImageUpload canvasRef={canvasRef} onPixelsReady={setPixels} />
                )}
              </div>

              <p className="text-2xs text-helix-dim text-center mb-4">
                {inputMode === 'draw'
                  ? 'Draw a digit (0-9) on the canvas above'
                  : 'Upload an image of a handwritten digit (0-9)'}
              </p>

              <button
                type="button"
                onClick={runInference}
                disabled={!selected || !hasDrawing || isRunning}
                className={cn(
                  'w-full flex items-center justify-center gap-2 py-3 rounded-lg font-medium text-sm transition-all',
                  !selected || !hasDrawing || isRunning
                    ? 'bg-helix-border text-helix-muted cursor-not-allowed'
                    : 'bg-white text-black hover:bg-white/90',
                )}
              >
                {isRunning ? (
                  <>
                    <Loader2 size={16} className="animate-spin" />
                    Running MPC Inference...
                  </>
                ) : (
                  <>
                    <Shield size={16} />
                    Classify with MPC
                  </>
                )}
              </button>

              {/* How it works */}
              <div className="mt-4 p-3 bg-helix-bg rounded-lg border border-helix-border">
                <p className="text-2xs font-medium text-helix-text2 mb-2">How distributed MPC inference works</p>
                <div className="space-y-1.5 text-2xs text-helix-muted">
                  <div className="flex items-center gap-2">
                    <Shield size={10} className="text-green-400 shrink-0" />
                    <span>Weights are <strong className="text-helix-text">secret-shared</strong> across independent workers</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Users size={10} className="text-blue-400 shrink-0" />
                    <span>Each worker computes on their <strong className="text-helix-text">share only</strong> via inter-party transport</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle size={10} className="text-cyan-400 shrink-0" />
                    <span>Workers <strong className="text-helix-text">sign attestations</strong> for on-chain verification</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <HardDrive size={10} className="text-purple-400 shrink-0" />
                    <span>Multi-party attestation recorded on <strong className="text-helix-text">HelixCoordinatorV4</strong></span>
                  </div>
                </div>
              </div>
            </Card>

            {/* Results Column */}
            <div className="space-y-4">
              {/* Submitting indicator */}
              {isRunning && (
                <Card variant="default">
                  <div className="flex items-center gap-3 py-4">
                    <Loader2 size={16} className="animate-spin text-white" />
                    <div>
                      <p className="text-sm text-white">Running distributed MPC inference...</p>
                      <p className="text-2xs text-helix-muted mt-0.5">
                        Secret-sharing weights across 3 independent workers, running forward pass via transport
                      </p>
                    </div>
                  </div>
                </Card>
              )}

              {/* Error */}
              {error && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3"
                >
                  <div className="flex items-start gap-2">
                    <AlertTriangle size={14} className="text-red-400 shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm text-red-300">{error}</p>
                      <button
                        type="button"
                        onClick={() => {
                          setError(null);
                          setPhase('idle');
                        }}
                        className="text-2xs text-red-400 hover:text-red-300 mt-1 underline"
                      >
                        Dismiss
                      </button>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Result */}
              {phase === 'done' && result && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <MPCResultCard result={result} />
                </motion.div>
              )}

              {/* Empty state */}
              {!isRunning && !error && phase === 'idle' && (
                <Card variant="default">
                  <div className="flex flex-col items-center justify-center py-16">
                    <Shield size={24} className="text-helix-dim mb-3" />
                    <p className="text-sm text-helix-text2">Draw a digit and click Classify</p>
                    <p className="text-2xs text-helix-muted mt-1">
                      Runs inference via secure multi-party computation
                    </p>
                  </div>
                </Card>
              )}
            </div>
          </div>
        </>
      )}
    </motion.div>
  );
}
