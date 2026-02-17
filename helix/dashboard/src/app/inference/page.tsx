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
  Zap,
  Database,
  Globe,
  Cpu,
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

interface LocalInferenceResult {
  prediction: number;
  confidence: number;
  probabilities: number[];
  model_source: string;
  cached: boolean;
}

interface ComputeInferenceResult {
  prediction: number;
  confidence: number;
  raw_response: string;
  model_used: string;
  provider: string;
  source: string;
}

type InferencePhase =
  | 'idle'
  | 'loading-model'
  | 'running-local'
  | 'calling-0g-compute'
  | 'done'
  | 'error';

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
// Progress Steps
// ============================================================================

function InferenceProgress({
  phase,
  localResult,
  computeResult,
}: {
  phase: InferencePhase;
  localResult: LocalInferenceResult | null;
  computeResult: ComputeInferenceResult | null;
}) {
  const steps = [
    {
      key: 'loading-model',
      label: 'Loading model from 0G Storage',
      icon: Database,
      detail: 'Downloading weights from decentralized storage',
    },
    {
      key: 'running-local',
      label: 'Running trained model inference',
      icon: Cpu,
      detail: 'Forward pass through HELIX-trained neural network',
    },
    {
      key: 'calling-0g-compute',
      label: 'Running 0G Compute inference',
      icon: Globe,
      detail: 'Classifying via 0G Compute Network LLM',
    },
  ] as const;

  const phaseOrder = ['loading-model', 'running-local', 'calling-0g-compute', 'done'];

  return (
    <div className="space-y-2">
      {steps.map((step) => {
        const stepIdx = phaseOrder.indexOf(step.key);
        const currentIdx = phaseOrder.indexOf(phase);
        const isActive = phase === step.key;
        const isDone = currentIdx > stepIdx;
        const Icon = step.icon;

        return (
          <div
            key={step.key}
            className={cn(
              'flex items-center gap-3 px-3 py-2 rounded-lg transition-all',
              isActive && 'bg-white/[0.04]',
            )}
          >
            {isActive ? (
              <Loader2 size={14} className="animate-spin text-white shrink-0" />
            ) : isDone ? (
              <CheckCircle size={14} className="text-green-400 shrink-0" />
            ) : (
              <Icon size={14} className="text-helix-dim shrink-0" />
            )}
            <div className="flex-1 min-w-0">
              <span
                className={cn(
                  'text-sm',
                  isActive ? 'text-white' : isDone ? 'text-helix-text' : 'text-helix-dim',
                )}
              >
                {step.label}
              </span>
              {isActive && (
                <p className="text-2xs text-helix-muted mt-0.5">{step.detail}</p>
              )}
            </div>
            {step.key === 'running-local' && isDone && localResult && (
              <Badge variant="default" className="text-2xs shrink-0">
                Predicted: {localResult.prediction}
              </Badge>
            )}
            {step.key === 'calling-0g-compute' && isDone && computeResult && (
              <Badge variant="default" className="text-2xs shrink-0">
                Predicted: {computeResult.prediction}
              </Badge>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ============================================================================
// Result Cards
// ============================================================================

function LocalResultCard({ result }: { result: LocalInferenceResult }) {
  return (
    <Card variant="glass">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Cpu size={14} className="text-blue-400" />
          <h3 className="text-sm font-medium text-white">HELIX Trained Model</h3>
        </div>
        <div className="flex items-center gap-2">
          {result.cached && (
            <Badge variant="default" className="text-helix-muted text-2xs">
              <Zap size={10} />
              Cached
            </Badge>
          )}
          {result.model_source.includes('0g') ? (
            <Badge variant="default" className="text-green-400">
              <HardDrive size={10} />
              0G Storage
            </Badge>
          ) : (
            <Badge variant="default">
              <Server size={10} />
              Backend
            </Badge>
          )}
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
    </Card>
  );
}

function ComputeResultCard({ result }: { result: ComputeInferenceResult }) {
  const [showRaw, setShowRaw] = useState(false);

  return (
    <Card variant="glass">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Globe size={14} className="text-purple-400" />
          <h3 className="text-sm font-medium text-white">0G Compute Network</h3>
        </div>
        <Badge variant="default" className="text-purple-400">
          <Globe size={10} />
          {result.model_used?.split('/').pop() || '0G LLM'}
        </Badge>
      </div>

      <div className="flex items-center gap-6 mb-4">
        <div className="w-20 h-20 rounded-xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center">
          <span className="text-4xl font-light text-white font-mono">{result.prediction}</span>
        </div>
        <div>
          <p className="text-2xl font-mono font-light text-white">
            {(result.confidence * 100).toFixed(0)}%
          </p>
          <p className="text-2xs text-helix-muted mt-0.5">LLM confidence</p>
        </div>
        <div className="ml-auto">
          <CheckCircle size={24} className="text-purple-400" />
        </div>
      </div>

      <div className="flex items-center gap-2 text-2xs text-helix-muted mb-2">
        <span className="font-mono truncate">Provider: {result.provider?.slice(0, 10)}...</span>
      </div>

      <button
        type="button"
        onClick={() => setShowRaw(!showRaw)}
        className="text-2xs text-helix-muted hover:text-white transition-colors"
      >
        {showRaw ? 'Hide' : 'Show'} raw LLM response
      </button>

      <AnimatePresence>
        {showRaw && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
          >
            <pre className="text-2xs font-mono text-helix-text bg-helix-bg px-3 py-2 rounded-md border border-helix-border mt-2 whitespace-pre-wrap">
              {result.raw_response}
            </pre>
          </motion.div>
        )}
      </AnimatePresence>
    </Card>
  );
}

function AgreementBadge({
  localResult,
  computeResult,
}: {
  localResult: LocalInferenceResult;
  computeResult: ComputeInferenceResult;
}) {
  const agree = localResult.prediction === computeResult.prediction;

  return (
    <div
      className={cn(
        'flex items-center justify-center gap-3 px-4 py-3 rounded-lg border',
        agree
          ? 'bg-green-500/5 border-green-500/20'
          : 'bg-yellow-500/5 border-yellow-500/20',
      )}
    >
      {agree ? (
        <CheckCircle size={16} className="text-green-400" />
      ) : (
        <AlertTriangle size={16} className="text-yellow-400" />
      )}
      <span className={cn('text-sm font-medium', agree ? 'text-green-300' : 'text-yellow-300')}>
        {agree
          ? `Both models agree: digit is ${localResult.prediction}`
          : `Models disagree: trained model says ${localResult.prediction}, 0G Compute says ${computeResult.prediction}`}
      </span>
    </div>
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
  const [localResult, setLocalResult] = useState<LocalInferenceResult | null>(null);
  const [computeResult, setComputeResult] = useState<ComputeInferenceResult | null>(null);
  const [phase, setPhase] = useState<InferencePhase>('idle');
  const [error, setError] = useState<string | null>(null);
  const [computeError, setComputeError] = useState<string | null>(null);
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
    setLocalResult(null);
    setComputeResult(null);
    setPhase('idle');
    setError(null);
    setComputeError(null);
  }, []);

  const runInference = useCallback(async () => {
    if (!selected || !pixels.length) return;

    setPhase('loading-model');
    setError(null);
    setComputeError(null);
    setLocalResult(null);
    setComputeResult(null);

    try {
      // Phase 1: Load model + run local inference
      setPhase('loading-model');
      const localRes = await fetch('/api/inference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: selected.sessionId,
          root_hash: selected.rootHash || undefined,
          pixels,
        }),
      });

      if (!localRes.ok) {
        const errBody = await localRes.json().catch(() => ({}));
        throw new Error(errBody.error || `Local inference HTTP ${localRes.status}`);
      }

      setPhase('running-local');
      const localData: LocalInferenceResult = await localRes.json();
      setLocalResult(localData);

      // Phase 2: Run 0G Compute inference in parallel
      setPhase('calling-0g-compute');
      try {
        const computeRes = await fetch('/api/0g-compute', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ pixels }),
        });

        if (!computeRes.ok) {
          const errBody = await computeRes.json().catch(() => ({}));
          setComputeError(errBody.error || `0G Compute HTTP ${computeRes.status}`);
        } else {
          const computeData: ComputeInferenceResult = await computeRes.json();
          setComputeResult(computeData);
        }
      } catch (err) {
        setComputeError(err instanceof Error ? err.message : '0G Compute failed');
      }

      setPhase('done');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Inference failed');
      setPhase('error');
    }
  }, [selected, pixels]);

  const hasDrawing = pixels.length > 0 && pixels.some((p) => p > 0.01);
  const isRunning =
    phase === 'loading-model' || phase === 'running-local' || phase === 'calling-0g-compute';

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
        <Badge variant="default" className="text-purple-400">
          <Globe size={10} />
          0G Compute
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
                setLocalResult(null);
                setComputeResult(null);
                setPhase('idle');
                setError(null);
                setComputeError(null);
              }}
            />
            {selected && (
              <div className="flex items-center gap-4 mt-3 text-2xs text-helix-muted">
                <span className="font-mono">784 → 128 → 10</span>
                <span>·</span>
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
                <span>·</span>
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
                    {phase === 'loading-model'
                      ? 'Loading Model...'
                      : phase === 'running-local'
                        ? 'Running Local Inference...'
                        : 'Querying 0G Compute...'}
                  </>
                ) : (
                  <>
                    <Sparkles size={16} />
                    Classify
                  </>
                )}
              </button>

              {/* How it works */}
              <div className="mt-4 p-3 bg-helix-bg rounded-lg border border-helix-border">
                <p className="text-2xs font-medium text-helix-text2 mb-2">How inference works</p>
                <div className="space-y-1.5 text-2xs text-helix-muted">
                  <div className="flex items-center gap-2">
                    <HardDrive size={10} className="text-green-400 shrink-0" />
                    <span>Model weights loaded from <strong className="text-helix-text">0G Storage</strong></span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Cpu size={10} className="text-blue-400 shrink-0" />
                    <span>Forward pass through <strong className="text-helix-text">HELIX trained model</strong></span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Globe size={10} className="text-purple-400 shrink-0" />
                    <span>Verified via <strong className="text-helix-text">0G Compute Network</strong> LLM</span>
                  </div>
                </div>
              </div>
            </Card>

            {/* Results Column */}
            <div className="space-y-4">
              {/* Progress */}
              {isRunning && (
                <Card variant="default">
                  <h3 className="text-sm font-medium text-white mb-3">Progress</h3>
                  <InferenceProgress
                    phase={phase}
                    localResult={localResult}
                    computeResult={computeResult}
                  />
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

              {/* Results */}
              {phase === 'done' && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                  className="space-y-4"
                >
                  {/* Agreement indicator */}
                  {localResult && computeResult && (
                    <AgreementBadge localResult={localResult} computeResult={computeResult} />
                  )}

                  {/* Local model result */}
                  {localResult && <LocalResultCard result={localResult} />}

                  {/* 0G Compute result */}
                  {computeResult && <ComputeResultCard result={computeResult} />}

                  {/* 0G Compute error (non-fatal) */}
                  {computeError && !computeResult && (
                    <Card variant="default">
                      <div className="flex items-center gap-2 mb-2">
                        <Globe size={14} className="text-purple-400" />
                        <h3 className="text-sm font-medium text-white">0G Compute Network</h3>
                      </div>
                      <div className="flex items-start gap-2 bg-yellow-500/5 border border-yellow-500/20 rounded-lg px-3 py-2">
                        <AlertTriangle size={12} className="text-yellow-400 shrink-0 mt-0.5" />
                        <div>
                          <p className="text-2xs text-yellow-300">{computeError}</p>
                          <p className="text-2xs text-helix-muted mt-1">
                            0G Compute requires ZG_COMPUTE_PRIVATE_KEY in .env.local with funded 0G tokens.
                          </p>
                        </div>
                      </div>
                    </Card>
                  )}
                </motion.div>
              )}

              {/* Empty state */}
              {!isRunning && !error && phase === 'idle' && (
                <Card variant="default">
                  <div className="flex flex-col items-center justify-center py-16">
                    <Sparkles size={24} className="text-helix-dim mb-3" />
                    <p className="text-sm text-helix-text2">Draw a digit and click Classify</p>
                    <p className="text-2xs text-helix-muted mt-1">
                      Runs inference via HELIX model + 0G Compute Network
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
