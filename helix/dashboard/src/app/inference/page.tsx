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

interface InferenceResult {
  prediction: number;
  confidence: number;
  probabilities: number[];
  model_source: string;
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

  const getCanvasPos = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
    return {
      x: (clientX - rect.left) * (canvas.width / rect.width),
      y: (clientY - rect.top) * (canvas.height / rect.height),
    };
  }, [canvasRef]);

  const draw = useCallback((x: number, y: number) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(x, y, BRUSH_RADIUS, 0, Math.PI * 2);
    ctx.fill();

    // Interpolate between last position for smooth lines
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
  }, [canvasRef]);

  const handleStart = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    isDrawing.current = true;
    const pos = getCanvasPos(e);
    lastPos.current = null;
    draw(pos.x, pos.y);
  }, [getCanvasPos, draw]);

  const handleMove = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    if (!isDrawing.current) return;
    const pos = getCanvasPos(e);
    draw(pos.x, pos.y);
  }, [getCanvasPos, draw]);

  const handleEnd = useCallback(() => {
    isDrawing.current = false;
    lastPos.current = null;
    extractPixels();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const extractPixels = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Create a temporary small canvas to downsample
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = GRID_SIZE;
    tempCanvas.height = GRID_SIZE;
    const tempCtx = tempCanvas.getContext('2d')!;

    // Use anti-aliasing when scaling down
    tempCtx.imageSmoothingEnabled = true;
    tempCtx.imageSmoothingQuality = 'high';
    tempCtx.drawImage(canvas, 0, 0, GRID_SIZE, GRID_SIZE);

    const imageData = tempCtx.getImageData(0, 0, GRID_SIZE, GRID_SIZE);
    const pixels: number[] = [];

    for (let i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
      // Use red channel (grayscale), normalize to [0, 1]
      pixels.push(imageData.data[i * 4] / 255);
    }

    onPixelsReady(pixels);
  }, [canvasRef, onPixelsReady]);

  // Initialize canvas with black background
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  }, [canvasRef]);

  return (
    <div className="relative">
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
      {/* Grid overlay (subtle) */}
      <div
        className="absolute inset-0 pointer-events-none rounded-lg opacity-[0.03]"
        style={{
          backgroundImage:
            `linear-gradient(rgba(255,255,255,0.5) 1px, transparent 1px),
             linear-gradient(90deg, rgba(255,255,255,0.5) 1px, transparent 1px)`,
          backgroundSize: `${CANVAS_SIZE / GRID_SIZE}px ${CANVAS_SIZE / GRID_SIZE}px`,
        }}
      />
    </div>
  );
}

// ============================================================================
// Probability Bars
// ============================================================================

interface ProbabilityBarsProps {
  probabilities: number[];
  prediction: number;
}

function ProbabilityBars({ probabilities, prediction }: ProbabilityBarsProps) {
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
          <div className="flex-1 h-5 bg-helix-border rounded-sm overflow-hidden relative">
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

interface ModelSelectorProps {
  models: TrainingHistoryEntry[];
  selected: TrainingHistoryEntry | null;
  onSelect: (entry: TrainingHistoryEntry) => void;
}

function ModelSelector({ models, selected, onSelect }: ModelSelectorProps) {
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
            <Badge variant="default" className="font-mono text-2xs">v{selected.version}</Badge>
            <span className="text-sm text-helix-text">
              {selected.accuracy !== null ? `${(selected.accuracy * 100).toFixed(1)}% accuracy` : 'No accuracy data'}
            </span>
            <span className="text-2xs text-helix-muted font-mono">
              {selected.sessionId.slice(0, 8)}
            </span>
          </div>
        ) : (
          <span className="text-sm text-helix-muted">Select a model version...</span>
        )}
        <ChevronDown size={14} className={cn(
          'text-helix-muted transition-transform',
          open && 'rotate-180',
        )} />
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
                <Badge variant="default" className="font-mono text-2xs shrink-0">v{entry.version}</Badge>
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
// Main Page
// ============================================================================

export default function InferencePage() {
  const searchParams = useSearchParams();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [models, setModels] = useState<TrainingHistoryEntry[]>([]);
  const [selected, setSelected] = useState<TrainingHistoryEntry | null>(null);
  const [pixels, setPixels] = useState<number[] | null>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load models that have retrievable weights
  useEffect(() => {
    const history = getTrainingHistory();
    // Only show models with stored weights:
    // - Stored on 0G with a root hash (persistent, always retrievable)
    // - From a real training session (backend may have weights)
    const withWeights = history.filter(
      (e) => e.status === 'complete' && (
        (e.storedOn0G && e.rootHash) ||
        (!e.sessionId.startsWith('upload-'))
      ),
    );
    // Sort: 0G-stored first, then by accuracy descending
    withWeights.sort((a, b) => {
      if (a.storedOn0G && !b.storedOn0G) return -1;
      if (!a.storedOn0G && b.storedOn0G) return 1;
      return (b.accuracy ?? 0) - (a.accuracy ?? 0);
    });
    setModels(withWeights);

    // Auto-select from URL params
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
    setPixels(null);
    setResult(null);
    setError(null);
  }, []);

  const runInference = useCallback(async () => {
    if (!selected || !pixels) return;

    setIsRunning(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch('/api/inference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: selected.sessionId,
          root_hash: selected.rootHash || undefined,
          pixels,
        }),
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.error || `HTTP ${res.status}`);
      }

      const data: InferenceResult = await res.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Inference failed');
    } finally {
      setIsRunning(false);
    }
  }, [selected, pixels]);

  const hasDrawing = pixels !== null && pixels.some((p) => p > 0.01);

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
              onSelect={setSelected}
            />
            {selected && (
              <div className="flex items-center gap-4 mt-3 text-2xs text-helix-muted">
                <span className="font-mono">784 → 128 → 10</span>
                <span>·</span>
                <span>{selected.storedOn0G ? 'Stored on 0G' : 'Local backend'}</span>
                <span>·</span>
                <span>Trained {new Date(selected.date).toLocaleDateString()}</span>
              </div>
            )}
          </Card>

          {/* Drawing + Results */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Drawing Canvas */}
            <Card variant="default">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium text-white">Draw a Digit</h3>
                <button
                  type="button"
                  onClick={clearCanvas}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-2xs text-helix-muted bg-helix-bg border border-helix-border hover:text-white hover:border-helix-border2 transition-colors"
                >
                  <Eraser size={12} />
                  Clear
                </button>
              </div>

              <div className="flex justify-center mb-4">
                <DrawingCanvas canvasRef={canvasRef} onPixelsReady={setPixels} />
              </div>

              <p className="text-2xs text-helix-dim text-center mb-4">
                Draw a digit (0-9) on the canvas above
              </p>

              <button
                type="button"
                onClick={runInference}
                disabled={!selected || !hasDrawing || isRunning}
                className={cn(
                  'w-full flex items-center justify-center gap-2 py-3 rounded-lg font-medium text-sm transition-all',
                  (!selected || !hasDrawing || isRunning)
                    ? 'bg-helix-border text-helix-muted cursor-not-allowed'
                    : 'bg-white text-black hover:bg-white/90',
                )}
              >
                {isRunning ? (
                  <>
                    <Loader2 size={16} className="animate-spin" />
                    Running Inference...
                  </>
                ) : (
                  <>
                    <Sparkles size={16} />
                    Classify
                  </>
                )}
              </button>
            </Card>

            {/* Results */}
            <div className="space-y-4">
              {error && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3"
                >
                  <div className="flex items-center gap-2">
                    <AlertTriangle size={14} className="text-red-400 shrink-0" />
                    <p className="text-sm text-red-300">{error}</p>
                  </div>
                </motion.div>
              )}

              {result ? (
                <>
                  {/* Prediction */}
                  <Card variant="glass">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-sm font-medium text-white">Prediction</h3>
                      <div className="flex items-center gap-2">
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
                        <span className="text-4xl font-light text-white font-mono">
                          {result.prediction}
                        </span>
                      </div>
                      <div>
                        <p className="text-2xl font-mono font-light text-white">
                          {(result.confidence * 100).toFixed(1)}%
                        </p>
                        <p className="text-2xs text-helix-muted">confidence</p>
                      </div>
                      <div className="ml-auto">
                        <CheckCircle size={24} className="text-green-400" />
                      </div>
                    </div>
                  </Card>

                  {/* Probability Distribution */}
                  <Card variant="default">
                    <h3 className="text-sm font-medium text-white mb-4">Probability Distribution</h3>
                    <ProbabilityBars
                      probabilities={result.probabilities}
                      prediction={result.prediction}
                    />
                  </Card>
                </>
              ) : (
                <Card variant="default">
                  <div className="flex flex-col items-center justify-center py-16">
                    <Sparkles size={24} className="text-helix-dim mb-3" />
                    <p className="text-sm text-helix-text2">Draw a digit and click Classify</p>
                    <p className="text-2xs text-helix-muted mt-1">
                      The model will predict which digit (0-9) you drew
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
