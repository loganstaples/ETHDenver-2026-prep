'use client';

import { useState, useMemo, useEffect, useCallback, useRef, Suspense } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { motion, AnimatePresence, useMotionValue, useSpring } from 'framer-motion';
import {
  Play,
  Loader2,
  CheckCircle,
  XCircle,
  AlertTriangle,
  AlertCircle,
  Shield,
  Zap,
  Activity,
  Wifi,
  WifiOff,
  Upload,
  Download,
  HardDrive,
  ExternalLink,
  Copy,
  ChevronDown,
  Link2,
  Layers,
  Search,
  Lock,
  History,
  Clock,
} from 'lucide-react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip as RechartsTooltip,
} from 'recharts';
import { useSignMessage, useAccount, useChainId, useWriteContract, useWaitForTransactionReceipt, useSwitchChain } from 'wagmi';
import { parseEther, keccak256, toBytes, decodeEventLog } from 'viem';
import { hardhat } from 'wagmi/chains';
import { Badge } from '@/components/ui/Badge';
import { CheaterToast } from '@/components/ui/CheaterToast';
import { cn } from '@/lib/utils';
import {
  useMpcTraining,
  type TrainingJobConfig,
  type UploadedData,
  type UploadedWeights,
  type ZeroGStorageResult,
  type TrainingSessionState,
} from '@/hooks/useMpcTraining';
import { useModelRegistry, type ModelWithVersions } from '@/hooks/useModelRegistry';
import { deriveModelKey, decryptWeights } from '@/lib/model-encryption';
import { HELIX_COORDINATOR_V4_ABI, getContractAddress } from '@/lib/contracts';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

function formatDuration(secs: number): string {
  if (secs < 60) return `${secs}s`;
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

function computeArchitectureHash(dims: number[]): `0x${string}` {
  const archString = `HELIX_ARCH:${dims[0]}:${dims[1]}:${dims[2]}`;
  return keccak256(toBytes(archString));
}

// ============================================================================
// Constants
// ============================================================================

const PHASE_DESCRIPTIONS: Record<number, string> = {
  1: 'Loading training data',
  2: 'Initializing model weights',
  3: 'Deploying contracts',
  4: 'Deploying coordinator',
  5: 'Registering training job',
  6: 'Staking workers',
  7: 'Preparing workers',
  8: 'Running MPC training',
  9: 'Submitting checkpoints',
  10: 'Verifying MACs',
  11: 'Completing on-chain',
  12: 'Evaluating accuracy',
  13: 'Complete',
};

const TOTAL_PHASES = 13;

// Per-step MPC operations — these all actually happen every training step.
// Keyed to step number (not a timer) so they advance with real progress.
const MPC_STEP_OPERATIONS = [
  'Secret-shared forward pass — layer 1 matmul',
  'Garbled-circuit ReLU activation via OT',
  'Secret-shared forward pass — layer 2 matmul',
  'Computing cross-entropy loss',
  'MPC backward pass — gradient computation',
  'Applying gradient update with LR decay',
];

type WeightFetchStatus = 'idle' | 'fetching' | 'decrypting' | 'uploading' | 'done' | 'error';

// ============================================================================
// Version & Training History
// ============================================================================

const VERSION_REGEX = /^\d+(\.\d+){0,2}$/;
function isValidVersion(v: string): boolean { return VERSION_REGEX.test(v.trim()); }

function toSlug(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}


// ============================================================================
// Shared micro-components
// ============================================================================

function Toggle({ on, onToggle, disabled }: { on: boolean; onToggle: () => void; disabled?: boolean }) {
  return (
    <button
      type="button"
      onClick={onToggle}
      disabled={disabled}
      className={cn(
        'relative w-11 h-6 rounded-full transition-colors shrink-0',
        on ? 'bg-white' : 'bg-helix-border2',
        disabled && 'opacity-40 cursor-not-allowed',
      )}
    >
      <motion.div
        className={cn('absolute top-1 w-4 h-4 rounded-full', on ? 'bg-black' : 'bg-helix-muted')}
        animate={{ left: on ? 24 : 4 }}
        transition={{ type: 'spring', stiffness: 500, damping: 35 }}
      />
    </button>
  );
}

function NumberInput({ value, onChange, ...rest }: { value: number; onChange: (v: number) => void } & Omit<React.InputHTMLAttributes<HTMLInputElement>, 'value' | 'onChange'>) {
  return (
    <input
      type="number"
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      onWheel={(e) => e.currentTarget.blur()}
      className="w-full bg-transparent text-right text-base text-white outline-none placeholder:text-helix-dim [-moz-appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
      {...rest}
    />
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-helix-surface2 border border-helix-border2 rounded-xl px-3 py-2 text-xs shadow-xl">
      <p className="text-helix-dim mb-0.5">Step {label}</p>
      {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
      {payload.map((entry: any, i: number) => (
        <p key={i} className={cn('font-medium', entry.name === 'Accuracy' ? 'text-green-400' : 'text-white')}>
          {entry.name === 'Accuracy' && typeof entry.value === 'number'
            ? `${(entry.value * 100).toFixed(1)}%`
            : typeof entry.value === 'number'
              ? entry.value.toFixed(4)
              : entry.value}
        </p>
      ))}
    </div>
  );
}

function PhaseRing({ phase, totalPhases, size = 180 }: { phase: number; totalPhases: number; size?: number }) {
  const pad = 20;
  const full = size + pad * 2;
  const strokeWidth = 6;
  const radius = (size - strokeWidth) / 2;
  const center = full / 2;
  const arcDeg = 270;
  const circumference = 2 * Math.PI * radius;
  const arcLength = (arcDeg / 360) * circumference;
  const progress = Math.min(100, (phase / totalPhases) * 100);
  const filledLength = (progress / 100) * arcLength;
  const rotation = 135;
  const yOffset = size * 0.08;
  const filterId = `phase-glow-${size}`;

  return (
    <svg
      width={full}
      height={full}
      viewBox={`0 0 ${full} ${full}`}
      style={{ margin: -pad, marginTop: -pad + yOffset, overflow: 'visible' }}
    >
      <defs>
        <filter id={filterId} x="-100%" y="-100%" width="400%" height="400%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="4" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <circle
        cx={center} cy={center} r={radius}
        fill="none" stroke="#1e1e22" strokeWidth={strokeWidth}
        strokeDasharray={`${arcLength} ${circumference}`}
        strokeLinecap="round"
        transform={`rotate(${rotation} ${center} ${center})`}
      />
      {progress > 0 && (
        <circle
          cx={center} cy={center} r={radius}
          fill="none" stroke="white" strokeWidth={strokeWidth}
          strokeDasharray={`${filledLength} ${circumference}`}
          strokeLinecap="round"
          transform={`rotate(${rotation} ${center} ${center})`}
          filter={`url(#${filterId})`}
          style={{ transition: 'stroke-dasharray 0.6s ease-out' }}
        />
      )}
    </svg>
  );
}

function SubStatusTicker({ session }: { session: TrainingSessionState }) {
  // During MPC training (phase 8), derive specific descriptions from real state
  const text = useMemo(() => {
    if (session.phase !== 8 || session.current_step === 0) {
      return session.phase_description || PHASE_DESCRIPTIONS[session.phase] || 'Processing...';
    }

    // Show real event-driven status when notable things happen
    if (session.cheater_detected) {
      return `Cheater detected — worker ${session.cheater_detected.party_index} at step ${session.cheater_detected.step}`;
    }

    // Cycle through per-step MPC operations keyed to step number (not a timer).
    // These operations actually happen every step; the rotation reflects real progress.
    const workers = session.workers_active ?? 3;
    const opIndex = session.current_step % MPC_STEP_OPERATIONS.length;
    const base = MPC_STEP_OPERATIONS[opIndex];

    // Augment with real metrics
    if (session.mac_checks_passed > 0 && session.current_step % 5 === 0) {
      return `Verifying SPDZ MACs — ${session.mac_checks_passed} checks passed across ${workers} parties`;
    }
    if (session.checkpoints_submitted > 0 && session.current_step % 50 === 0) {
      return `Checkpoint ${session.checkpoints_submitted} attested on-chain — Pedersen commitment`;
    }

    return `${base} — ${workers} workers`;
  }, [session.phase, session.current_step, session.phase_description, session.cheater_detected,
      session.mac_checks_passed, session.checkpoints_submitted, session.workers_active]);

  return (
    <AnimatePresence mode="wait">
      <motion.span
        key={text}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -6 }}
        transition={{ duration: 0.2 }}
        className="text-sm text-helix-muted"
      >
        {text}
      </motion.span>
    </AnimatePresence>
  );
}

function AnimatedNumber({ value, className }: { value: number; className?: string }) {
  const motionVal = useMotionValue(value);
  const spring = useSpring(motionVal, { stiffness: 300, damping: 30 });
  const [display, setDisplay] = useState(value);

  useEffect(() => { motionVal.set(value); }, [value, motionVal]);
  useEffect(() => {
    const unsubscribe = spring.on('change', (v) => setDisplay(Math.round(v)));
    return unsubscribe;
  }, [spring]);

  return <span className={className}>{display}</span>;
}

// ============================================================================
// SettingRow — clean list row
// ============================================================================

function SettingRow({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between px-5 py-4">
      <div>
        <span className="text-base text-helix-text2">{label}</span>
        {hint && <p className="text-xs text-red-400 mt-0.5">{hint}</p>}
      </div>
      <div className="w-32 flex justify-end">{children}</div>
    </div>
  );
}

// ============================================================================
// UploadRow — minimal file upload
// ============================================================================

function UploadRow({ label, accept, onUpload, uploaded, optional }: {
  label: string;
  accept: string;
  onUpload: (file: File) => Promise<void>;
  uploaded: string | null;
  optional?: boolean;
}) {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  const handleUpload = async (file: File) => {
    setIsUploading(true);
    setUploadError(null);
    setFileName(file.name);
    try {
      await onUpload(file);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Upload failed');
      setFileName(null);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="space-y-1.5">
      <div className={cn(
        'flex items-center justify-between px-5 py-3.5 rounded-2xl border transition-colors',
        uploaded
          ? 'bg-green-500/[0.04] border-green-500/15'
          : uploadError
            ? 'bg-red-500/[0.04] border-red-500/20'
            : 'bg-helix-surface border-helix-border',
      )}>
        <div className="flex items-center gap-3 min-w-0">
          {isUploading ? (
            <Loader2 size={16} className="text-helix-accent shrink-0 animate-spin" />
          ) : uploaded ? (
            <CheckCircle size={16} className="text-green-400 shrink-0" />
          ) : uploadError ? (
            <AlertCircle size={16} className="text-red-400 shrink-0" />
          ) : (
            <Upload size={16} className="text-helix-muted shrink-0" />
          )}
          <div className="min-w-0">
            <p className={cn('text-base truncate', uploaded ? 'text-green-300' : uploadError ? 'text-red-300' : 'text-helix-text2')}>
              {isUploading ? 'Uploading…' : label}
              {optional && !uploaded && !isUploading && <span className="text-helix-dim ml-1.5">optional</span>}
            </p>
            {uploaded && (
              <p className="text-xs text-green-400/60 truncate">{fileName ? `${fileName} · ${uploaded}` : uploaded}</p>
            )}
          </div>
        </div>
        <label className={cn(
          'shrink-0 px-3.5 py-1.5 rounded-xl text-xs transition-colors',
          isUploading
            ? 'bg-white/[0.04] text-helix-dim cursor-wait'
            : 'bg-white/[0.06] text-helix-text2 hover:text-white hover:bg-white/[0.1] cursor-pointer',
        )}>
          {isUploading ? 'Uploading…' : uploaded ? 'Replace' : 'Upload'}
          <input type="file" accept={accept} className="hidden" disabled={isUploading} onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleUpload(file);
          }} />
        </label>
      </div>
      {uploadError && (
        <p className="text-xs text-red-400 px-5">{uploadError}</p>
      )}
    </div>
  );
}

// ============================================================================
// PaymentNumber — Cash App style animated payment display
// ============================================================================

function PaymentNumber({
  value,
  color,
  editable,
  onChange,
}: {
  value: number;
  color: 'green' | 'yellow' | 'red';
  editable: boolean;
  onChange?: (v: number) => void;
}) {
  const [display, setDisplay] = useState(value);
  const [editText, setEditText] = useState(value.toFixed(4));
  const motionVal = useMotionValue(value);
  const spring = useSpring(motionVal, { stiffness: 80, damping: 20, mass: 0.5 });
  const wasEditable = useRef(editable);

  useEffect(() => {
    if (!editable) motionVal.set(value);
  }, [value, editable, motionVal]);

  useEffect(() => {
    if (editable) return;
    return spring.on('change', (v) => setDisplay(v));
  }, [spring, editable]);

  // When entering edit mode, seed editText with the formatted value
  useEffect(() => {
    if (editable && !wasEditable.current) {
      setEditText(value.toFixed(4));
    }
    wasEditable.current = editable;
  }, [editable, value]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const text = e.target.value;
    setEditText(text);
    const num = parseFloat(text);
    if (!isNaN(num) && num >= 0) {
      onChange?.(num);
    }
  };

  const colorClass = color === 'green' ? 'text-green-400'
    : color === 'yellow' ? 'text-yellow-400'
    : 'text-red-400';

  const numberClass = cn(
    'text-7xl font-bold tracking-tighter tabular-nums transition-colors duration-500',
    colorClass,
  );

  const displayText = editable ? editText : display.toFixed(4);

  return (
    <div className="flex items-baseline justify-center gap-4">
      <div className="relative">
        {/* Always-visible span sets the size — never swapped out */}
        <span className={cn(numberClass, 'invisible')} aria-hidden>
          {displayText}
        </span>
        {editable ? (
          <input
            type="text"
            inputMode="decimal"
            value={editText}
            onChange={handleChange}
            className={cn(numberClass, 'absolute inset-0 bg-transparent text-center outline-none p-0 border-0 w-full h-full')}
          />
        ) : (
          <span className={cn(numberClass, 'absolute inset-0')}>
            {displayText}
          </span>
        )}
      </div>
      <span className={cn('text-2xl font-semibold transition-colors duration-500', colorClass)}>
        ADI
      </span>
    </div>
  );
}

// ============================================================================
// Config Form — two-column full-width layout
// ============================================================================

interface ConfigFormProps {
  onStart: (config: TrainingJobConfig, opts: { storeOn0G: boolean; version: string; modelName: string; modelSlug: string }) => void;
  isStarting: boolean;
  onUploadData: (file: File) => Promise<void>;
  onUploadWeights: (file: File) => Promise<void>;
  uploadedData: UploadedData | null;
  uploadedWeights: UploadedWeights | null;
  workersOnline: number;
  defaultVersion: string;
  models: ModelWithVersions[];
  selectedModelId: number | null;
  onSelectModel: (tokenId: number | null) => void;
  isFetchingWeights: boolean;
  fetchedModelName: string | null;
  isWalletPrompting: boolean;
  isConfirmingPayment: boolean;
  walletConnected: boolean;
}

function ConfigForm({
  onStart, isStarting, onUploadData, onUploadWeights,
  uploadedData, uploadedWeights, workersOnline, defaultVersion,
  models, selectedModelId, onSelectModel, isFetchingWeights, fetchedModelName,
  isWalletPrompting, isConfirmingPayment, walletConnected,
}: ConfigFormProps) {
  const [numSteps, setNumSteps] = useState(200);
  const [learningRate, setLearningRate] = useState(0.05);
  const [checkpointFreq] = useState(50);
  const [zkMode, setZkMode] = useState<'off' | 'always' | 'risk'>('off');
  const [zkCheckpointFreq, setZkCheckpointFreq] = useState(5);
  const [minWorkersForMpc, setMinWorkersForMpc] = useState(2);
  const [paymentEth, setPaymentEth] = useState(0.0001);
  const [stakePerWorkerEth, setStakePerWorkerEth] = useState(0.001);
  const [storeOn0G, setStoreOn0G] = useState(false);
  const [version, setVersion] = useState(defaultVersion);
  const [versionError, setVersionError] = useState<string | null>(null);
  const [autoPropose, setAutoPropose] = useState(true);
  const [simulateCheater, setSimulateCheater] = useState(false);
  const [cheaterParty, setCheaterParty] = useState<number | undefined>(undefined);
  const [cheaterStep, setCheaterStep] = useState<number | undefined>(undefined);

  // Model identity
  const [modelMode, setModelMode] = useState<'new' | 'existing'>(selectedModelId !== null ? 'existing' : 'new');
  const [modelName, setModelName] = useState('');
  const [modelSlug, setModelSlug] = useState('');
  const [modelSearch, setModelSearch] = useState('');

  const handleModelNameChange = (name: string) => {
    setModelName(name);
    setModelSlug(toSlug(name));
  };

  const filteredModels = useMemo(() => {
    if (!modelSearch.trim()) return models;
    const q = modelSearch.toLowerCase();
    return models.filter(
      (m) => m.name.toLowerCase().includes(q) || (m.slug && m.slug.toLowerCase().includes(q)),
    );
  }, [models, modelSearch]);

  // Auto-compute recommended payment: base rate × steps × workers × ZK overhead
  const recommendedPayment = useMemo(() => {
    const workers = workersOnline > 0 ? workersOnline : 3;
    const zkMult = zkMode === 'always' ? 1.75 : zkMode === 'risk' ? 1.25 : 1.0;
    return parseFloat((0.0000001 * numSteps * workers * zkMult).toFixed(6));
  }, [numSteps, workersOnline, zkMode]);

  const effectivePayment = autoPropose ? recommendedPayment : paymentEth;

  const paymentColor = (() => {
    if (recommendedPayment <= 0) return 'green' as const;
    const ratio = effectivePayment / recommendedPayment;
    if (ratio >= 0.9) return 'green' as const;
    if (ratio >= 0.6) return 'yellow' as const;
    return 'red' as const;
  })();

  const handleToggleAutoPropose = () => {
    if (autoPropose) setPaymentEth(recommendedPayment);
    setAutoPropose(!autoPropose);
  };

  const handleVersionChange = (v: string) => {
    setVersion(v);
    setVersionError(v.trim() && !isValidVersion(v) ? 'Must be x.y.z format' : null);
  };

  const handleSubmit = () => {
    const v = version.trim() || defaultVersion;
    if (!isValidVersion(v)) return;

    const effectiveModelName = modelMode === 'existing' && selectedModel
      ? selectedModel.name
      : modelName.trim();
    const effectiveModelSlug = modelMode === 'existing' && selectedModel
      ? (selectedModel.slug || '')
      : modelSlug.trim();

    onStart({
      architecture: [784, 128, 10],
      num_workers: workersOnline > 0 ? workersOnline : 3,
      num_steps: numSteps,
      learning_rate: learningRate,
      checkpoint_freq: checkpointFreq,
      mac_interval: 0,
      zk_mode: zkMode,
      zk_checkpoint_freq: zkCheckpointFreq,
      min_workers_for_mpc: minWorkersForMpc,
      train_size: 5000,
      test_size: 500,
      use_real_mnist: true,
      payment_eth: effectivePayment,
      stake_per_worker_eth: stakePerWorkerEth,
      simulate_cheater: simulateCheater,
      cheater_party: simulateCheater ? cheaterParty : undefined,
      cheater_step: simulateCheater ? cheaterStep : undefined,
      seed: 42,
      model_name: effectiveModelName || undefined,
      model_slug: effectiveModelSlug || undefined,
      model_token_id: modelMode === 'existing' && selectedModelId !== null ? selectedModelId : undefined,
    }, { storeOn0G, version: v, modelName: effectiveModelName, modelSlug: effectiveModelSlug });
  };

  const selectedModel = models.find((m) => m.tokenId === selectedModelId) ?? null;
  const latestVersion = selectedModel?.versions?.length
    ? selectedModel.versions[selectedModel.versions.length - 1]
    : null;

  const canStart = !isStarting && !isWalletPrompting && !isConfirmingPayment
    && !versionError && !isFetchingWeights && walletConnected
    && (modelMode === 'existing'
      ? selectedModelId !== null
      : (modelName.trim() !== '' && modelSlug.trim() !== ''));

  return (
    <div className="space-y-6">
      {/* ── Header ────────────────────────────────────────────── */}
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-4xl font-semibold tracking-tight text-white">
            New Training Run
          </h1>
          <p className="text-base text-helix-muted mt-1">
            MNIST 784 → 128 → 10 · ~102K params · Verifiable MPC
          </p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/[0.04] border border-white/[0.06]">
          <span className={cn('w-2 h-2 rounded-full', workersOnline >= 2 ? 'bg-green-400 animate-pulse' : 'bg-helix-dim')} />
          <span className="text-sm text-helix-text2">
            {workersOnline} worker{workersOnline !== 1 ? 's' : ''}
          </span>
        </div>
      </div>

      {/* ── Two-column grid ──────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* ── LEFT COLUMN ──────────────────────────────────────── */}
        <div className="flex flex-col gap-5">

          {/* Model Identity */}
          <div className="space-y-3">
            {/* Segmented toggle */}
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => { setModelMode('new'); onSelectModel(null); }}
                className={cn(
                  'py-2.5 rounded-xl text-base font-medium transition-all',
                  modelMode === 'new'
                    ? 'bg-white text-black shadow-lg shadow-white/5'
                    : 'bg-helix-surface border border-helix-border text-helix-muted hover:text-white hover:border-helix-border2',
                )}
              >
                New Model
              </button>
              <button
                type="button"
                onClick={() => models.length > 0 && setModelMode('existing')}
                disabled={models.length === 0}
                className={cn(
                  'py-2.5 rounded-xl text-base font-medium transition-all',
                  modelMode === 'existing'
                    ? 'bg-white text-black shadow-lg shadow-white/5'
                    : 'bg-helix-surface border border-helix-border text-helix-muted hover:text-white hover:border-helix-border2',
                  models.length === 0 && 'opacity-40 cursor-not-allowed',
                )}
              >
                Continue Existing
              </button>
            </div>

            <AnimatePresence mode="wait">
              {modelMode === 'new' ? (
                <motion.div
                  key="new-model"
                  initial={{ opacity: 0, y: -4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -4 }}
                  className="space-y-3"
                >
                  {/* Model Name + read-only slug */}
                  <div className="grid grid-cols-2 gap-2">
                    <input
                      type="text"
                      value={modelName}
                      onChange={(e) => handleModelNameChange(e.target.value)}
                      placeholder="Model Name"
                      className="min-w-0 px-4 py-3.5 bg-helix-surface border border-helix-border rounded-2xl text-base text-white placeholder:text-helix-dim focus:outline-none focus:border-helix-border2 transition-colors"
                    />
                    <div className="min-w-0 px-4 py-3.5 bg-black border border-white/20 rounded-2xl text-base text-helix-muted select-none truncate">
                      {modelSlug || '\u00A0'}
                    </div>
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  key="existing-model"
                  initial={{ opacity: 0, y: -4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -4 }}
                  className="space-y-3"
                >
                  {selectedModel ? (
                    <>
                      {/* Selected model — locked display */}
                      <div className="flex items-center gap-3 px-4 py-3.5 rounded-2xl bg-white/[0.03] border border-white/[0.08]">
                        <Lock size={14} className="text-helix-muted shrink-0" />
                        <div className="flex-1 min-w-0">
                          <p className="text-base text-white font-medium truncate">{selectedModel.name}</p>
                          <p className="text-xs text-helix-dim truncate">
                            {selectedModel.slug}
                            {latestVersion && <span className="ml-1.5">· v{latestVersion.semver}</span>}
                          </p>
                        </div>
                        <button
                          type="button"
                          onClick={() => onSelectModel(null)}
                          className="text-xs text-helix-dim hover:text-white transition-colors shrink-0"
                        >
                          Change
                        </button>
                      </div>

                      {/* Weight fetch status */}
                      <AnimatePresence mode="wait">
                        {isFetchingWeights && (
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
                        {!isFetchingWeights && fetchedModelName && uploadedWeights && (
                          <motion.div
                            key="loaded"
                            initial={{ opacity: 0, y: -4 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -4 }}
                            className="flex items-center gap-3 px-4 py-3 rounded-2xl bg-green-500/[0.06] border border-green-500/20"
                          >
                            <CheckCircle size={14} className="text-green-400" />
                            <span className="text-sm text-green-300">
                              <span className="font-medium">{fetchedModelName}</span>
                              <span className="text-green-400/60 ml-1.5">
                                {uploadedWeights.totalParams.toLocaleString()} params
                              </span>
                            </span>
                          </motion.div>
                        )}
                        {selectedModel && latestVersion && !latestVersion.weightsStored && !isFetchingWeights && (
                          <motion.div
                            key="no-weights"
                            initial={{ opacity: 0, y: -4 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -4 }}
                            className="flex items-center gap-3 px-4 py-3 rounded-2xl bg-yellow-500/[0.06] border border-yellow-500/20"
                          >
                            <AlertTriangle size={14} className="text-yellow-400" />
                            <span className="text-sm text-yellow-300/80">No stored weights — upload manually or start fresh</span>
                          </motion.div>
                        )}
                        {selectedModel && !latestVersion && !isFetchingWeights && (
                          <motion.div
                            key="no-versions"
                            initial={{ opacity: 0, y: -4 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -4 }}
                            className="flex items-center gap-3 px-4 py-3 rounded-2xl bg-yellow-500/[0.06] border border-yellow-500/20"
                          >
                            <AlertTriangle size={14} className="text-yellow-400" />
                            <span className="text-sm text-yellow-300/80">No versions yet — upload weights or start fresh</span>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </>
                  ) : (
                    <>
                      {/* Search input */}
                      <div className="relative">
                        <Search size={14} className="absolute left-4 top-1/2 -translate-y-1/2 text-helix-muted pointer-events-none" />
                        <input
                          type="text"
                          value={modelSearch}
                          onChange={(e) => setModelSearch(e.target.value)}
                          placeholder="Search your models..."
                          className="w-full pl-10 pr-4 py-3.5 bg-helix-surface border border-helix-border rounded-2xl text-base text-white placeholder:text-helix-dim focus:outline-none focus:border-helix-border2 transition-colors"
                        />
                      </div>

                      {/* Model list */}
                      <div className="max-h-48 overflow-y-auto space-y-1.5 rounded-2xl">
                        {filteredModels.map((m) => {
                          const latest = m.versions.length ? m.versions[m.versions.length - 1] : null;
                          return (
                            <button
                              key={m.tokenId}
                              type="button"
                              onClick={() => onSelectModel(m.tokenId)}
                              className="w-full flex items-center gap-3 px-4 py-3 rounded-xl bg-helix-surface border border-helix-border hover:border-helix-border2 transition-colors text-left"
                            >
                              <Layers size={14} className="text-helix-muted shrink-0" />
                              <div className="flex-1 min-w-0">
                                <p className="text-base text-white truncate">{m.name}</p>
                                <p className="text-xs text-helix-dim truncate">
                                  {m.slug}
                                  {latest && <span className="ml-1.5">· v{latest.semver}</span>}
                                </p>
                              </div>
                              <ChevronDown size={12} className="text-helix-dim -rotate-90 shrink-0" />
                            </button>
                          );
                        })}
                        {filteredModels.length === 0 && (
                          <p className="text-sm text-helix-dim text-center py-4">No models found</p>
                        )}
                      </div>
                    </>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Core training settings */}
          <div className="rounded-2xl bg-helix-surface border border-helix-border overflow-hidden divide-y divide-helix-border">
            <SettingRow label="Version" hint={versionError ?? undefined}>
              <input
                type="text"
                value={version}
                onChange={(e) => handleVersionChange(e.target.value)}
                placeholder={defaultVersion}
                className={cn(
                  'bg-transparent text-right text-base outline-none placeholder:text-helix-dim w-24',
                  versionError ? 'text-red-400' : 'text-white',
                )}
              />
            </SettingRow>

            <SettingRow label="Training Steps">
              <NumberInput value={numSteps} onChange={setNumSteps} min={10} max={10000} />
            </SettingRow>

            <SettingRow label="Learning Rate">
              <NumberInput value={learningRate} onChange={setLearningRate} step={0.0001} min={0.00001} max={1} />
            </SettingRow>

            <SettingRow label="Stake / Worker">
              <div className="flex items-center gap-1.5">
                <NumberInput value={stakePerWorkerEth} onChange={setStakePerWorkerEth} step={0.01} min={0} />
                <span className="text-sm text-helix-dim shrink-0">ADI</span>
              </div>
            </SettingRow>
          </div>

          {/* Uploads (optional — MNIST is loaded automatically) */}
          <div className="space-y-3">
            <UploadRow
              label="Training Data (optional — MNIST auto-loaded)"
              accept=".json,.csv,.idx,.idx3-ubyte,.idx1-ubyte,.gz"
              onUpload={onUploadData}
              uploaded={uploadedData ? `${uploadedData.samples} samples · ${uploadedData.inputDim}D` : null}
              optional
            />
            <UploadRow
              label={fetchedModelName ? `Weights · ${fetchedModelName}` : 'Initial Weights'}
              accept=".json"
              onUpload={onUploadWeights}
              uploaded={uploadedWeights ? `${uploadedWeights.totalParams.toLocaleString()} params` : null}
              optional
            />
          </div>
        </div>

        {/* ── RIGHT COLUMN ─────────────────────────────────────── */}
        <div className="flex flex-col gap-5">

          {/* Payment + Start — Cash App style, connected */}
          <div className="rounded-2xl bg-helix-surface border border-helix-border overflow-hidden">
            <div className="p-6">
              {/* Header with auto toggle */}
              <div className="flex items-center justify-between mb-5">
                <span className="text-base font-medium text-helix-text2">Payment</span>
                <div className="flex items-center gap-2.5">
                  <span className="text-sm text-helix-dim">Auto</span>
                  <Toggle on={autoPropose} onToggle={handleToggleAutoPropose} />
                </div>
              </div>

              {/* Big number — same component for both modes */}
              <div className="py-3">
                <PaymentNumber
                  value={autoPropose ? recommendedPayment : paymentEth}
                  color={paymentColor}
                  editable={!autoPropose}
                  onChange={setPaymentEth}
                />
              </div>

              {/* Footer — fixed height, breakdown slides horizontally, rec fades in */}
              <div className="relative h-5 mt-4">
                <motion.span
                  className="absolute top-0 text-sm text-helix-dim whitespace-nowrap"
                  animate={{ left: autoPropose ? '50%' : 0, x: autoPropose ? '-50%' : 0 }}
                  transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                >
                  {workersOnline > 0 ? workersOnline : 3} workers × {numSteps} steps
                  {zkMode !== 'off' && ` × ${zkMode === 'always' ? '1.75' : '1.25'}× ZK`}
                </motion.span>
                <AnimatePresence>
                  {!autoPropose && (
                    <motion.span
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.2 }}
                      className="absolute top-0 right-0 text-sm text-helix-dim whitespace-nowrap"
                    >
                      Rec: {recommendedPayment.toFixed(4)} ADI
                    </motion.span>
                  )}
                </AnimatePresence>
              </div>
            </div>

            {/* Start button — attached to payment card */}
            <motion.button
              type="button"
              onClick={handleSubmit}
              disabled={!canStart}
              whileTap={canStart ? { scale: 0.98 } : {}}
              className={cn(
                'w-full py-5 text-xl font-bold tracking-tight transition-all',
                'flex items-center justify-center gap-3 border-t',
                canStart
                  ? 'bg-zinc-200 text-black border-zinc-200 hover:bg-zinc-300 active:bg-zinc-400'
                  : 'bg-helix-border text-helix-muted border-helix-border cursor-not-allowed',
              )}
            >
              {!walletConnected ? (
                <><Lock size={22} /> Connect Wallet</>
              ) : isWalletPrompting ? (
                <><Loader2 size={22} className="animate-spin" /> Confirm in Wallet...</>
              ) : isConfirmingPayment ? (
                <><Loader2 size={22} className="animate-spin" /> Confirming Payment...</>
              ) : isStarting ? (
                <><Loader2 size={22} className="animate-spin" /> Starting...</>
              ) : isFetchingWeights ? (
                <><Loader2 size={22} className="animate-spin" /> Loading Weights...</>
              ) : (
                <><Play size={22} /> Start Training</>
              )}
            </motion.button>
          </div>

          {/* ZK Mode */}
          <div className="space-y-3">
            <span className="text-base font-medium text-helix-text2">ZK Proofs</span>
            <div className="grid grid-cols-3 gap-2">
              {(['off', 'always', 'risk'] as const).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => setZkMode(mode)}
                  className={cn(
                    'py-2.5 rounded-xl text-base font-medium transition-all',
                    zkMode === mode
                      ? 'bg-white text-black shadow-lg shadow-white/5'
                      : 'bg-helix-surface border border-helix-border text-helix-muted hover:text-white hover:border-helix-border2',
                  )}
                >
                  {mode === 'off' ? 'Off' : mode === 'always' ? 'Always' : 'Risk-Based'}
                </button>
              ))}
            </div>

            <AnimatePresence>
              {zkMode === 'always' && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <div className="flex items-center justify-between py-3.5 px-4 bg-helix-surface border border-helix-border rounded-xl">
                    <span className="text-base text-helix-text2">ZK Checkpoint Freq</span>
                    <NumberInput value={zkCheckpointFreq} onChange={setZkCheckpointFreq} min={1} max={100} />
                  </div>
                </motion.div>
              )}
              {zkMode === 'risk' && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <div className="flex items-center justify-between py-3.5 px-4 bg-helix-surface border border-helix-border rounded-xl">
                    <span className="text-base text-helix-text2">Min Workers for MPC</span>
                    <NumberInput value={minWorkersForMpc} onChange={setMinWorkersForMpc} min={2} max={workersOnline > 2 ? workersOnline : 7} />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Simulate Cheater (demo) */}
          <div className="rounded-2xl bg-helix-surface border border-helix-border overflow-hidden">
            <div className="flex items-center justify-between px-5 py-4">
              <div>
                <p className="text-base text-white">Simulate Cheater</p>
                <p className="text-xs text-helix-dim mt-0.5">Inject a byzantine worker for demo</p>
              </div>
              <Toggle on={simulateCheater} onToggle={() => setSimulateCheater(!simulateCheater)} />
            </div>
            <AnimatePresence>
              {simulateCheater && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="overflow-hidden"
                >
                  <div className="px-5 pb-4 space-y-2">
                    <div className="flex items-center justify-between py-3 px-4 bg-red-500/[0.04] border border-red-500/15 rounded-xl">
                      <span className="text-sm text-red-300/80">Cheater Party</span>
                      <NumberInput
                        value={cheaterParty ?? (workersOnline > 0 ? workersOnline - 1 : 2)}
                        onChange={(v) => setCheaterParty(v)}
                        min={0}
                        max={workersOnline > 0 ? workersOnline - 1 : 9}
                      />
                    </div>
                    <div className="flex items-center justify-between py-3 px-4 bg-red-500/[0.04] border border-red-500/15 rounded-xl">
                      <span className="text-sm text-red-300/80">Corrupt at Step</span>
                      <NumberInput
                        value={cheaterStep ?? Math.floor(numSteps / 2)}
                        onChange={(v) => setCheaterStep(v)}
                        min={1}
                        max={numSteps}
                      />
                    </div>
                    <p className="text-[11px] text-red-300/40 px-1">
                      The selected worker will inject corrupt weight shares at the specified step. SPDZ MAC verification will detect and report it.
                    </p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Store on 0G */}
          <div className="rounded-2xl bg-helix-surface border border-helix-border overflow-hidden">
            <div className="flex items-center justify-between px-5 py-4">
              <div>
                <p className="text-base text-white">Store on 0G</p>
                <p className="text-xs text-helix-dim mt-0.5">Save model to decentralized storage</p>
              </div>
              <Toggle on={storeOn0G} onToggle={() => setStoreOn0G(!storeOn0G)} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Live Progress — full-width two-column training view
// ============================================================================

interface LiveProgressProps {
  session: NonNullable<ReturnType<typeof useMpcTraining>['session']>;
  losses: { step: number; loss: number; accuracy?: number }[];
  isConnected: boolean;
  error: string | null;
  version: string;
  modelName: string;
  onDownloadModel: (sessionId: string) => Promise<void>;
  onStoreOnZeroG: (sessionId: string, version?: string) => Promise<void>;
  isStoringOnZeroG: boolean;
  zeroGResult: ZeroGStorageResult | null;
  showStoreOn0G: boolean;
  elapsedTime: number;
}

function LiveProgress({ session, losses, isConnected, error, version, modelName, onDownloadModel, onStoreOnZeroG, isStoringOnZeroG, zeroGResult, showStoreOn0G, elapsedTime }: LiveProgressProps) {
  const stepProgress = session.total_steps > 0
    ? (session.current_step / session.total_steps) * 100
    : 0;
  const isTerminal = session.status === 'complete' || session.status === 'failed';
  const isComplete = session.status === 'complete';

  const lastLoss = losses.length > 0 ? losses[losses.length - 1].loss : 0;
  const prevLoss = losses.length > 10 ? losses[losses.length - 11].loss : lastLoss;
  const lossDelta = lastLoss - prevLoss;

  const lastAcc = losses.length > 0 ? (losses[losses.length - 1].accuracy ?? 0) : 0;
  const prevAcc = losses.length > 10 ? (losses[losses.length - 11].accuracy ?? 0) : lastAcc;
  const accDelta = lastAcc - prevAcc;

  const totalCheckpoints = session.total_steps > 0 ? Math.floor(session.total_steps / 50) : 0;

  return (
    <div className="space-y-6">
      {/* Header bar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {modelName && <h2 className="text-2xl font-semibold tracking-tight text-white">{modelName}</h2>}
          <Badge variant="default">v{version}</Badge>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            {isConnected ? (
              <Wifi size={14} className="text-green-400" />
            ) : (
              <WifiOff size={14} className="text-red-400" />
            )}
            <span className="text-xs text-helix-dim font-mono">{session.session_id.slice(0, 8)}</span>
          </div>
          <div className={cn(
            'flex items-center gap-2 px-3.5 py-1.5 rounded-full text-sm font-medium',
            isComplete ? 'bg-green-500/10 text-green-400'
              : session.status === 'failed' ? 'bg-red-500/10 text-red-400'
              : 'bg-white/[0.06] text-white',
          )}>
            {!isTerminal && <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />}
            {session.status === 'running' ? 'Training' : session.status}
          </div>
        </div>
      </div>

      {error && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="px-4 py-3 rounded-2xl bg-red-500/[0.06] border border-red-500/20">
          <p className="text-sm text-red-300">{error}</p>
        </motion.div>
      )}

      {/* Hero Status Card — during training */}
      {!isTerminal && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative rounded-3xl bg-helix-surface border border-helix-border overflow-hidden"
        >
          <div
            className="absolute top-0 left-0 h-[2px] bg-gradient-to-r from-transparent via-white/30 to-transparent transition-all duration-500"
            style={{ width: `${stepProgress}%` }}
          />

          <div className="p-8">
            <div className="flex items-center gap-8">
              {/* Phase ring */}
              <div className="relative shrink-0 flex items-center justify-center">
                <div className="absolute inset-0 rounded-full bg-white/[0.02] blur-xl scale-110" />
                <PhaseRing phase={session.phase} totalPhases={TOTAL_PHASES} size={160} />
                <div className="absolute inset-0 flex flex-col items-center justify-center" style={{ marginTop: 160 * 0.08 }}>
                  <span className="text-3xl font-bold text-white tabular-nums">{session.phase}</span>
                  <span className="text-[10px] uppercase tracking-widest text-helix-muted mt-0.5">of {TOTAL_PHASES}</span>
                </div>
              </div>

              {/* Right side */}
              <div className="flex-1 min-w-0 space-y-5">
                <div>
                  <h3 className="text-lg font-semibold text-white mb-1">
                    {PHASE_DESCRIPTIONS[session.phase] || 'Processing...'}
                  </h3>
                  <SubStatusTicker session={session} />
                </div>

                <div className="flex items-baseline gap-3">
                  <span className="text-5xl font-bold text-white tabular-nums font-mono tracking-tighter">
                    <AnimatedNumber value={session.current_step} />
                  </span>
                  <span className="text-xl text-helix-dim font-mono">/ {session.total_steps}</span>
                  <span className="text-sm text-helix-muted ml-2">steps</span>
                </div>

                <div className="space-y-1.5">
                  <div className="h-2.5 w-full rounded-full bg-helix-border overflow-hidden">
                    <motion.div
                      className="h-full rounded-full bg-white"
                      animate={{ width: `${stepProgress}%` }}
                      transition={{ duration: 0.4, ease: 'easeOut' }}
                      style={{ boxShadow: stepProgress > 0 && stepProgress < 100 ? '0 0 12px rgba(255,255,255,0.4), 0 0 4px rgba(255,255,255,0.6)' : 'none' }}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-helix-dim font-mono tabular-nums">{stepProgress.toFixed(1)}%</span>
                    <span className="text-xs text-helix-dim">
                      {session.elapsed_secs > 0
                        ? formatDuration(Math.round(session.elapsed_secs))
                        : formatDuration(elapsedTime)} elapsed
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Quick stats — loss & accuracy */}
            <div className="flex items-center gap-6 mt-6 pt-5 border-t border-white/[0.04]">
              <div className="flex items-baseline gap-2">
                <span className="text-[10px] uppercase tracking-wider text-helix-muted">Loss</span>
                <span className="text-lg font-mono font-semibold text-white tabular-nums">
                  {lastLoss > 0 ? lastLoss.toFixed(4) : '\u2014'}
                </span>
                {lastLoss > 0 && (
                  <span className={cn('text-xs font-mono tabular-nums', lossDelta <= 0 ? 'text-green-400' : 'text-red-400')}>
                    {lossDelta <= 0 ? '\u2193' : '\u2191'}{Math.abs(lossDelta).toFixed(4)}
                  </span>
                )}
              </div>
              <div className="w-px h-5 bg-white/[0.06]" />
              <div className="flex items-baseline gap-2">
                <span className="text-[10px] uppercase tracking-wider text-helix-muted">Accuracy</span>
                <span className="text-lg font-mono font-semibold text-white tabular-nums">
                  {lastAcc > 0 ? `${(lastAcc * 100).toFixed(1)}%` : '\u2014'}
                </span>
                {lastAcc > 0 && (
                  <span className={cn('text-xs font-mono tabular-nums', accDelta >= 0 ? 'text-green-400' : 'text-red-400')}>
                    {accDelta >= 0 ? '\u2191' : '\u2193'}{Math.abs(accDelta * 100).toFixed(1)}%
                  </span>
                )}
              </div>
              <div className="w-px h-5 bg-white/[0.06]" />
              <div className="flex items-baseline gap-2">
                <Shield size={12} className="text-green-400/60" />
                <span className="text-xs text-helix-dim font-mono tabular-nums">{session.mac_checks_passed} MACs</span>
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Completion state */}
      {isComplete && session.accuracy !== null && (
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: 'spring', stiffness: 200, damping: 20 }}
          className="relative rounded-3xl bg-helix-surface border border-green-500/20 overflow-hidden"
        >
          <div className="absolute top-0 left-0 w-full h-[2px] bg-gradient-to-r from-transparent via-green-400/40 to-transparent" />
          <div className="p-8 text-center">
            <p className="text-sm text-green-400/60 uppercase tracking-wider mb-3">Training Complete</p>
            <p className="text-7xl font-bold tracking-tighter text-white">
              {(session.accuracy * 100).toFixed(1)}
              <span className="text-3xl text-helix-muted ml-1">%</span>
            </p>
            <p className="text-sm text-helix-muted mt-3">
              {session.current_step} steps · {session.mac_checks_passed} MAC checks passed · {session.checkpoints_submitted} checkpoints
            </p>
          </div>
        </motion.div>
      )}

      {session.status === 'failed' && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="rounded-3xl bg-helix-surface border border-red-500/20 p-8 text-center">
          <XCircle size={48} className="text-red-400 mx-auto mb-3" />
          <p className="text-xl font-medium text-red-300">Training Failed</p>
        </motion.div>
      )}

      {/* Detail grid */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-5">
        {/* Left column — chart & big metrics */}
        <div className="lg:col-span-3 space-y-5">
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-helix-surface border border-helix-border rounded-2xl p-5">
              <div className="text-[10px] uppercase tracking-wider text-helix-muted mb-2">Current Loss</div>
              <div className="flex items-baseline gap-2">
                <span className="text-4xl font-bold font-mono text-white tabular-nums">
                  {lastLoss > 0 ? lastLoss.toFixed(4) : '\u2014'}
                </span>
                {lastLoss > 0 && (
                  <span className={cn('text-sm font-mono tabular-nums', lossDelta <= 0 ? 'text-green-400' : 'text-red-400')}>
                    {lossDelta <= 0 ? '\u2193' : '\u2191'}{Math.abs(lossDelta).toFixed(4)}
                  </span>
                )}
              </div>
            </div>
            <div className="bg-helix-surface border border-helix-border rounded-2xl p-5">
              <div className="text-[10px] uppercase tracking-wider text-helix-muted mb-2">Accuracy</div>
              <div className="flex items-baseline gap-2">
                <span className="text-4xl font-bold font-mono text-white tabular-nums">
                  {lastAcc > 0 ? `${(lastAcc * 100).toFixed(1)}%` : '\u2014'}
                </span>
                {lastAcc > 0 && (
                  <span className={cn('text-sm font-mono tabular-nums', accDelta >= 0 ? 'text-green-400' : 'text-red-400')}>
                    {accDelta >= 0 ? '\u2191' : '\u2193'}{Math.abs(accDelta * 100).toFixed(1)}%
                  </span>
                )}
              </div>
            </div>
          </div>
          <LossCurve data={losses} />
        </div>

        {/* Right column — stats & info */}
        <div className="lg:col-span-2 space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <StatCard label="Workers" value={session.workers_active ?? '\u2014'} icon={<Activity size={14} />} />
            <StatCard label="MAC Checks" value={session.mac_checks_passed} icon={<Shield size={14} />} />
            <StatCard label="Checkpoints" value={`${session.checkpoints_submitted}${totalCheckpoints > 0 ? ` / ${totalCheckpoints}` : ''}`} icon={<Layers size={14} />} />
            <StatCard label="Elapsed" value={session.elapsed_secs > 0 ? formatDuration(Math.round(session.elapsed_secs)) : formatDuration(elapsedTime)} icon={<Clock size={14} />} />
          </div>

          <div className="bg-helix-surface border border-helix-border rounded-2xl p-4">
            <h4 className="text-[10px] uppercase tracking-wider text-helix-muted mb-3">Security Protocol</h4>
            <div className="space-y-2.5">
              {[
                { label: 'SPDZ MAC Verification', active: session.mac_checks_passed > 0 },
                { label: 'Additive Secret Sharing', active: (session.workers_active ?? 0) > 0 },
                { label: 'Zero Weight Leakage', active: true },
                { label: 'Cheater Detection', active: session.phase >= 8 },
              ].map((item) => (
                <div key={item.label} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className={cn('w-1.5 h-1.5 rounded-full', item.active ? 'bg-green-400' : 'bg-helix-muted/40')} />
                    <span className="text-xs text-helix-dim">{item.label}</span>
                  </div>
                  <span className={cn('text-[10px] font-mono', item.active ? 'text-green-400/80' : 'text-helix-muted')}>
                    {item.active ? 'ACTIVE' : 'STANDBY'}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-helix-surface border border-helix-border rounded-2xl p-4">
            <h4 className="text-[10px] uppercase tracking-wider text-helix-muted mb-3">Session</h4>
            <div className="space-y-2">
              {[
                { label: 'ID', value: session.session_id.slice(0, 12) + '\u2026' },
                ...(session.coordinator_address ? [{ label: 'Coordinator', value: session.coordinator_address.slice(0, 8) + '\u2026' + session.coordinator_address.slice(-4) }] : []),
                ...(session.job_id !== undefined ? [{ label: 'Job ID', value: String(session.job_id) }] : []),
                { label: 'Architecture', value: '784 \u2192 32 \u2192 10' },
              ].map((item) => (
                <div key={item.label} className="flex items-center justify-between">
                  <span className="text-xs text-helix-muted">{item.label}</span>
                  <span className="text-xs text-helix-dim font-mono">{item.value}</span>
                </div>
              ))}
            </div>
          </div>

          {session.cheater_detected && (
            <motion.div
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center gap-3 px-4 py-3.5 rounded-2xl bg-red-500/[0.08] border border-red-500/25"
            >
              <AlertTriangle size={16} className="text-red-400 shrink-0" />
              <div>
                <p className="text-sm font-medium text-red-300">Cheater Detected</p>
                <p className="text-xs text-red-300/60 mt-0.5">
                  Worker {session.cheater_detected.party_index} · Step {session.cheater_detected.step} · Stake slashed
                </p>
              </div>
            </motion.div>
          )}
        </div>
      </div>

      {/* Results actions */}
      {isTerminal && (
        <ResultsActions
          session={session}
          version={version}
          onDownloadModel={onDownloadModel}
          onStoreOnZeroG={onStoreOnZeroG}
          isStoringOnZeroG={isStoringOnZeroG}
          zeroGResult={zeroGResult}
          showStoreOn0G={showStoreOn0G}
        />
      )}
    </div>
  );
}

// ============================================================================
// StatCard — clean metric card
// ============================================================================

function StatCard({ label, value, icon }: { label: string; value: string | number; icon: React.ReactNode | null }) {
  return (
    <div className="bg-helix-surface border border-helix-border rounded-2xl px-4 py-3.5">
      <div className="flex items-center gap-1.5 mb-1.5">
        {icon && <span className="text-helix-dim">{icon}</span>}
        <span className="text-xs text-helix-dim uppercase tracking-wider">{label}</span>
      </div>
      <p className="text-2xl font-semibold text-white tabular-nums">{value}</p>
    </div>
  );
}

// ============================================================================
// Loss Curve — area chart
// ============================================================================

function LossCurve({ data }: { data: { step: number; loss: number; accuracy?: number }[] }) {
  const chartData = useMemo(() => {
    if (data.length === 0) return [];
    // EMA smoothing — alpha high enough to track reality, low enough to remove per-step noise
    const lossAlpha = 0.35;
    const accAlpha = 0.3;
    let lossEma = data[0].loss;
    let accEma = (data[0].accuracy ?? 0) * 100;
    const smoothed = data.map((d) => {
      lossEma = lossAlpha * d.loss + (1 - lossAlpha) * lossEma;
      const rawAcc = (d.accuracy ?? 0) * 100;
      accEma = accAlpha * rawAcc + (1 - accAlpha) * accEma;
      return { step: d.step, loss: lossEma, accuracy: accEma / 100 };
    });
    if (smoothed.length <= 200) return smoothed;
    const step = Math.ceil(smoothed.length / 200);
    return smoothed.filter((_, i) => i % step === 0 || i === smoothed.length - 1);
  }, [data]);

  const hasAccuracy = chartData.some(d => d.accuracy !== undefined && d.accuracy > 0);

  if (chartData.length < 2) {
    return (
      <div className="flex items-center justify-center h-48 rounded-2xl bg-helix-surface border border-helix-border text-helix-dim text-sm">
        Waiting for data...
      </div>
    );
  }

  // Use raw (unsmoothed) last point for headline numbers
  const lastPoint = data[data.length - 1];

  return (
    <div className="rounded-2xl bg-helix-surface border border-helix-border p-5 pt-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-4">
          <span className="text-sm text-helix-muted">Loss</span>
          {hasAccuracy && (
            <span className="text-sm text-green-400/70">Accuracy</span>
          )}
        </div>
        <div className="flex items-center gap-4">
          <span className="text-sm text-helix-dim tabular-nums">
            {lastPoint.loss.toFixed(4)}
          </span>
          {hasAccuracy && lastPoint.accuracy !== undefined && (
            <span className="text-sm text-green-400/60 tabular-nums">
              {(lastPoint.accuracy * 100).toFixed(1)}%
            </span>
          )}
        </div>
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={chartData}>
          <defs>
            <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="rgba(255,255,255,0.12)" />
              <stop offset="100%" stopColor="rgba(255,255,255,0)" />
            </linearGradient>
            <linearGradient id="accGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="rgba(74,222,128,0.15)" />
              <stop offset="100%" stopColor="rgba(74,222,128,0)" />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="step"
            stroke="transparent"
            tick={{ fill: '#3e3e44', fontSize: 11 }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            yAxisId="loss"
            stroke="transparent"
            tick={{ fill: '#3e3e44', fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={40}
            tickFormatter={(v: number) => v.toFixed(2)}
          />
          {hasAccuracy && (
            <YAxis
              yAxisId="accuracy"
              orientation="right"
              stroke="transparent"
              tick={{ fill: '#4ade80', fontSize: 11, opacity: 0.5 }}
              tickLine={false}
              axisLine={false}
              width={40}
              domain={[0, 1]}
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            />
          )}
          <RechartsTooltip content={<ChartTooltip />} cursor={{ stroke: '#2a2a2e', strokeWidth: 1 }} />
          <Area
            yAxisId="loss"
            type="monotone"
            dataKey="loss"
            name="Loss"
            stroke="#ffffff"
            strokeWidth={1.5}
            fill="url(#lossGrad)"
            dot={false}
            activeDot={{ r: 3, fill: '#ffffff', stroke: '#111113', strokeWidth: 2 }}
            isAnimationActive={false}
          />
          {hasAccuracy && (
            <Area
              yAxisId="accuracy"
              type="monotone"
              dataKey="accuracy"
              name="Accuracy"
              stroke="#4ade80"
              strokeWidth={1.5}
              fill="url(#accGrad)"
              dot={false}
              activeDot={{ r: 3, fill: '#4ade80', stroke: '#111113', strokeWidth: 2 }}
              isAnimationActive={false}
            />
          )}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

// ============================================================================
// Results Actions — post-training buttons
// ============================================================================

function ResultsActions({ session, version, onDownloadModel, onStoreOnZeroG, isStoringOnZeroG, zeroGResult, showStoreOn0G }: {
  session: LiveProgressProps['session'];
  version: string;
  onDownloadModel: (sessionId: string) => Promise<void>;
  onStoreOnZeroG: (sessionId: string, version?: string) => Promise<void>;
  isStoringOnZeroG: boolean;
  zeroGResult: ZeroGStorageResult | null;
  showStoreOn0G: boolean;
}) {
  const isSuccess = session.status === 'complete';
  const [copied, setCopied] = useState(false);

  const copyHash = (hash: string) => {
    navigator.clipboard.writeText(hash);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!isSuccess) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-3"
    >
      {/* Summary stats */}
      <div className="flex items-center gap-6 py-3 text-sm text-helix-muted">
        <span>{session.current_step}/{session.total_steps} steps</span>
        <span className="w-px h-3 bg-helix-border" />
        <span>
          {session.elapsed_secs > 0
            ? `${session.elapsed_secs.toFixed(1)}s`
            : session.started_at > 0
              ? `${((Date.now() / 1000) - session.started_at).toFixed(1)}s`
              : '--'}
        </span>
        <span className="w-px h-3 bg-helix-border" />
        <span>{session.checkpoints_submitted} checkpoints</span>
      </div>

      {/* Action buttons */}
      <div className="flex gap-3">
        <button
          type="button"
          onClick={() => onDownloadModel(session.session_id)}
          className="flex-1 flex items-center justify-center gap-2 py-3.5 rounded-2xl bg-white text-black text-sm font-semibold hover:shadow-lg hover:shadow-white/10 transition-all"
        >
          <Download size={16} />
          Download Model
        </button>

        {showStoreOn0G && !zeroGResult && (
          <button
            type="button"
            onClick={() => onStoreOnZeroG(session.session_id, version)}
            disabled={isStoringOnZeroG}
            className={cn(
              'flex-1 flex items-center justify-center gap-2 py-3.5 rounded-2xl border text-sm font-semibold transition-all',
              isStoringOnZeroG
                ? 'bg-helix-surface border-helix-border text-helix-muted cursor-wait'
                : 'bg-helix-surface border-helix-border text-white hover:border-helix-border2',
            )}
          >
            {isStoringOnZeroG ? (
              <><Loader2 size={16} className="animate-spin" /> Uploading...</>
            ) : (
              <><HardDrive size={16} /> Store on 0G</>
            )}
          </button>
        )}
      </div>

      {/* 0G result */}
      {zeroGResult && (
        <motion.div
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          className="rounded-2xl bg-green-500/[0.04] border border-green-500/15 p-4 space-y-3"
        >
          <div className="flex items-center gap-2">
            <CheckCircle size={14} className="text-green-400" />
            <span className="text-sm text-green-300 font-medium">Stored on 0G</span>
          </div>

          <div className="flex items-center gap-2">
            <code className="flex-1 text-xs text-green-300/70 bg-green-500/[0.06] px-3 py-2 rounded-xl truncate">
              {zeroGResult.rootHash}
            </code>
            <button
              type="button"
              onClick={() => copyHash(zeroGResult.rootHash)}
              className="shrink-0 p-2 rounded-xl bg-green-500/[0.06] text-green-400 hover:text-green-300 transition-colors"
            >
              {copied ? <CheckCircle size={14} /> : <Copy size={14} />}
            </button>
          </div>

          <div className="flex items-center gap-4">
            <a
              href={zeroGResult.explorerUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-green-400/60 hover:text-green-400 transition-colors flex items-center gap-1"
            >
              View on Explorer <ExternalLink size={10} />
            </a>

            <Link
              href={`/my-models?action=add-version&session=${session.session_id}&hash=${zeroGResult.rootHash}&version=${version}&accuracy=${session.accuracy !== null ? session.accuracy * 100 : ''}`}
              className="text-xs text-green-400/60 hover:text-green-400 transition-colors flex items-center gap-1"
            >
              Save to Registry <Link2 size={10} />
            </Link>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}


// ============================================================================
// History Card Grid — responsive grid of past training sessions
// ============================================================================

function HistoryCardGrid({ sessions, onSelect }: {
  sessions: TrainingSessionState[];
  onSelect: (s: TrainingSessionState) => void;
}) {
  if (sessions.length === 0) {
    return (
      <div className="text-center py-16 text-helix-dim">
        <History size={32} className="mx-auto mb-3 opacity-40" />
        <p>No training sessions yet</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {sessions.map((s) => (
        <button
          key={s.session_id}
          type="button"
          onClick={() => onSelect(s)}
          className="text-left p-5 rounded-2xl bg-helix-surface border border-helix-border hover:border-white/20 transition-all group"
        >
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-white truncate max-w-[60%]">
              {s.model_name || 'Unnamed Model'}
            </span>
            <span className={cn(
              'text-xs px-2 py-0.5 rounded-full',
              s.status === 'complete' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
            )}>
              {s.status}
            </span>
          </div>

          {/* Mini sparkline */}
          {s.losses.length > 1 && (
            <div className="h-12 mb-3 opacity-60 group-hover:opacity-100 transition-opacity">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={s.losses.slice(-50).map((l, i) => ({ i, l }))}>
                  <Area type="monotone" dataKey="l" stroke="#ffffff40" fill="#ffffff10" strokeWidth={1} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}

          <div className="flex items-center gap-3 text-sm text-helix-dim">
            {s.accuracy !== null && s.accuracy !== undefined && (
              <span className="text-white font-medium">{(s.accuracy * 100).toFixed(1)}%</span>
            )}
            <span>{s.current_step}/{s.total_steps} steps</span>
            <span className="ml-auto">{formatDuration(Math.round(s.elapsed_secs))}</span>
          </div>

          <div className="text-xs text-helix-dim mt-2">
            {new Date(s.started_at * 1000).toLocaleDateString()} {new Date(s.started_at * 1000).toLocaleTimeString()}
          </div>
        </button>
      ))}
    </div>
  );
}

// ============================================================================
// History Detail Modal — full-screen overlay with session details
// ============================================================================

function HistoryDetailModal({ session, onClose }: {
  session: TrainingSessionState;
  onClose: () => void;
}) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  const lossData = session.losses.map((l, i) => ({ step: i + 1, loss: l }));

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" />

        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.95, opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="relative w-full max-w-5xl max-h-[90vh] overflow-y-auto rounded-3xl bg-[#0a0a0a] border border-helix-border p-8"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Close button */}
          <button
            type="button"
            onClick={onClose}
            className="absolute top-4 right-4 p-2 rounded-xl text-helix-dim hover:text-white hover:bg-helix-surface transition-colors"
          >
            <XCircle size={20} />
          </button>

          {/* Header */}
          <div className="flex items-center gap-3 mb-6">
            <h2 className="text-xl font-semibold text-white">
              {session.model_name || 'Training Session'}
            </h2>
            <span className={cn(
              'text-xs px-2 py-0.5 rounded-full',
              session.status === 'complete' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
            )}>
              {session.status}
            </span>
            <span className="text-sm text-helix-dim ml-auto font-mono">
              {session.session_id.slice(0, 8)}
            </span>
          </div>

          {/* Two-column layout */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* LEFT: Hero metric + loss curve */}
            <div className="space-y-4">
              <div className="p-6 rounded-2xl bg-helix-surface border border-helix-border text-center">
                {session.status === 'complete' && session.accuracy !== null ? (
                  <>
                    <p className="text-5xl font-bold text-white tabular-nums">
                      {(session.accuracy! * 100).toFixed(1)}
                      <span className="text-2xl text-helix-dim">%</span>
                    </p>
                    <p className="text-sm text-helix-dim mt-1">Final Accuracy</p>
                  </>
                ) : (
                  <>
                    <p className="text-5xl font-bold text-white tabular-nums">
                      {session.current_loss.toFixed(4)}
                    </p>
                    <p className="text-sm text-helix-dim mt-1">Final Loss</p>
                  </>
                )}
              </div>

              {lossData.length > 1 && (
                <div className="h-64">
                  <LossCurve data={lossData} />
                </div>
              )}
            </div>

            {/* RIGHT: Stats + metadata */}
            <div className="space-y-4">
              {/* Progress bar */}
              <div>
                <div className="h-1.5 rounded-full bg-helix-border overflow-hidden">
                  <div
                    className="h-full rounded-full bg-white"
                    style={{ width: `${(session.current_step / Math.max(session.total_steps, 1)) * 100}%` }}
                  />
                </div>
                <p className="text-sm text-helix-dim mt-1 tabular-nums">
                  Step {session.current_step} / {session.total_steps}
                </p>
              </div>

              {/* Stats grid */}
              <div className="grid grid-cols-2 gap-3">
                <StatCard label="MAC Checks" value={session.mac_checks_passed} icon={<Shield size={14} />} />
                <StatCard label="Checkpoints" value={session.checkpoints_submitted} icon={<Zap size={14} />} />
                <StatCard label="ZK Proofs" value={session.zk_proofs_generated} icon={<Activity size={14} />} />
                <StatCard label="Elapsed" value={formatDuration(Math.round(session.elapsed_secs))} icon={<Clock size={14} />} />
              </div>

              {/* Cheater alert */}
              {session.cheater_detected && (
                <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
                  <strong>Cheater Detected</strong>
                  <span className="ml-2">Worker {session.cheater_detected.party_index}</span>
                </div>
              )}

              {/* Session metadata */}
              <div className="p-4 rounded-xl bg-helix-surface border border-helix-border space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-helix-dim">Session ID</span>
                  <span className="text-helix-text2 font-mono">{session.session_id.slice(0, 16)}...</span>
                </div>
                {session.coordinator_address && (
                  <div className="flex justify-between">
                    <span className="text-helix-dim">Coordinator</span>
                    <span className="text-helix-text2 font-mono">{session.coordinator_address.slice(0, 10)}...</span>
                  </div>
                )}
                {session.job_id > 0 && (
                  <div className="flex justify-between">
                    <span className="text-helix-dim">Job ID</span>
                    <span className="text-helix-text2 font-mono">{session.job_id}</span>
                  </div>
                )}
                <div className="flex justify-between">
                  <span className="text-helix-dim">Started</span>
                  <span className="text-helix-text2">
                    {new Date(session.started_at * 1000).toLocaleString()}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}


// ============================================================================
// Inner page (needs useSearchParams, so must be inside Suspense)
// ============================================================================

function TrainPageInner() {
  const searchParams = useSearchParams();
  const queryModelId = searchParams.get('model');

  const {
    startTraining, uploadData, uploadWeights, downloadModel, storeOnZeroG,
    session, losses, isConnected, isStarting, error: trainingError,
    uploadedData, uploadedWeights, workersOnline, zeroGResult, isStoringOnZeroG,
    elapsedTime, history, fetchHistory, isLoadingHistory,
  } = useMpcTraining();

  const { models, isLoading: isLoadingModels } = useModelRegistry();
  const { signMessageAsync } = useSignMessage();

  // Wallet payment flow
  const { address } = useAccount();
  const chainId = useChainId();
  const { switchChain } = useSwitchChain();
  const { writeContract, data: paymentTxHash, isPending: isWalletPrompting, error: walletError, reset: resetWalletWrite } = useWriteContract();
  const { isLoading: isConfirmingPayment, isSuccess: paymentConfirmed } = useWaitForTransactionReceipt({ hash: paymentTxHash });

  // Determine expected chain from env (31337 for local Anvil, 99999 for ADI testnet)
  const expectedChainId = Number(process.env.NEXT_PUBLIC_DEFAULT_CHAIN_ID || '31337');
  const isWrongChain = address && chainId !== expectedChainId;

  const [operatorAddress, setOperatorAddress] = useState<string | null>(null);
  const [walletPaymentError, setWalletPaymentError] = useState<string | null>(null);
  const pendingConfigRef = useRef<{ config: TrainingJobConfig; opts: { storeOn0G: boolean; version: string; modelName: string; modelSlug: string } } | null>(null);

  // Fetch operator address from backend on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/operator-address`)
      .then((res) => res.ok ? res.json() : null)
      .then((data) => {
        if (data?.address) setOperatorAddress(data.address);
      })
      .catch(() => {});
  }, []);

  // When payment is confirmed, extract jobId and start training
  useEffect(() => {
    if (!paymentConfirmed || !paymentTxHash || !pendingConfigRef.current) return;
    const { config, opts } = pendingConfigRef.current;
    pendingConfigRef.current = null;

    // Extract jobId from transaction receipt logs
    const extractJobId = async () => {
      try {
        const { createPublicClient, http } = await import('viem');
        const rpcUrl = process.env.NEXT_PUBLIC_ETH_RPC_URL || 'http://localhost:8545';
        // Dynamically determine the chain
        const publicClient = createPublicClient({ transport: http(rpcUrl) });
        const receipt = await publicClient.getTransactionReceipt({ hash: paymentTxHash });

        let jobId: number | undefined;
        for (const log of receipt.logs) {
          try {
            const decoded = decodeEventLog({
              abi: HELIX_COORDINATOR_V4_ABI,
              data: log.data,
              topics: log.topics,
            });
            if (decoded.eventName === 'JobRegistered') {
              jobId = Number((decoded.args as { jobId: bigint }).jobId);
              break;
            }
          } catch {
            // Not our event, skip
          }
        }

        if (jobId !== undefined) {
          setWalletPaymentError(null);
          startTraining({ ...config, job_id: jobId, payment_eth: 0 });
        } else {
          setWalletPaymentError('Could not extract job ID from transaction');
        }
      } catch (err) {
        setWalletPaymentError(err instanceof Error ? err.message : 'Failed to read transaction receipt');
      }
    };

    setWantsStoreOn0G(opts.storeOn0G);
    setCurrentVersion(opts.version);
    setCurrentModelName(opts.modelName);
    extractJobId();
  }, [paymentConfirmed, paymentTxHash, startTraining]);

  const hasSession = session !== null;
  const [viewMode, setViewMode] = useState<'live' | 'history'>('live');
  const [selectedHistorySession, setSelectedHistorySession] = useState<TrainingSessionState | null>(null);
  const [wantsStoreOn0G, setWantsStoreOn0G] = useState(false);
  const [currentVersion, setCurrentVersion] = useState('1.0.0');
  const [currentModelName, setCurrentModelName] = useState('');

  const [selectedModelId, setSelectedModelId] = useState<number | null>(null);
  const [weightFetchStatus, setWeightFetchStatus] = useState<WeightFetchStatus>('idle');
  const [weightFetchError, setWeightFetchError] = useState<string | null>(null);
  const [fetchedModelName, setFetchedModelName] = useState<string | null>(null);
  const fetchingForRef = useRef<number | null>(null);
  const autoSelectAppliedRef = useRef(false);

  // Auto-select from query param
  useEffect(() => {
    if (autoSelectAppliedRef.current) return;
    if (!queryModelId || isLoadingModels) return;
    const tokenId = Number(queryModelId);
    if (isNaN(tokenId)) return;
    const found = models.find((m) => m.tokenId === tokenId);
    if (found) {
      autoSelectAppliedRef.current = true;
      setSelectedModelId(tokenId);
    }
  }, [queryModelId, models, isLoadingModels]);

  // Fetch weights when selectedModelId changes
  useEffect(() => {
    if (selectedModelId === null) {
      if (fetchingForRef.current !== null) fetchingForRef.current = null;
      setWeightFetchStatus('idle');
      setWeightFetchError(null);
      setFetchedModelName(null);
      return;
    }

    const model = models.find((m) => m.tokenId === selectedModelId);
    if (!model) return;

    const lv = model.versions.length ? model.versions[model.versions.length - 1] : null;
    if (!lv || !lv.weightsStored || !lv.rootHash) {
      setWeightFetchStatus('idle');
      setWeightFetchError(null);
      setFetchedModelName(null);
      return;
    }

    if (fetchingForRef.current === selectedModelId) return;
    fetchingForRef.current = selectedModelId;

    const fetchWeights = async () => {
      setWeightFetchStatus('fetching');
      setWeightFetchError(null);
      setFetchedModelName(null);

      try {
        const res = await fetch('/api/fetch-from-0g', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ rootHash: lv.rootHash }),
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
          const key = await deriveModelKey(signMsg, selectedModelId);
          const decryptedJson = await decryptWeights(key, rawBytes);
          weightsData = JSON.parse(decryptedJson);
        } else {
          weightsData = result.data;
        }

        setWeightFetchStatus('uploading');
        const weightsJson = JSON.stringify(weightsData);
        const blob = new Blob([weightsJson], { type: 'application/json' });
        const syntheticFile = new File([blob], `weights-from-${model.slug}.json`, { type: 'application/json' });
        await uploadWeights(syntheticFile);

        setFetchedModelName(model.name);
        setWeightFetchStatus('done');
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to fetch weights';
        setWeightFetchError(message);
        setWeightFetchStatus('error');
        fetchingForRef.current = null;
      }
    };

    fetchWeights();
  }, [selectedModelId, models, signMessageAsync, uploadWeights]);

  const handleSelectModel = useCallback((tokenId: number | null) => {
    fetchingForRef.current = null;
    setSelectedModelId(tokenId);
  }, []);

  const handleStart = async (config: TrainingJobConfig, opts: { storeOn0G: boolean; version: string; modelName: string; modelSlug: string }) => {
    pendingConfigRef.current = { config, opts };
    setWalletPaymentError(null);

    // On local Anvil (31337), skip the on-chain payment and go straight to backend.
    // The contract call is expensive in gas and MetaMask prices local ETH at real USD rates.
    const isLocalChain = expectedChainId === 31337;
    if (isLocalChain) {
      setWantsStoreOn0G(opts.storeOn0G);
      setCurrentVersion(opts.version);
      setCurrentModelName(opts.modelName);
      pendingConfigRef.current = null;
      startTraining({ ...config, job_id: undefined, payment_eth: 0 });
      return;
    }

    // On real chains, ensure wallet is on the correct chain before sending tx
    if (chainId !== expectedChainId) {
      try {
        switchChain({ chainId: expectedChainId });
        setWalletPaymentError(`Please switch to chain ${expectedChainId} in your wallet, then try again.`);
        return;
      } catch {
        setWalletPaymentError(`Please switch your wallet to chain ${expectedChainId} manually.`);
        return;
      }
    }

    const coordinatorAddress = getContractAddress(expectedChainId, 'helixCoordinator') as `0x${string}`;
    const archHash = computeArchitectureHash(config.architecture);
    const paymentWei = parseEther(String(config.payment_eth));

    writeContract({
      chainId: expectedChainId,
      address: coordinatorAddress,
      abi: HELIX_COORDINATOR_V4_ABI,
      functionName: 'registerTrainingJob',
      args: [
        archHash,
        BigInt(config.checkpoint_freq),
        BigInt(config.num_steps),
        paymentWei,
        config.zk_mode === 'always',
        BigInt(config.zk_checkpoint_freq),
        config.zk_mode === 'risk',
        BigInt(config.min_workers_for_mpc),
        (operatorAddress ?? '0x0000000000000000000000000000000000000000') as `0x${string}`,
      ],
      value: paymentWei,
    });
  };

  const defaultVersion = useMemo(() => {
    if (history.length === 0) return '1.0.0';
    return `${history.length + 1}.0.0`;
  }, [history]);
  const isFetchingWeights = weightFetchStatus === 'fetching' || weightFetchStatus === 'decrypting' || weightFetchStatus === 'uploading';

  return (
    <>
    {/* Global cheater toast notification */}
    <CheaterToast cheater={session?.cheater_detected ?? null} />

    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
      className="py-4 space-y-8"
    >
      {/* Wrong chain warning */}
      {isWrongChain && (
        <motion.div
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between px-4 py-3 rounded-2xl bg-yellow-500/[0.08] border border-yellow-500/20"
        >
          <div className="flex items-center gap-2.5">
            <XCircle size={14} className="text-yellow-400 shrink-0" />
            <p className="text-sm text-yellow-300">
              Wrong network. Connected to chain {chainId}, expected {expectedChainId}.
            </p>
          </div>
          <button
            type="button"
            onClick={() => switchChain({ chainId: expectedChainId })}
            className="px-3 py-1 text-xs font-medium text-yellow-300 bg-yellow-500/10 border border-yellow-500/20 rounded-lg hover:bg-yellow-500/20 transition-colors"
          >
            Switch Network
          </button>
        </motion.div>
      )}

      {/* Weight fetch error */}
      {weightFetchStatus === 'error' && weightFetchError && (
        <motion.div
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-2.5 px-4 py-3 rounded-2xl bg-red-500/[0.06] border border-red-500/20"
        >
          <XCircle size={14} className="text-red-400 shrink-0" />
          <p className="text-sm text-red-300">{weightFetchError}</p>
        </motion.div>
      )}

      {/* Wallet / payment errors */}
      {(walletError || walletPaymentError) && (
        <motion.div
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-2.5 px-4 py-3 rounded-2xl bg-red-500/[0.06] border border-red-500/20"
        >
          <XCircle size={14} className="text-red-400 shrink-0" />
          <p className="text-sm text-red-300">
            {walletPaymentError || (walletError instanceof Error ? walletError.message : 'Wallet transaction failed')}
          </p>
          <button
            type="button"
            onClick={() => { resetWalletWrite(); setWalletPaymentError(null); }}
            className="ml-auto text-xs text-red-400/60 hover:text-red-300 transition-colors shrink-0"
          >
            Dismiss
          </button>
        </motion.div>
      )}

      {/* View mode toggle */}
      <div className="flex items-center gap-1 p-1 rounded-xl bg-helix-surface border border-helix-border w-fit">
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
          onClick={() => { setViewMode('history'); fetchHistory(); }}
          className={cn(
            'px-4 py-1.5 rounded-lg text-sm font-medium transition-all',
            viewMode === 'history' ? 'bg-white text-black' : 'text-helix-dim hover:text-helix-text2'
          )}
        >
          History
        </button>
      </div>

      {viewMode === 'live' ? (
        <>
          {!hasSession ? (
            <ConfigForm
              onStart={handleStart}
              isStarting={isStarting}
              onUploadData={uploadData}
              onUploadWeights={uploadWeights}
              uploadedData={uploadedData}
              uploadedWeights={uploadedWeights}
              workersOnline={workersOnline}
              defaultVersion={defaultVersion}
              models={models}
              selectedModelId={selectedModelId}
              onSelectModel={handleSelectModel}
              isFetchingWeights={isFetchingWeights}
              fetchedModelName={fetchedModelName}
              isWalletPrompting={isWalletPrompting}
              isConfirmingPayment={isConfirmingPayment}
              walletConnected={!!address}
            />
          ) : (
            <LiveProgress
              session={session}
              losses={losses}
              isConnected={isConnected}
              error={trainingError}
              version={currentVersion}
              modelName={currentModelName}
              onDownloadModel={downloadModel}
              onStoreOnZeroG={storeOnZeroG}
              isStoringOnZeroG={isStoringOnZeroG}
              zeroGResult={zeroGResult}
              showStoreOn0G={wantsStoreOn0G}
              elapsedTime={elapsedTime}
            />
          )}
        </>
      ) : (
        <HistoryCardGrid
          sessions={history}
          onSelect={setSelectedHistorySession}
        />
      )}

      {/* Detail modal */}
      {selectedHistorySession && (
        <HistoryDetailModal
          session={selectedHistorySession}
          onClose={() => setSelectedHistorySession(null)}
        />
      )}
    </motion.div>
    </>
  );
}

// ============================================================================
// Main Page (wrapped in Suspense for useSearchParams)
// ============================================================================

export default function TrainPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center py-20">
          <Loader2 size={24} className="animate-spin text-helix-muted" />
        </div>
      }
    >
      <TrainPageInner />
    </Suspense>
  );
}
