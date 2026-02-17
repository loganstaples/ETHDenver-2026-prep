'use client';

import { useState, useEffect, useCallback } from 'react';
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
  Hash,
} from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { StatCard } from '@/components/ui/StatCard';
import { EmptyState } from '@/components/ui/EmptyState';
import { cn } from '@/lib/utils';

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

// ============================================================================
// Version Detail Expansion
// ============================================================================

interface VersionRowProps {
  entry: TrainingHistoryEntry;
  isExpanded: boolean;
  onToggle: () => void;
  onDelete: (sessionId: string) => void;
}

function VersionRow({ entry, isExpanded, onToggle, onDelete }: VersionRowProps) {
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

        <div className="flex items-center gap-4 ml-auto shrink-0">
          {entry.accuracy !== null && (
            <span className="text-sm font-mono text-white">
              {(entry.accuracy * 100).toFixed(1)}%
            </span>
          )}

          <span className="text-xs font-mono text-helix-muted">
            {entry.steps}/{entry.totalSteps} steps
          </span>

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
                    {entry.steps} / {entry.totalSteps}
                  </p>
                </div>
                <div className="bg-helix-surface rounded-lg p-3">
                  <p className="label-text mb-1">Status</p>
                  <p className={cn(
                    'text-lg font-mono font-light',
                    isComplete ? 'text-green-400' : 'text-red-400',
                  )}>
                    {isComplete ? 'Complete' : 'Failed'}
                  </p>
                </div>
                <div className="bg-helix-surface rounded-lg p-3">
                  <p className="label-text mb-1">Trained</p>
                  <p className="text-lg font-mono font-light text-white">
                    {new Date(entry.date).toLocaleDateString('en-US', {
                      month: 'short',
                      day: 'numeric',
                      year: 'numeric',
                    })}
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
              <div className="flex items-center gap-3 pt-1">
                {isComplete && (
                  <Link
                    href={`/inference?session=${entry.sessionId}${entry.rootHash ? `&hash=${entry.rootHash}` : ''}&version=${entry.version}`}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white text-black text-sm font-medium hover:bg-white/90 transition-colors"
                  >
                    <Sparkles size={14} />
                    Run Inference
                  </Link>
                )}
                {isComplete && (
                  <a
                    href={`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001'}/api/training/sessions/${entry.sessionId}/model`}
                    download={`helix-model-v${entry.version}-${entry.sessionId.slice(0, 8)}.json`}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text hover:border-helix-border2 hover:text-white transition-colors"
                  >
                    <Download size={14} />
                    Download
                  </a>
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

  useEffect(() => {
    setHistory(getTrainingHistory());
  }, []);

  const handleDelete = useCallback((sessionId: string) => {
    setHistory((prev) => {
      const updated = prev.filter((e) => e.sessionId !== sessionId);
      saveTrainingHistory(updated);
      return updated;
    });
    if (expandedId === sessionId) setExpandedId(null);
  }, [expandedId]);

  // Compute aggregate stats
  const completedVersions = history.filter((e) => e.status === 'complete');
  const storedOn0G = history.filter((e) => e.storedOn0G);
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
        <Link
          href="/train"
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white text-black text-sm font-medium hover:bg-white/90 transition-colors"
        >
          Train New Version
        </Link>
      </div>

      {history.length === 0 ? (
        <EmptyState
          icon={<Layers size={32} />}
          title="No models yet"
          description="Train your first MNIST model to see it here. Each training session creates a new version."
          action={
            <Link
              href="/train"
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white text-black text-sm font-medium hover:bg-white/90 transition-colors"
            >
              Start Training
            </Link>
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

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
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
                label="Completed"
                value={`${completedVersions.length}/${history.length}`}
                icon={<BarChart3 size={14} />}
              />
            </div>
          </Card>

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
