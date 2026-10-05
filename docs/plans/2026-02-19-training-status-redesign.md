# Training Status Redesign — Implementation Plan

**Goal:** Make the training status indicator bigger, more detailed, animated, and live-updating on both the train page and dashboard — CashApp/Apple meets Bloomberg terminal aesthetic.

**Architecture:** Fix the root cause (polling overwrites WS data) in both hooks, then rebuild the `LiveProgress` component as a hero status card with phase ring, sub-status ticker, animated step counter, and inline stat row with deltas. Dashboard gets the same live-data fixes.

**Tech Stack:** React 18, Next.js, Framer Motion (already installed), Recharts, Tailwind CSS, WebSocket

---

### Task 1: Fix polling overwrite in `useMpcTraining`

**Files:**
- Modify: `helix/dashboard/src/hooks/useMpcTraining.ts:392-430`

**Step 1: Fix the polling `poll()` function to merge instead of overwrite**

Replace the entire `poll` function body inside `startPolling`. The fix: only update session state if the polled data has a higher `current_step` than what WS already set, and never shrink the `losses` array.

```typescript
const poll = async () => {
  try {
    const res = await fetch(`${API_BASE}/api/training/sessions/${sessionId}`);
    if (res.ok) {
      const data: TrainingSessionState = await res.json();

      // Merge: only apply poll data if it's newer than what WS set
      setSession((prev) => {
        if (!prev) return data;
        // If polled data is ahead (or terminal), use it
        if (data.current_step > prev.current_step || data.status === 'complete' || data.status === 'failed') {
          return data;
        }
        // Otherwise keep WS-driven state but merge non-step fields
        return {
          ...prev,
          coordinator_address: data.coordinator_address || prev.coordinator_address,
          job_id: data.job_id || prev.job_id,
          workers_active: data.workers_active ?? prev.workers_active,
          status: data.status === 'complete' || data.status === 'failed' ? data.status : prev.status,
        };
      });

      // Only update losses if poll has more data points than WS
      setLosses((prev) => {
        if (data.losses && data.losses.length > prev.length) {
          return data.losses.map((loss, i) => ({
            step: i + 1,
            loss,
            accuracy: Math.max(0, Math.min(1, 1 - loss / 2.302585)),
          }));
        }
        return prev;
      });

      // Stop polling if terminal state
      if (data.status === 'complete' || data.status === 'failed') {
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }
      }
    }
  } catch {
    // Polling failure is non-fatal — WebSocket is primary
  }
};
```

**Step 2: Verify the app builds**

Run: `cd helix/dashboard && npx next build 2>&1 | tail -5`
Expected: Build succeeds (or only pre-existing warnings)

**Step 3: Commit**

```bash
git add helix/dashboard/src/hooks/useMpcTraining.ts
git commit -m "fix: prevent polling from overwriting live WebSocket data in useMpcTraining"
```

---

### Task 2: Fix polling overwrite in `useDashboardSessions`

**Files:**
- Modify: `helix/dashboard/src/hooks/useDashboardSessions.ts:57-71`

**Step 1: Fix `fetchActiveSession` to merge instead of overwrite**

Replace the `fetchActiveSession` callback. Same pattern — only apply poll data if it's newer:

```typescript
const fetchActiveSession = useCallback(async (sessionId: string) => {
  try {
    const res = await fetch(`${API_BASE}/api/training/sessions/${sessionId}`);
    if (!res.ok) return;
    const data: TrainingSessionState = await res.json();

    // Merge: only overwrite if poll data is ahead of WS-driven state
    setActiveSession((prev) => {
      if (!prev) return data;
      if (data.current_step > prev.current_step || data.status === 'complete' || data.status === 'failed') {
        return data;
      }
      return {
        ...prev,
        coordinator_address: data.coordinator_address || prev.coordinator_address,
        job_id: data.job_id || prev.job_id,
        workers_active: data.workers_active ?? prev.workers_active,
        status: data.status === 'complete' || data.status === 'failed' ? data.status : prev.status,
      };
    });

    // Only update losses if poll has more
    setLosses((prev) => {
      if (data.losses && data.losses.length > prev.length) {
        return data.losses.map((loss: number, i: number) => ({ step: i + 1, loss }));
      }
      return prev;
    });
  } catch {
    // Non-fatal — WebSocket is primary
  }
}, []);
```

**Step 2: Verify the app builds**

Run: `cd helix/dashboard && npx next build 2>&1 | tail -5`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add helix/dashboard/src/hooks/useDashboardSessions.ts
git commit -m "fix: prevent polling from overwriting live WebSocket data in useDashboardSessions"
```

---

### Task 3: Build the Hero Status Card on the Train Page

**Files:**
- Modify: `helix/dashboard/src/app/train/page.tsx`

This is the main visual redesign. Replace the current `LiveProgress` component (lines 897-1095) with a new hero-first layout. This task covers:
- Phase ring (animated SVG arc)
- Sub-status ticker (cycling MPC operations)
- Animated step counter
- Glowing progress bar
- Inline stat row with deltas
- Improved checkpoint display

**Step 1: Add new constants and sub-status messages**

Replace the existing `PHASE_DESCRIPTIONS` (lines 77-91) and add sub-status cycling:

```typescript
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

const MPC_SUB_OPERATIONS = [
  'Distributing secret shares across workers',
  'Computing forward pass — layer 1',
  'Garbled-circuit ReLU activation',
  'Computing forward pass — layer 2',
  'Computing backward pass — gradients',
  'Verifying SPDZ MACs — integrity check',
  'Aggregating gradient updates',
];
```

**Step 2: Add PhaseRing component**

Add this new component after the existing `ChartTooltip` component (~line 168). This is a 270° arc SVG similar to the existing `HorseshoeProgress` on the dashboard, but styled for the train page:

```tsx
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

  return (
    <svg
      width={full}
      height={full}
      viewBox={`0 0 ${full} ${full}`}
      style={{ margin: -pad, marginTop: -pad + yOffset, overflow: 'visible' }}
    >
      <defs>
        <filter id="phase-glow" x="-100%" y="-100%" width="400%" height="400%">
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
          filter="url(#phase-glow)"
          style={{ transition: 'stroke-dasharray 0.6s ease-out' }}
        />
      )}
    </svg>
  );
}
```

**Step 3: Add SubStatusTicker component**

This cycles through MPC sub-operations during phase 8:

```tsx
function SubStatusTicker({ phase, workers }: { phase: number; workers: number }) {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (phase !== 8) { setIndex(0); return; }
    const interval = setInterval(() => {
      setIndex((i) => (i + 1) % MPC_SUB_OPERATIONS.length);
    }, 2000);
    return () => clearInterval(interval);
  }, [phase]);

  const text = phase === 8
    ? MPC_SUB_OPERATIONS[index].replace('across workers', `across ${workers} workers`)
    : PHASE_DESCRIPTIONS[phase] || 'Processing...';

  return (
    <AnimatePresence mode="wait">
      <motion.span
        key={`${phase}-${index}`}
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
```

**Step 4: Add AnimatedNumber component**

Spring-animated number display for step counter and stats:

```tsx
function AnimatedNumber({ value, className }: { value: number; className?: string }) {
  const motionVal = useMotionValue(value);
  const spring = useSpring(motionVal, { stiffness: 300, damping: 30 });
  const [display, setDisplay] = useState(value);

  useEffect(() => { motionVal.set(value); }, [value, motionVal]);
  useEffect(() => spring.on('change', (v) => setDisplay(Math.round(v))), [spring]);

  return <span className={className}>{display}</span>;
}
```

**Step 5: Replace the `LiveProgress` component**

Replace the entire `LiveProgress` function (lines 916-1095) with the new hero layout. Keep the same props interface (lines 901-914).

```tsx
function LiveProgress({ session, losses, isConnected, error, version, modelName, onDownloadModel, onStoreOnZeroG, isStoringOnZeroG, zeroGResult, showStoreOn0G, elapsedTime }: LiveProgressProps) {
  const stepProgress = session.total_steps > 0
    ? (session.current_step / session.total_steps) * 100
    : 0;
  const isTerminal = session.status === 'complete' || session.status === 'failed';
  const isComplete = session.status === 'complete';

  // Compute deltas for stats
  const lastLoss = losses.length > 0 ? losses[losses.length - 1].loss : 0;
  const prevLoss = losses.length > 10 ? losses[losses.length - 11].loss : lastLoss;
  const lossDelta = lastLoss - prevLoss;

  const lastAcc = losses.length > 0 ? (losses[losses.length - 1].accuracy ?? 0) : 0;
  const prevAcc = losses.length > 10 ? (losses[losses.length - 11].accuracy ?? 0) : lastAcc;
  const accDelta = lastAcc - prevAcc;

  const totalCheckpoints = session.total_steps > 0 ? Math.floor(session.total_steps / 50) : 0;

  return (
    <div className="space-y-6">
      {/* ── Header bar ─────────────────────────── */}
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

      {/* ── Error ─────────────────────────── */}
      {error && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="px-4 py-3 rounded-2xl bg-red-500/[0.06] border border-red-500/20">
          <p className="text-sm text-red-300">{error}</p>
        </motion.div>
      )}

      {/* ── Hero Status Card ─────────────────────────── */}
      {!isTerminal && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative rounded-3xl bg-helix-surface border border-helix-border overflow-hidden"
        >
          {/* Subtle top glow line */}
          <div
            className="absolute top-0 left-0 h-[2px] bg-gradient-to-r from-transparent via-white/30 to-transparent transition-all duration-500"
            style={{ width: `${stepProgress}%` }}
          />

          <div className="p-8">
            {/* Phase ring + info */}
            <div className="flex items-center gap-8">
              {/* Phase ring */}
              <div className="relative shrink-0 flex items-center justify-center">
                <div className="absolute inset-0 rounded-full bg-white/[0.02] blur-xl scale-110" />
                <PhaseRing phase={session.phase} totalPhases={TOTAL_PHASES} size={160} />
                {/* Center text */}
                <div className="absolute inset-0 flex flex-col items-center justify-center" style={{ marginTop: 160 * 0.08 }}>
                  <span className="text-3xl font-bold text-white tabular-nums">{session.phase}</span>
                  <span className="text-[10px] uppercase tracking-widest text-helix-muted mt-0.5">
                    of {TOTAL_PHASES}
                  </span>
                </div>
              </div>

              {/* Right side: status details */}
              <div className="flex-1 min-w-0 space-y-5">
                {/* Phase name + sub-status */}
                <div>
                  <h3 className="text-lg font-semibold text-white mb-1">
                    {PHASE_DESCRIPTIONS[session.phase] || 'Processing...'}
                  </h3>
                  <SubStatusTicker phase={session.phase} workers={session.workers_active ?? 3} />
                </div>

                {/* Step counter */}
                <div className="flex items-baseline gap-3">
                  <span className="text-5xl font-bold text-white tabular-nums font-mono tracking-tighter">
                    <AnimatedNumber value={session.current_step} />
                  </span>
                  <span className="text-xl text-helix-dim font-mono">/ {session.total_steps}</span>
                  <span className="text-sm text-helix-muted ml-2">steps</span>
                </div>

                {/* Glowing progress bar */}
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

            {/* Stat row */}
            <div className="grid grid-cols-5 gap-4 mt-6 pt-6 border-t border-white/[0.04]">
              <div>
                <div className="text-[10px] uppercase tracking-wider text-helix-muted mb-1">Loss</div>
                <div className="flex items-baseline gap-1.5">
                  <span className="text-lg font-mono font-semibold text-white tabular-nums">
                    {lastLoss > 0 ? lastLoss.toFixed(4) : '—'}
                  </span>
                  {lastLoss > 0 && (
                    <span className={cn('text-xs font-mono tabular-nums', lossDelta <= 0 ? 'text-green-400' : 'text-red-400')}>
                      {lossDelta <= 0 ? '↓' : '↑'}{Math.abs(lossDelta).toFixed(4)}
                    </span>
                  )}
                </div>
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-wider text-helix-muted mb-1">Accuracy</div>
                <div className="flex items-baseline gap-1.5">
                  <span className="text-lg font-mono font-semibold text-white tabular-nums">
                    {lastAcc > 0 ? `${(lastAcc * 100).toFixed(1)}%` : '—'}
                  </span>
                  {lastAcc > 0 && (
                    <span className={cn('text-xs font-mono tabular-nums', accDelta >= 0 ? 'text-green-400' : 'text-red-400')}>
                      {accDelta >= 0 ? '↑' : '↓'}{Math.abs(accDelta * 100).toFixed(1)}%
                    </span>
                  )}
                </div>
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-wider text-helix-muted mb-1">MAC Checks</div>
                <div className="flex items-baseline gap-1.5">
                  <span className="text-lg font-mono font-semibold text-white tabular-nums">{session.mac_checks_passed}</span>
                  <Shield size={12} className="text-green-400/60" />
                </div>
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-wider text-helix-muted mb-1">Checkpoints</div>
                <span className="text-lg font-mono font-semibold text-white tabular-nums">
                  {session.checkpoints_submitted}{totalCheckpoints > 0 ? ` / ${totalCheckpoints}` : ''}
                </span>
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-wider text-helix-muted mb-1">Workers</div>
                <span className="text-lg font-mono font-semibold text-white tabular-nums">
                  {session.workers_active ?? '—'}
                </span>
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* ── Completion state ─────────────────────────── */}
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

      {/* Failed state */}
      {session.status === 'failed' && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="rounded-3xl bg-helix-surface border border-red-500/20 p-8 text-center">
          <XCircle size={48} className="text-red-400 mx-auto mb-3" />
          <p className="text-xl font-medium text-red-300">Training Failed</p>
        </motion.div>
      )}

      {/* ── Charts (side by side) ─────────────────────────── */}
      <LossCurve data={losses} />

      {/* ── Cheater alert ─────────────────────────── */}
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

      {/* ── Results actions ─────────────────────────── */}
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
```

**Step 6: Remove the old `StatCard` component**

The `StatCard` component (lines 1102-1112) is no longer used by `LiveProgress` (it's still used by `HistoryDetailModal` so keep it, but verify). Check usage: if only `HistoryDetailModal` uses it, keep it; if nothing uses it, remove it.

**Step 7: Verify the app builds**

Run: `cd helix/dashboard && npx next build 2>&1 | tail -10`
Expected: Build succeeds

**Step 8: Commit**

```bash
git add helix/dashboard/src/app/train/page.tsx
git commit -m "feat: hero status card with phase ring, animated step counter, and inline stats"
```

---

### Task 4: Fix dashboard real-time display

**Files:**
- Modify: `helix/dashboard/src/app/dashboard/page.tsx:764-789`

The dashboard already gets WS data via `useDashboardSessions`, and we fixed the polling overwrite in Task 2. The remaining issue is the progress section needs a live step display.

**Step 1: Update the progress section to show more detail**

Replace the progress card (lines 764-789) with a version that shows the sub-status and animated step count:

```tsx
{/* Progress */}
<div className="bg-helix-surface border border-helix-border rounded-2xl p-6">
  <div className="flex items-center justify-between mb-1">
    <div className="flex items-center gap-3">
      <span className="text-3xl font-bold text-white tabular-nums font-mono">
        {activeSession?.current_step ?? 0}
      </span>
      <span className="text-lg text-helix-dim font-mono">/ {activeSession?.total_steps ?? 0}</span>
      <span className="text-sm text-helix-muted">steps</span>
    </div>
    <span className="text-sm text-helix-muted tabular-nums font-mono">{progress.toFixed(0)}%</span>
  </div>
  {activeSession?.phase_description && (
    <p className="text-sm text-helix-muted mb-3">{activeSession.phase_description}</p>
  )}
  <div className="h-2.5 bg-helix-border rounded-full overflow-hidden">
    <motion.div
      className="h-full rounded-full bg-white"
      initial={{ width: 0 }}
      animate={{ width: `${progress}%` }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      style={{ boxShadow: progress > 0 && progress < 100 ? '0 0 12px rgba(255,255,255,0.4)' : 'none' }}
    />
  </div>
  {activeSession?.status === 'complete' && (
    <div className="mt-2 text-xs text-green-400">Training complete</div>
  )}
  {activeSession?.status === 'failed' && (
    <div className="mt-2 text-xs text-red-400">Training failed</div>
  )}
</div>
```

**Step 2: Verify the app builds**

Run: `cd helix/dashboard && npx next build 2>&1 | tail -5`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add helix/dashboard/src/app/dashboard/page.tsx
git commit -m "feat: dashboard progress shows live step count with animated bar"
```

---

### Task 5: Visual polish and smoke test

**Files:**
- All 4 files from Tasks 1-4

**Step 1: Run the dev server and visually verify**

Run: `cd helix/dashboard && npm run dev`

Open `http://localhost:3000/train` and start a training session. Verify:
- Phase ring animates as phases progress
- Sub-status text cycles through MPC operations during phase 8
- Step counter updates in real-time (not stuck at 0)
- Loss and accuracy values update live
- Glowing progress bar advances smoothly
- Stat row shows loss/accuracy deltas with green/red colors
- Checkpoints show "X / Y" format

Open `http://localhost:3000/dashboard` and verify:
- Step counter updates live
- Loss/accuracy charts update in real-time
- Progress bar matches train page

**Step 2: Final commit**

```bash
git add -A
git commit -m "chore: training status redesign — hero card, live updates, animated stats"
```
