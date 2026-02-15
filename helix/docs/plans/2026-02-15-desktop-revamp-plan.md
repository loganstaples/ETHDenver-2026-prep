# Desktop App Revamp Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete frontend rebuild of the HELIX desktop app — single-view architecture with orb launch animation, expandable metric cards, monochrome + emerald palette, spring animations.

**Architecture:** Keep entire Tauri Rust backend, all TypeScript types/wrappers/hooks. Rebuild all React components from scratch. Single-view (no routing). Two app states: Idle (orb) and Active (dashboard). Progressive disclosure via click-to-expand cards.

**Tech Stack:** Tauri v2, React 18, TypeScript, Tailwind v4, Framer Motion 12, Lucide icons

---

### Task 1: Clean Scaffold — Remove Old Components and Pages

**Files:**
- Delete: `helix/desktop/src/pages/` (entire directory)
- Delete: `helix/desktop/src/components/` (entire directory)
- Modify: `helix/desktop/src/App.tsx`
- Modify: `helix/desktop/package.json`

**Step 1: Remove react-router-dom**

Run: `cd /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop && npm uninstall react-router-dom`

**Step 2: Delete old components and pages**

Run:
```bash
rm -rf /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop/src/pages
rm -rf /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop/src/components
```

**Step 3: Create new directory structure**

```bash
mkdir -p /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop/src/components/idle
mkdir -p /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop/src/components/active
mkdir -p /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop/src/components/active/cards
mkdir -p /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop/src/components/ui
mkdir -p /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop/src/components/settings
```

**Step 4: Create placeholder App.tsx**

Write `helix/desktop/src/App.tsx`:
```tsx
export default function App() {
  return (
    <div className="h-full w-full flex items-center justify-center bg-[#050505]">
      <p className="text-white/30 text-sm">HELIX — rebuilding...</p>
    </div>
  );
}
```

**Step 5: Verify build compiles**

Run: `cd /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop && npx tsc --noEmit`
Expected: No errors

**Step 6: Commit**

```bash
git add -A helix/desktop/src/
git add helix/desktop/package.json helix/desktop/package-lock.json
git commit -m "chore(desktop): clean scaffold for frontend rebuild

Remove all old components, pages, and react-router-dom.
Create new directory structure for single-view architecture.
Keep all hooks, types, tauri wrappers, and Rust backend."
```

---

### Task 2: Design Tokens — New globals.css

**Files:**
- Rewrite: `helix/desktop/src/styles/globals.css`

**Step 1: Write new globals.css**

Replace the entire file with new design tokens matching the design doc:

```css
@import "tailwindcss";
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@200;300;400;500;600&display=swap');

@theme {
  /* Surfaces */
  --color-bg-void: #050505;
  --color-bg-primary: #0A0A0A;
  --color-bg-card: #0A0A0A;
  --color-bg-card-hover: #111111;
  --color-bg-elevated: #141414;
  --color-bg-overlay: rgba(5, 5, 5, 0.85);

  /* Borders */
  --color-border-subtle: #1A1A1A;
  --color-border-hover: #252525;
  --color-border-focus: #333333;

  /* Text */
  --color-text-primary: #FFFFFF;
  --color-text-secondary: #A0A0A0;
  --color-text-tertiary: #505050;
  --color-text-disabled: #333333;

  /* Accent - Emerald */
  --color-accent: #34D399;
  --color-accent-dim: rgba(52, 211, 153, 0.15);
  --color-accent-glow: rgba(52, 211, 153, 0.08);

  /* Status */
  --color-status-healthy: #22C55E;
  --color-status-warning: #EAB308;
  --color-status-error: #EF4444;
}

/* Fonts */
@font-face {
  font-family: 'Inter';
  font-style: normal;
  font-weight: 100 900;
  font-display: swap;
  src: url('https://rsms.me/inter/font-files/InterVariable.woff2') format('woff2');
}

/* Reset */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html, body, #root {
  height: 100%;
  width: 100%;
  overflow: hidden;
}

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 13px;
  line-height: 1.4;
  color: var(--color-text-primary);
  background-color: var(--color-bg-void);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* Typography */
.font-mono {
  font-family: 'JetBrains Mono', 'SF Mono', monospace;
}

.text-metric-hero {
  font-family: 'JetBrains Mono', monospace;
  font-size: 42px;
  font-weight: 200;
  font-variant-numeric: tabular-nums;
  line-height: 1;
  letter-spacing: -0.02em;
}

.text-metric-lg {
  font-family: 'JetBrains Mono', monospace;
  font-size: 28px;
  font-weight: 300;
  font-variant-numeric: tabular-nums;
  line-height: 1.1;
}

.text-metric-md {
  font-family: 'JetBrains Mono', monospace;
  font-size: 18px;
  font-weight: 400;
  font-variant-numeric: tabular-nums;
  line-height: 1.2;
}

.text-metric-sm {
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  font-weight: 400;
  font-variant-numeric: tabular-nums;
  line-height: 1.3;
}

.text-label {
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  line-height: 1;
}

.text-label-sm {
  font-family: 'Inter', sans-serif;
  font-size: 10px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  line-height: 1;
}

.text-body {
  font-family: 'Inter', sans-serif;
  font-size: 13px;
  font-weight: 400;
  line-height: 1.5;
}

.tabular-nums {
  font-variant-numeric: tabular-nums;
}

/* Scrollbar */
::-webkit-scrollbar {
  width: 4px;
  height: 4px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: var(--color-border-subtle);
  border-radius: 2px;
}

::-webkit-scrollbar-thumb:hover {
  background: var(--color-border-hover);
}

/* Orb Animations */
@keyframes orb-glow-pulse {
  0%, 100% { opacity: 0.4; }
  50% { opacity: 0.7; }
}

@keyframes orb-ring-rotate {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

@keyframes orb-ring-spin-up {
  from { transform: rotate(0deg); }
  to { transform: rotate(1080deg); }
}

/* Status Dot */
@keyframes pulse-dot {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

.animate-pulse-dot {
  animation: pulse-dot 2s ease-in-out infinite;
}
```

**Step 2: Verify build**

Run: `cd /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop && npx tsc --noEmit`
Expected: No errors

**Step 3: Commit**

```bash
git add helix/desktop/src/styles/globals.css
git commit -m "style(desktop): new design tokens — monochrome + emerald

Lighter font weights for metric numbers (200-300 vs 600-700).
Emerald accent #34D399. Near-black void #050505.
Orb animation keyframes. Thinner scrollbar."
```

---

### Task 3: App State Hooks

**Files:**
- Create: `helix/desktop/src/hooks/useAppState.ts`
- Create: `helix/desktop/src/hooks/useExpandedCard.ts`

**Step 1: Write useAppState hook**

This manages the idle/active state and transition phases.

Write `helix/desktop/src/hooks/useAppState.ts`:
```tsx
import { useState, useCallback } from "react";

export type AppPhase = "idle" | "starting" | "active" | "stopping";

export function useAppState() {
  const [phase, setPhase] = useState<AppPhase>("idle");

  const beginStart = useCallback(() => {
    setPhase("starting");
  }, []);

  const completeStart = useCallback(() => {
    setPhase("active");
  }, []);

  const beginStop = useCallback(() => {
    setPhase("stopping");
  }, []);

  const completeStop = useCallback(() => {
    setPhase("idle");
  }, []);

  return { phase, beginStart, completeStart, beginStop, completeStop };
}
```

**Step 2: Write useExpandedCard hook**

Write `helix/desktop/src/hooks/useExpandedCard.ts`:
```tsx
import { useState, useCallback, useEffect } from "react";

export type CardId = "earnings" | "training" | "resources" | "network" | "loss" | "rounds" | null;

export function useExpandedCard() {
  const [expandedCard, setExpandedCard] = useState<CardId>(null);

  const expand = useCallback((id: CardId) => {
    setExpandedCard((prev) => (prev === id ? null : id));
  }, []);

  const collapse = useCallback(() => {
    setExpandedCard(null);
  }, []);

  // Escape key to collapse
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") collapse();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [collapse]);

  return { expandedCard, expand, collapse };
}
```

**Step 3: Verify build**

Run: `cd /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop && npx tsc --noEmit`
Expected: No errors

**Step 4: Commit**

```bash
git add helix/desktop/src/hooks/useAppState.ts helix/desktop/src/hooks/useExpandedCard.ts
git commit -m "feat(desktop): add useAppState and useExpandedCard hooks

useAppState: idle/starting/active/stopping phases.
useExpandedCard: single card expansion with Escape to collapse."
```

---

### Task 4: The Orb Component

**Files:**
- Create: `helix/desktop/src/components/idle/Orb.tsx`

**Step 1: Write the Orb**

This is the hero element. A 200px circle with emerald gradient, one orbital ring, glow pulse. Click triggers spin-up callback.

Write `helix/desktop/src/components/idle/Orb.tsx`:
```tsx
import { motion } from "framer-motion";

interface OrbProps {
  onActivate: () => void;
  phase: "idle" | "starting";
}

export function Orb({ onActivate, phase }: OrbProps) {
  const isStarting = phase === "starting";

  return (
    <div className="relative flex items-center justify-center">
      {/* Glow backdrop */}
      <div
        className="absolute rounded-full"
        style={{
          width: 280,
          height: 280,
          background: "radial-gradient(circle, rgba(52,211,153,0.08) 0%, transparent 70%)",
          animation: isStarting ? "none" : "orb-glow-pulse 3s ease-in-out infinite",
        }}
      />

      {/* Orbital ring */}
      <motion.div
        className="absolute rounded-full border"
        style={{
          width: 280,
          height: 280,
          borderColor: "rgba(52,211,153,0.15)",
        }}
        animate={{
          rotate: isStarting ? 1080 : 360,
          scale: isStarting ? 3 : 1,
          opacity: isStarting ? 0 : 1,
        }}
        transition={
          isStarting
            ? { duration: 0.8, ease: "easeIn" }
            : { rotate: { duration: 15, repeat: Infinity, ease: "linear" }, scale: { duration: 0 }, opacity: { duration: 0 } }
        }
      />

      {/* Core orb */}
      <motion.button
        onClick={onActivate}
        className="relative z-10 rounded-full cursor-pointer focus:outline-none"
        style={{
          width: 200,
          height: 200,
          background: "radial-gradient(circle at 50% 45%, rgba(52,211,153,0.12) 0%, #0A0A0A 60%, #050505 100%)",
          border: "1px solid rgba(52,211,153,0.1)",
        }}
        whileHover={{
          boxShadow: "0 0 60px rgba(52,211,153,0.15)",
          borderColor: "rgba(52,211,153,0.25)",
        }}
        whileTap={{ scale: 0.97 }}
        animate={
          isStarting
            ? { scale: 0, opacity: 0 }
            : { scale: 1, opacity: 1 }
        }
        transition={
          isStarting
            ? { duration: 0.6, delay: 0.2, ease: "easeIn" }
            : { type: "spring", stiffness: 300, damping: 30 }
        }
      >
        {/* Inner glow dot */}
        <div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full"
          style={{
            width: 4,
            height: 4,
            backgroundColor: "rgba(52,211,153,0.6)",
            boxShadow: "0 0 20px rgba(52,211,153,0.3)",
          }}
        />
      </motion.button>
    </div>
  );
}
```

**Step 2: Verify build**

Run: `cd /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop && npx tsc --noEmit`
Expected: No errors

**Step 3: Commit**

```bash
git add helix/desktop/src/components/idle/Orb.tsx
git commit -m "feat(desktop): add Orb component — hero launch button

200px circle with emerald gradient, orbital ring, glow pulse.
Spin-up animation on activate: ring expands and fades, orb shrinks."
```

---

### Task 5: IdleView — Orb + Wordmark

**Files:**
- Create: `helix/desktop/src/components/idle/IdleView.tsx`

**Step 1: Write IdleView**

Write `helix/desktop/src/components/idle/IdleView.tsx`:
```tsx
import { motion } from "framer-motion";
import { Settings } from "lucide-react";
import { Orb } from "./Orb";

interface IdleViewProps {
  onActivate: () => void;
  phase: "idle" | "starting";
  onOpenSettings: () => void;
}

export function IdleView({ onActivate, phase, onOpenSettings }: IdleViewProps) {
  return (
    <motion.div
      className="h-full w-full flex flex-col items-center justify-center bg-bg-void relative"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.4 }}
    >
      {/* Settings button — top right */}
      <button
        onClick={onOpenSettings}
        className="absolute top-5 right-5 p-2 text-text-tertiary hover:text-text-secondary transition-colors cursor-pointer"
      >
        <Settings size={18} />
      </button>

      {/* Orb */}
      <Orb onActivate={onActivate} phase={phase} />

      {/* Wordmark */}
      <motion.div
        className="mt-10 flex flex-col items-center gap-2"
        animate={phase === "starting" ? { opacity: 0, y: 20 } : { opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <h1
          className="text-text-primary tracking-[0.3em] text-sm font-light"
          style={{ fontFamily: "'Inter', sans-serif" }}
        >
          HELIX
        </h1>
        <p className="text-text-tertiary text-xs">
          Ready to compute
        </p>
      </motion.div>
    </motion.div>
  );
}
```

**Step 2: Verify build**

Run: `cd /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop && npx tsc --noEmit`

**Step 3: Commit**

```bash
git add helix/desktop/src/components/idle/IdleView.tsx
git commit -m "feat(desktop): add IdleView — orb + wordmark + settings"
```

---

### Task 6: Header Component for Active State

**Files:**
- Create: `helix/desktop/src/components/active/Header.tsx`

**Step 1: Write Header**

Minimal: HELIX wordmark left, session timer center-right, settings gear, stop button.

Write `helix/desktop/src/components/active/Header.tsx`:
```tsx
import { Settings, Square } from "lucide-react";
import { formatSessionTimer } from "../../lib/utils";

interface HeaderProps {
  sessionSeconds: number;
  onStop: () => void;
  onOpenSettings: () => void;
}

export function Header({ sessionSeconds, onStop, onOpenSettings }: HeaderProps) {
  return (
    <header className="h-12 flex items-center justify-between px-5 flex-shrink-0">
      {/* Left: wordmark */}
      <span className="text-text-tertiary tracking-[0.25em] text-xs font-light select-none">
        HELIX
      </span>

      {/* Right: timer + controls */}
      <div className="flex items-center gap-4">
        <span className="font-mono text-text-secondary text-xs tabular-nums">
          {formatSessionTimer(sessionSeconds)}
        </span>
        <button
          onClick={onOpenSettings}
          className="p-1.5 text-text-tertiary hover:text-text-secondary transition-colors cursor-pointer"
        >
          <Settings size={15} />
        </button>
        <button
          onClick={onStop}
          className="p-1.5 text-text-tertiary hover:text-status-error transition-colors cursor-pointer"
        >
          <Square size={13} fill="currentColor" />
        </button>
      </div>
    </header>
  );
}
```

**Step 2: Verify build**

Run: `cd /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop && npx tsc --noEmit`

**Step 3: Commit**

```bash
git add helix/desktop/src/components/active/Header.tsx
git commit -m "feat(desktop): add minimal Header — wordmark, timer, controls"
```

---

### Task 7: Sparkline UI Component

**Files:**
- Create: `helix/desktop/src/components/ui/Sparkline.tsx`

**Step 1: Write Sparkline**

Reusable SVG sparkline for metric history. Used in expanded resource and loss cards.

Write `helix/desktop/src/components/ui/Sparkline.tsx`:
```tsx
interface SparklineProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  fillOpacity?: number;
}

export function Sparkline({
  data,
  width = 200,
  height = 40,
  color = "#34D399",
  fillOpacity = 0.1,
}: SparklineProps) {
  if (data.length < 2) return null;

  const max = Math.max(...data, 1);
  const min = Math.min(...data, 0);
  const range = max - min || 1;

  const points = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * width;
      const y = height - ((v - min) / range) * (height - 4) - 2;
      return `${x},${y}`;
    })
    .join(" ");

  const fillPoints = `0,${height} ${points} ${width},${height}`;

  return (
    <svg width={width} height={height} className="overflow-visible">
      {/* Fill area */}
      <polygon points={fillPoints} fill={color} opacity={fillOpacity} />
      {/* Line */}
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
```

**Step 2: Verify build + Commit**

```bash
cd /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop && npx tsc --noEmit
git add helix/desktop/src/components/ui/Sparkline.tsx
git commit -m "feat(desktop): add Sparkline SVG component"
```

---

### Task 8: MetricCard — Reusable Expandable Card

**Files:**
- Create: `helix/desktop/src/components/active/cards/MetricCard.tsx`

**Step 1: Write MetricCard**

The core reusable card. Compact by default (big number + label). Clickable to expand. When another card is expanded, this one shrinks to a pill.

Write `helix/desktop/src/components/active/cards/MetricCard.tsx`:
```tsx
import { motion } from "framer-motion";
import { cn } from "../../../lib/utils";
import type { CardId } from "../../../hooks/useExpandedCard";
import type { ReactNode } from "react";

interface MetricCardProps {
  id: CardId;
  expandedCard: CardId;
  onExpand: (id: CardId) => void;
  value: string;
  label: string;
  expandedContent: ReactNode;
}

export function MetricCard({
  id,
  expandedCard,
  onExpand,
  value,
  label,
  expandedContent,
}: MetricCardProps) {
  const isExpanded = expandedCard === id;
  const isOtherExpanded = expandedCard !== null && expandedCard !== id;

  return (
    <motion.div
      layout
      onClick={() => onExpand(id)}
      className={cn(
        "rounded-xl cursor-pointer overflow-hidden transition-colors",
        isExpanded
          ? "bg-bg-card border border-accent/10 backdrop-blur-sm col-span-2 row-span-2"
          : isOtherExpanded
            ? "bg-bg-card/50"
            : "bg-bg-card hover:bg-bg-card-hover border border-transparent hover:border-border-subtle"
      )}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
    >
      {isOtherExpanded ? (
        /* Pill mode: just label */
        <div className="px-3 py-2.5 flex items-center gap-2">
          <span className="text-label text-text-tertiary">{label}</span>
          <span className="font-mono text-text-secondary text-xs">{value}</span>
        </div>
      ) : isExpanded ? (
        /* Expanded mode */
        <div className="p-5">
          <div className="flex items-baseline justify-between mb-4">
            <div>
              <span className="text-metric-lg text-text-primary">{value}</span>
              <span className="text-label text-text-tertiary ml-3">{label}</span>
            </div>
            <span className="text-label-sm text-text-tertiary">ESC to close</span>
          </div>
          <div>{expandedContent}</div>
        </div>
      ) : (
        /* Compact mode: big number + label */
        <div className="p-4 flex flex-col gap-1.5">
          <span className="text-metric-hero text-text-primary">{value}</span>
          <span className="text-label text-text-tertiary">{label}</span>
        </div>
      )}
    </motion.div>
  );
}
```

**Step 2: Verify build + Commit**

```bash
cd /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop && npx tsc --noEmit
git add helix/desktop/src/components/active/cards/MetricCard.tsx
git commit -m "feat(desktop): add MetricCard — expandable card with 3 modes

Compact (big number + label), Pill (when another card expanded),
Expanded (full detail content). Spring layout animations."
```

---

### Task 9: Card Detail Components — Earnings, Training, Resources

**Files:**
- Create: `helix/desktop/src/components/active/cards/EarningsDetail.tsx`
- Create: `helix/desktop/src/components/active/cards/TrainingDetail.tsx`
- Create: `helix/desktop/src/components/active/cards/ResourcesDetail.tsx`

**Step 1: Write EarningsDetail**

Write `helix/desktop/src/components/active/cards/EarningsDetail.tsx`:
```tsx
import type { TrainingStatus, SessionInfo } from "../../../lib/types";
import { Sparkline } from "../../ui/Sparkline";
import { formatNumber } from "../../../lib/utils";

interface EarningsDetailProps {
  training: TrainingStatus;
  sessionInfo: SessionInfo | null;
  earningsHistory: number[];
}

export function EarningsDetail({ training, sessionInfo, earningsHistory }: EarningsDetailProps) {
  return (
    <div className="space-y-5">
      {/* Earnings chart */}
      <div>
        <p className="text-label text-text-tertiary mb-2">Session earnings</p>
        <Sparkline data={earningsHistory} width={460} height={80} />
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-3 gap-4">
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">Rate</p>
          <p className="text-metric-md text-text-primary">
            {sessionInfo ? formatNumber(sessionInfo.earnings_rate_per_hour) : "—"}
            <span className="text-text-tertiary text-xs ml-1">HLX/hr</span>
          </p>
        </div>
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">Rounds</p>
          <p className="text-metric-md text-text-primary">{training.rounds_completed}</p>
        </div>
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">Proofs</p>
          <p className="text-metric-md text-text-primary">{training.proofs_generated}</p>
        </div>
      </div>

      {/* Total vs session */}
      <div className="flex gap-6 pt-2 border-t border-border-subtle">
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">Total earned</p>
          <p className="text-metric-sm text-text-secondary">{formatNumber(training.total_earned, 4)} HLX</p>
        </div>
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">This session</p>
          <p className="text-metric-sm text-accent">{formatNumber(training.session_earned, 4)} HLX</p>
        </div>
      </div>
    </div>
  );
}
```

**Step 2: Write TrainingDetail**

Write `helix/desktop/src/components/active/cards/TrainingDetail.tsx`:
```tsx
import type { TrainingStatus } from "../../../lib/types";
import { Sparkline } from "../../ui/Sparkline";

const PHASES = ["Assigned", "Training", "Proving", "Submitted", "Verified"];

interface TrainingDetailProps {
  training: TrainingStatus;
  lossHistory: number[];
}

export function TrainingDetail({ training, lossHistory }: TrainingDetailProps) {
  const phaseIndex = PHASES.findIndex(
    (p) => p.toLowerCase() === training.phase.toLowerCase()
  );

  return (
    <div className="space-y-5">
      {/* Phase timeline */}
      <div>
        <p className="text-label text-text-tertiary mb-3">Phase</p>
        <div className="flex items-center gap-2">
          {PHASES.map((p, i) => (
            <div key={p} className="flex items-center gap-2">
              <div
                className="w-2 h-2 rounded-full"
                style={{
                  backgroundColor:
                    i <= phaseIndex ? "#34D399" : "#252525",
                }}
              />
              <span
                className="text-xs"
                style={{
                  color: i <= phaseIndex ? "#A0A0A0" : "#505050",
                }}
              >
                {p}
              </span>
              {i < PHASES.length - 1 && (
                <div
                  className="w-4 h-px"
                  style={{
                    backgroundColor: i < phaseIndex ? "#34D399" : "#252525",
                  }}
                />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Progress */}
      <div>
        <div className="flex justify-between mb-1.5">
          <p className="text-label-sm text-text-tertiary">Progress</p>
          <p className="text-metric-sm text-text-secondary">
            {training.current_step ?? 0}/{training.total_steps ?? 0}
          </p>
        </div>
        <div className="h-1 bg-border-subtle rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${training.progress * 100}%`,
              backgroundColor: "#34D399",
            }}
          />
        </div>
      </div>

      {/* Loss curve */}
      <div>
        <p className="text-label text-text-tertiary mb-2">Loss curve</p>
        <Sparkline data={lossHistory} width={460} height={80} color="#EAB308" />
      </div>

      {/* Info grid */}
      <div className="grid grid-cols-2 gap-4 pt-2 border-t border-border-subtle">
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">Model</p>
          <p className="text-body text-text-secondary">{training.model_name || "—"}</p>
        </div>
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">Current loss</p>
          <p className="text-metric-sm text-text-secondary">
            {training.current_loss?.toFixed(4) ?? "—"}
          </p>
        </div>
      </div>
    </div>
  );
}
```

**Step 3: Write ResourcesDetail**

Write `helix/desktop/src/components/active/cards/ResourcesDetail.tsx`:
```tsx
import type { SystemMetrics } from "../../../lib/types";
import { Sparkline } from "../../ui/Sparkline";
import { formatBytes } from "../../../lib/utils";

interface ResourcesDetailProps {
  metrics: SystemMetrics;
  cpuHistory: number[];
  memHistory: number[];
  gpuHistory: number[];
}

export function ResourcesDetail({ metrics, cpuHistory, memHistory, gpuHistory }: ResourcesDetailProps) {
  const memPercent = metrics.memory_total_mb > 0
    ? (metrics.memory_used_mb / metrics.memory_total_mb) * 100
    : 0;

  return (
    <div className="space-y-5">
      {/* CPU */}
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-baseline gap-3 mb-2">
            <p className="text-label text-text-tertiary">CPU</p>
            <p className="text-metric-md text-text-primary">
              {metrics.cpu_usage_percent.toFixed(0)}%
            </p>
          </div>
          <Sparkline data={cpuHistory} width={200} height={32} />
        </div>
      </div>

      {/* Memory */}
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-baseline gap-3 mb-2">
            <p className="text-label text-text-tertiary">Memory</p>
            <p className="text-metric-md text-text-primary">
              {formatBytes(metrics.memory_used_mb)}
              <span className="text-text-tertiary text-xs ml-1">
                / {formatBytes(metrics.memory_total_mb)}
              </span>
            </p>
          </div>
          <Sparkline data={memHistory} width={200} height={32} color="#3B82F6" />
        </div>
        <span className="text-metric-sm text-text-tertiary ml-4">{memPercent.toFixed(0)}%</span>
      </div>

      {/* GPU (if available) */}
      {metrics.gpu_usage_percent !== null && (
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-baseline gap-3 mb-2">
              <p className="text-label text-text-tertiary">GPU</p>
              <p className="text-metric-md text-text-primary">
                {metrics.gpu_usage_percent?.toFixed(0) ?? "—"}%
              </p>
            </div>
            <Sparkline data={gpuHistory} width={200} height={32} color="#A855F7" />
          </div>
        </div>
      )}

      {metrics.gpu_usage_percent === null && (
        <div className="pt-2 border-t border-border-subtle">
          <p className="text-label-sm text-text-tertiary">No GPU detected</p>
        </div>
      )}
    </div>
  );
}
```

**Step 4: Verify build + Commit**

```bash
cd /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop && npx tsc --noEmit
git add helix/desktop/src/components/active/cards/EarningsDetail.tsx \
        helix/desktop/src/components/active/cards/TrainingDetail.tsx \
        helix/desktop/src/components/active/cards/ResourcesDetail.tsx
git commit -m "feat(desktop): add detail components — Earnings, Training, Resources

Expanded views for first 3 metric cards with charts, stats, and sparklines."
```

---

### Task 10: Card Detail Components — Network, Loss, Rounds

**Files:**
- Create: `helix/desktop/src/components/active/cards/NetworkDetail.tsx`
- Create: `helix/desktop/src/components/active/cards/LossDetail.tsx`
- Create: `helix/desktop/src/components/active/cards/RoundsDetail.tsx`

**Step 1: Write NetworkDetail**

Write `helix/desktop/src/components/active/cards/NetworkDetail.tsx`:
```tsx
import type { PeerInfo } from "../../../lib/types";
import { truncateHash } from "../../../lib/utils";

interface NetworkDetailProps {
  peers: PeerInfo[];
}

export function NetworkDetail({ peers }: NetworkDetailProps) {
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-5 gap-2 text-label-sm text-text-tertiary pb-2 border-b border-border-subtle">
        <span>Peer</span>
        <span>Address</span>
        <span>Role</span>
        <span className="text-right">Reputation</span>
        <span className="text-right">Last seen</span>
      </div>
      {peers.length === 0 ? (
        <p className="text-body text-text-tertiary py-4 text-center">No peers connected</p>
      ) : (
        peers.map((peer) => (
          <div key={peer.id} className="grid grid-cols-5 gap-2 items-center py-1.5">
            <span className="text-metric-sm text-text-secondary font-mono">
              {truncateHash(peer.id)}
            </span>
            <span className="text-xs text-text-tertiary truncate">{peer.address}</span>
            <span className="text-xs text-text-secondary">{peer.role}</span>
            <span className="text-metric-sm text-text-secondary text-right">{peer.reputation}</span>
            <span className="text-xs text-text-tertiary text-right">
              {peer.last_seen_secs_ago < 5 ? "now" : `${peer.last_seen_secs_ago}s ago`}
            </span>
          </div>
        ))
      )}
    </div>
  );
}
```

**Step 2: Write LossDetail**

Write `helix/desktop/src/components/active/cards/LossDetail.tsx`:
```tsx
import type { TrainingStatus } from "../../../lib/types";
import { Sparkline } from "../../ui/Sparkline";

interface LossDetailProps {
  training: TrainingStatus;
  lossHistory: number[];
}

export function LossDetail({ training, lossHistory }: LossDetailProps) {
  const recentLoss = lossHistory.slice(-10);
  const avgRecent = recentLoss.length > 0
    ? recentLoss.reduce((a, b) => a + b, 0) / recentLoss.length
    : 0;
  const trend = recentLoss.length >= 2
    ? recentLoss[recentLoss.length - 1] - recentLoss[0]
    : 0;

  return (
    <div className="space-y-5">
      {/* Loss curve — full width */}
      <div>
        <p className="text-label text-text-tertiary mb-2">Loss over training</p>
        <Sparkline data={lossHistory} width={460} height={100} color="#EAB308" fillOpacity={0.05} />
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 pt-2 border-t border-border-subtle">
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">Current</p>
          <p className="text-metric-md text-text-primary">
            {training.current_loss?.toFixed(4) ?? "—"}
          </p>
        </div>
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">Avg (last 10)</p>
          <p className="text-metric-md text-text-secondary">{avgRecent.toFixed(4)}</p>
        </div>
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">Trend</p>
          <p className={`text-metric-md ${trend <= 0 ? "text-status-healthy" : "text-status-warning"}`}>
            {trend <= 0 ? "↓" : "↑"} {Math.abs(trend).toFixed(4)}
          </p>
        </div>
      </div>
    </div>
  );
}
```

**Step 3: Write RoundsDetail**

Write `helix/desktop/src/components/active/cards/RoundsDetail.tsx`:
```tsx
import type { TrainingStatus } from "../../../lib/types";

interface RoundsDetailProps {
  training: TrainingStatus;
}

export function RoundsDetail({ training }: RoundsDetailProps) {
  return (
    <div className="space-y-4">
      {/* Summary stats */}
      <div className="grid grid-cols-3 gap-4">
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">Completed</p>
          <p className="text-metric-lg text-text-primary">{training.rounds_completed}</p>
        </div>
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">Proofs generated</p>
          <p className="text-metric-lg text-text-primary">{training.proofs_generated}</p>
        </div>
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">Current round</p>
          <p className="text-metric-lg text-accent">
            #{training.current_round ?? "—"}
          </p>
        </div>
      </div>

      {/* Status */}
      <div className="pt-3 border-t border-border-subtle">
        <div className="flex items-center gap-2">
          <div
            className="w-2 h-2 rounded-full"
            style={{
              backgroundColor: training.active ? "#22C55E" : "#505050",
            }}
          />
          <span className="text-body text-text-secondary">
            {training.active ? `Round #${training.current_round} — ${training.phase}` : "Idle"}
          </span>
        </div>
      </div>
    </div>
  );
}
```

**Step 4: Verify build + Commit**

```bash
cd /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop && npx tsc --noEmit
git add helix/desktop/src/components/active/cards/NetworkDetail.tsx \
        helix/desktop/src/components/active/cards/LossDetail.tsx \
        helix/desktop/src/components/active/cards/RoundsDetail.tsx
git commit -m "feat(desktop): add detail components — Network, Loss, Rounds

Peer table, loss convergence chart, round summary for remaining cards."
```

---

### Task 11: ActivityFeed Component

**Files:**
- Create: `helix/desktop/src/components/active/ActivityFeed.tsx`

**Step 1: Write ActivityFeed**

Compact scrolling event log at the bottom of the dashboard.

Write `helix/desktop/src/components/active/ActivityFeed.tsx`:
```tsx
import { motion, AnimatePresence } from "framer-motion";
import type { ActivityEvent } from "../../lib/types";
import { cn } from "../../lib/utils";

interface ActivityFeedProps {
  events: ActivityEvent[];
}

const typeColor: Record<string, string> = {
  info: "text-text-tertiary",
  success: "text-status-healthy",
  warning: "text-status-warning",
  error: "text-status-error",
  earn: "text-accent",
};

export function ActivityFeed({ events }: ActivityFeedProps) {
  const recent = events.slice(0, 8);

  return (
    <div className="bg-bg-card rounded-xl p-3 overflow-hidden">
      <div className="flex items-center justify-between mb-2 px-1">
        <span className="text-label text-text-tertiary">Activity</span>
        <span className="text-label-sm text-text-tertiary">{events.length} events</span>
      </div>
      <div className="space-y-0.5 max-h-[140px] overflow-y-auto">
        <AnimatePresence mode="popLayout">
          {recent.map((event) => {
            const time = new Date(event.timestamp);
            const timeStr = time.toLocaleTimeString("en-US", {
              hour12: false,
              hour: "2-digit",
              minute: "2-digit",
              second: "2-digit",
            });
            return (
              <motion.div
                key={event.id}
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ type: "spring", stiffness: 500, damping: 40 }}
                className="flex items-center gap-3 px-1 py-1 rounded"
              >
                <span className="font-mono text-[10px] text-text-tertiary tabular-nums flex-shrink-0">
                  {timeStr}
                </span>
                <span className="text-xs text-text-secondary truncate flex-1">
                  {event.message}
                </span>
                <span className={cn("text-[10px] uppercase font-medium flex-shrink-0", typeColor[event.event_type] || "text-text-tertiary")}>
                  {event.event_type}
                </span>
              </motion.div>
            );
          })}
        </AnimatePresence>
        {events.length === 0 && (
          <p className="text-xs text-text-tertiary text-center py-3">No activity yet</p>
        )}
      </div>
    </div>
  );
}
```

**Step 2: Verify build + Commit**

```bash
cd /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop && npx tsc --noEmit
git add helix/desktop/src/components/active/ActivityFeed.tsx
git commit -m "feat(desktop): add ActivityFeed — compact animated event log"
```

---

### Task 12: CardGrid — Dashboard Layout with Expand/Collapse

**Files:**
- Create: `helix/desktop/src/components/active/CardGrid.tsx`

**Step 1: Write CardGrid**

This is the core layout. Arranges 6 MetricCards in a 3-column grid + ActivityFeed at bottom. Manages the expand/collapse layout transitions.

Write `helix/desktop/src/components/active/CardGrid.tsx`:
```tsx
import { LayoutGroup } from "framer-motion";
import { MetricCard } from "./cards/MetricCard";
import { EarningsDetail } from "./cards/EarningsDetail";
import { TrainingDetail } from "./cards/TrainingDetail";
import { ResourcesDetail } from "./cards/ResourcesDetail";
import { NetworkDetail } from "./cards/NetworkDetail";
import { LossDetail } from "./cards/LossDetail";
import { RoundsDetail } from "./cards/RoundsDetail";
import { ActivityFeed } from "./ActivityFeed";
import { useTrainingStatus } from "../../hooks/useTrainingStatus";
import { useSystemMetrics } from "../../hooks/useSystemMetrics";
import { useNetworkPeers } from "../../hooks/useNetworkPeers";
import { useActivityLog } from "../../hooks/useActivityLog";
import { useMetricHistory, useMultiMetricHistory } from "../../hooks/useMetricHistory";
import { formatNumber, formatBytes } from "../../lib/utils";
import type { CardId } from "../../hooks/useExpandedCard";
import type { SessionInfo } from "../../lib/types";
import { useEffect, useState } from "react";
import { getSessionInfo } from "../../lib/tauri";

interface CardGridProps {
  expandedCard: CardId;
  onExpand: (id: CardId) => void;
}

export function CardGrid({ expandedCard, onExpand }: CardGridProps) {
  const training = useTrainingStatus();
  const metrics = useSystemMetrics();
  const peers = useNetworkPeers();
  const events = useActivityLog(true);
  const { history: earningsHistory, push: pushEarnings } = useMetricHistory();
  const { history: lossHistory, push: pushLoss } = useMetricHistory();
  const { cpu, memory, gpu, pushAll } = useMultiMetricHistory();
  const [sessionInfo, setSessionInfo] = useState<SessionInfo | null>(null);

  // Push metric history on updates
  useEffect(() => {
    pushEarnings(training.session_earned);
  }, [training.session_earned, pushEarnings]);

  useEffect(() => {
    if (training.current_loss !== null) pushLoss(training.current_loss);
  }, [training.current_loss, pushLoss]);

  useEffect(() => {
    const memPercent = metrics.memory_total_mb > 0
      ? (metrics.memory_used_mb / metrics.memory_total_mb) * 100
      : 0;
    pushAll(metrics.cpu_usage_percent, memPercent, metrics.gpu_usage_percent ?? 0);
  }, [metrics, pushAll]);

  useEffect(() => {
    getSessionInfo().then(setSessionInfo).catch(() => {});
    const interval = setInterval(() => {
      getSessionInfo().then(setSessionInfo).catch(() => {});
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const cards = [
    {
      id: "earnings" as CardId,
      value: formatNumber(training.session_earned),
      label: "HLX earned",
      detail: (
        <EarningsDetail
          training={training}
          sessionInfo={sessionInfo}
          earningsHistory={earningsHistory}
        />
      ),
    },
    {
      id: "training" as CardId,
      value: training.current_round !== null ? `#${training.current_round}` : "—",
      label: training.phase,
      detail: <TrainingDetail training={training} lossHistory={lossHistory} />,
    },
    {
      id: "resources" as CardId,
      value: `${metrics.cpu_usage_percent.toFixed(0)}%`,
      label: `CPU · ${formatBytes(metrics.memory_used_mb)} mem`,
      detail: (
        <ResourcesDetail
          metrics={metrics}
          cpuHistory={cpu}
          memHistory={memory}
          gpuHistory={gpu}
        />
      ),
    },
    {
      id: "network" as CardId,
      value: `${peers.length}`,
      label: "peers",
      detail: <NetworkDetail peers={peers} />,
    },
    {
      id: "loss" as CardId,
      value: training.current_loss?.toFixed(4) ?? "—",
      label: "loss",
      detail: <LossDetail training={training} lossHistory={lossHistory} />,
    },
    {
      id: "rounds" as CardId,
      value: `${training.rounds_completed}`,
      label: "rounds",
      detail: <RoundsDetail training={training} />,
    },
  ];

  return (
    <div className="flex-1 flex flex-col gap-3 px-5 pb-4 overflow-hidden">
      <LayoutGroup>
        <div className="grid grid-cols-3 gap-3 flex-1 min-h-0">
          {cards.map((card) => (
            <MetricCard
              key={card.id}
              id={card.id}
              expandedCard={expandedCard}
              onExpand={onExpand}
              value={card.value}
              label={card.label}
              expandedContent={card.detail}
            />
          ))}
        </div>
      </LayoutGroup>
      <ActivityFeed events={events} />
    </div>
  );
}
```

**Step 2: Verify build + Commit**

```bash
cd /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop && npx tsc --noEmit
git add helix/desktop/src/components/active/CardGrid.tsx
git commit -m "feat(desktop): add CardGrid — 3-col grid with expand/collapse

Wires all 6 metric cards with data hooks, history tracking,
and session info polling. Activity feed at bottom."
```

---

### Task 13: ActiveView — Full Dashboard Composition

**Files:**
- Create: `helix/desktop/src/components/active/ActiveView.tsx`

**Step 1: Write ActiveView**

Composes Header + CardGrid. Handles stop action.

Write `helix/desktop/src/components/active/ActiveView.tsx`:
```tsx
import { motion } from "framer-motion";
import { Header } from "./Header";
import { CardGrid } from "./CardGrid";
import { useNodeStatus } from "../../hooks/useNodeStatus";
import { useSessionTimer } from "../../hooks/useSessionTimer";
import { useExpandedCard } from "../../hooks/useExpandedCard";

interface ActiveViewProps {
  onStop: () => void;
  onOpenSettings: () => void;
}

export function ActiveView({ onStop, onOpenSettings }: ActiveViewProps) {
  const { status } = useNodeStatus();
  const sessionSeconds = useSessionTimer(status.running, status.uptime_secs);
  const { expandedCard, expand } = useExpandedCard();

  return (
    <motion.div
      className="h-full w-full flex flex-col bg-bg-void"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3, delay: 0.2 }}
    >
      <Header
        sessionSeconds={sessionSeconds}
        onStop={onStop}
        onOpenSettings={onOpenSettings}
      />
      <CardGrid expandedCard={expandedCard} onExpand={expand} />
    </motion.div>
  );
}
```

**Step 2: Verify build + Commit**

```bash
cd /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop && npx tsc --noEmit
git add helix/desktop/src/components/active/ActiveView.tsx
git commit -m "feat(desktop): add ActiveView — dashboard composition

Header + CardGrid with session timer and expand state."
```

---

### Task 14: SettingsOverlay

**Files:**
- Create: `helix/desktop/src/components/settings/SettingsOverlay.tsx`

**Step 1: Write SettingsOverlay**

Full-screen overlay that slides up. Glassmorphism backdrop. Expandable sections with toggles and inputs.

Write `helix/desktop/src/components/settings/SettingsOverlay.tsx`:
```tsx
import { motion, AnimatePresence } from "framer-motion";
import { X } from "lucide-react";
import { useConfig } from "../../hooks/useConfig";
import { useState } from "react";
import type { NodeConfig } from "../../lib/types";

interface SettingsOverlayProps {
  open: boolean;
  onClose: () => void;
}

function SettingRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between py-2.5">
      <span className="text-body text-text-secondary">{label}</span>
      {children}
    </div>
  );
}

function Toggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className="w-9 h-5 rounded-full transition-colors relative cursor-pointer"
      style={{ backgroundColor: checked ? "#34D399" : "#252525" }}
    >
      <div
        className="absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform"
        style={{ transform: checked ? "translateX(17px)" : "translateX(2px)" }}
      />
    </button>
  );
}

function NumberInput({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  return (
    <input
      type="number"
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className="w-20 bg-bg-card border border-border-subtle rounded px-2 py-1 text-metric-sm text-text-primary text-right focus:outline-none focus:border-accent/30"
    />
  );
}

function TextInput({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <input
      type="text"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-48 bg-bg-card border border-border-subtle rounded px-2 py-1 text-metric-sm text-text-primary focus:outline-none focus:border-accent/30"
    />
  );
}

export function SettingsOverlay({ open, onClose }: SettingsOverlayProps) {
  const { config, updateConfig } = useConfig();
  const [local, setLocal] = useState<NodeConfig | null>(null);

  // Initialize local state from config
  const current = local ?? config;

  const update = (patch: Partial<NodeConfig>) => {
    if (!current) return;
    const next = { ...current, ...patch };
    setLocal(next);
    updateConfig(next);
  };

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-end justify-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
        >
          {/* Backdrop */}
          <div
            className="absolute inset-0 backdrop-blur-md"
            style={{ backgroundColor: "rgba(5,5,5,0.85)" }}
            onClick={onClose}
          />

          {/* Panel */}
          <motion.div
            className="relative w-full max-w-xl bg-bg-primary/95 backdrop-blur-xl rounded-t-2xl border-t border-border-subtle max-h-[80vh] overflow-y-auto"
            initial={{ y: "100%" }}
            animate={{ y: 0 }}
            exit={{ y: "100%" }}
            transition={{ type: "spring", stiffness: 300, damping: 35 }}
          >
            {/* Header */}
            <div className="sticky top-0 flex items-center justify-between px-6 py-4 bg-bg-primary/95 backdrop-blur-xl border-b border-border-subtle z-10">
              <span className="text-label text-text-secondary">Settings</span>
              <button onClick={onClose} className="p-1 text-text-tertiary hover:text-text-secondary cursor-pointer">
                <X size={16} />
              </button>
            </div>

            {current && (
              <div className="px-6 py-4 space-y-6">
                {/* Resources */}
                <section>
                  <h3 className="text-label text-text-tertiary mb-3">Resources</h3>
                  <div className="divide-y divide-border-subtle">
                    <SettingRow label="CPU Threads">
                      <NumberInput value={current.cpu_threads} onChange={(v) => update({ cpu_threads: v })} />
                    </SettingRow>
                    <SettingRow label="GPU Memory Limit (MB)">
                      <NumberInput value={current.gpu_memory_limit_mb} onChange={(v) => update({ gpu_memory_limit_mb: v })} />
                    </SettingRow>
                    <SettingRow label="Max Concurrent Tasks">
                      <NumberInput value={current.max_concurrent_tasks} onChange={(v) => update({ max_concurrent_tasks: v })} />
                    </SettingRow>
                  </div>
                </section>

                {/* Network */}
                <section>
                  <h3 className="text-label text-text-tertiary mb-3">Network</h3>
                  <div className="divide-y divide-border-subtle">
                    <SettingRow label="Listen Address">
                      <TextInput value={current.listen_address} onChange={(v) => update({ listen_address: v })} />
                    </SettingRow>
                    <SettingRow label="Aggregator Address">
                      <TextInput value={current.aggregator_address} onChange={(v) => update({ aggregator_address: v })} />
                    </SettingRow>
                    <SettingRow label="RPC Port">
                      <NumberInput value={current.rpc_port} onChange={(v) => update({ rpc_port: v })} />
                    </SettingRow>
                    <SettingRow label="TLS">
                      <Toggle checked={current.use_tls} onChange={(v) => update({ use_tls: v })} />
                    </SettingRow>
                  </div>
                </section>

                {/* Training */}
                <section>
                  <h3 className="text-label text-text-tertiary mb-3">Training</h3>
                  <div className="divide-y divide-border-subtle">
                    <SettingRow label="Local Epochs">
                      <NumberInput value={current.local_epochs} onChange={(v) => update({ local_epochs: v })} />
                    </SettingRow>
                    <SettingRow label="Batch Size">
                      <NumberInput value={current.batch_size} onChange={(v) => update({ batch_size: v })} />
                    </SettingRow>
                    <SettingRow label="Generate Proofs">
                      <Toggle checked={current.generate_proofs} onChange={(v) => update({ generate_proofs: v })} />
                    </SettingRow>
                  </div>
                </section>

                {/* Version */}
                <section className="pb-4">
                  <p className="text-label-sm text-text-tertiary">HELIX v0.1.0</p>
                </section>
              </div>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
```

**Step 2: Verify build + Commit**

```bash
cd /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop && npx tsc --noEmit
git add helix/desktop/src/components/settings/SettingsOverlay.tsx
git commit -m "feat(desktop): add SettingsOverlay — full-screen slide-up panel

Glassmorphism backdrop, spring animation, resources/network/training sections.
Clean toggles and number inputs."
```

---

### Task 15: App.tsx — Wire Everything Together

**Files:**
- Rewrite: `helix/desktop/src/App.tsx`

**Step 1: Write final App.tsx**

This is the root component. Manages idle↔active transitions, settings overlay, and delegates to IdleView/ActiveView.

Write `helix/desktop/src/App.tsx`:
```tsx
import { AnimatePresence } from "framer-motion";
import { useState, useCallback } from "react";
import { IdleView } from "./components/idle/IdleView";
import { ActiveView } from "./components/active/ActiveView";
import { SettingsOverlay } from "./components/settings/SettingsOverlay";
import { useAppState } from "./hooks/useAppState";
import { useNodeStatus } from "./hooks/useNodeStatus";

export default function App() {
  const { phase, beginStart, completeStart, beginStop, completeStop } = useAppState();
  const { start, stop } = useNodeStatus();
  const [settingsOpen, setSettingsOpen] = useState(false);

  const handleActivate = useCallback(async () => {
    beginStart();
    await start();
    // Wait for orb animation to finish before showing dashboard
    setTimeout(() => {
      completeStart();
    }, 1000);
  }, [beginStart, completeStart, start]);

  const handleStop = useCallback(async () => {
    beginStop();
    await stop();
    setTimeout(() => {
      completeStop();
    }, 600);
  }, [beginStop, completeStop, stop]);

  const isIdle = phase === "idle" || phase === "starting";
  const isActive = phase === "active" || phase === "stopping";

  return (
    <div className="h-full w-full bg-bg-void">
      <AnimatePresence mode="wait">
        {isIdle && (
          <IdleView
            key="idle"
            onActivate={handleActivate}
            phase={phase as "idle" | "starting"}
            onOpenSettings={() => setSettingsOpen(true)}
          />
        )}
        {isActive && (
          <ActiveView
            key="active"
            onStop={handleStop}
            onOpenSettings={() => setSettingsOpen(true)}
          />
        )}
      </AnimatePresence>
      <SettingsOverlay open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  );
}
```

**Step 2: Verify build**

Run: `cd /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop && npx tsc --noEmit`
Expected: No errors

**Step 3: Verify dev server starts**

Run: `cd /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop && npx vite build`
Expected: Build succeeds without errors

**Step 4: Commit**

```bash
git add helix/desktop/src/App.tsx
git commit -m "feat(desktop): wire App.tsx — idle/active transitions + settings

AnimatePresence for smooth view switching. Orb triggers start_node,
stop button triggers stop_node. Settings overlay accessible from both states."
```

---

### Task 16: Visual Polish Pass

**Files:**
- Modify: Various components for visual refinement

**Step 1: Test the full app visually**

Run: `cd /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop && npm run dev`

Open in browser at localhost:1420 and verify:
1. Idle view: orb centered, wordmark below, settings gear top-right
2. Click orb: spin-up animation → dashboard appears
3. Dashboard: 6 cards in 3-column grid, activity feed at bottom
4. Click a card: it expands, others shrink to pills
5. Press Escape: card collapses
6. Settings gear: overlay slides up from bottom
7. Stop button: dashboard fades, back to orb

**Step 2: Fix any visual issues found during testing**

Adjust spacing, font sizes, animation timing, and colors as needed. Common adjustments:
- Card grid height/spacing
- Animation timing (spring stiffness/damping)
- Font weight on metric numbers
- Orb glow intensity

**Step 3: Commit**

```bash
git add -A helix/desktop/src/
git commit -m "style(desktop): visual polish pass

Tuned animations, spacing, and typography after visual testing."
```

---

### Task 17: Final Build Verification

**Step 1: Clean build check**

```bash
cd /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop
rm -rf dist
npx tsc --noEmit && npx vite build
```

Expected: Build succeeds, no TypeScript errors, dist/ populated

**Step 2: Verify Tauri build compiles** (optional, takes longer)

```bash
cd /Users/loganstaples/hackathons/ethdenver26/prep-work/helix/desktop
npm run tauri build -- --debug 2>&1 | head -20
```

Expected: At minimum the frontend build step succeeds

**Step 3: Final commit if any remaining changes**

```bash
git add -A helix/desktop/
git commit -m "chore(desktop): final build verification — all clear"
```
