# Desktop App Revamp Design

## Overview

Complete frontend rebuild of the HELIX desktop app for compute providers. Keep the Tauri v2 Rust backend, all data hooks, and TypeScript types. Rebuild all React components for a single-view architecture with progressive disclosure.

**Design pillars:** Apple minimalism meets Bloomberg terminal. Monochrome + emerald accent. Spring animations. Maximally simple by default, powerful on demand.

## App States

### Idle (LaunchPad)
- Full-screen near-black void (#050505)
- Centered: the Orb (200px circle, subtle emerald radial gradient, one thin orbital ring rotating at 15s period, gentle glow pulse at 3s)
- Below: "HELIX" wordmark, thin uppercase
- Top-right: settings gear icon
- Orb hover: glow brightens

### Transition (Idle → Active)
- Press orb: ring accelerates → expands outward → glow flashes → orb fades
- Dashboard cards bloom in from center with staggered spring animation
- ~1s total transition

### Active (Dashboard)
- Single view. No pages, no routing, no sidebar.
- Minimal header: HELIX wordmark (left), session timer (center-right), settings gear, stop button
- 3-column card grid (6 metric cards) + bottom activity feed
- Cards are ultra-compact by default: one big number + one label
- Click any card to expand it in-place (~60-70% of grid area, others shrink to pills)
- Escape or click expanded card to collapse

### Transition (Active → Idle)
- Press stop: reverse animation, panels collapse toward center, orb reforms
- Back to idle state

## Dashboard Cards (6 total)

### Row 1
1. **Earnings** — compact: total HLX earned. Expanded: rate/hr, earnings chart, rounds breakdown, proof count
2. **Training** — compact: current round + phase. Expanded: phase dots, progress bar, step count, loss curve, model info
3. **Resources** — compact: CPU % + MEM usage. Expanded: CPU/MEM/GPU sparklines, detailed usage

### Row 2
4. **Network** — compact: peer count. Expanded: peer table (roles, latency, reputation)
5. **Loss** — compact: current loss value. Expanded: loss curve over steps, convergence metrics
6. **Rounds** — compact: rounds completed count. Expanded: round history table (status, duration, earnings)

### Bottom
- **Activity Feed** — compact scrolling event log (timestamp + message + type badge)

## Visual Language

### Colors
- Background: #050505 (main), #0A0A0A (cards), #111111 (card hover)
- Accent: Emerald #34D399 — ONLY for: orb glow, active states, primary metric in expanded cards, interactive elements
- Text: #FFFFFF (primary numbers), #A0A0A0 (labels), #505050 (disabled)
- Status: green #22C55E (healthy), yellow #EAB308 (warning), red #EF4444 (error) — only when relevant
- Borders: #1A1A1A (default), #252525 (hover). Extremely subtle.

### Typography
- Inter: all UI text (clean, Apple-like)
- JetBrains Mono: numbers/metrics only (terminal density)
- Big numbers: 32-48px, weight 200-300. Labels: 11-12px, uppercase, letter-spaced

### Animation
- Spring physics for all layout transitions (Framer Motion, stiffness ~300, damping ~30)
- 50ms stagger between card entrances
- No easing curves — springs only
- Orb: CSS keyframes for rotation

### Cards
- No visible borders by default — background difference only
- Hover: barely perceptible border (#1A1A1A → #252525)
- Rounded corners: 12px
- No shadows. Darkness is the depth.
- Expanded state: faint glassmorphism (backdrop-blur + emerald border glow)

### Settings
- Full-screen overlay sliding up from below
- Glassmorphism backdrop (blur + dark overlay)
- Expandable sections. Clean toggles and sliders.
- Close with X or Escape

## Component Architecture

```
App.tsx
├── IdleView
│   ├── Orb (circle + ring + glow)
│   ├── Wordmark
│   └── SettingsButton
├── ActiveView
│   ├── Header (wordmark, timer, settings, stop)
│   ├── CardGrid
│   │   ├── MetricCard (x6, reusable wrapper)
│   │   │   ├── CompactView (number + label)
│   │   │   └── ExpandedView (per-card detail component)
│   │   └── ActivityFeed
│   └── SettingsOverlay
└── TransitionManager (idle↔active orchestration)
```

## What to Keep

- `src-tauri/` — entire Rust backend (all commands, state, simulation)
- `src/lib/types.ts` — TypeScript interfaces
- `src/lib/tauri.ts` — Tauri invoke wrappers
- `src/lib/utils.ts` — formatting helpers
- `src/hooks/` — all existing data hooks (useNodeStatus, useTrainingStatus, etc.)

## What to Rebuild

- All components in `src/components/`
- `src/App.tsx`
- `src/styles/globals.css` (new design tokens)
- Remove `src/pages/` (single view, no routing)

## What to Remove

- `react-router-dom` dependency (no page navigation)

## New Hooks

- `useExpandedCard` — manages which card is expanded (null or card ID)
- `useAppState` — manages idle/active state + transition phase
- `useOrbAnimation` — controls orb animation phases (idle, spin-up, dissolve)

## Tech Stack (unchanged)

- Tauri v2 + React 18 + TypeScript + Tailwind v4 + Framer Motion
- Lucide icons, clsx, tailwind-merge
