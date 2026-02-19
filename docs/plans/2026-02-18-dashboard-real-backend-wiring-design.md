# Dashboard Real Backend Wiring & Worker Reputation

**Date:** 2026-02-18
**Status:** Approved

## Summary

Remove useless marketplace stats, wire all dashboard frontend to real Rust backend data, rewrite Activity page with clique subgraphs grouped by inference request, add worker panel with reputation scores on Dashboard page.

## Section 1: Marketplace Cleanup & API Client

**Marketplace page (`/`):**
- Remove `QuickStats.tsx` rendering from models marketplace page
- Keep mode toggle, filter tabs, and model cards (already on-chain/real)

**New `useBackendApi` hook:**
- Unified REST client wrapping all backend calls to `NEXT_PUBLIC_API_URL`
- Endpoints: `/health`, `/workers`, `/peers`, `/metrics`, `/round/status`, `/api/events`, `/api/training/:id/losses`
- Auto-refresh on 5-10s interval
- Error/loading states for backend-unreachable scenarios

## Section 2: Dashboard Worker Panel & Reputation

**Worker Panel on Dashboard page (`/dashboard`):**
- Collapsible panel below training session monitoring
- Two tabs: Active Workers | Inactive Workers
- Table columns: Worker ID, Status badge, Reputation Score (0-100), CPU Load, Memory, Rounds Completed/Participated, Success Rate, Earnings, Last Heartbeat

**Reputation Score (0-100):**
- Backend has 0.0-1.0 success rate + multi-dimensional peer reputation
- Formula: `(success_rate * 0.4) + (validity * 0.25) + (responsiveness * 0.15) + (uptime * 0.15) + (bandwidth * 0.05)` scaled to 0-100
- Color coding: green 80+, yellow 50-79, red <50

**Worker Detail (expandable row):**
- Resource metrics: CPU, Memory, GPU, Temperature, Network I/O, Disk
- Capabilities: canTrain, canProve, canAggregate, GPU model, CUDA, max batch size
- Round history (last 10 rounds)
- Reputation breakdown by dimension
- Stake amount and slashing history

**Data:** `GET /workers` + `GET /peers` + WebSocket for real-time updates

## Section 3: Activity Page Rewrite

**Network Graph:**
- Remove randomly-generated mesh topology
- Source: `GET /api/events` filtered for inference requests on user-owned models
- Group workers into clique subgraphs per active inference request:
  - Each inference request = one cluster
  - Central node per cluster = model being inferred
  - Worker nodes show ID, status, latency
  - Edge thickness = request volume
  - Cluster color = unique per model
- Force-directed layout with inter-cluster repulsion
- Click cluster for request details; click worker for mini-profile

**Stats & Charts:**
- Request Volume chart from real inference events aggregated by hour
- Total Revenue from completed inference fees
- Total Requests from real event count
- Avg Latency from actual inference duration
- Request table from backend events (not localStorage)

**Filtering:** Only show activity for models the connected wallet owns.

## Section 4: Wire All Remaining Graphs & Stats

**Dashboard page:**
- Loss Curve → `GET /api/training/:id/losses`
- Training metrics → `GET /round/status`
- Proof timeline → `GET /metrics` Prometheus counters
- Network stats → `GET /health`

**Network page:** Verify `NodeGrid`, `NetworkHealthBar`, `NodeDetailSheet` use real backend data.

**Train page:** Verify `useMpcTraining` reports all 13 phases accurately; wire loss charts to backend.

**Inference page:** Verify end-to-end proxy works with real backend response.

**Backend endpoint extensions needed:**
- Extend `WorkerInfo` response to include: CPU load, memory, reputation score, capabilities, earnings
- Add `GET /api/events?owner=<address>` filter
- Add reputation breakdown to worker responses
