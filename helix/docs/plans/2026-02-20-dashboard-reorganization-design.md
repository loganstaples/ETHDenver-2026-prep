# Dashboard Reorganization Design

## Context

The current dashboard has overlapping concerns: training monitor, inference history, and worker panels all live on one page, duplicating functionality from the Train, Inference, and (unused) Network pages. Model ownership is scattered across `/my-models`, `/my-models/[id]`, and `/models/[id]`. Workers don't use the web app (terminal-only for demo), so worker panels on the dashboard serve no primary user.

## Users

Primary persona: **Consumers** who browse models, run inference (paying commission), and can also be model owners who create/train/sell models. Workers are terminal-only and don't use the web app.

## Design

### New Route Structure

| Route | Purpose | Auth Required |
|-------|---------|---------------|
| `/` | Marketplace — browse models (inference-first, toggle to buyer mode) | No |
| `/dashboard` | Portfolio — models owned, revenue, active jobs, activity feed | Yes (wallet) |
| `/models/[id]` | Smart model detail — adapts UI based on ownership | No (owner features need wallet) |
| `/train` | Launch & monitor training sessions | Yes |
| `/inference` | Run MPC inference (draw/upload) | Yes |
| `/network` | Public worker directory with reputation, resources, status | No |
| `/settings` | Wallet config, trusted nodes | Yes |

### Removed Routes

- `/my-models` — model list absorbed into Dashboard
- `/my-models/[id]` — merged into `/models/[id]` with ownership detection

### Navigation

**Logged in sidebar:**
- Dashboard (LayoutDashboard)
- Marketplace (Boxes)
- Train (Cpu)
- Inference (Sparkles)
- Network (Globe)
- Settings (Settings)

**Logged out sidebar:**
- Marketplace (Boxes)
- Network (Globe)
- Settings (Settings) — shows connect wallet prompt

### Dashboard Page (Portfolio)

Top stat cards:
- Models Owned (count)
- Total Revenue (ETH earned from commissions)
- Active Training Jobs (count)
- Inferences Served (total across all models)

Revenue section:
- 30-day line chart of commission earnings
- Commission breakdown by model (pie/bar)

My Models list:
- Each row: model name, latest version, accuracy, total revenue
- Links to `/models/[id]`
- "+ Create" button for new model creation

Active Training section:
- Compact cards for in-progress jobs with progress bar
- Links to Train page for full details

Recent Activity feed:
- Unified chronological feed: inferences served (with earnings), model purchases, training events

### Marketplace (Root `/`)

Minimal change from current root page:
- Default mode: Inference marketplace (browse to use, pay commission)
- Toggle to Buyer mode (browse to acquire NFT)
- Search, category filters, sort options (quality score, accuracy, price, newest)
- Unauthenticated landing page

### Merged Model Detail (`/models/[id]`)

**All users see:**
- Model name, description, stats (versions, accuracy, inferences served, fee rate)
- Version history with accuracy progression
- Creator/owner info
- "Run Inference" action

**Owner additionally sees:**
- Inline edit for name/description
- Visibility toggle (public/private)
- For-sale toggle + price setting
- Inference fee configuration
- "Add Version" button with weight upload/encrypt/0G storage
- Revenue stats for this specific model

### Network Page (Public Worker Directory)

Summary bar: workers online, total compute capacity, network uptime

Worker list (filterable, sortable):
- Address, reputation score (stars + percentage)
- GPU/resource info, utilization percentage
- Jobs completed, uptime, latency, online/offline status
- Filter by status, sort by reputation/utilization/jobs

Network stats section: total jobs processed, average latency, cheaters detected

### Design Goals

- Beautiful, smooth UI — polished transitions, clean typography, thoughtful spacing
- No generic AI aesthetics — distinctive visual identity
- Consistent with existing component library (Card, Badge, Modal, Tabs patterns)
