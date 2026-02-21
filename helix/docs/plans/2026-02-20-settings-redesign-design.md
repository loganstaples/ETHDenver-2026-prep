# Settings Page Redesign

**Date:** 2026-02-20
**Direction:** CashApp/Fintech aesthetic — monochrome with functional color only

## Problem

The current settings page is plain and utilitarian. It lacks visual hierarchy, personality, and consistency with the more polished pages (dashboard, training). All four sections look identical (stacked cards with key-value rows), making nothing feel important.

## Design

### Layout

Single-column flow. Wallet gets a hero glass card at the top for hierarchy. All other sections use open borderless layouts with uppercase section headers and thin separators — no more identical stacked cards.

### 1. Wallet Hero Card

- `Card variant="glass"` with `bg-gradient-to-b from-white/[0.03] to-transparent`
- Larger padding (`p-8`) and typography (`text-lg` address) for visual weight
- Top row: 36px monochrome avatar circle (`border-white/10 bg-white/[0.06]`, first 2 chars of address) + truncated address in large mono + copy button
- Bottom row: network name, chain ID, green pulse "Connected" badge — separated by `·` dots
- Coordinator address as a small label below
- Disconnected state: same glass card, centered wallet icon + "Connect your wallet"

### 2. Security — Trusted Nodes (Open Layout)

- Section header: `SECURITY` uppercase tracking-wider with thin border-b separator
- Subheader: "Trusted Nodes" left + `[+ Add]` button right, inline
- Explainer text in `text-helix-dim`
- Add input: full-width monospace with icon button inside right edge
- Node list: hover-revealing rows, address mono left, X remove on hover right
- Node count footer in `text-helix-dim`

### 3. Network (Open Layout, 2x2 Grid)

- Section header: `NETWORK` uppercase with separator
- 2x2 grid of key-value pairs:
  - Chain / name | Chain ID / number (mono)
  - RPC / truncated endpoint (mono) | Coordinator / AddressDisplay
- Labels: `text-helix-muted text-[11px] uppercase`
- Values: `text-[13px] text-white` or `font-mono`

### 4. Preferences (Open Layout)

- Section header: `PREFERENCES` uppercase with separator
- SettingRow components for Auto-refresh (toggle), Theme (read-only), Version (badge)

### Color Palette

Strictly monochrome. Color only for functional status:
- Green pulse dot for "Connected" status
- Red for errors/alerts
- No decorative gradients or accent colors

### Animation

- Page fade-in: `initial={{ opacity: 0, y: 6 }}` (existing pattern)
- Staggered section entrance: each section delays by 0.05s
- Error messages: AnimatePresence slide (existing pattern)

### Components Reused

- `Card` (glass variant for hero only)
- `Badge` (pulse variant for connected status, outline for network/version)
- `AddressDisplay` (wallet + coordinator)
- Existing `Toggle` component (cleaned up slightly)

### Components Modified

- `SectionHeader` — redesigned to be an open uppercase divider instead of card-internal header
- `SettingRow` — same pattern but used outside cards now
