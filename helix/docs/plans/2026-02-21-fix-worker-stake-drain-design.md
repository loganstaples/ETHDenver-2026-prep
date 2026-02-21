# Fix Worker Stake Drain

**Date:** 2026-02-21

## Problem

Workers are getting drained of all their ADI when they should only be staking a user-specified amount. Multiple compounding issues:

1. `config/training.rs` defaults stake to 0.5 ETH — should have no default (must come from frontend)
2. Workers pay their own gas — owner should pay all gas
3. Stakes are never auto-returned — locked in contract with 7-day cooldown and `enable_withdrawal=false`
4. Phase 12.9 sweep takes remaining ETH from pre-existing worker wallets (their money)

## Economic Model

- **Owner pays**: gas for all transactions, payment pool (worker earnings)
- **Worker pays**: stake deposit only (their own money, as collateral)
- **Worker receives**: stake back (automatic, if honest) + earnings from payment pool

## Changes

### Contract (HelixCoordinatorV4.sol)

Add `_autoReturnStakes(jobId)` internal function that iterates active workers and sends `stakeAmount` back to each non-slashed, non-pool worker. Called from:

- `completeTraining` — after payment distribution
- `stopTraining` — after owner refund
- `pauseTraining` — after releasing workers

Keep `withdrawStake` as manual fallback.

### Config (config/training.rs)

Remove `default_stake()` entirely. `stake_amount` has no default — must be set explicitly from the frontend train page.

### Demo (demo/mod.rs)

Set demo stake amounts to 0.001 (minimum viable for demos).

### Orchestration (full_orchestration.rs)

- **Phase 5.5 (pre-existing workers)**: Owner sends gas money (~0.15 ETH) to worker wallets. Workers stake their own ETH.
- **Phase 11.5**: Remove — contract auto-returns stakes now.
- **Phase 12.9**: Only sweep generated wallets. Skip pre-existing worker wallets.

### Dashboard

No changes needed — already sends `stake_per_worker_eth` from frontend input.
