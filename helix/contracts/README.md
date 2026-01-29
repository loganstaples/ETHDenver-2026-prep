# HELIX Contracts

Solidity smart contracts for on-chain coordination of decentralized training.

## Contracts

### Core
- **HelixCoordinator** - Main coordinator, round management, participant registry
- **TrainingRound** - Single round state machine, gradient submission
- **ModelRegistry** - Model state commitments, version history

### Verification
- **HelixVerifier** - Halo2 proof verification (generated from circuits)
- **BoundsChecker** - Verify error bound claims
- **AggregationVerifier** - Verify gradient aggregation

### Token
- **HelixToken** - ERC20 token for rewards
- **Staking** - Stake to participate, slashing conditions
- **Rewards** - Reward distribution per round

## Development

```bash
# Build
forge build

# Test
forge test

# Deploy
forge script script/Deploy.s.sol --broadcast
```
