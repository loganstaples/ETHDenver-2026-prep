# HELIX Dashboard

Next.js frontend for visualizing live training with proofs.

## Features

- **Training Progress** - Loss curves, metrics, real-time updates
- **Proof Explorer** - View any training step's ZK proof
- **Error Bounds Visualization** - See bounds tightening over training
- **Model Chat** - Query the trained model
- **Node Network** - Geographic view of participants

## Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

## Tech Stack

- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS
- ethers.js for contract interaction
