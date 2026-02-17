/**
 * Mock model data for UI development.
 *
 * ┌─────────────────────────────────────────────────┐
 * │  TO DISABLE: set USE_MOCK_DATA = false below    │
 * └─────────────────────────────────────────────────┘
 */

export const USE_MOCK_DATA = true;

// Fake addresses
const MOCK_USER = '0x742d35Cc6634C0532925a3b844Bc9e7595f2bD18';
const MOCK_ALICE = '0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B';
const MOCK_BOB = '0x1Db3439a222C519ab44bb1144fC28167b4Fa6EE6';
const MOCK_CAROL = '0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe';

const now = Math.floor(Date.now() / 1000);
const day = 86400;

export const MOCK_MODELS = [
  {
    tokenId: 0,
    slug: 'mnist-classifier',
    name: 'MNIST Digit Classifier',
    description: 'A 784→32→10 feedforward neural network trained on MNIST via distributed MPC. Achieves 97.2% accuracy on the test set with privacy-preserving training.',
    creator: MOCK_USER,
    owner: MOCK_USER,
    createdAt: now - 14 * day,
    isPublic: true,
    inferenceFee: 250, // 2.5%
    forSale: true,
    salePrice: 0.5, // ETH
    versions: [
      {
        semver: '1.0.0',
        rootHash: '0x4a2b8c1d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b',
        accuracy: 0.934,
        timestamp: now - 14 * day,
        sessionId: 'v1.0.0-1739500000000',
        weightsStored: true,
      },
      {
        semver: '1.1.0',
        rootHash: '0x5b3c9d2e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c',
        accuracy: 0.958,
        timestamp: now - 7 * day,
        sessionId: 'v1.1.0-1740100000000',
        weightsStored: true,
      },
      {
        semver: '1.2.0',
        rootHash: '0x6c4da3f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d',
        accuracy: 0.972,
        timestamp: now - 2 * day,
        sessionId: 'v1.2.0-1740500000000',
        weightsStored: true,
      },
    ],
  },
  {
    tokenId: 1,
    slug: 'sentiment-bert-tiny',
    name: 'Sentiment Analysis (BERT-tiny)',
    description: 'Lightweight sentiment classifier fine-tuned from BERT-tiny. Binary positive/negative classification on movie reviews. Encrypted weights stored on 0G.',
    creator: MOCK_ALICE,
    owner: MOCK_ALICE,
    createdAt: now - 10 * day,
    isPublic: true,
    inferenceFee: 500, // 5%
    forSale: true,
    salePrice: 1.2, // ETH
    versions: [
      {
        semver: '0.1.0',
        rootHash: '0x7d5eb4a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e',
        accuracy: 0.821,
        timestamp: now - 10 * day,
        sessionId: 'v0.1.0-1739800000000',
        weightsStored: true,
      },
      {
        semver: '0.2.0',
        rootHash: '0x8e6fc5b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f',
        accuracy: 0.887,
        timestamp: now - 5 * day,
        sessionId: 'v0.2.0-1740300000000',
        weightsStored: true,
      },
    ],
  },
  {
    tokenId: 2,
    slug: 'cifar10-resnet',
    name: 'CIFAR-10 ResNet Mini',
    description: 'Small ResNet variant for CIFAR-10 image classification. 10 classes, 92.1% test accuracy. Training verified via MPC attestation.',
    creator: MOCK_BOB,
    owner: MOCK_BOB,
    createdAt: now - 21 * day,
    isPublic: true,
    inferenceFee: 0, // free
    forSale: false,
    salePrice: 0,
    versions: [
      {
        semver: '1.0.0',
        rootHash: '0x9f7ad6c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a',
        accuracy: 0.874,
        timestamp: now - 21 * day,
        sessionId: 'v1.0.0-1738900000000',
        weightsStored: true,
      },
      {
        semver: '2.0.0',
        rootHash: '0xa08be7d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b',
        accuracy: 0.921,
        timestamp: now - 3 * day,
        sessionId: 'v2.0.0-1740400000000',
        weightsStored: true,
      },
    ],
  },
  {
    tokenId: 3,
    slug: 'fraud-detector-v1',
    name: 'Transaction Fraud Detector',
    description: 'Anomaly detection model for financial transactions. Trained on synthetic fraud datasets with MPC privacy guarantees. High precision, low recall tuning.',
    creator: MOCK_CAROL,
    owner: MOCK_CAROL,
    createdAt: now - 5 * day,
    isPublic: true,
    inferenceFee: 1000, // 10%
    forSale: true,
    salePrice: 2.5, // ETH
    versions: [
      {
        semver: '1.0.0',
        rootHash: '0xb19cf8e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c',
        accuracy: 0.946,
        timestamp: now - 5 * day,
        sessionId: 'v1.0.0-1740200000000',
        weightsStored: true,
      },
    ],
  },
  {
    tokenId: 4,
    slug: 'text-embeddings-sm',
    name: 'Text Embeddings (Small)',
    description: 'Compact text embedding model, 128-dim output. Useful for semantic search and clustering. Distilled from larger teacher model.',
    creator: MOCK_ALICE,
    owner: MOCK_ALICE,
    createdAt: now - 3 * day,
    isPublic: true,
    inferenceFee: 150, // 1.5%
    forSale: false,
    salePrice: 0,
    versions: [
      {
        semver: '1.0.0',
        rootHash: '',
        accuracy: 0,
        timestamp: now - 3 * day,
        sessionId: 'v1.0.0-1740400000000',
        weightsStored: false,
      },
    ],
  },
  {
    tokenId: 5,
    slug: 'private-medical-model',
    name: 'Medical Imaging Classifier',
    description: 'Private model for X-ray anomaly detection. Restricted access.',
    creator: MOCK_BOB,
    owner: MOCK_BOB,
    createdAt: now - 8 * day,
    isPublic: false,
    inferenceFee: 2000, // 20%
    forSale: false,
    salePrice: 0,
    versions: [
      {
        semver: '1.0.0',
        rootHash: '0xc2ade9f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d',
        accuracy: 0.912,
        timestamp: now - 8 * day,
        sessionId: 'v1.0.0-1739900000000',
        weightsStored: true,
      },
    ],
  },
  {
    tokenId: 6,
    slug: 'mnist-autoencoder',
    name: 'MNIST Autoencoder',
    description: 'Autoencoder for MNIST digit reconstruction and denoising. Latent space of 16 dimensions. No version with weights stored yet — training in progress.',
    creator: MOCK_USER,
    owner: MOCK_USER,
    createdAt: now - 1 * day,
    isPublic: false,
    inferenceFee: 0,
    forSale: false,
    salePrice: 0,
    versions: [],
  },
];

/**
 * The "current user" address for mock mode.
 * Models owned by this address will appear in "My Models".
 */
export const MOCK_CURRENT_USER = MOCK_USER;
