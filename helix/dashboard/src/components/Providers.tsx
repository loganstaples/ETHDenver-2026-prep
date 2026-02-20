'use client';

import * as React from 'react';
import {
    RainbowKitProvider,
    getDefaultConfig,
    darkTheme,
} from '@rainbow-me/rainbowkit';
import {
    metaMaskWallet,
    coinbaseWallet,
    argentWallet,
    trustWallet,
    ledgerWallet,
} from '@rainbow-me/rainbowkit/wallets';
import {
    mainnet,
    sepolia,
    hardhat,
    localhost,
} from 'wagmi/chains';
import { defineChain, http } from 'viem';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { WagmiProvider } from 'wagmi';
import '@rainbow-me/rainbowkit/styles.css';

// ADI Network Testnet (chain 99999)
const adiTestnet = defineChain({
    id: 99999,
    name: 'ADI Network Testnet',
    nativeCurrency: { name: 'ADI', symbol: 'ADI', decimals: 18 },
    rpcUrls: {
        default: { http: ['https://rpc.ab.testnet.adifoundation.ai'] },
    },
    blockExplorers: {
        default: { name: 'ADI Explorer', url: 'https://explorer.ab.testnet.adifoundation.ai' },
    },
    testnet: true,
});

// WalletConnect requires a valid projectId. With a placeholder, only injected wallets work.
const projectId = process.env.NEXT_PUBLIC_WALLETCONNECT_PROJECT_ID || 'PLACEHOLDER_PROJECT_ID';
const hasRealProjectId = projectId !== 'PLACEHOLDER_PROJECT_ID' && projectId !== 'placeholder';

if (!hasRealProjectId && typeof window !== 'undefined') {
    console.warn(
        '[HELIX] Missing NEXT_PUBLIC_WALLETCONNECT_PROJECT_ID. ' +
        'Only injected wallets (MetaMask, Coinbase) will work. See .env.example for setup.'
    );
}

// ADI testnet first so it's the default chain for public reads (even without wallet)
const enableTestnets = process.env.NEXT_PUBLIC_ENABLE_TESTNETS === 'true';
const chains = enableTestnets
    ? ([adiTestnet, hardhat, localhost, mainnet, sepolia] as const)
    : ([mainnet, sepolia] as const);

const config = getDefaultConfig({
    appName: 'Helix Dashboard',
    projectId,
    wallets: [
        {
            groupName: 'Popular',
            wallets: [metaMaskWallet, coinbaseWallet],
        },
        {
            groupName: 'Other',
            wallets: [argentWallet, trustWallet, ledgerWallet],
        },
    ],
    chains,
    transports: {
        [mainnet.id]: http(),
        [sepolia.id]: http(),
        [hardhat.id]: http(process.env.NEXT_PUBLIC_ETH_RPC_URL || 'http://127.0.0.1:8545', { timeout: 10_000 }),
        [localhost.id]: http(process.env.NEXT_PUBLIC_ETH_RPC_URL || 'http://127.0.0.1:8545', { timeout: 10_000 }),
        [adiTestnet.id]: http('https://rpc.ab.testnet.adifoundation.ai', { timeout: 15_000 }),
    },
    ssr: true,
});

const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            staleTime: 30_000,
            retry: 1,
            retryDelay: 2_000,
            refetchOnWindowFocus: false,
            // Prevent infinite loading: fail after 15s
            networkMode: 'online',
        },
    },
});

export function Providers({ children }: { children: React.ReactNode }) {
    const [mounted, setMounted] = React.useState(false);
    React.useEffect(() => setMounted(true), []);
    return (
        <WagmiProvider config={config}>
            <QueryClientProvider client={queryClient}>
                <RainbowKitProvider
                    initialChain={adiTestnet}
                    theme={darkTheme({
                        accentColor: '#ffffff',
                        accentColorForeground: 'black',
                        borderRadius: 'medium',
                    })}
                >
                    {mounted && children}
                </RainbowKitProvider>
            </QueryClientProvider>
        </WagmiProvider>
    );
}
