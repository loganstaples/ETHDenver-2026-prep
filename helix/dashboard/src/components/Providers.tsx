'use client';

import * as React from 'react';
import {
    RainbowKitProvider,
    getDefaultWallets,
    getDefaultConfig,
    darkTheme,
} from '@rainbow-me/rainbowkit';
import {
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

const { wallets } = getDefaultWallets();

// WalletConnect requires a non-empty projectId even during SSR/build.
// Use a placeholder to avoid build failures; real connections need a valid ID.
const projectId = process.env.NEXT_PUBLIC_WALLETCONNECT_PROJECT_ID || 'PLACEHOLDER_PROJECT_ID';
if (projectId === 'PLACEHOLDER_PROJECT_ID' && typeof window !== 'undefined') {
    console.warn(
        '[HELIX] Missing NEXT_PUBLIC_WALLETCONNECT_PROJECT_ID. ' +
        'Wallet connection will not work. See .env.example for setup.'
    );
}

const chains = [
    mainnet,
    sepolia,
    ...(process.env.NEXT_PUBLIC_ENABLE_TESTNETS === 'true' ? [hardhat, localhost, adiTestnet] : []),
] as const;

const config = getDefaultConfig({
    appName: 'Helix Dashboard',
    projectId,
    wallets: [
        ...wallets,
        {
            groupName: 'Other',
            wallets: [
                argentWallet,
                trustWallet,
                ledgerWallet,
            ],
        },
    ],
    chains,
    transports: {
        [mainnet.id]: http(),
        [sepolia.id]: http(),
        [hardhat.id]: http(process.env.NEXT_PUBLIC_ETH_RPC_URL || 'http://127.0.0.1:8545'),
        [localhost.id]: http(process.env.NEXT_PUBLIC_ETH_RPC_URL || 'http://127.0.0.1:8545'),
        [adiTestnet.id]: http('https://rpc.ab.testnet.adifoundation.ai'),
    },
    ssr: true,
});

const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            staleTime: 30_000,
            retry: 2,
            refetchOnWindowFocus: false,
        },
    },
});

export function Providers({ children }: { children: React.ReactNode }) {
    const [mounted, setMounted] = React.useState(false);
    React.useEffect(() => setMounted(true), []);
    return (
        <WagmiProvider config={config}>
            <QueryClientProvider client={queryClient}>
                <RainbowKitProvider theme={darkTheme({
                    accentColor: '#ffffff',
                    accentColorForeground: 'black',
                    borderRadius: 'medium',
                })}>
                    {mounted && children}
                </RainbowKitProvider>
            </QueryClientProvider>
        </WagmiProvider>
    );
}
