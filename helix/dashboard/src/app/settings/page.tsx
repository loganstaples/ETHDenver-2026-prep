'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Globe,
  SlidersHorizontal,
  WalletCards,
  FileCode2,
  Shield,
  Plus,
  X,
  AlertCircle,
} from 'lucide-react';
import { useAccount, useChainId } from 'wagmi';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { AddressDisplay } from '@/components/ui/AddressDisplay';
import { useContract } from '@/hooks/useContract';
import { useTrustedNodes } from '@/hooks/useTrustedNodes';

// ---------------------------------------------------------------------------
// Chain metadata
// ---------------------------------------------------------------------------

const CHAIN_NAMES: Record<number, string> = {
  1: 'Ethereum Mainnet',
  11155111: 'Sepolia',
  31337: 'Localhost (Hardhat)',
  99999: 'ADI Network Testnet',
};

function getChainName(chainId: number): string {
  return CHAIN_NAMES[chainId] ?? `Chain ${chainId}`;
}

const RPC_ENDPOINTS: Record<number, string> = {
  1: 'https://eth-mainnet.g.alchemy.com/v2/***',
  11155111: 'https://eth-sepolia.g.alchemy.com/v2/***',
  31337: process.env.NEXT_PUBLIC_ETH_RPC_URL || 'http://127.0.0.1:8545',
  99999: process.env.NEXT_PUBLIC_ETH_RPC_URL || 'https://rpc.ab.testnet.adifoundation.ai',
};

function getRpcEndpoint(chainId: number): string {
  return (
    process.env.NEXT_PUBLIC_RPC_URL ??
    RPC_ENDPOINTS[chainId] ??
    'Not configured'
  );
}

// ---------------------------------------------------------------------------
// Animation helpers
// ---------------------------------------------------------------------------

function stagger(i: number) {
  return {
    initial: { opacity: 0, y: 12 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.4, delay: i * 0.07, ease: [0.25, 0.46, 0.45, 0.94] as const },
  };
}

// ---------------------------------------------------------------------------
// Toggle switch — larger, more satisfying
// ---------------------------------------------------------------------------

function Toggle({
  enabled,
  onToggle,
}: {
  enabled: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={enabled}
      onClick={onToggle}
      className={`
        relative inline-flex h-7 w-12 shrink-0 cursor-pointer items-center rounded-full
        transition-all duration-200
        ${enabled ? 'bg-white/20 border border-white/20' : 'bg-white/[0.04] border border-helix-border'}
      `}
    >
      <motion.span
        layout
        transition={{ type: 'spring', stiffness: 500, damping: 35 }}
        className={`
          pointer-events-none inline-block h-4.5 w-4.5 rounded-full shadow-sm
          ${enabled ? 'bg-white ml-[26px]' : 'bg-helix-dim ml-[4px]'}
        `}
      />
    </button>
  );
}

// ---------------------------------------------------------------------------
// Wallet avatar
// ---------------------------------------------------------------------------

function WalletAvatar({ address }: { address: string }) {
  const chars = address.slice(2, 4).toUpperCase();
  return (
    <div className="relative">
      <div className="w-14 h-14 rounded-2xl border border-white/10 bg-white/[0.06] flex items-center justify-center">
        <span className="font-mono text-lg text-helix-text2 font-semibold">
          {chars}
        </span>
      </div>
      <div className="absolute -inset-1 rounded-2xl border border-white/[0.03]" />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function SettingsPage() {
  const { address, isConnected, chain } = useAccount();
  const chainId = useChainId();
  const { contractAddress } = useContract();

  const [autoRefresh, setAutoRefresh] = useState(true);

  // Trusted nodes
  const { trustedNodes, addNode, removeNode } = useTrustedNodes();
  const [newAddress, setNewAddress] = useState('');
  const [trustedError, setTrustedError] = useState<string | null>(null);

  const handleAddTrustedNode = () => {
    setTrustedError(null);
    const result = addNode(newAddress);
    if (result.ok) {
      setNewAddress('');
    } else {
      setTrustedError(result.error ?? 'Failed to add address');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleAddTrustedNode();
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
      className="space-y-8"
    >
      {/* Page title */}
      <motion.h1
        {...stagger(0)}
        className="text-3xl font-semibold tracking-tight text-white"
      >
        Settings
      </motion.h1>

      {/* ================================================================= */}
      {/* WALLET HERO — full width glass card                               */}
      {/* ================================================================= */}
      <motion.div {...stagger(1)}>
        <Card variant="glass" className="!p-0 overflow-hidden">
          <div className="relative px-10 py-10">
            {/* Ambient glow */}
            <div className="absolute top-0 right-0 w-72 h-72 bg-white/[0.012] rounded-full blur-[100px] pointer-events-none" />
            <div className="absolute bottom-0 left-1/3 w-56 h-56 bg-white/[0.008] rounded-full blur-[80px] pointer-events-none" />

            {isConnected && address ? (
              <div className="relative flex items-start justify-between gap-10">
                {/* Left — identity */}
                <div className="flex items-start gap-5 min-w-0">
                  <WalletAvatar address={address} />
                  <div className="min-w-0 pt-0.5">
                    <AddressDisplay
                      address={address}
                      className="!text-xl !text-white font-semibold"
                    />
                    <div className="flex items-center gap-3 mt-2">
                      <span className="text-base text-helix-text2">
                        {chain?.name ?? getChainName(chainId)}
                      </span>
                      <span className="text-helix-dim text-lg">&middot;</span>
                      <span className="font-mono text-base text-helix-muted tabular-nums">
                        {chainId}
                      </span>
                      <span className="text-helix-dim text-lg">&middot;</span>
                      <Badge variant="pulse" className="!text-emerald-400/80 !bg-emerald-400/[0.08] !text-sm !py-1 !px-2.5">
                        Connected
                      </Badge>
                    </div>
                  </div>
                </div>

                {/* Right — coordinator */}
                <div className="shrink-0 text-right pt-0.5">
                  <span className="text-sm uppercase tracking-[0.08em] text-helix-muted block mb-2">
                    Coordinator
                  </span>
                  <AddressDisplay address={contractAddress} className="!text-base" />
                </div>
              </div>
            ) : (
              <div className="relative py-10 text-center">
                <div className="w-20 h-20 mx-auto mb-6 rounded-2xl border border-white/[0.08] bg-white/[0.03] flex items-center justify-center">
                  <WalletCards size={32} className="text-helix-dim" />
                </div>
                <p className="text-xl text-helix-text2 font-medium">
                  No wallet connected
                </p>
                <p className="text-base text-helix-muted mt-2 max-w-sm mx-auto leading-relaxed">
                  Connect your wallet using the button in the header to view account details.
                </p>
              </div>
            )}
          </div>
        </Card>
      </motion.div>

      {/* ================================================================= */}
      {/* TWO-COLUMN GRID: Security (left) | Network + Preferences (right)  */}
      {/* ================================================================= */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* -------------------------------------------------------------- */}
        {/* LEFT COLUMN: Security / Trusted Nodes                          */}
        {/* -------------------------------------------------------------- */}
        <motion.div {...stagger(2)}>
          <Card className="!p-0 h-full">
            <div className="p-7">
              {/* Header */}
              <div className="flex items-center gap-3 mb-2">
                <div className="w-9 h-9 rounded-xl bg-white/[0.04] border border-helix-border flex items-center justify-center">
                  <Shield size={18} className="text-helix-muted" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-white">Trusted Nodes</h2>
                  <p className="text-xs text-helix-muted uppercase tracking-wide">Security</p>
                </div>
              </div>

              <p className="text-sm text-helix-muted mt-4 leading-relaxed">
                Restrict which workers handle your model weights. In additive
                secret sharing, each worker holds only a fragment &mdash; as long
                as one share holder is honest, full weights remain private.
              </p>

              {/* Add address input */}
              <div className="relative mt-5">
                <input
                  type="text"
                  value={newAddress}
                  onChange={(e) => {
                    setNewAddress(e.target.value);
                    setTrustedError(null);
                  }}
                  onKeyDown={handleKeyDown}
                  placeholder="0x..."
                  className="w-full bg-white/[0.03] border border-helix-border rounded-xl px-5 py-3.5 pr-14
                             text-base text-white placeholder-helix-dim font-mono
                             focus:outline-none focus:border-helix-border2 focus:bg-white/[0.04]
                             transition-all duration-200"
                />
                <button
                  type="button"
                  onClick={handleAddTrustedNode}
                  disabled={!newAddress.trim()}
                  className="absolute right-2 top-1/2 -translate-y-1/2 p-2.5 rounded-lg
                             text-helix-muted hover:text-white hover:bg-white/[0.08]
                             disabled:opacity-20 disabled:cursor-not-allowed
                             transition-all duration-200"
                >
                  <Plus size={18} />
                </button>
              </div>

              {/* Error message */}
              <AnimatePresence>
                {trustedError && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="flex items-center gap-2 mt-3 text-sm text-red-400"
                  >
                    <AlertCircle size={14} />
                    {trustedError}
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Trusted nodes list */}
              <div className="mt-5">
                {trustedNodes.length === 0 ? (
                  <div className="py-10 text-center border border-dashed border-helix-border/60 rounded-xl">
                    <div className="w-12 h-12 mx-auto mb-4 rounded-xl bg-white/[0.03] border border-white/[0.06] flex items-center justify-center">
                      <Shield size={20} className="text-helix-dim" />
                    </div>
                    <p className="text-base text-helix-muted">
                      No trusted nodes configured
                    </p>
                    <p className="text-sm text-helix-dim mt-1.5">
                      Add Ethereum addresses of nodes you trust.
                    </p>
                  </div>
                ) : (
                  <div className="rounded-xl border border-helix-border/60 overflow-hidden">
                    {trustedNodes.map((addr, i) => (
                      <div
                        key={addr}
                        className={`flex items-center justify-between px-5 py-3.5 group
                          hover:bg-white/[0.02] transition-colors
                          ${i < trustedNodes.length - 1 ? 'border-b border-helix-border/40' : ''}`}
                      >
                        <div className="flex items-center gap-3">
                          <div className="w-2 h-2 rounded-full bg-helix-dim" />
                          <span className="font-mono text-base text-helix-text2">
                            {addr}
                          </span>
                        </div>
                        <button
                          type="button"
                          onClick={() => removeNode(addr)}
                          className="p-1.5 rounded-lg text-helix-dim hover:text-red-400 hover:bg-red-400/[0.06]
                                     opacity-0 group-hover:opacity-100 transition-all duration-200"
                          title="Remove trusted node"
                        >
                          <X size={15} />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {trustedNodes.length > 0 && (
                <p className="text-sm text-helix-dim mt-4 tabular-nums">
                  {trustedNodes.length} node{trustedNodes.length !== 1 ? 's' : ''} configured
                </p>
              )}
            </div>
          </Card>
        </motion.div>

        {/* -------------------------------------------------------------- */}
        {/* RIGHT COLUMN: Network + Preferences stacked                    */}
        {/* -------------------------------------------------------------- */}
        <div className="space-y-6">

          {/* Network card */}
          <motion.div {...stagger(3)}>
            <Card className="!p-0">
              <div className="p-7">
                {/* Header */}
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-9 h-9 rounded-xl bg-white/[0.04] border border-helix-border flex items-center justify-center">
                    <Globe size={18} className="text-helix-muted" />
                  </div>
                  <h2 className="text-xl font-semibold text-white">Network</h2>
                </div>

                {/* 2x2 grid filling full card width */}
                <div className="grid grid-cols-2 gap-6">
                  <div>
                    <span className="text-sm uppercase tracking-[0.06em] text-helix-muted block mb-1.5">
                      Chain
                    </span>
                    <span className="text-lg text-white font-medium">
                      {chain?.name ?? getChainName(chainId)}
                    </span>
                  </div>

                  <div>
                    <span className="text-sm uppercase tracking-[0.06em] text-helix-muted block mb-1.5">
                      Chain ID
                    </span>
                    <span className="font-mono text-lg text-white tabular-nums">
                      {chainId}
                    </span>
                  </div>

                  <div className="col-span-2 pt-2 border-t border-helix-border/50">
                    <span className="text-sm uppercase tracking-[0.06em] text-helix-muted block mb-1.5">
                      RPC Endpoint
                    </span>
                    <span className="font-mono text-base text-helix-text2 break-all">
                      {getRpcEndpoint(chainId)}
                    </span>
                  </div>

                  <div className="col-span-2 pt-2 border-t border-helix-border/50">
                    <span className="text-sm uppercase tracking-[0.06em] text-helix-muted block mb-1.5">
                      Coordinator
                    </span>
                    <AddressDisplay address={contractAddress} className="!text-base" />
                  </div>
                </div>
              </div>
            </Card>
          </motion.div>

          {/* Preferences card */}
          <motion.div {...stagger(4)}>
            <Card className="!p-0">
              <div className="p-7">
                {/* Header */}
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-9 h-9 rounded-xl bg-white/[0.04] border border-helix-border flex items-center justify-center">
                    <SlidersHorizontal size={18} className="text-helix-muted" />
                  </div>
                  <h2 className="text-xl font-semibold text-white">Preferences</h2>
                </div>

                <div className="space-y-0">
                  {/* Auto-refresh */}
                  <div className="flex items-center justify-between py-4 border-b border-helix-border/50">
                    <div>
                      <span className="text-base text-white block">Auto-refresh</span>
                      <span className="text-sm text-helix-muted mt-0.5 block">
                        Poll for updates every 5 seconds
                      </span>
                    </div>
                    <Toggle
                      enabled={autoRefresh}
                      onToggle={() => setAutoRefresh((v) => !v)}
                    />
                  </div>

                  {/* Theme */}
                  <div className="flex items-center justify-between py-4 border-b border-helix-border/50">
                    <span className="text-base text-white">Theme</span>
                    <span className="text-base text-helix-text2">Dark</span>
                  </div>

                  {/* Version */}
                  <div className="flex items-center justify-between py-4">
                    <span className="text-base text-white">Version</span>
                    <Badge variant="outline" className="!text-sm !py-1 !px-3 gap-2">
                      <FileCode2 size={12} className="opacity-50" />
                      v0.1.0-alpha
                    </Badge>
                  </div>
                </div>
              </div>
            </Card>
          </motion.div>

        </div>
      </div>
    </motion.div>
  );
}
