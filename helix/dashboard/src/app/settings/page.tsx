'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Wallet, Globe, SlidersHorizontal, WalletCards, FileCode2 } from 'lucide-react';
import { useAccount, useChainId } from 'wagmi';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { AddressDisplay } from '@/components/ui/AddressDisplay';
import { useContract } from '@/hooks/useContract';

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
  31337: 'http://127.0.0.1:8545',
  99999: 'https://rpc.ab.testnet.adifoundation.ai',
};

function getRpcEndpoint(chainId: number): string {
  return (
    process.env.NEXT_PUBLIC_RPC_URL ??
    RPC_ENDPOINTS[chainId] ??
    'Not configured'
  );
}

// ---------------------------------------------------------------------------
// Section header used inside each card
// ---------------------------------------------------------------------------

function SectionHeader({ icon: Icon, label }: { icon: React.ComponentType<{ size?: number | string; className?: string }>; label: string }) {
  return (
    <div className="flex items-center gap-2 mb-4">
      <Icon size={14} className="text-[#555]" />
      <h3 className="text-[11px] font-medium uppercase tracking-wider text-[#666]">{label}</h3>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Row: key-value display used throughout
// ---------------------------------------------------------------------------

function SettingRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between py-2.5 border-b border-helix-border last:border-0">
      <span className="text-[13px] text-[#888]">{label}</span>
      <div className="text-[13px] text-white">{children}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Toggle switch (visual only, local state)
// ---------------------------------------------------------------------------

function Toggle({ enabled, onToggle }: { enabled: boolean; onToggle: () => void }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={enabled}
      onClick={onToggle}
      className={`
        relative inline-flex h-5 w-9 shrink-0 cursor-pointer items-center rounded-full
        border border-helix-border transition-colors
        ${enabled ? 'bg-white/20' : 'bg-white/[0.04]'}
      `}
    >
      <span
        className={`
          pointer-events-none inline-block h-3 w-3 rounded-full transition-transform
          ${enabled ? 'translate-x-[18px] bg-white' : 'translate-x-[3px] bg-[#555]'}
        `}
      />
    </button>
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

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="space-y-6"
    >
      {/* Page title */}
      <h1 className="text-[13px] font-medium text-white">Settings</h1>

      {/* ----------------------------------------------------------------- */}
      {/* 1. Wallet                                                         */}
      {/* ----------------------------------------------------------------- */}
      <Card>
        <SectionHeader icon={Wallet} label="Wallet" />

        {isConnected && address ? (
          <div className="space-y-0 divide-y divide-helix-border">
            <SettingRow label="Address">
              <AddressDisplay address={address} />
            </SettingRow>

            <SettingRow label="Network">
              <Badge variant="outline">
                {chain?.name ?? getChainName(chainId)}
              </Badge>
            </SettingRow>

            <SettingRow label="Chain ID">
              <span className="font-mono text-[#aaa]">{chainId}</span>
            </SettingRow>

            <SettingRow label="Status">
              <Badge variant="pulse">Connected</Badge>
            </SettingRow>

            <SettingRow label="Coordinator">
              <AddressDisplay address={contractAddress} />
            </SettingRow>
          </div>
        ) : (
          <div className="py-6 text-center">
            <WalletCards size={28} className="mx-auto mb-2 text-[#444]" />
            <p className="text-[13px] text-[#666]">Wallet not connected</p>
            <p className="text-[11px] text-[#555] mt-1">
              Use the connect button in the header to link your wallet.
            </p>
          </div>
        )}
      </Card>

      {/* ----------------------------------------------------------------- */}
      {/* 2. Network                                                        */}
      {/* ----------------------------------------------------------------- */}
      <Card>
        <SectionHeader icon={Globe} label="Network" />

        <div className="space-y-0 divide-y divide-helix-border">
          <SettingRow label="Current Chain">
            <span className="text-[#aaa]">{chain?.name ?? getChainName(chainId)}</span>
          </SettingRow>

          <SettingRow label="Chain ID">
            <span className="font-mono text-[#aaa]">{chainId}</span>
          </SettingRow>

          <SettingRow label="RPC Endpoint">
            <span className="font-mono text-[11px] text-[#666] max-w-[260px] truncate inline-block">
              {getRpcEndpoint(chainId)}
            </span>
          </SettingRow>

          <SettingRow label="Coordinator Address">
            <AddressDisplay address={contractAddress} />
          </SettingRow>
        </div>
      </Card>

      {/* ----------------------------------------------------------------- */}
      {/* 3. Preferences                                                    */}
      {/* ----------------------------------------------------------------- */}
      <Card>
        <SectionHeader icon={SlidersHorizontal} label="Preferences" />

        <div className="space-y-0 divide-y divide-helix-border">
          <SettingRow label="Auto-refresh">
            <div className="flex items-center gap-2">
              {autoRefresh && (
                <span className="text-[11px] text-[#555]">5 s</span>
              )}
              <Toggle enabled={autoRefresh} onToggle={() => setAutoRefresh((v) => !v)} />
            </div>
          </SettingRow>

          <SettingRow label="Theme">
            <span className="text-[#aaa]">Dark</span>
          </SettingRow>

          <SettingRow label="Version">
            <Badge variant="outline">
              <FileCode2 size={10} className="opacity-60" />
              v0.1.0-alpha
            </Badge>
          </SettingRow>
        </div>
      </Card>
    </motion.div>
  );
}
