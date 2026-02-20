'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Wallet, Globe, SlidersHorizontal, WalletCards, FileCode2, Shield, Plus, X, AlertCircle } from 'lucide-react';
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
      {/* 2. Trusted Nodes                                                  */}
      {/* ----------------------------------------------------------------- */}
      <Card>
        <SectionHeader icon={Shield} label="Trusted Nodes" />

        <p className="text-[11px] text-[#555] mb-4">
          Restrict which workers can handle your model weights during training or
          inference. At least one trusted node must be active for a job to start.
          In HELIX&apos;s additive secret sharing, each worker holds only a fragment
          of your model &mdash; as long as even one share holder is honest, no
          coalition of other participants can reconstruct the full weights.
        </p>

        {/* Add address input */}
        <div className="flex gap-2 mb-3">
          <input
            type="text"
            value={newAddress}
            onChange={(e) => { setNewAddress(e.target.value); setTrustedError(null); }}
            onKeyDown={handleKeyDown}
            placeholder="0x..."
            className="flex-1 bg-white/[0.04] border border-helix-border rounded px-3 py-1.5
                       text-[13px] text-white placeholder-[#444] font-mono
                       focus:outline-none focus:border-[#555] transition-colors"
          />
          <button
            type="button"
            onClick={handleAddTrustedNode}
            disabled={!newAddress.trim()}
            className="flex items-center gap-1 px-3 py-1.5 rounded border border-helix-border
                       text-[12px] text-[#aaa] hover:text-white hover:border-[#555]
                       disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            <Plus size={12} />
            Add
          </button>
        </div>

        {/* Error message */}
        <AnimatePresence>
          {trustedError && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="flex items-center gap-1.5 mb-3 text-[11px] text-red-400"
            >
              <AlertCircle size={11} />
              {trustedError}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Trusted nodes list */}
        {trustedNodes.length === 0 ? (
          <div className="py-4 text-center border border-dashed border-helix-border rounded">
            <Shield size={20} className="mx-auto mb-1.5 text-[#333]" />
            <p className="text-[11px] text-[#555]">No trusted nodes configured</p>
            <p className="text-[10px] text-[#444] mt-0.5">
              Add Ethereum addresses of nodes you trust to handle your model weights.
            </p>
          </div>
        ) : (
          <div className="space-y-0 divide-y divide-helix-border">
            {trustedNodes.map((addr) => (
              <div
                key={addr}
                className="flex items-center justify-between py-2 group"
              >
                <span className="font-mono text-[12px] text-[#aaa]">
                  {addr}
                </span>
                <button
                  type="button"
                  onClick={() => removeNode(addr)}
                  className="p-1 rounded text-[#444] hover:text-red-400 hover:bg-white/[0.04]
                             opacity-0 group-hover:opacity-100 transition-all"
                  title="Remove trusted node"
                >
                  <X size={12} />
                </button>
              </div>
            ))}
          </div>
        )}

        {trustedNodes.length > 0 && (
          <div className="mt-3 pt-3 border-t border-helix-border">
            <span className="text-[11px] text-[#555]">
              {trustedNodes.length} trusted node{trustedNodes.length !== 1 ? 's' : ''} configured
            </span>
          </div>
        )}
      </Card>

      {/* ----------------------------------------------------------------- */}
      {/* 3. Network                                                        */}
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
      {/* 4. Preferences                                                    */}
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
