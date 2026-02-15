import { motion, AnimatePresence } from "framer-motion";
import { X } from "lucide-react";
import { useConfig } from "../../hooks/useConfig";
import { useState } from "react";
import type { NodeConfig } from "../../lib/types";

interface SettingsOverlayProps {
  open: boolean;
  onClose: () => void;
}

function SettingRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between py-2.5">
      <span className="text-body text-text-secondary">{label}</span>
      {children}
    </div>
  );
}

function Toggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className="w-9 h-5 rounded-full transition-colors relative cursor-pointer"
      style={{ backgroundColor: checked ? "#34D399" : "#252525" }}
    >
      <div
        className="absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform"
        style={{ transform: checked ? "translateX(17px)" : "translateX(2px)" }}
      />
    </button>
  );
}

function NumberInput({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  return (
    <input
      type="number"
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className="w-20 bg-bg-card border border-border-subtle rounded px-2 py-1 text-metric-sm text-text-primary text-right focus:outline-none focus:border-accent/30"
    />
  );
}

function TextInput({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <input
      type="text"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-48 bg-bg-card border border-border-subtle rounded px-2 py-1 text-metric-sm text-text-primary focus:outline-none focus:border-accent/30"
    />
  );
}

export function SettingsOverlay({ open, onClose }: SettingsOverlayProps) {
  const { config, updateConfig } = useConfig();
  const [local, setLocal] = useState<NodeConfig | null>(null);
  const current = local ?? config;

  const update = (patch: Partial<NodeConfig>) => {
    const next = { ...current, ...patch };
    setLocal(next);
    updateConfig(next);
  };

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-end justify-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
        >
          {/* Backdrop */}
          <div
            className="absolute inset-0 backdrop-blur-md"
            style={{ backgroundColor: "rgba(5,5,5,0.85)" }}
            onClick={onClose}
          />

          {/* Panel */}
          <motion.div
            className="relative w-full max-w-xl bg-bg-primary/95 backdrop-blur-xl rounded-t-2xl border-t border-border-subtle max-h-[80vh] overflow-y-auto"
            initial={{ y: "100%" }}
            animate={{ y: 0 }}
            exit={{ y: "100%" }}
            transition={{ type: "spring", stiffness: 300, damping: 35 }}
          >
            {/* Header */}
            <div className="sticky top-0 flex items-center justify-between px-6 py-4 bg-bg-primary/95 backdrop-blur-xl border-b border-border-subtle z-10">
              <span className="text-label text-text-secondary">Settings</span>
              <button onClick={onClose} className="p-1 text-text-tertiary hover:text-text-secondary cursor-pointer">
                <X size={16} />
              </button>
            </div>

            <div className="px-6 py-4 space-y-6">
                {/* Resources */}
                <section>
                  <h3 className="text-label text-text-tertiary mb-3">Resources</h3>
                  <div className="divide-y divide-border-subtle">
                    <SettingRow label="CPU Threads">
                      <NumberInput value={current.cpu_threads} onChange={(v) => update({ cpu_threads: v })} />
                    </SettingRow>
                    <SettingRow label="GPU Memory Limit (MB)">
                      <NumberInput value={current.gpu_memory_limit_mb} onChange={(v) => update({ gpu_memory_limit_mb: v })} />
                    </SettingRow>
                    <SettingRow label="Max Concurrent Tasks">
                      <NumberInput value={current.max_concurrent_tasks} onChange={(v) => update({ max_concurrent_tasks: v })} />
                    </SettingRow>
                  </div>
                </section>

                {/* Network */}
                <section>
                  <h3 className="text-label text-text-tertiary mb-3">Network</h3>
                  <div className="divide-y divide-border-subtle">
                    <SettingRow label="Listen Address">
                      <TextInput value={current.listen_address} onChange={(v) => update({ listen_address: v })} />
                    </SettingRow>
                    <SettingRow label="Aggregator Address">
                      <TextInput value={current.aggregator_address} onChange={(v) => update({ aggregator_address: v })} />
                    </SettingRow>
                    <SettingRow label="RPC Port">
                      <NumberInput value={current.rpc_port} onChange={(v) => update({ rpc_port: v })} />
                    </SettingRow>
                    <SettingRow label="TLS">
                      <Toggle checked={current.use_tls} onChange={(v) => update({ use_tls: v })} />
                    </SettingRow>
                  </div>
                </section>

                {/* Training */}
                <section>
                  <h3 className="text-label text-text-tertiary mb-3">Training</h3>
                  <div className="divide-y divide-border-subtle">
                    <SettingRow label="Local Epochs">
                      <NumberInput value={current.local_epochs} onChange={(v) => update({ local_epochs: v })} />
                    </SettingRow>
                    <SettingRow label="Batch Size">
                      <NumberInput value={current.batch_size} onChange={(v) => update({ batch_size: v })} />
                    </SettingRow>
                    <SettingRow label="Generate Proofs">
                      <Toggle checked={current.generate_proofs} onChange={(v) => update({ generate_proofs: v })} />
                    </SettingRow>
                  </div>
                </section>

                {/* Version */}
                <section className="pb-4">
                  <p className="text-label-sm text-text-tertiary">HELIX v0.1.0</p>
                </section>
              </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
