import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Card, CardHeader } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { Toggle } from "../components/ui/Toggle";
import { useConfig } from "../hooks/useConfig";
import { useNodeStatus } from "../hooks/useNodeStatus";
import type { NodeConfig } from "../lib/types";
import { truncateHash } from "../lib/utils";
import { Copy, FolderOpen } from "lucide-react";

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.03 },
  },
};

const item = {
  hidden: { opacity: 0, y: 4 },
  show: { opacity: 1, y: 0, transition: { duration: 0.15, ease: "easeOut" as const } },
};

function SliderField({
  label,
  value,
  min,
  max,
  step,
  unit,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between">
        <span className="text-sm text-text-secondary">{label}</span>
        <span className="text-sm text-text-primary tabular-nums">
          {value}
          {unit ? ` ${unit}` : ""}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1 bg-border-primary rounded-full appearance-none cursor-pointer
          [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:w-3.5
          [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-accent
          [&::-webkit-slider-thumb]:cursor-pointer"
      />
    </div>
  );
}

function TextInputField({
  label,
  value,
  onChange,
  mono,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  mono?: boolean;
}) {
  return (
    <div className="space-y-1.5">
      <label className="text-sm text-text-secondary">{label}</label>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={`w-full bg-bg-tertiary border border-border-primary rounded-md px-3 py-1.5 text-sm text-text-primary
          focus:outline-none focus:border-border-focus transition-colors duration-150
          ${mono ? "font-mono" : ""}`}
      />
    </div>
  );
}

function NumberInputField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-1.5">
      <label className="text-sm text-text-secondary">{label}</label>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full bg-bg-tertiary border border-border-primary rounded-md px-3 py-1.5 text-sm text-text-primary
          focus:outline-none focus:border-border-focus transition-colors duration-150 tabular-nums"
      />
    </div>
  );
}

export function SettingsPage() {
  const { config, loading, updateConfig } = useConfig();
  const { status } = useNodeStatus();
  const [local, setLocal] = useState<NodeConfig | null>(null);
  const [dirty, setDirty] = useState(false);

  useEffect(() => {
    if (config && !local) {
      setLocal(config);
    }
  }, [config, local]);

  if (loading || !local) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-sm text-text-tertiary">Loading settings...</p>
      </div>
    );
  }

  const update = (partial: Partial<NodeConfig>) => {
    setLocal((prev) => (prev ? { ...prev, ...partial } : prev));
    setDirty(true);
  };

  const save = async () => {
    if (local) {
      await updateConfig(local);
      setDirty(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold text-text-primary tracking-tight">
            Settings
          </h1>
          <p className="text-sm text-text-tertiary">
            Node configuration and preferences
          </p>
        </div>
        {dirty && (
          <Button variant="primary" size="md" onClick={save}>
            Save Changes
          </Button>
        )}
      </div>

      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="space-y-4"
      >
        {/* Resource Allocation */}
        <motion.div variants={item}>
          <Card>
            <CardHeader title="Resource Allocation" />
            <div className="space-y-5">
              <SliderField
                label="CPU Threads"
                value={local.cpu_threads}
                min={1}
                max={32}
                step={1}
                onChange={(v) => update({ cpu_threads: v })}
              />
              <SliderField
                label="GPU Memory Limit"
                value={local.gpu_memory_limit_mb}
                min={1024}
                max={32768}
                step={1024}
                unit="MB"
                onChange={(v) => update({ gpu_memory_limit_mb: v })}
              />
              <SliderField
                label="Max Concurrent Tasks"
                value={local.max_concurrent_tasks}
                min={1}
                max={8}
                step={1}
                onChange={(v) => update({ max_concurrent_tasks: v })}
              />
            </div>
          </Card>
        </motion.div>

        {/* Network */}
        <motion.div variants={item}>
          <Card>
            <CardHeader title="Network" />
            <div className="space-y-4">
              <TextInputField
                label="Listen Address"
                value={local.listen_address}
                onChange={(v) => update({ listen_address: v })}
                mono
              />
              <TextInputField
                label="Aggregator Address"
                value={local.aggregator_address}
                onChange={(v) => update({ aggregator_address: v })}
                mono
              />
              <NumberInputField
                label="RPC Port"
                value={local.rpc_port}
                onChange={(v) => update({ rpc_port: v })}
              />
              <Toggle
                label="Use TLS"
                checked={local.use_tls}
                onChange={(v) => update({ use_tls: v })}
              />
            </div>
          </Card>
        </motion.div>

        {/* Training */}
        <motion.div variants={item}>
          <Card>
            <CardHeader title="Training" />
            <div className="space-y-4">
              <NumberInputField
                label="Local Epochs"
                value={local.local_epochs}
                onChange={(v) => update({ local_epochs: v })}
              />
              <NumberInputField
                label="Batch Size"
                value={local.batch_size}
                onChange={(v) => update({ batch_size: v })}
              />
              <Toggle
                label="Generate Proofs"
                checked={local.generate_proofs}
                onChange={(v) => update({ generate_proofs: v })}
              />
            </div>
          </Card>
        </motion.div>

        {/* Identity */}
        <motion.div variants={item}>
          <Card>
            <CardHeader title="Node Identity" />
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm text-text-secondary">Peer ID</span>
                <button
                  onClick={() =>
                    navigator.clipboard.writeText(status.peer_id)
                  }
                  className="flex items-center gap-1.5 text-sm text-text-tertiary hover:text-text-secondary transition-colors font-mono"
                >
                  {truncateHash(status.peer_id)}
                  <Copy size={12} strokeWidth={1.5} />
                </button>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-text-secondary">Data Dir</span>
                <button className="flex items-center gap-1.5 text-sm text-text-tertiary hover:text-text-secondary transition-colors">
                  ~/.helix
                  <FolderOpen size={12} strokeWidth={1.5} />
                </button>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-text-secondary">Role</span>
                <span className="text-sm text-text-primary">Compute</span>
              </div>
            </div>
          </Card>
        </motion.div>

        {/* About */}
        <motion.div variants={item}>
          <Card>
            <CardHeader title="About" />
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-text-tertiary">HELIX Desktop</span>
                <span className="text-text-secondary">v0.1.0</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-tertiary">helix-node</span>
                <span className="text-text-secondary">v0.1.0</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-tertiary">Tauri</span>
                <span className="text-text-secondary">v2</span>
              </div>
            </div>
          </Card>
        </motion.div>
      </motion.div>
    </div>
  );
}
