import { useState, useEffect } from "react";
import type { NodeConfig } from "../../lib/types";
import { useConfig } from "../../hooks/useConfig";

export function SettingsPanel() {
  const { config, updateConfig } = useConfig();
  const [local, setLocal] = useState<NodeConfig | null>(null);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (config && !local) {
      setLocal(config);
    }
  }, [config, local]);

  if (!local) {
    return (
      <div className="text-body text-text-tertiary">Loading settings...</div>
    );
  }

  const update = <K extends keyof NodeConfig>(key: K, value: NodeConfig[K]) => {
    setLocal((prev) => prev && { ...prev, [key]: value });
    setDirty(true);
  };

  const save = async () => {
    if (!local) return;
    setSaving(true);
    try {
      await updateConfig(local);
      setDirty(false);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="flex flex-col gap-5">
      {/* Resource Allocation */}
      <Section title="RESOURCES">
        <SliderField
          label="CPU Threads"
          value={local.cpu_threads}
          min={1}
          max={32}
          onChange={(v) => update("cpu_threads", v)}
        />
        <SliderField
          label="GPU Memory (GB)"
          value={Math.round(local.gpu_memory_limit_mb / 1024)}
          min={1}
          max={32}
          onChange={(v) => update("gpu_memory_limit_mb", v * 1024)}
        />
        <SliderField
          label="Max Tasks"
          value={local.max_concurrent_tasks}
          min={1}
          max={8}
          onChange={(v) => update("max_concurrent_tasks", v)}
        />
      </Section>

      {/* Network */}
      <Section title="NETWORK">
        <TextField
          label="Listen Address"
          value={local.listen_address}
          onChange={(v) => update("listen_address", v)}
        />
        <TextField
          label="Aggregator"
          value={local.aggregator_address}
          onChange={(v) => update("aggregator_address", v)}
        />
        <NumberField
          label="RPC Port"
          value={local.rpc_port}
          onChange={(v) => update("rpc_port", v)}
        />
        <ToggleField
          label="Use TLS"
          checked={local.use_tls}
          onChange={(v) => update("use_tls", v)}
        />
      </Section>

      {/* Training */}
      <Section title="TRAINING">
        <NumberField
          label="Local Epochs"
          value={local.local_epochs}
          onChange={(v) => update("local_epochs", v)}
        />
        <NumberField
          label="Batch Size"
          value={local.batch_size}
          onChange={(v) => update("batch_size", v)}
        />
        <ToggleField
          label="Generate Proofs"
          checked={local.generate_proofs}
          onChange={(v) => update("generate_proofs", v)}
        />
      </Section>

      {/* Save button */}
      {dirty && (
        <button
          onClick={save}
          disabled={saving}
          className="w-full h-8 bg-text-bright text-bg-primary text-body font-medium hover:bg-text-primary transition-colors disabled:opacity-40"
        >
          {saving ? "Saving..." : "Save Changes"}
        </button>
      )}
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-3">
      <span className="text-label text-text-tertiary border-b border-border-grid pb-1">
        {title}
      </span>
      {children}
    </div>
  );
}

function SliderField({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-body text-text-secondary">{label}</span>
      <div className="flex items-center gap-2">
        <input
          type="range"
          min={min}
          max={max}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-24 accent-white h-0.5"
        />
        <span className="text-mono-data text-text-primary w-8 text-right tabular-nums">
          {value}
        </span>
      </div>
    </div>
  );
}

function TextField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-body text-text-secondary">{label}</span>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="bg-bg-primary border border-border-primary text-mono-data text-text-primary px-2 py-1 w-44 focus:border-border-focus outline-none"
      />
    </div>
  );
}

function NumberField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-body text-text-secondary">{label}</span>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="bg-bg-primary border border-border-primary text-mono-data text-text-primary px-2 py-1 w-24 text-right focus:border-border-focus outline-none tabular-nums"
      />
    </div>
  );
}

function ToggleField({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-body text-text-secondary">{label}</span>
      <button
        onClick={() => onChange(!checked)}
        className={`w-8 h-4 rounded-none border relative transition-colors ${
          checked
            ? "bg-text-bright border-text-bright"
            : "bg-bg-primary border-border-primary"
        }`}
      >
        <div
          className={`w-3 h-3 absolute top-0.5 transition-transform ${
            checked
              ? "translate-x-4 bg-bg-primary"
              : "translate-x-0.5 bg-text-tertiary"
          }`}
        />
      </button>
    </div>
  );
}
