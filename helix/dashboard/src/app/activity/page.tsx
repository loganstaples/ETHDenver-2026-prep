'use client';

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  Search,
  X,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import {
  forceSimulation,
  forceManyBody,
  forceCenter,
  forceLink,
  forceCollide,
  forceX,
  forceY,
} from 'd3-force';
import type { SimulationNodeDatum, SimulationLinkDatum, Simulation } from 'd3-force';
import { cn } from '@/lib/utils';
import { useNodes, type WorkerNode, type NetworkConnection } from '@/hooks/useNodes';
import { usePublicModels } from '@/hooks/usePublicModels';
import { useAccount } from 'wagmi';

// ============================================================================
// Network Graph
// ============================================================================

interface GNode extends SimulationNodeDatum {
  id: string;
  type: WorkerNode['type'];
  radius: number;
  cluster: number;
}

interface GLink extends SimulationLinkDatum<GNode> {
  latency: number;
  connectionStatus: NetworkConnection['status'];
}

/**
 * Force-directed network graph with white-ish nodes, soft glow,
 * centered layout, auto-fit, no zoom/pan. Nodes are grouped by
 * cluster (inference request they serve).
 */
function NetworkGraph({
  nodes,
  connections,
}: {
  nodes: WorkerNode[];
  connections: NetworkConnection[];
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const simRef = useRef<Simulation<GNode, GLink> | null>(null);
  const graphNodesRef = useRef<GNode[]>([]);
  const graphLinksRef = useRef<GLink[]>([]);
  const animFrameRef = useRef<number>(0);
  const prevTopologyRef = useRef<string>('');
  const [canvasSize, setCanvasSize] = useState({ width: 600, height: 360 });

  const latestNodesRef = useRef<WorkerNode[]>(nodes);
  useEffect(() => {
    latestNodesRef.current = nodes;
  }, [nodes]);

  // Resize
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      if (width > 0 && height > 0) setCanvasSize({ width: Math.floor(width), height: Math.floor(height) });
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  // Topology key — only recreate sim when node IDs or connections change
  const topologyKey = useMemo(() => {
    const ids = nodes.map((n) => n.id).sort().join(',');
    const conns = connections.map((c) => `${c.from}-${c.to}`).sort().join(',');
    return `${ids}|${conns}`;
  }, [nodes, connections]);

  // Build simulation
  useEffect(() => {
    if (nodes.length === 0) return;
    if (topologyKey === prevTopologyRef.current && simRef.current) return;
    prevTopologyRef.current = topologyKey;

    const oldPositions = new Map<string, { x: number; y: number }>();
    graphNodesRef.current.forEach((n) => {
      if (n.x != null && n.y != null) oldPositions.set(n.id, { x: n.x, y: n.y });
    });

    const w = canvasSize.width;
    const h = canvasSize.height;
    const cx = w / 2;
    const cy = h / 2;

    // Assign clusters (simulate grouping by request)
    const clusterCount = Math.max(2, Math.ceil(nodes.length / 3));
    const gNodes: GNode[] = nodes.map((n, i) => {
      const old = oldPositions.get(n.id);
      const cluster = i % clusterCount;
      return {
        id: n.id,
        type: n.type,
        radius: n.type === 'aggregator' ? 10 : n.type === 'verifier' ? 8 : 7,
        cluster,
        x: old?.x ?? cx + (Math.random() - 0.5) * w * 0.3,
        y: old?.y ?? cy + (Math.random() - 0.5) * h * 0.3,
      };
    });

    const nodeIds = new Set(gNodes.map((n) => n.id));
    const gLinks: GLink[] = connections
      .filter((c) => nodeIds.has(c.from) && nodeIds.has(c.to))
      .map((c) => ({
        source: c.from,
        target: c.to,
        latency: c.latency,
        connectionStatus: c.status,
      }));

    graphNodesRef.current = gNodes;
    graphLinksRef.current = gLinks;
    simRef.current?.stop();

    // Cluster center positions (spread evenly in a circle)
    const clusterCenters = Array.from({ length: clusterCount }, (_, i) => {
      const angle = (i / clusterCount) * Math.PI * 2 - Math.PI / 2;
      const radius = Math.min(w, h) * 0.2;
      return { x: cx + Math.cos(angle) * radius, y: cy + Math.sin(angle) * radius };
    });

    const sim = forceSimulation(gNodes)
      .force('charge', forceManyBody().strength(-120))
      .force('center', forceCenter(cx, cy).strength(0.05))
      .force(
        'link',
        forceLink<GNode, GLink>(gLinks)
          .id((d) => d.id)
          .distance(80)
          .strength(0.3),
      )
      .force('collision', forceCollide<GNode>().radius((d) => d.radius + 10))
      .force('clusterX', forceX<GNode>((d) => clusterCenters[d.cluster]?.x ?? cx).strength(0.08))
      .force('clusterY', forceY<GNode>((d) => clusterCenters[d.cluster]?.y ?? cy).strength(0.08))
      .alphaDecay(0.04)
      .on('tick', () => {
        const m = 40;
        gNodes.forEach((n) => {
          n.x = Math.max(m, Math.min(w - m, n.x!));
          n.y = Math.max(m, Math.min(h - m, n.y!));
        });
      });

    simRef.current = sim;
    return () => { sim.stop(); };
  }, [topologyKey, canvasSize.width, canvasSize.height, nodes, connections]);

  // Render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let pulse = 0;
    const render = () => {
      pulse += 0.012;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = canvasSize.width * dpr;
      canvas.height = canvasSize.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      ctx.fillStyle = '#09090b';
      ctx.fillRect(0, 0, canvasSize.width, canvasSize.height);

      const statusMap = new Map<string, WorkerNode['status']>();
      latestNodesRef.current.forEach((n) => statusMap.set(n.id, n.status));

      const gNodes = graphNodesRef.current;
      const gLinks = graphLinksRef.current;

      // Links
      gLinks.forEach((link) => {
        const s = link.source as GNode;
        const t = link.target as GNode;
        if (s.x == null || s.y == null || t.x == null || t.y == null) return;

        ctx.beginPath();
        ctx.moveTo(s.x, s.y);
        ctx.lineTo(t.x, t.y);

        const off = link.connectionStatus === 'offline';
        const deg = link.connectionStatus === 'degraded';
        const alpha = off ? 0.03 : deg ? 0.06 : 0.12;
        ctx.strokeStyle = `rgba(255, 255, 255, ${alpha})`;
        ctx.lineWidth = 0.8;

        if (link.connectionStatus === 'active') {
          ctx.setLineDash([2, 4]);
          ctx.lineDashOffset = -pulse * 15;
        } else {
          ctx.setLineDash([]);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      });

      // Nodes — all white-ish
      gNodes.forEach((node) => {
        if (node.x == null || node.y == null) return;
        const status = statusMap.get(node.id) ?? 'active';
        const isOffline = status === 'offline';
        const isActive = status === 'active' || status === 'proving' || status === 'training';

        // Soft outer glow
        if (!isOffline) {
          const glowR = node.radius * 4;
          const glow = ctx.createRadialGradient(
            node.x, node.y, node.radius * 0.5,
            node.x, node.y, glowR,
          );
          const pa = isActive ? 0.06 + Math.sin(pulse * 1.5 + node.x * 0.01) * 0.03 : 0.04;
          glow.addColorStop(0, `rgba(255, 255, 255, ${pa})`);
          glow.addColorStop(1, 'rgba(255, 255, 255, 0)');
          ctx.beginPath();
          ctx.arc(node.x, node.y, glowR, 0, Math.PI * 2);
          ctx.fillStyle = glow;
          ctx.fill();
        }

        // Inner glow ring
        if (!isOffline && isActive) {
          const ringR = node.radius * 2;
          const ring = ctx.createRadialGradient(
            node.x, node.y, node.radius,
            node.x, node.y, ringR,
          );
          ring.addColorStop(0, 'rgba(255, 255, 255, 0.08)');
          ring.addColorStop(1, 'rgba(255, 255, 255, 0)');
          ctx.beginPath();
          ctx.arc(node.x, node.y, ringR, 0, Math.PI * 2);
          ctx.fillStyle = ring;
          ctx.fill();
        }

        // Core circle
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        if (isOffline) {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
        } else {
          const brightness = isActive ? 0.9 : 0.5;
          ctx.fillStyle = `rgba(255, 255, 255, ${brightness})`;
        }
        ctx.fill();

        // Subtle border
        if (!isOffline) {
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      });

      animFrameRef.current = requestAnimationFrame(render);
    };

    animFrameRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animFrameRef.current);
  }, [canvasSize]);

  return (
    <div ref={containerRef} className="relative w-full h-full min-h-[340px] rounded-2xl bg-helix-bg overflow-hidden">
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ width: canvasSize.width, height: canvasSize.height }}
      />
      <div className="absolute top-4 right-4 flex items-center gap-2 px-3 py-1.5 rounded-xl bg-helix-surface/80 backdrop-blur-sm border border-helix-border/50">
        <div className="w-1.5 h-1.5 rounded-full bg-white/60 animate-pulse" />
        <span className="text-xs text-helix-muted">Live</span>
      </div>
    </div>
  );
}

// ============================================================================
// Mock inference request data
// ============================================================================

interface InferenceRequest {
  id: string;
  timestamp: number;
  requester: string;
  modelTokenId: number;
  modelName: string;
  prediction: number;
  confidence: number;
  fee: number;
  ownerRevenue: number;
  latency: number;
  workers: number;
  status: 'completed' | 'pending' | 'failed';
}

function generateMockRequests(modelNames: Map<number, string>): InferenceRequest[] {
  const requesters = [
    '0x742d35Cc6634C0532925a3b844Bc9e7595f01231',
    '0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199',
    '0xdD2FD4581271e230360230F9337D5c0430Bf44C0',
    '0xbDA5747bFD65F08deb54cb465eB87D40e51B197E',
    '0x2546BcD3c84621e976D8185a91A922aE77ECEc30',
  ];

  const tokenIds = Array.from(modelNames.keys());
  if (tokenIds.length === 0) tokenIds.push(1, 2, 3);

  return Array.from({ length: 40 }, (_, i) => {
    const tokenId = tokenIds[Math.floor(Math.random() * tokenIds.length)];
    const fee = 0.001;
    const ownerRevenue = fee * (0.01 + Math.random() * 0.04);

    return {
      id: `inf-${Date.now()}-${i}`,
      timestamp: Date.now() - i * (30000 + Math.random() * 60000),
      requester: requesters[Math.floor(Math.random() * requesters.length)],
      modelTokenId: tokenId,
      modelName: modelNames.get(tokenId) ?? `Model #${tokenId}`,
      prediction: Math.floor(Math.random() * 10),
      confidence: 0.75 + Math.random() * 0.24,
      fee,
      ownerRevenue,
      latency: 80 + Math.random() * 200,
      workers: 3,
      status: Math.random() > 0.05 ? 'completed' : Math.random() > 0.5 ? 'pending' : 'failed',
    };
  });
}

function generateTimelineData(): Array<{ time: string; requests: number }> {
  return Array.from({ length: 24 }, (_, i) => {
    const h = (new Date().getHours() - (23 - i) + 24) % 24;
    return {
      time: `${h.toString().padStart(2, '0')}:00`,
      requests: Math.floor(5 + Math.random() * 20),
    };
  });
}

// ============================================================================
// Page
// ============================================================================

export default function ActivityPage() {
  const { address } = useAccount();
  const { allModels } = usePublicModels();
  const { nodes, topology } = useNodes();
  const [search, setSearch] = useState('');
  const [sortField, setSortField] = useState<'timestamp' | 'fee' | 'confidence'>('timestamp');
  const [sortAsc, setSortAsc] = useState(false);

  const modelNames = useMemo(() => {
    const map = new Map<number, string>();
    allModels.forEach((m) => map.set(m.tokenId, m.name));
    return map;
  }, [allModels]);

  const myModels = useMemo(
    () => allModels.filter((m) => address && m.owner.toLowerCase() === address.toLowerCase()),
    [allModels, address],
  );

  const requests = useMemo(() => generateMockRequests(modelNames), [modelNames]);
  const timelineData = useMemo(() => generateTimelineData(), []);

  const filtered = useMemo(() => {
    let result = requests;
    if (search.trim()) {
      const q = search.toLowerCase();
      result = result.filter(
        (r) =>
          r.requester.toLowerCase().includes(q) ||
          r.modelName.toLowerCase().includes(q) ||
          r.prediction.toString() === q,
      );
    }
    result.sort((a, b) => {
      const mul = sortAsc ? 1 : -1;
      if (sortField === 'timestamp') return (a.timestamp - b.timestamp) * mul;
      if (sortField === 'fee') return (a.fee - b.fee) * mul;
      return (a.confidence - b.confidence) * mul;
    });
    return result;
  }, [requests, search, sortField, sortAsc]);

  const totalRevenue = requests.reduce((s, r) => s + r.ownerRevenue, 0);
  const totalRequests = requests.length;
  const avgLatency = requests.reduce((s, r) => s + r.latency, 0) / (requests.length || 1);

  const toggleSort = (field: typeof sortField) => {
    if (sortField === field) setSortAsc(!sortAsc);
    else { setSortField(field); setSortAsc(false); }
  };

  const SortIcon = sortAsc ? ChevronUp : ChevronDown;

  const chartTooltipStyle = {
    backgroundColor: '#111113',
    border: '1px solid #1e1e22',
    borderRadius: 12,
    fontSize: 12,
    fontFamily: 'var(--font-geist-mono)',
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
      className="space-y-6 max-w-[1200px]"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight text-white">Activity</h1>
          <p className="text-sm text-helix-muted mt-1">
            Inference requests on your models
          </p>
        </div>
        <div className="flex items-center gap-4 text-sm text-helix-muted">
          <span>{myModels.length} models</span>
          <span>{totalRequests} requests</span>
        </div>
      </div>

      {/* Network Graph + Summary */}
      <div className="grid grid-cols-3 gap-4">
        {/* Graph */}
        <div className="col-span-2 bg-helix-surface border border-helix-border rounded-2xl overflow-hidden" style={{ height: 340 }}>
          <NetworkGraph nodes={nodes} connections={topology.connections} />
        </div>

        {/* Summary */}
        <div className="bg-helix-surface border border-helix-border rounded-2xl p-6 flex flex-col justify-between">
          <div className="space-y-6">
            <div>
              <div className="text-xs text-helix-muted mb-1">Total Revenue</div>
              <div className="text-3xl font-semibold text-white tabular-nums font-mono">
                {totalRevenue.toFixed(4)}
              </div>
              <div className="text-xs text-helix-muted mt-0.5">ADI</div>
            </div>
            <div>
              <div className="text-xs text-helix-muted mb-1">Total Requests</div>
              <div className="text-3xl font-semibold text-white tabular-nums font-mono">
                {totalRequests}
              </div>
            </div>
            <div>
              <div className="text-xs text-helix-muted mb-1">Avg Latency</div>
              <div className="text-3xl font-semibold text-white tabular-nums font-mono">
                {avgLatency.toFixed(0)}<span className="text-lg text-helix-muted">ms</span>
              </div>
            </div>
          </div>
          <div className="text-xs text-helix-dim mt-4">
            {nodes.length} nodes serving requests
          </div>
        </div>
      </div>

      {/* Timeline Chart */}
      <div className="bg-helix-surface border border-helix-border rounded-2xl p-6">
        <div className="text-sm font-medium text-white mb-4">Request Volume (24h)</div>
        <div className="h-36">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={timelineData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="volG" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#ffffff" stopOpacity={0.08} />
                  <stop offset="100%" stopColor="#ffffff" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
              <XAxis dataKey="time" tick={{ fill: '#3e3e44', fontSize: 10 }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fill: '#3e3e44', fontSize: 10 }} tickLine={false} axisLine={false} width={30} />
              <Tooltip contentStyle={chartTooltipStyle} labelStyle={{ color: '#63636e' }} />
              <Area type="monotone" dataKey="requests" stroke="rgba(255,255,255,0.4)" fill="url(#volG)" strokeWidth={1.5} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Request Table */}
      <div className="bg-helix-surface border border-helix-border rounded-2xl overflow-hidden">
        {/* Search + header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-helix-border">
          <span className="text-sm font-medium text-white">Recent Requests</span>
          <div className="relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-helix-dim" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search..."
              className="pl-8 pr-7 py-1.5 bg-helix-bg border border-helix-border rounded-xl text-xs text-white placeholder:text-helix-dim focus:outline-none focus:border-helix-border2 w-48"
            />
            {search && (
              <button type="button" onClick={() => setSearch('')} className="absolute right-2 top-1/2 -translate-y-1/2 text-helix-dim hover:text-white">
                <X size={12} />
              </button>
            )}
          </div>
        </div>

        {/* Table header */}
        <div className="grid grid-cols-[1fr_90px_90px_60px_60px_60px_50px] gap-3 px-6 py-2.5 border-b border-helix-border bg-helix-bg/30 text-xs text-helix-muted">
          <button type="button" onClick={() => toggleSort('timestamp')} className="flex items-center gap-1 text-left hover:text-helix-text2 transition-colors">
            Time {sortField === 'timestamp' && <SortIcon size={10} />}
          </button>
          <span>Requester</span>
          <span>Model</span>
          <span>Result</span>
          <button type="button" onClick={() => toggleSort('confidence')} className="flex items-center gap-1 hover:text-helix-text2 transition-colors">
            Conf {sortField === 'confidence' && <SortIcon size={10} />}
          </button>
          <button type="button" onClick={() => toggleSort('fee')} className="flex items-center gap-1 hover:text-helix-text2 transition-colors">
            Fee {sortField === 'fee' && <SortIcon size={10} />}
          </button>
          <span>Status</span>
        </div>

        {/* Rows */}
        <div className="max-h-[360px] overflow-y-auto">
          {filtered.map((r) => (
            <div
              key={r.id}
              className="grid grid-cols-[1fr_90px_90px_60px_60px_60px_50px] gap-3 px-6 py-2.5 border-b border-helix-border/50 hover:bg-white/[0.015] text-xs transition-colors"
            >
              <span className="text-helix-text2 font-mono tabular-nums">
                {new Date(r.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
              </span>
              <span className="text-helix-text2 font-mono truncate">
                {r.requester.slice(0, 6)}..{r.requester.slice(-3)}
              </span>
              <span className="text-helix-text2 truncate">{r.modelName}</span>
              <div className="flex items-center">
                <span className="w-6 h-6 rounded-lg bg-white/[0.06] flex items-center justify-center text-xs font-mono font-medium text-white">
                  {r.prediction}
                </span>
              </div>
              <span className="text-white font-mono tabular-nums">
                {(r.confidence * 100).toFixed(1)}%
              </span>
              <span className="text-white font-mono tabular-nums">
                {r.fee.toFixed(3)}
              </span>
              <span
                className={cn(
                  'font-mono text-xs',
                  r.status === 'completed' ? 'text-green-400'
                    : r.status === 'pending' ? 'text-yellow-400'
                      : 'text-red-400',
                )}
              >
                {r.status === 'completed' ? 'OK' : r.status === 'pending' ? 'PEND' : 'FAIL'}
              </span>
            </div>
          ))}
          {filtered.length === 0 && (
            <div className="flex items-center justify-center py-12 text-sm text-helix-muted">
              No requests found
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
