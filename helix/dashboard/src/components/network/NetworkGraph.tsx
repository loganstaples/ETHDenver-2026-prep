'use client';

import { useEffect, useRef, useCallback } from 'react';
import type { ActiveModel } from '@/hooks/useActiveTopology';

// ============================================================================
// Types
// ============================================================================

interface NetworkGraphProps {
    activeModels: ActiveModel[];
}

/** Internal node representation with animated position + opacity. */
interface GraphNode {
    id: string;
    type: 'model' | 'worker';
    /** Target X (layout-computed). */
    tx: number;
    /** Target Y (layout-computed). */
    ty: number;
    /** Current rendered X (lerped toward tx). */
    x: number;
    /** Current rendered Y (lerped toward ty). */
    y: number;
    /** Current opacity (0-1). */
    opacity: number;
    /** Target opacity: 1 for alive, 0 for leaving. */
    targetOpacity: number;
    radius: number;
    label: string;
    /** Which model this node belongs to (modelId). */
    modelId: string;
    /** Task type, only meaningful for model nodes. */
    taskType?: 'training' | 'inference';
}

// ============================================================================
// Constants
// ============================================================================

const LERP_FACTOR = 0.08;
const OPACITY_LERP = 0.04; // ~500ms to fade in/out at 60fps
const MODEL_RADIUS = 20;
const WORKER_RADIUS = 10;
const WORKER_RING_BASE_RADIUS = 90;
const MODEL_RING_RADIUS_FRACTION = 0.28; // fraction of min(w,h) for model ring

// Colors
const MODEL_FILL = 'rgba(255, 255, 255, 0.95)';
const WORKER_FILL = 'rgba(16, 185, 129, 0.85)'; // emerald
const WORKER_GLOW_OUTER = 'rgba(16, 185, 129, 0)';
const MODEL_GLOW_OUTER = 'rgba(255, 255, 255, 0)';

// ============================================================================
// Component
// ============================================================================

export default function NetworkGraph({ activeModels }: NetworkGraphProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const animFrameRef = useRef<number>(0);
    const nodesRef = useRef<Map<string, GraphNode>>(new Map());
    const sizeRef = useRef({ width: 600, height: 400 });
    const dashOffsetRef = useRef(0);

    // ========================================================================
    // Layout computation
    // ========================================================================

    const computeLayout = useCallback(
        (models: ActiveModel[], w: number, h: number): Map<string, { tx: number; ty: number }> => {
            const positions = new Map<string, { tx: number; ty: number }>();
            const cx = w / 2;
            const cy = h / 2;

            if (models.length === 0) return positions;

            if (models.length === 1) {
                // Single model: center of canvas, workers in ring around it
                const model = models[0];
                positions.set(`model-${model.modelId}`, { tx: cx, ty: cy });

                const workerCount = model.workerIds.length;
                const ringR = Math.min(WORKER_RING_BASE_RADIUS, Math.min(w, h) * 0.3);

                model.workerIds.forEach((wid, i) => {
                    const angle = (i * 2 * Math.PI) / Math.max(workerCount, 1);
                    positions.set(`worker-${wid}`, {
                        tx: cx + ringR * Math.cos(angle),
                        ty: cy + ringR * Math.sin(angle),
                    });
                });
            } else {
                // Multiple models: models in a circle, each with own worker ring
                const modelRingR = Math.min(w, h) * MODEL_RING_RADIUS_FRACTION;
                const workerRingR = Math.max(
                    40,
                    Math.min(60, (Math.min(w, h) * 0.18)),
                );

                models.forEach((model, mi) => {
                    const modelAngle =
                        (mi * 2 * Math.PI) / models.length - Math.PI / 2;
                    const mx = cx + modelRingR * Math.cos(modelAngle);
                    const my = cy + modelRingR * Math.sin(modelAngle);
                    positions.set(`model-${model.modelId}`, { tx: mx, ty: my });

                    const workerCount = model.workerIds.length;
                    model.workerIds.forEach((wid, wi) => {
                        const wAngle = (wi * 2 * Math.PI) / Math.max(workerCount, 1);
                        positions.set(`worker-${wid}`, {
                            tx: mx + workerRingR * Math.cos(wAngle),
                            ty: my + workerRingR * Math.sin(wAngle),
                        });
                    });
                });
            }

            return positions;
        },
        [],
    );

    // ========================================================================
    // Sync nodes map with activeModels (join / update / leave)
    // ========================================================================

    const syncNodes = useCallback(
        (models: ActiveModel[], w: number, h: number) => {
            const layoutPositions = computeLayout(models, w, h);
            const nodeMap = nodesRef.current;

            // Collect all expected node IDs
            const expectedIds = new Set<string>();

            for (const model of models) {
                const modelNodeId = `model-${model.modelId}`;
                expectedIds.add(modelNodeId);

                const pos = layoutPositions.get(modelNodeId);
                if (!nodeMap.has(modelNodeId)) {
                    // New model node: start at target pos, opacity 0
                    nodeMap.set(modelNodeId, {
                        id: modelNodeId,
                        type: 'model',
                        tx: pos?.tx ?? w / 2,
                        ty: pos?.ty ?? h / 2,
                        x: pos?.tx ?? w / 2,
                        y: pos?.ty ?? h / 2,
                        opacity: 0,
                        targetOpacity: 1,
                        radius: MODEL_RADIUS,
                        label: model.modelName,
                        modelId: model.modelId,
                        taskType: model.taskType,
                    });
                } else {
                    const node = nodeMap.get(modelNodeId)!;
                    node.tx = pos?.tx ?? node.tx;
                    node.ty = pos?.ty ?? node.ty;
                    node.targetOpacity = 1;
                    node.label = model.modelName;
                    node.taskType = model.taskType;
                }

                for (const wid of model.workerIds) {
                    const workerNodeId = `worker-${wid}`;
                    expectedIds.add(workerNodeId);

                    const wpos = layoutPositions.get(workerNodeId);
                    if (!nodeMap.has(workerNodeId)) {
                        nodeMap.set(workerNodeId, {
                            id: workerNodeId,
                            type: 'worker',
                            tx: wpos?.tx ?? w / 2,
                            ty: wpos?.ty ?? h / 2,
                            x: wpos?.tx ?? w / 2,
                            y: wpos?.ty ?? h / 2,
                            opacity: 0,
                            targetOpacity: 1,
                            radius: WORKER_RADIUS,
                            label: wid.length > 10 ? `${wid.slice(0, 6)}..${wid.slice(-4)}` : wid,
                            modelId: model.modelId,
                        });
                    } else {
                        const node = nodeMap.get(workerNodeId)!;
                        node.tx = wpos?.tx ?? node.tx;
                        node.ty = wpos?.ty ?? node.ty;
                        node.targetOpacity = 1;
                        node.modelId = model.modelId;
                    }
                }
            }

            // Mark nodes that are no longer expected as leaving (targetOpacity 0)
            nodeMap.forEach((node, id) => {
                if (!expectedIds.has(id)) {
                    node.targetOpacity = 0;
                }
            });
        },
        [computeLayout],
    );

    // ========================================================================
    // ResizeObserver
    // ========================================================================

    useEffect(() => {
        const el = containerRef.current;
        if (!el) return;

        const obs = new ResizeObserver((entries) => {
            const { width, height } = entries[0].contentRect;
            if (width > 0 && height > 0) {
                sizeRef.current = { width: Math.floor(width), height: Math.floor(height) };
            }
        });
        obs.observe(el);
        return () => obs.disconnect();
    }, []);

    // ========================================================================
    // Render loop
    // ========================================================================

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let pulse = 0;

        // Keep a ref-local copy of activeModels so the render loop always
        // has the latest value without re-starting the effect.
        let latestModels = activeModels;

        const render = () => {
            pulse += 0.012;
            dashOffsetRef.current += 0.5;

            const { width: w, height: h } = sizeRef.current;
            const dpr = window.devicePixelRatio || 1;
            canvas.width = w * dpr;
            canvas.height = h * dpr;
            canvas.style.width = `${w}px`;
            canvas.style.height = `${h}px`;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

            // Clear
            ctx.fillStyle = '#09090b';
            ctx.fillRect(0, 0, w, h);

            // Sync layout from latest models
            syncNodes(latestModels, w, h);

            const nodeMap = nodesRef.current;

            // Animate positions + opacity
            nodeMap.forEach((node) => {
                node.x += (node.tx - node.x) * LERP_FACTOR;
                node.y += (node.ty - node.y) * LERP_FACTOR;
                node.opacity += (node.targetOpacity - node.opacity) * OPACITY_LERP;
            });

            // Remove fully faded-out nodes
            const toDelete: string[] = [];
            nodeMap.forEach((node, id) => {
                if (node.targetOpacity === 0 && node.opacity < 0.01) {
                    toDelete.push(id);
                }
            });
            toDelete.forEach((id) => nodeMap.delete(id));

            // ---- Draw edges ----

            // Collect workers by modelId for ring edges
            const workersByModel = new Map<string, GraphNode[]>();
            nodeMap.forEach((node) => {
                if (node.type === 'worker') {
                    const list = workersByModel.get(node.modelId) ?? [];
                    list.push(node);
                    workersByModel.set(node.modelId, list);
                }
            });

            // Model <-> Worker edges (solid white, animated dash)
            workersByModel.forEach((workers, modelId) => {
                const modelNode = nodeMap.get(`model-${modelId}`);
                if (!modelNode) return;

                for (const worker of workers) {
                    const alpha = Math.min(modelNode.opacity, worker.opacity) * 0.35;
                    if (alpha < 0.005) continue;

                    ctx.beginPath();
                    ctx.moveTo(modelNode.x, modelNode.y);
                    ctx.lineTo(worker.x, worker.y);
                    ctx.strokeStyle = `rgba(255, 255, 255, ${alpha})`;
                    ctx.lineWidth = 2;
                    ctx.setLineDash([8, 6]);
                    // Dash flows worker -> model (negative offset = toward source)
                    ctx.lineDashOffset = -dashOffsetRef.current;
                    ctx.stroke();
                    ctx.setLineDash([]);
                }
            });

            // Worker <-> Worker ring edges (dotted gray, static)
            workersByModel.forEach((workers) => {
                if (workers.length < 2) return;

                for (let i = 0; i < workers.length; i++) {
                    const a = workers[i];
                    const b = workers[(i + 1) % workers.length];
                    const alpha = Math.min(a.opacity, b.opacity) * 0.2;
                    if (alpha < 0.005) continue;

                    ctx.beginPath();
                    ctx.moveTo(a.x, a.y);
                    ctx.lineTo(b.x, b.y);
                    ctx.strokeStyle = `rgba(200, 200, 200, ${alpha})`;
                    ctx.lineWidth = 1;
                    ctx.setLineDash([3, 5]);
                    ctx.stroke();
                    ctx.setLineDash([]);
                }
            });

            // ---- Draw nodes ----

            nodeMap.forEach((node) => {
                if (node.opacity < 0.005) return;

                if (node.type === 'model') {
                    // Radial glow
                    const glowR = node.radius * 4;
                    const glow = ctx.createRadialGradient(
                        node.x, node.y, node.radius * 0.5,
                        node.x, node.y, glowR,
                    );
                    const glowAlpha = node.opacity * (0.14 + Math.sin(pulse * 1.5) * 0.04);
                    glow.addColorStop(0, `rgba(255,255,255,${glowAlpha})`);
                    glow.addColorStop(1, MODEL_GLOW_OUTER);
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, glowR, 0, Math.PI * 2);
                    ctx.fillStyle = glow;
                    ctx.fill();

                    // Core circle
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
                    ctx.fillStyle = MODEL_FILL.replace('0.95', String(0.95 * node.opacity));
                    ctx.fill();
                    ctx.strokeStyle = `rgba(255,255,255,${0.3 * node.opacity})`;
                    ctx.lineWidth = 1;
                    ctx.stroke();

                    // Label below
                    ctx.fillStyle = `rgba(255,255,255,${0.6 * node.opacity})`;
                    ctx.font = '10px monospace';
                    ctx.textAlign = 'center';
                    ctx.fillText(
                        node.label.length > 18 ? node.label.slice(0, 18) : node.label,
                        node.x,
                        node.y + node.radius + 14,
                    );
                } else {
                    // Worker node: emerald glow
                    const glowR = node.radius * 3.5;
                    const glow = ctx.createRadialGradient(
                        node.x, node.y, node.radius * 0.3,
                        node.x, node.y, glowR,
                    );
                    const ga = node.opacity * (0.12 + Math.sin(pulse * 1.5 + node.x * 0.02) * 0.03);
                    glow.addColorStop(0, `rgba(16,185,129,${ga})`);
                    glow.addColorStop(1, WORKER_GLOW_OUTER);
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, glowR, 0, Math.PI * 2);
                    ctx.fillStyle = glow;
                    ctx.fill();

                    // Core
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
                    ctx.fillStyle = WORKER_FILL.replace('0.85', String(0.85 * node.opacity));
                    ctx.fill();
                    ctx.strokeStyle = `rgba(16,185,129,${0.25 * node.opacity})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            });

            // ---- Empty state ----

            if (latestModels.length === 0 && nodeMap.size === 0) {
                ctx.fillStyle = 'rgba(255,255,255,0.4)';
                ctx.font = '14px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('No active tasks', w / 2, h / 2 - 10);

                ctx.fillStyle = 'rgba(255,255,255,0.2)';
                ctx.font = '12px sans-serif';
                ctx.fillText(
                    'Start training or inference to see the network topology',
                    w / 2,
                    h / 2 + 14,
                );
            }

            animFrameRef.current = requestAnimationFrame(render);
        };

        animFrameRef.current = requestAnimationFrame(render);

        // Provide a way for the models-update effect to feed new data
        // into the render loop without restarting it.
        modelsUpdateRef.current = (m: ActiveModel[]) => {
            latestModels = m;
        };

        return () => cancelAnimationFrame(animFrameRef.current);
        // Only restart the render loop when syncNodes changes (stable) — not on every activeModels change
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [syncNodes]);

    // Feed new activeModels into the running render loop
    const modelsUpdateRef = useRef<((m: ActiveModel[]) => void) | null>(null);
    useEffect(() => {
        modelsUpdateRef.current?.(activeModels);
    }, [activeModels]);

    // ========================================================================
    // Render
    // ========================================================================

    return (
        <div
            ref={containerRef}
            className="relative w-full h-full min-h-[340px] rounded-2xl bg-helix-bg overflow-hidden"
        >
            <canvas ref={canvasRef} className="w-full h-full" />

            {/* Live badge */}
            <div className="absolute top-4 right-4 flex items-center gap-2 px-3 py-1.5 rounded-xl bg-helix-surface/80 backdrop-blur-sm border border-helix-border/50">
                <div className="w-1.5 h-1.5 rounded-full bg-white/60 animate-pulse" />
                <span className="text-xs text-helix-muted">Live</span>
            </div>
        </div>
    );
}
