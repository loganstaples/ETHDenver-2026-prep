import re

with open('src/app/network/page.tsx', 'r') as f:
    text = f.read()

new_worker_card = """\
function WorkerCard({
  node,
  onClick,
}: {
  node: WorkerNode;
  onClick: () => void;
}) {
  const status = node.slashed ? 'offline' : node.status;
  const dotColor = STATUS_DOT[status] ?? 'bg-helix-dim';
  const badgeStyle = node.slashed
    ? 'bg-red-500/10 text-red-400 border border-red-500/20'
    : (STATUS_BADGE[status] ?? 'bg-white/[0.02] text-helix-dim border border-white/[0.04]');
  const badgeLabel = node.slashed ? 'Slashed' : (STATUS_LABEL[status] ?? status);
  const isNodeActive = isActive(node.status) && !node.slashed;
  const repClamped = Math.max(0, Math.min(100, node.reputation));
  
  const typeStyle = TYPE_STYLE[node.type] ?? 'text-helix-dim';

  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'w-full h-full relative group text-left transition-all duration-300',
        'rounded-xl border border-white/[0.06] hover:border-white/[0.12]',
        'bg-[#0C0C0C]/50 hover:bg-[#111111]/80 backdrop-blur-md overflow-hidden flex flex-col',
        'shadow-sm shadow-black/20'
      )}
    >
      <div className={cn(
        'absolute top-0 inset-x-0 h-[2px] opacity-60 group-hover:opacity-100 transition-opacity duration-300',
        node.type === 'compute' ? 'bg-blue-500' :
        node.type === 'aggregator' ? 'bg-amber-500' :
        node.type === 'verifier' ? 'bg-violet-500' : 'bg-white/20'
      )} />

      <div className="p-5 flex-1 flex flex-col">
        <div className="flex justify-between items-start mb-6">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="relative flex h-2 w-2">
                {isNodeActive && (
                  <span className={cn('animate-ping absolute inline-flex h-full w-full rounded-full opacity-60', dotColor)} />
                )}
                <span className={cn('relative inline-flex rounded-full h-2 w-2', dotColor)} />
              </span>
              <span className="font-mono text-[13px] font-medium tracking-tight text-white mb-0.5">
                {formatAddr(node.address)}
              </span>
            </div>
            <div className="flex items-center gap-2 pl-4">
              <span className={cn('text-[9px] uppercase tracking-widest font-semibold', typeStyle)}>
                {TYPE_LABEL[node.type] ?? node.type}
              </span>
              {node.partyIndex != null && (
                <>
                  <span className="w-[3px] h-[3px] rounded-full bg-white/20" />
                  <span className="text-[10px] font-mono text-helix-muted">P{node.partyIndex}</span>
                </>
              )}
            </div>
          </div>
          <div className="flex flex-col items-end gap-1.5 pt-0.5">
             <span className={cn('text-[9px] font-mono uppercase tracking-widest px-2 py-0.5 rounded-md backdrop-blur-sm', badgeStyle)}>
              {badgeLabel}
            </span>
            <span className="text-[10px] text-helix-dim font-mono">
              {formatTimeSince(node.lastHeartbeat)}
            </span>
          </div>
        </div>

        <div className="mt-auto">
          <div className="mb-5 pl-4">
            <div className="flex justify-between items-end mb-1.5">
              <span className="text-[9px] text-helix-dim uppercase tracking-widest font-medium">Reputation</span>
              <span className={cn(
                'font-mono text-xs',
                repClamped >= 80 ? 'text-emerald-400' :
                repClamped >= 50 ? 'text-white/70' :
                'text-white/40',
              )}>
                {Math.round(repClamped)}<span className="text-white/30 text-[10px]">/100</span>
              </span>
            </div>
            <div className="h-[2px] w-full bg-white/[0.04] flex rounded-full overflow-hidden">
              <motion.div
                className={cn(
                  'h-full',
                  repClamped >= 80 ? 'bg-emerald-400' :
                  repClamped >= 50 ? 'bg-white/50' :
                  'bg-white/20',
                )}
                initial={{ width: 0 }}
                animate={{ width: `${repClamped}%` }}
                transition={{ duration: 0.8, ease: 'easeOut' }}
              />
            </div>
          </div>

          <div className="grid grid-cols-3 gap-2 pl-4 pt-4 border-t border-white/[0.04]">
            <div>
              <p className="text-[9px] text-helix-dim uppercase tracking-widest mb-1.5 font-medium">Rounds</p>
              <p className="font-mono text-sm text-white/90">{node.roundsParticipated}</p>
            </div>
            <div>
              <p className="text-[9px] text-helix-dim uppercase tracking-widest mb-1.5 font-medium">Proofs</p>
              <p className="font-mono text-sm text-white/90">{node.proofsSubmitted}</p>
            </div>
            <div>
              <p className="text-[9px] text-helix-dim uppercase tracking-widest mb-1.5 font-medium">Staked</p>
              <p className="font-mono text-sm text-white/90">{formatStake(node.stakedAmount)} <span className="text-[10px] text-white/40">HLX</span></p>
            </div>
          </div>
        </div>
      </div>
    </button>
  );
}
"""

text = re.sub(r'const TYPE_ACCENT.*?^};\n\nconst TYPE_GLOW.*?^};\n\n', '', text, flags=re.MULTILINE|re.DOTALL)
text = re.sub(r'function WorkerCard\(.*?\n}\n\nfunction WorkerDetailModal', new_worker_card + '\nfunction WorkerDetailModal', text, flags=re.DOTALL)

with open('src/app/network/page.tsx', 'w') as f:
    f.write(text)

print("Replacement complete")

