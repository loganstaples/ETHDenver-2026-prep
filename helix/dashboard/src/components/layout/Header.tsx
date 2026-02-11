export default function Header() {
    return (
        <header className="bg-helix-surface border-b border-helix-border h-12 px-5 flex items-center justify-between">
            <span className="text-[13px] font-semibold text-white tracking-tight">HELIX</span>
            <button className="bg-white text-black text-[13px] font-medium px-3 py-1.5 rounded-[4px] hover:bg-white/90 transition-colors">
                Connect Wallet
            </button>
        </header>
    );
}
