'use client';

import classNames from 'clsx';
import { motion, AnimatePresence } from 'framer-motion';
import {
    LayoutDashboard,
    Upload,
    Database,
    Boxes,
    Activity,
    Wallet,
    MonitorDot,
    Server,
    Zap,
    ChevronDown,
} from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';
import { ConnectButton } from '@rainbow-me/rainbowkit';

const primaryNav = [
    { name: 'Dashboard', href: '/', icon: LayoutDashboard },
    { name: 'Submit Model', href: '/submit', icon: Upload },
    { name: 'Datasets', href: '/data', icon: Database },
    { name: 'My Models', href: '/models', icon: Boxes },
    { name: 'Training', href: '/training', icon: Activity },
    { name: 'Billing', href: '/billing', icon: Wallet },
];

const advancedNav = [
    { name: 'Status', href: '/status', icon: MonitorDot },
    { name: 'Nodes', href: '/nodes', icon: Server },
    { name: 'Proofs', href: '/proofs', icon: Zap },
];

const pageMeta: Record<string, { title: string; subtitle: string }> = {
    '/': { title: 'Dashboard', subtitle: 'Network overview' },
    '/submit': { title: 'Submit Model', subtitle: 'Upload and register a model' },
    '/data': { title: 'Datasets', subtitle: 'Training data' },
    '/models': { title: 'My Models', subtitle: 'Registered models' },
    '/training': { title: 'Training', subtitle: 'Active jobs' },
    '/billing': { title: 'Billing', subtitle: 'Staking & costs' },
    '/status': { title: 'Status', subtitle: 'Network monitor' },
    '/nodes': { title: 'Nodes', subtitle: 'Node topology' },
    '/proofs': { title: 'Proofs', subtitle: 'Proof explorer' },
};

export function Layout({ children }: { children: React.ReactNode }) {
    const pathname = usePathname();
    const [advancedOpen, setAdvancedOpen] = useState(false);
    const page = pageMeta[pathname] || { title: 'HELIX', subtitle: '' };

    return (
        <div className="min-h-screen bg-helix-bg text-white flex">
            {/* Sidebar */}
            <aside className="w-[200px] border-r border-helix-border bg-helix-bg flex flex-col fixed h-full z-10">
                {/* Logo */}
                <div className="h-12 flex items-center px-5 border-b border-helix-border">
                    <Link href="/" className="flex items-center gap-2.5">
                        <div className="w-5 h-5 border border-white/40 flex items-center justify-center font-mono text-[10px] font-bold tracking-tighter text-white/90">
                            H
                        </div>
                        <span className="text-[13px] font-semibold tracking-tight">HELIX</span>
                    </Link>
                </div>

                {/* Nav */}
                <nav className="flex-1 px-2 pt-3 space-y-0.5">
                    {primaryNav.map((item) => {
                        const active = pathname === item.href;
                        return (
                            <Link
                                key={item.name}
                                href={item.href}
                                className={classNames(
                                    'flex items-center gap-2.5 px-3 py-[7px] rounded-[4px] text-[13px] transition-colors relative',
                                    active
                                        ? 'bg-white/[0.07] text-white'
                                        : 'text-[#888] hover:text-white hover:bg-white/[0.04]'
                                )}
                            >
                                {active && (
                                    <motion.div
                                        layoutId="navIndicator"
                                        className="absolute left-0 w-[2px] h-3.5 bg-white rounded-r-sm"
                                        transition={{ type: 'spring', stiffness: 500, damping: 35 }}
                                    />
                                )}
                                <item.icon className="w-[14px] h-[14px] flex-shrink-0" strokeWidth={1.5} />
                                {item.name}
                            </Link>
                        );
                    })}

                    {/* Advanced */}
                    <div className="pt-3 mt-1">
                        <button
                            onClick={() => setAdvancedOpen(!advancedOpen)}
                            className="flex items-center justify-between w-full px-3 py-1.5 text-[11px] text-[#555] uppercase tracking-widest font-medium hover:text-[#888] transition-colors"
                        >
                            Advanced
                            <ChevronDown className={classNames(
                                'w-3 h-3 transition-transform duration-200',
                                advancedOpen && 'rotate-180'
                            )} />
                        </button>
                        <AnimatePresence>
                            {advancedOpen && (
                                <motion.div
                                    initial={{ height: 0, opacity: 0 }}
                                    animate={{ height: 'auto', opacity: 1 }}
                                    exit={{ height: 0, opacity: 0 }}
                                    transition={{ duration: 0.15 }}
                                    className="overflow-hidden space-y-0.5 mt-0.5"
                                >
                                    {advancedNav.map((item) => {
                                        const active = pathname === item.href;
                                        return (
                                            <Link
                                                key={item.name}
                                                href={item.href}
                                                className={classNames(
                                                    'flex items-center gap-2.5 px-3 py-[7px] rounded-[4px] text-[13px] transition-colors',
                                                    active
                                                        ? 'bg-white/[0.07] text-white'
                                                        : 'text-[#888] hover:text-white hover:bg-white/[0.04]'
                                                )}
                                            >
                                                <item.icon className="w-[14px] h-[14px] flex-shrink-0" strokeWidth={1.5} />
                                                {item.name}
                                            </Link>
                                        );
                                    })}
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                </nav>

                {/* Footer */}
                <div className="px-5 py-3 border-t border-helix-border">
                    <div className="flex items-center gap-1.5 text-[11px] text-[#555]">
                        <span className="w-1.5 h-1.5 rounded-full bg-white/50 animate-pulse-slow" />
                        Online
                    </div>
                </div>
            </aside>

            {/* Main */}
            <main className="flex-1 ml-[200px]">
                <header className="h-12 border-b border-helix-border flex items-center justify-between px-6 sticky top-0 bg-helix-bg/80 backdrop-blur-sm z-20">
                    <div className="flex items-baseline gap-3">
                        <h1 className="text-[13px] font-semibold text-white">{page.title}</h1>
                        <span className="text-[11px] text-[#555]">{page.subtitle}</span>
                    </div>
                    <ConnectButton />
                </header>

                <div className="p-6">
                    {children}
                </div>
            </main>
        </div>
    );
}
