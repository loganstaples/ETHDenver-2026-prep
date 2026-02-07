'use client';

import classNames from 'clsx';
import { motion } from 'framer-motion';
import { Boxes, LayoutDashboard, MonitorDot, GraduationCap, Server, Zap } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ConnectButton } from '@rainbow-me/rainbowkit';

const sidebarItems = [
    { name: 'Overview', href: '/', icon: LayoutDashboard },
    { name: 'Status', href: '/status', icon: MonitorDot },
    { name: 'Models', href: '/models', icon: Boxes },
    { name: 'Training', href: '/training', icon: GraduationCap },
    { name: 'Nodes', href: '/nodes', icon: Server },
    { name: 'Proofs', href: '/proofs', icon: Zap },
];

export function Layout({ children }: { children: React.ReactNode }) {
    const pathname = usePathname();

    return (
        <div className="min-h-screen bg-neutral-950 text-white flex">
            {/* Sidebar */}
            <motion.aside
                initial={{ x: -100, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                className="w-64 border-r border-neutral-800 bg-neutral-900/50 backdrop-blur-xl p-6 flex flex-col fixed h-full z-10"
            >
                <div className="flex items-center gap-2 mb-10">
                    <div className="w-8 h-8 bg-emerald-500 rounded-lg flex items-center justify-center font-bold text-black">
                        H
                    </div>
                    <span className="text-xl font-bold tracking-tight">HELIX</span>
                </div>

                <nav className="flex-1 space-y-2">
                    {sidebarItems.map((item) => {
                        const isActive = pathname === item.href;
                        return (
                            <Link
                                key={item.name}
                                href={item.href}
                                className={classNames(
                                    'flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 group',
                                    isActive
                                        ? 'bg-emerald-500/10 text-emerald-400 font-medium shadow-[0_0_20px_rgba(16,185,129,0.1)]'
                                        : 'text-neutral-400 hover:bg-neutral-800/50 hover:text-white'
                                )}
                            >
                                <item.icon className={classNames("w-5 h-5", isActive ? "stroke-[2.5px]" : "stroke-2")} />
                                {item.name}
                                {isActive && (
                                    <motion.div
                                        layoutId="activeTab"
                                        className="absolute left-0 w-1 h-8 bg-emerald-500 rounded-r-full"
                                    />
                                )}
                            </Link>
                        );
                    })}
                </nav>

                <div className="pt-6 border-t border-neutral-800">
                    <div className="text-xs text-neutral-500 uppercase font-semibold tracking-wider mb-4">Network Status</div>
                    <div className="flex items-center gap-2 text-sm text-emerald-400">
                        <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                        </span>
                        Online
                    </div>
                </div>
            </motion.aside>

            {/* Main Content */}
            <main className="flex-1 ml-64 p-8 relative">
                <header className="flex justify-between items-center mb-10">
                    <div>
                        <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-neutral-500">
                            Dashboard
                        </h1>
                        <p className="text-neutral-400 text-sm">Welcome back, Coordinator.</p>
                    </div>
                    <ConnectButton />
                </header>

                <div className="grid gap-6">
                    {children}
                </div>
            </main>
        </div>
    );
}
