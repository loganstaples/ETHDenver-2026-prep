'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion } from 'framer-motion';
import { Boxes, Globe, Settings } from 'lucide-react';
import { cn } from '@/lib/utils';

const navItems = [
  { name: 'Models', href: '/', icon: Boxes },
  { name: 'Network', href: '/network', icon: Globe },
  { name: 'Settings', href: '/settings', icon: Settings },
] as const;

export default function Sidebar() {
  const pathname = usePathname();
  const [expanded, setExpanded] = useState(false);

  return (
    <aside
      onMouseEnter={() => setExpanded(true)}
      onMouseLeave={() => setExpanded(false)}
      className={cn(
        'fixed top-0 left-0 h-full z-50 flex flex-col',
        'bg-helix-bg border-r border-helix-border',
        'transition-all duration-200 ease-out',
        expanded ? 'w-[200px]' : 'w-16',
      )}
    >
      {/* Logo */}
      <div className="h-14 flex items-center px-5 shrink-0">
        <Link href="/" className="text-sm font-semibold tracking-widest text-white">
          {expanded ? 'HELIX' : 'H'}
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-2 pt-1 space-y-0.5">
        {navItems.map((item) => {
          const active =
            item.href === '/'
              ? pathname === '/' || pathname.startsWith('/models')
              : pathname.startsWith(item.href);

          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                'relative flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors',
                active
                  ? 'text-white'
                  : 'text-helix-muted hover:text-helix-text2',
              )}
            >
              {active && (
                <motion.div
                  layoutId="sidebar-active"
                  className="absolute left-0 top-1/2 -translate-y-1/2 w-[2px] h-4 bg-white rounded-r-sm"
                  transition={{ type: 'spring', stiffness: 500, damping: 35 }}
                />
              )}
              <item.icon size={18} className="shrink-0" />
              <span
                className={cn(
                  'whitespace-nowrap transition-opacity duration-200',
                  expanded ? 'opacity-100' : 'opacity-0',
                )}
              >
                {item.name}
              </span>
            </Link>
          );
        })}
      </nav>

      {/* Footer: online status */}
      <div className="px-5 py-4 border-t border-helix-border">
        <div
          className={cn(
            'flex items-center gap-1.5 text-2xs text-helix-muted transition-opacity duration-200',
            expanded ? 'opacity-100' : 'opacity-0',
          )}
        >
          <div className="w-1.5 h-1.5 rounded-full bg-white/40" />
          Online
        </div>
      </div>
    </aside>
  );
}
