'use client';

import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

interface TabItem {
  id: string;
  label: string;
}

interface TabsProps {
  tabs: TabItem[];
  activeTab: string;
  onChange: (id: string) => void;
  className?: string;
}

export function Tabs({ tabs, activeTab, onChange, className }: TabsProps) {
  return (
    <div className={cn('flex items-center gap-1', className)}>
      {tabs.map((tab) => (
        <Tab
          key={tab.id}
          id={tab.id}
          label={tab.label}
          isActive={tab.id === activeTab}
          onClick={() => onChange(tab.id)}
        />
      ))}
    </div>
  );
}

interface TabProps {
  id: string;
  label: string;
  isActive: boolean;
  onClick: () => void;
}

export function Tab({ label, isActive, onClick }: TabProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'relative px-3 py-2 text-sm transition-colors',
        isActive ? 'text-helix-text' : 'text-helix-muted hover:text-helix-text2',
      )}
    >
      {label}
      {isActive && (
        <motion.div
          layoutId="activeTab"
          className="absolute inset-x-0 bottom-0 h-0.5 rounded bg-white"
          transition={{
            type: 'spring',
            stiffness: 500,
            damping: 35,
          }}
        />
      )}
    </button>
  );
}
