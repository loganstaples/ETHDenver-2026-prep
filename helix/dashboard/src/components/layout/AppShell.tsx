'use client';

import Sidebar from './Sidebar';
import Header from './Header';
import { TrainingWidget } from '@/components/ui/TrainingWidget';

interface AppShellProps {
  children: React.ReactNode;
}

export default function AppShell({ children }: AppShellProps) {
  return (
    <div className="flex min-h-screen bg-helix-bg">
      <Sidebar />
      <div className="flex-1 ml-16">
        <Header />
        <main className="p-6">
          {children}
        </main>
      </div>
      <TrainingWidget />
    </div>
  );
}
