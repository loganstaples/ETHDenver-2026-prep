'use client';

import { useState } from 'react';
import { Plus } from 'lucide-react';
import { useContractState } from '@/hooks/useContractState';
import { QuickStats } from '@/components/models/QuickStats';
import { ModelCardGrid } from '@/components/models/ModelCardGrid';
import { RegisterModelDialog } from '@/components/models/RegisterModelDialog';

export default function Page() {
  const { models, isLoading } = useContractState();
  const [registerOpen, setRegisterOpen] = useState(false);

  return (
    <div className="space-y-6">
      {/* Page Title */}
      <h1 className="page-title">Models</h1>

      {/* Quick Stats */}
      <QuickStats />

      {/* Section Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-white">All Models</h2>
        <button
          type="button"
          onClick={() => setRegisterOpen(true)}
          className="inline-flex items-center gap-1.5 bg-white text-black font-medium text-sm px-3 py-1.5 rounded-lg hover:bg-white/90 transition-colors"
        >
          <Plus size={14} strokeWidth={2} />
          Register Model
        </button>
      </div>

      {/* Model Grid */}
      <ModelCardGrid models={models} isLoading={isLoading} />

      {/* Register Dialog */}
      <RegisterModelDialog
        isOpen={registerOpen}
        onClose={() => setRegisterOpen(false)}
      />
    </div>
  );
}
