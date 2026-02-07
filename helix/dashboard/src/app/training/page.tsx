'use client';

import TrainingProgress from '@/components/training/TrainingProgress';
import TrainingMetrics from '@/components/training/TrainingMetrics';

export default function TrainingPage() {
    return (
        <div className="space-y-6">
            <TrainingProgress />
            <TrainingMetrics />
        </div>
    );
}
