'use client';

import NodeNetworkView from '@/components/network/NodeNetworkView';
import NodeStats from '@/components/network/NodeStats';

export default function NodesPage() {
    return (
        <div className="space-y-6">
            <NodeNetworkView />
            <NodeStats />
        </div>
    );
}
