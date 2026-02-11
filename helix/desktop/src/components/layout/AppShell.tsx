import { Outlet } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { StatusBar } from "./StatusBar";
import { useNodeStatus } from "../../hooks/useNodeStatus";
import { formatUptime } from "../../lib/utils";

export function AppShell() {
  const { status } = useNodeStatus();

  return (
    <div className="h-full flex flex-col bg-bg-primary">
      <div className="flex flex-1 overflow-hidden">
        <Sidebar nodeRunning={status.running} />
        <main className="flex-1 overflow-y-auto p-6">
          <Outlet />
        </main>
      </div>
      <StatusBar
        nodeRunning={status.running}
        peerCount={status.connected_peers}
        uptime={formatUptime(status.uptime_secs)}
      />
    </div>
  );
}
