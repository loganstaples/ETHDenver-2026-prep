import { motion } from "framer-motion";
import { Header } from "./Header";
import { CardGrid } from "./CardGrid";
import { useNodeStatus } from "../../hooks/useNodeStatus";
import { useSessionTimer } from "../../hooks/useSessionTimer";
import { useWindowManager } from "../../hooks/useWindowManager";

interface ActiveViewProps {
  onStop: () => void;
  onOpenSettings: () => void;
}

export function ActiveView({ onStop, onOpenSettings }: ActiveViewProps) {
  const { status } = useNodeStatus();
  const sessionSeconds = useSessionTimer(status.running, status.uptime_secs);
  const { windows, focusedCard, focus, unfocus, minimize, restore } = useWindowManager();

  return (
    <motion.div
      className="h-full w-full flex flex-col bg-bg-void"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3, delay: 0.2 }}
    >
      <Header
        sessionSeconds={sessionSeconds}
        onStop={onStop}
        onOpenSettings={onOpenSettings}
      />
      <CardGrid
        windows={windows}
        focusedCard={focusedCard}
        onFocus={focus}
        onUnfocus={unfocus}
        onMinimize={minimize}
        onRestore={restore}
      />
    </motion.div>
  );
}
