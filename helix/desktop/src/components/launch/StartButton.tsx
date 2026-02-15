import { useState } from "react";
import { motion } from "framer-motion";
import { cn } from "../../lib/utils";

interface StartButtonProps {
  onStart: () => void;
  starting?: boolean;
}

export function StartButton({ onStart, starting = false }: StartButtonProps) {
  const [hovered, setHovered] = useState(false);

  return (
    <motion.button
      onClick={onStart}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      disabled={starting}
      className="relative w-[180px] h-[180px] rounded-full border-2 border-border-focus bg-transparent cursor-pointer transition-all focus:outline-none disabled:cursor-wait"
      whileHover={{
        borderColor: "#FFFFFF",
        boxShadow: "0 0 40px rgba(255,255,255,0.06)",
      }}
      whileTap={{ scale: 0.95 }}
      layoutId="start-circle"
    >
      {/* Orbital dot */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div
          className={cn(
            "absolute w-full h-full",
            starting ? "animate-orbit-fast" : "animate-orbit"
          )}
          style={{ transformOrigin: "center center" }}
        >
          <div
            className={cn(
              "w-1 h-1 rounded-full absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2",
              hovered || starting ? "bg-white" : "bg-text-secondary"
            )}
          />
        </div>
      </div>

      {/* Label */}
      <span className="font-mono text-[16px] font-medium tracking-[0.15em] text-text-primary select-none">
        {starting ? "..." : "START"}
      </span>
    </motion.button>
  );
}
