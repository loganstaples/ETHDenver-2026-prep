import { motion } from "framer-motion";

interface OrbProps {
  onActivate: () => void;
  phase: "idle" | "starting";
}

export function Orb({ onActivate, phase }: OrbProps) {
  const isStarting = phase === "starting";

  return (
    <div className="relative flex items-center justify-center">
      {/* Glow backdrop */}
      <div
        className="absolute rounded-full"
        style={{
          width: 280,
          height: 280,
          background: "radial-gradient(circle, rgba(52,211,153,0.08) 0%, transparent 70%)",
          animation: isStarting ? "none" : "orb-glow-pulse 3s ease-in-out infinite",
        }}
      />

      {/* Orbital ring */}
      <motion.div
        className="absolute rounded-full border"
        style={{
          width: 280,
          height: 280,
          borderColor: "rgba(52,211,153,0.15)",
        }}
        animate={{
          rotate: isStarting ? 1080 : 360,
          scale: isStarting ? 3 : 1,
          opacity: isStarting ? 0 : 1,
        }}
        transition={
          isStarting
            ? { duration: 0.8, ease: "easeIn" }
            : { rotate: { duration: 15, repeat: Infinity, ease: "linear" }, scale: { duration: 0 }, opacity: { duration: 0 } }
        }
      />

      {/* Core orb */}
      <motion.button
        onClick={onActivate}
        className="relative z-10 rounded-full cursor-pointer focus:outline-none"
        style={{
          width: 200,
          height: 200,
          background: "radial-gradient(circle at 50% 45%, rgba(52,211,153,0.12) 0%, #0A0A0A 60%, #050505 100%)",
          border: "1px solid rgba(52,211,153,0.1)",
        }}
        whileHover={{
          boxShadow: "0 0 60px rgba(52,211,153,0.15)",
          borderColor: "rgba(52,211,153,0.25)",
        }}
        whileTap={{ scale: 0.97 }}
        animate={
          isStarting
            ? { scale: 0, opacity: 0 }
            : { scale: 1, opacity: 1 }
        }
        transition={
          isStarting
            ? { duration: 0.6, delay: 0.2, ease: "easeIn" }
            : { type: "spring", stiffness: 300, damping: 30 }
        }
      >
        {/* Inner glow dot */}
        <div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full"
          style={{
            width: 4,
            height: 4,
            backgroundColor: "rgba(52,211,153,0.6)",
            boxShadow: "0 0 20px rgba(52,211,153,0.3)",
          }}
        />
      </motion.button>
    </div>
  );
}
