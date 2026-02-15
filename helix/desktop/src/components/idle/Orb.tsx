import { motion } from "framer-motion";

interface OrbProps {
  onActivate: () => void;
  phase: "idle" | "starting";
}

export function Orb({ onActivate, phase }: OrbProps) {
  const isStarting = phase === "starting";

  return (
    <div className="relative flex items-center justify-center" style={{ width: 420, height: 420 }}>
      {/* Glow backdrop */}
      <div
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full"
        style={{
          width: 420,
          height: 420,
          background: "radial-gradient(circle, rgba(52,211,153,0.08) 0%, transparent 70%)",
          animation: isStarting ? "none" : "orb-glow-pulse 3s ease-in-out infinite",
        }}
      />

      {/* Ripple rings — only appear on activation */}
      {isStarting && (
        <>
          {[0, 150, 300].map((delay, i) => (
            <motion.div
              key={i}
              className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full border"
              style={{
                width: 280,
                height: 280,
                borderColor: "rgba(52,211,153,0.3)",
              }}
              initial={{ scale: 1, opacity: 0.5 }}
              animate={{ scale: 3, opacity: 0 }}
              transition={{ duration: 0.8, delay: delay / 1000, ease: "easeOut" }}
            />
          ))}
        </>
      )}

      {/* Outer orbital ring */}
      <motion.div
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full border"
        style={{
          width: 380,
          height: 380,
          borderColor: "rgba(52,211,153,0.1)",
        }}
        animate={{
          rotate: isStarting ? 1080 : 360,
          scale: isStarting ? 2.5 : 1,
          opacity: isStarting ? 0 : 1,
        }}
        transition={
          isStarting
            ? { duration: 0.8, ease: "easeIn" }
            : { rotate: { duration: 20, repeat: Infinity, ease: "linear" }, scale: { duration: 0 }, opacity: { duration: 0 } }
        }
      />

      {/* Inner orbital ring */}
      <motion.div
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full border"
        style={{
          width: 320,
          height: 320,
          borderColor: "rgba(52,211,153,0.06)",
        }}
        animate={{
          rotate: isStarting ? -720 : -360,
          opacity: isStarting ? 0 : 0.6,
        }}
        transition={
          isStarting
            ? { duration: 0.7, ease: "easeIn" }
            : { rotate: { duration: 30, repeat: Infinity, ease: "linear" }, opacity: { duration: 0 } }
        }
      />

      {/* Core orb */}
      <motion.button
        onClick={onActivate}
        className="relative z-10 rounded-full cursor-pointer focus:outline-none"
        style={{
          width: 280,
          height: 280,
          background: "radial-gradient(circle at 50% 45%, rgba(52,211,153,0.12) 0%, #0A0A0A 55%, #050505 100%)",
          border: "1px solid rgba(52,211,153,0.1)",
        }}
        whileHover={{
          boxShadow: "0 0 80px rgba(52,211,153,0.15)",
          borderColor: "rgba(52,211,153,0.25)",
        }}
        whileTap={{ scale: 0.97 }}
        animate={
          isStarting
            ? { scale: 0.3, opacity: 0 }
            : { scale: 1, opacity: 1 }
        }
        transition={
          isStarting
            ? { duration: 0.6, delay: 0.15, ease: [0.4, 0, 1, 1] }
            : { type: "spring", stiffness: 200, damping: 25 }
        }
      >
        {/* Inner glow dot */}
        <div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full"
          style={{
            width: 6,
            height: 6,
            backgroundColor: "rgba(52,211,153,0.6)",
            boxShadow: "0 0 30px rgba(52,211,153,0.3)",
          }}
        />
      </motion.button>

      {/* Wordmark — positioned inside the ring area */}
      <motion.div
        className="absolute bottom-[30px] left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 z-20"
        animate={
          isStarting
            ? { opacity: 0, scale: 0.8, filter: "blur(4px)" }
            : { opacity: 1, scale: 1, filter: "blur(0px)" }
        }
        transition={{ duration: 0.35 }}
      >
        <h1
          className="text-text-primary tracking-[0.3em] text-sm font-light"
          style={{ fontFamily: "'Inter', sans-serif" }}
        >
          HELIX
        </h1>
        <p className="text-text-tertiary text-xs">
          Ready to compute
        </p>
      </motion.div>
    </div>
  );
}
