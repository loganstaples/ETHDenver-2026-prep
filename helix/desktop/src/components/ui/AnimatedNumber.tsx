import { useEffect, useRef, useState } from "react";
import { cn } from "../../lib/utils";

interface AnimatedNumberProps {
  value: number;
  decimals?: number;
  className?: string;
  prefix?: string;
  suffix?: string;
}

export function AnimatedNumber({
  value,
  decimals = 2,
  className,
  prefix,
  suffix,
}: AnimatedNumberProps) {
  const [display, setDisplay] = useState(value.toFixed(decimals));
  const currentRef = useRef(value);
  const rafRef = useRef<number>();

  useEffect(() => {
    const start = currentRef.current;
    const end = value;
    const duration = 400;
    const startTime = performance.now();

    const animate = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = start + (end - start) * eased;
      currentRef.current = current;
      setDisplay(current.toFixed(decimals));

      if (progress < 1) {
        rafRef.current = requestAnimationFrame(animate);
      }
    };

    rafRef.current = requestAnimationFrame(animate);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [value, decimals]);

  return (
    <span className={cn("tabular-nums", className)}>
      {prefix}
      {display}
      {suffix}
    </span>
  );
}

export function AnimatedInteger({
  value,
  className,
  prefix,
  suffix,
}: Omit<AnimatedNumberProps, "decimals">) {
  const [display, setDisplay] = useState(Math.round(value).toLocaleString());
  const currentRef = useRef(value);
  const rafRef = useRef<number>();

  useEffect(() => {
    const start = currentRef.current;
    const end = value;
    const duration = 400;
    const startTime = performance.now();

    const animate = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = start + (end - start) * eased;
      currentRef.current = current;
      setDisplay(Math.round(current).toLocaleString());

      if (progress < 1) {
        rafRef.current = requestAnimationFrame(animate);
      }
    };

    rafRef.current = requestAnimationFrame(animate);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [value]);

  return (
    <span className={cn("tabular-nums", className)}>
      {prefix}
      {display}
      {suffix}
    </span>
  );
}
