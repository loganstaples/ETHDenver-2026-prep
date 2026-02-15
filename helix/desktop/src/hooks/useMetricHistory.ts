import { useState, useCallback, useRef } from "react";

const MAX_POINTS = 30;

export function useMetricHistory(initialValue = 0) {
  const [history, setHistory] = useState<number[]>(() =>
    Array(MAX_POINTS).fill(initialValue)
  );
  const bufferRef = useRef<number[]>(Array(MAX_POINTS).fill(initialValue));

  const push = useCallback((value: number) => {
    const buf = bufferRef.current;
    buf.push(value);
    if (buf.length > MAX_POINTS) buf.shift();
    bufferRef.current = [...buf];
    setHistory(bufferRef.current);
  }, []);

  return { history, push };
}

export function useMultiMetricHistory() {
  const cpuBuf = useRef<number[]>(Array(MAX_POINTS).fill(0));
  const memBuf = useRef<number[]>(Array(MAX_POINTS).fill(0));
  const gpuBuf = useRef<number[]>(Array(MAX_POINTS).fill(0));

  const [cpu, setCpu] = useState<number[]>(() => Array(MAX_POINTS).fill(0));
  const [memory, setMemory] = useState<number[]>(() => Array(MAX_POINTS).fill(0));
  const [gpu, setGpu] = useState<number[]>(() => Array(MAX_POINTS).fill(0));

  const pushAll = useCallback(
    (cpuVal: number, memVal: number, gpuVal: number) => {
      cpuBuf.current = [...cpuBuf.current.slice(-(MAX_POINTS - 1)), cpuVal];
      memBuf.current = [...memBuf.current.slice(-(MAX_POINTS - 1)), memVal];
      gpuBuf.current = [...gpuBuf.current.slice(-(MAX_POINTS - 1)), gpuVal];
      setCpu(cpuBuf.current);
      setMemory(memBuf.current);
      setGpu(gpuBuf.current);
    },
    []
  );

  return { cpu, memory, gpu, pushAll };
}
