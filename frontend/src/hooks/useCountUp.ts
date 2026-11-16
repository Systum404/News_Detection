"use client";

import { useState, useEffect } from "react";

/**
 * High-performance smooth count-up animation hook using requestAnimationFrame
 * with cubic ease-out acceleration.
 */
export function useCountUp(target: number, durationMs: number = 850): number {
  const [current, setCurrent] = useState<number>(0);

  useEffect(() => {
    let startTimestamp: number | null = null;
    let animationFrameId: number;

    const startVal = 0;
    const endVal = target;

    const step = (timestamp: number) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / durationMs, 1);
      // Cubic ease-out: 1 - (1 - progress)^3
      const easeProgress = 1 - Math.pow(1 - progress, 3);
      const nextVal = startVal + (endVal - startVal) * easeProgress;
      setCurrent(nextVal);

      if (progress < 1) {
        animationFrameId = requestAnimationFrame(step);
      } else {
        setCurrent(endVal);
      }
    };

    animationFrameId = requestAnimationFrame(step);

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [target, durationMs]);

  return current;
}

