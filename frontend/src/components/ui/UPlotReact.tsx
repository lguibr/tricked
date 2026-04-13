import React, { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

interface UPlotReactProps {
  options: uPlot.Options;
  data: uPlot.AlignedData;
  onInstance?: (u: uPlot) => void;
  className?: string;
  onUnmount?: (u: uPlot) => void;
}

export const UPlotReact: React.FC<UPlotReactProps> = ({ options, data, onInstance, onUnmount, className }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<uPlot | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    
    // Auto-size on mount
    options.width = containerRef.current.clientWidth || 400;
    options.height = containerRef.current.clientHeight || 200;

    // Create new instance
    chartRef.current = new uPlot(options, data, containerRef.current);
    if (onInstance) {
      onInstance(chartRef.current);
    }

    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (chartRef.current) {
          chartRef.current.setSize({
            width: entry.contentRect.width,
            height: entry.contentRect.height
          });
        }
      }
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      // Cleanup to prevent memory leaks
      if (chartRef.current) {
        if (onUnmount) {
          onUnmount(chartRef.current);
        }
        chartRef.current.destroy();
        chartRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Intentionally strict on mount/unmount.

  useEffect(() => {
    if (chartRef.current && data) {
      chartRef.current.setData(data);
    }
  }, [data]);

  return <div ref={containerRef} className={className || "w-full h-full min-h-0"} />;
};
