import { useEffect, useState } from "react";

export const LayerNormsDisplay = ({ runIds, metricsDataRef }: any) => {
  const [data, setData] = useState<string>("Waiting for telemetry...");

  useEffect(() => {
    let unmounted = false;
    const interval = setInterval(() => {
      if (unmounted) return;
      const latest = runIds
        .map((id: string) => {
          const arr = metricsDataRef.current[id];
          if (!arr || arr.length === 0) return null;
          for (let i = arr.length - 1; i >= 0; i--) {
            if (arr[i].layer_gradient_norms) {
              return arr[i].layer_gradient_norms;
            }
          }
          return null;
        })
        .filter(Boolean);

      if (latest.length > 0) {
        setData(latest[latest.length - 1]);
      }
    }, 1000);
    return () => {
      unmounted = true;
      clearInterval(interval);
    };
  }, [runIds, metricsDataRef]);

  return (
    <div className="flex flex-col h-full w-full bg-[#050505] p-2">
      <div className="text-[9.5px] font-bold text-emerald-400 uppercase tracking-widest mb-1 flex items-center gap-1.5 border-b border-white/5 pb-1">
        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_5px_rgba(16,185,129,0.5)]" />
        Layer-Wise Gradient Norms (Top 3)
      </div>
      <div className="flex-1 flex items-center justify-center font-mono text-[9px] text-zinc-400 text-center px-2 overflow-y-auto leading-tight">
        {data || "N/A"}
      </div>
    </div>
  );
};
