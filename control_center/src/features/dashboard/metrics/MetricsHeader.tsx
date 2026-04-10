import {
  VscGraph,
} from "react-icons/vsc";
import { Slider } from "@/components/ui/slider";

interface MetricsHeaderProps {
  runIds: string[];
  activeJobs: any[];
  smoothingWeight: number;
  setSmoothingWeight: (val: number) => void;
  xAxisMode: "step" | "relative" | "absolute";
  setXAxisMode: (mode: "step" | "relative" | "absolute") => void;
}

export function MetricsHeader({
  runIds,
  activeJobs,
  smoothingWeight,
  setSmoothingWeight,
  xAxisMode,
  setXAxisMode
}: MetricsHeaderProps) {
  return (
    <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-[#080808]/95 backdrop-blur-md sticky top-0 z-20 shrink-0">
      <div className="flex items-center gap-3">
        <h2 className="text-[10px] font-bold text-zinc-200 uppercase tracking-widest flex items-center gap-1.5">
          <VscGraph className="w-3.5 h-3.5 text-primary" />
          Engine Observability
        </h2>
        {runIds.length > 0 && (
          <div className="flex items-center gap-1.5 border-l border-white/10 pl-3">
            {runIds.map((id: string) => {
              const isRunning = activeJobs.some((j: any) => j.id === id);
              return (
                <span
                  key={id}
                  className={`text-[8px] font-black tracking-widest uppercase px-1.5 py-0.5 rounded border ${
                    isRunning
                      ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.2)]"
                      : "bg-zinc-500/10 text-zinc-400 border-zinc-500/20"
                  }`}
                >
                  {id.substring(0, 4)}: {isRunning ? "RUNNING" : "STOPPED"}
                </span>
              );
            })}
          </div>
        )}
      </div>
      <div className="flex items-center space-x-6">
        <div className="flex items-center space-x-2">
          <span className="text-[8.5px] font-bold uppercase tracking-widest text-zinc-500">
            SMOOTHING
          </span>
          <div className="w-24 px-1 flex items-center">
            <Slider
              defaultValue={[smoothingWeight]}
              max={0.999}
              min={0.0}
              step={0.001}
              onValueChange={(val: any) => setSmoothingWeight(val[0])}
            />
          </div>
          <span className="text-[8px] font-mono text-zinc-400 w-6 text-right">
            {smoothingWeight.toFixed(2)}
          </span>
        </div>

        <div className="flex items-center space-x-2">
          <span className="text-[8.5px] font-bold uppercase tracking-widest text-zinc-500">
            X-Axis
          </span>
          <div className="flex bg-black/80 p-0.5 rounded border border-white/10 shadow-inner">
            <button
              onClick={() => setXAxisMode("step")}
              className={`px-2 py-0.5 text-[8.5px] uppercase tracking-widest font-black rounded-sm transition-all ${xAxisMode === "step" ? "bg-primary/20 text-primary border border-primary/40 shadow-sm" : "text-zinc-500 hover:text-zinc-300 border border-transparent hover:bg-white/5"}`}
            >
              Step
            </button>
            <button
              onClick={() => setXAxisMode("relative")}
              className={`px-2 py-0.5 text-[8.5px] uppercase tracking-widest font-black rounded-sm transition-all ${xAxisMode === "relative" ? "bg-primary/20 text-primary border border-primary/40 shadow-sm" : "text-zinc-500 hover:text-zinc-300 border border-transparent hover:bg-white/5"}`}
            >
              Relative
            </button>
            <button
              onClick={() => setXAxisMode("absolute")}
              className={`px-2 py-0.5 text-[8.5px] uppercase tracking-widest font-black rounded-sm transition-all ${xAxisMode === "absolute" ? "bg-primary/20 text-primary border border-primary/40 shadow-sm" : "text-zinc-500 hover:text-zinc-300 border border-transparent hover:bg-white/5"}`}
            >
              Absolute
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
