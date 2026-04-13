import { Target } from "lucide-react";
import { VscCopy, VscCheck } from "react-icons/vsc";
import { GlassCard } from "@/components/dashboard/GlassCard";
import { Trial } from "@/hooks/useOptimizerStudy";

interface StatsProps {
  trials: Trial[];
  completeTrials: Trial[];
  prunedTrials: Trial[];
  runningTrials: Trial[];
  failedTrials: Trial[];
  bestTrial: Trial | null;
  copiedConfig: boolean;
  handleCopyBestConfig: () => void;
}

export function OptimizerStudyHeaderStats({
  trials,
  completeTrials,
  prunedTrials,
  runningTrials,
  failedTrials,
  bestTrial,
  copiedConfig,
  handleCopyBestConfig,
}: StatsProps) {
  return (
    <GlassCard className="flex items-center justify-between p-5 shrink-0">
      <div className="flex items-center gap-10">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-full bg-emerald-500/10 flex items-center justify-center border border-emerald-500/20 shadow-[0_0_15px_rgba(16,185,129,0.15)]">
            <Target className="w-6 h-6 text-emerald-400" />
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] uppercase font-black text-zinc-500 tracking-widest mb-0.5">
              Total Trials
            </span>
            <span className="text-3xl font-black text-zinc-100 tracking-tight leading-none drop-shadow-md">
              {trials.length}
            </span>
          </div>
        </div>

        <div className="h-10 w-px bg-white/10 shadow-lg" />

        <div className="flex gap-8">
          <div className="flex flex-col">
            <span className="text-[10px] uppercase font-bold text-blue-500/70 tracking-widest mb-1.5 flex items-center gap-1.5">
              <div className="w-1.5 h-1.5 rounded-full bg-blue-500 shadow-[0_0_5px_rgba(59,130,246,0.8)]"></div>{" "}
              Complete
            </span>
            <span className="text-xl font-black text-zinc-200 leading-none">
              {completeTrials.length}
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] uppercase font-bold text-zinc-500/70 tracking-widest mb-1.5 flex items-center gap-1.5">
              <div className="w-1.5 h-1.5 rounded-full bg-zinc-500"></div>{" "}
              Pruned
            </span>
            <span className="text-xl font-black text-zinc-300 leading-none">
              {prunedTrials.length}
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] uppercase font-bold text-emerald-500/70 tracking-widest mb-1.5 flex items-center gap-1.5">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.8)]"></div>{" "}
              Running
            </span>
            <span className="text-xl font-black text-zinc-200 leading-none">
              {runningTrials.length}
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] uppercase font-bold text-red-500/70 tracking-widest mb-1.5 flex items-center gap-1.5">
              <div className="w-1.5 h-1.5 rounded-full bg-red-500 shadow-[0_0_5px_rgba(239,68,68,0.8)]"></div>{" "}
              Failed
            </span>
            <span className="text-xl font-black text-zinc-200 leading-none">
              {failedTrials.length}
            </span>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-5">
        {bestTrial && (
          <div className="flex flex-col items-end mr-2">
            <span className="text-[10px] uppercase font-black text-amber-500/80 tracking-widest mb-1 shadow-amber-500/20 drop-shadow-sm">
              Best Value
            </span>
            <span className="text-sm font-mono text-amber-400 font-bold bg-amber-500/10 px-2.5 py-1 rounded shadow-[inset_0_0_8px_rgba(245,158,11,0.1)] border border-amber-500/20">
              {Array.isArray(bestTrial.value) &&
              bestTrial.value.length >= 2 &&
              bestTrial.value[0] != null &&
              bestTrial.value[1] != null
                ? `[${Number(bestTrial.value[0]).toFixed(2)}, ${Number(bestTrial.value[1]).toFixed(4)}]`
                : Array.isArray(bestTrial.value) &&
                    bestTrial.value.length > 0 &&
                    bestTrial.value[0] != null
                  ? Number(bestTrial.value[0]).toFixed(4)
                  : bestTrial.value != null && !Array.isArray(bestTrial.value)
                    ? Number(bestTrial.value).toFixed(4)
                    : "-"}
            </span>
          </div>
        )}
        <button
          onClick={handleCopyBestConfig}
          disabled={!bestTrial}
          className={`flex items-center justify-center gap-2 px-5 py-3 rounded-lg text-[10px] font-black uppercase tracking-widest transition-all duration-300 ${copiedConfig ? "bg-emerald-500/20 border-emerald-500/50 text-emerald-400 shadow-[0_0_20px_rgba(16,185,129,0.25)]" : "bg-white/5 border border-white/10 text-zinc-300 hover:bg-white/10 hover:text-white hover:shadow-[0_0_15px_rgba(255,255,255,0.05)]"} disabled:opacity-30 disabled:cursor-not-allowed group`}
        >
          {copiedConfig ? (
            <VscCheck className="w-4 h-4" />
          ) : (
            <VscCopy className="w-4 h-4 group-hover:scale-110 transition-transform" />
          )}
          {copiedConfig ? "Copied!" : "Copy Best Params"}
        </button>
      </div>
    </GlassCard>
  );
}
