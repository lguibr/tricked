import { useState, useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { Server, Cpu, Database, Network, Layers, Target } from "lucide-react";
import { CompactTrialParams } from "@/components/execution/HydraConfigViewer";
import { VscCopy, VscCheck, VscSettingsGear } from "react-icons/vsc";
import { useAppStore } from "@/store/useAppStore";
import { useOptimizerStudy } from "@/hooks/useOptimizerStudy";
import { useTuningStore } from "@/store/useTuningStore";
import { GlassCard } from "@/components/dashboard/GlassCard";
import {
  getHistoryOption,
  getImportanceOption,
  getParallelOption,
} from "@/components/dashboard/OptimizerChartOptions";

export function OptimizerStudyDashboard() {
  const selectedRunId = useAppStore((state) => state.selectedRunId);
  const study = useOptimizerStudy(selectedRunId);
  const tuningConfig = useTuningStore((state) => state.config);
  const [copiedConfig, setCopiedConfig] = useState(false);

  const trials = study?.trials || [];
  const importance = study?.importance || {};

  const hasValidValue = (v: any) => {
    if (v == null) return false;
    if (typeof v === "number" && isNaN(v)) return false;
    if (Array.isArray(v)) {
      if (v.length === 0) return false;
      if (v[0] == null || isNaN(Number(v[0]))) return false;
      if (v[0] > 1e100) return false;
      if (v[1] != null && (isNaN(Number(v[1])) || v[1] > 1e100)) return false;
    } else {
      if (typeof v === "number" && v > 1e100) return false;
    }
    return true;
  };
  const completeTrials = trials.filter(
    (t) => t.state === "COMPLETE" && hasValidValue(t.value),
  );
  const prunedTrials = trials.filter(
    (t) => t.state === "PRUNED" && hasValidValue(t.value),
  );
  const runningTrials = trials.filter((t) => t.state === "RUNNING");
  const failedTrials = trials.filter(
    (t) => t.state === "FAIL" || t.state === "FAILED",
  );

  const bestTrial =
    completeTrials.length > 0
      ? completeTrials.reduce((best, t) => {
          const bestVal = Array.isArray(best.value) ? best.value : [best.value];
          const tVal = Array.isArray(t.value) ? t.value : [t.value];
          const currentBest = bestVal[bestVal.length > 1 ? 1 : 0] ?? Infinity;
          const candidate = tVal[tVal.length > 1 ? 1 : 0] ?? Infinity;
          return candidate < currentBest ? t : best;
        }, completeTrials[0])
      : null;

  const handleCopyBestConfig = () => {
    if (bestTrial && bestTrial.params) {
      navigator.clipboard.writeText(JSON.stringify(bestTrial.params, null, 2));
      setCopiedConfig(true);
      setTimeout(() => setCopiedConfig(false), 2000);
    }
  };

  const isMultiObjective = trials.some((t) => Array.isArray(t.value));

  // --- ECharts Options ---
  const historyOption = useMemo(
    () => getHistoryOption(completeTrials, prunedTrials, isMultiObjective),
    [completeTrials, prunedTrials, isMultiObjective],
  );
  const importanceOption = useMemo(
    () => getImportanceOption(importance),
    [importance],
  );
  const parallelOption = useMemo(() => getParallelOption(trials), [trials]);

  if (!selectedRunId) {
    return (
      <div className="flex flex-col items-center justify-center w-full h-full text-zinc-600 gap-4 bg-[#050505]">
        <div className="animate-pulse bg-zinc-800/30 p-6 rounded-full border border-white/5">
          <VscSettingsGear size={48} className="text-zinc-700" />
        </div>
        <div className="text-center">
          <p className="font-bold text-zinc-400 uppercase tracking-widest text-sm mb-1">
            No Study Selected
          </p>
          <p className="text-[10px] max-w-sm text-zinc-600">
            Create a new tuning study from the sidebar or select an existing one
            to view its optimization metrics and trials.
          </p>
        </div>
      </div>
    );
  }

  if (!study || !study.trials || study.trials.length === 0) {
    return (
      <div className="p-8 w-full h-full flex flex-col items-center justify-center bg-[#0a0a0a] text-zinc-400 relative overflow-hidden">
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[600px] h-[400px] bg-emerald-500/5 blur-[120px] rounded-full pointer-events-none" />

        <Server className="w-16 h-16 text-zinc-700 mb-6 block relative z-10" />
        <h2 className="text-xl font-black text-zinc-200 mb-2 uppercase tracking-widest text-center relative z-10">
          Diagnostics Uninitialized
        </h2>
        <p className="text-sm max-w-md text-center text-zinc-500 mb-8 leading-relaxed relative z-10">
          Baseline tuning relies on recursive optimization to precisely estimate
          maximal hardware concurrency limits, memory alignment paths, and deep
          Monte Carlo search capacities.
        </p>

        <div className="grid grid-cols-3 gap-6 max-w-3xl w-full relative z-10">
          <div className="bg-black/40 backdrop-blur-md border border-white/5 shadow-2xl rounded-xl p-5 flex flex-col items-center hover:border-emerald-500/30 transition-colors">
            <Cpu className="w-8 h-8 text-emerald-500/80 mb-4" />
            <h3 className="text-[11px] font-black text-zinc-200 uppercase tracking-widest mb-1.5">
              Compute Estimates
            </h3>
            <p className="text-[10px] text-zinc-500 text-center uppercase tracking-wider">
              Batch: 256-4096
              <br />
              Workers: 8-32
            </p>
          </div>
          <div className="bg-black/40 backdrop-blur-md border border-white/5 shadow-2xl rounded-xl p-5 flex flex-col items-center hover:border-amber-500/30 transition-colors">
            <Network className="w-8 h-8 text-amber-500/80 mb-4" />
            <h3 className="text-[11px] font-black text-zinc-200 uppercase tracking-widest mb-1.5">
              Search Space
            </h3>
            <p className="text-[10px] text-zinc-500 text-center uppercase tracking-wider">
              Simulations: 10-200
              <br />
              C_puct: 1.0-5.0
            </p>
          </div>
          <div className="bg-black/40 backdrop-blur-md border border-white/5 shadow-2xl rounded-xl p-5 flex flex-col items-center hover:border-purple-500/30 transition-colors">
            <Database className="w-8 h-8 text-purple-500/80 mb-4" />
            <h3 className="text-[11px] font-black text-zinc-200 uppercase tracking-widest mb-1.5">
              Memory Constraints
            </h3>
            <p className="text-[10px] text-zinc-500 text-center uppercase tracking-wider">
              Buffer: 100k
              <br />
              Max Gumbel: 16
            </p>
          </div>
        </div>

        <p className="text-[10px] font-mono tracking-widest uppercase bg-black border border-white/10 p-3 rounded-lg text-center max-w-sm mt-12 text-zinc-500 relative z-10">
          Deploy a scan to begin empirical measurement.
        </p>
      </div>
    );
  }

  return (
    <div className="w-full h-full bg-[#050505] flex flex-col p-4 gap-4 overflow-y-auto custom-scrollbar">
      {/* Header Stats */}
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

      {/* Visualizations Grid */}
      <div className="grid grid-cols-3 gap-4 h-[360px] shrink-0">
        <GlassCard className="col-span-2 w-full h-full flex flex-col p-1">
          <ReactECharts
            option={historyOption}
            style={{ width: "100%", height: "100%" }}
            className="flex-1 w-full min-h-0"
            notMerge={false}
            lazyUpdate={true}
          />
        </GlassCard>

        {/* Parameter Weight Goals Table */}
        <GlassCard className="w-full h-full flex flex-col p-4 overflow-y-auto custom-scrollbar bg-[#0a0a0c]">
          <h3 className="text-zinc-100 font-bold text-[11px] uppercase tracking-widest text-shadow-sm mb-4 border-b border-white/5 pb-2">
            Parameter Search Space Goals
          </h3>
          <div className="flex flex-col gap-2.5">
            {Object.entries(tuningConfig).map(([key, value]) => {
              if (
                typeof value === "object" &&
                value !== null &&
                "min" in value
              ) {
                return (
                  <div
                    key={key}
                    className="flex flex-col gap-1 bg-white/[0.02] p-2 rounded-md border border-white/5"
                  >
                    <span className="text-[10px] uppercase font-bold text-zinc-400 tracking-wider truncate">
                      {key}
                    </span>
                    <div className="flex items-center justify-between text-xs font-mono">
                      <span className="text-blue-400">{value.min}</span>
                      <span className="text-zinc-600 text-[10px]">TO</span>
                      <span className="text-purple-400">{value.max}</span>
                    </div>
                  </div>
                );
              }
              return null;
            })}
          </div>
        </GlassCard>
      </div>

      <div className="grid grid-cols-2 gap-4 h-[360px] shrink-0 mt-4">
        <GlassCard className="w-full h-full flex flex-col p-1">
          <ReactECharts
            option={importanceOption}
            style={{ width: "100%", height: "100%" }}
            className="flex-1 w-full min-h-0"
            notMerge={false}
            lazyUpdate={true}
          />
        </GlassCard>

        <GlassCard className="w-full h-full flex flex-col p-1">
          <ReactECharts
            option={parallelOption}
            style={{ width: "100%", height: "100%" }}
            className="flex-1 w-full min-h-0"
            notMerge={false}
            lazyUpdate={true}
          />
        </GlassCard>
      </div>

      {/* Modern Data Grid */}
      <GlassCard className="flex flex-col shrink-0 min-h-[450px] mb-4">
        <div className="p-4 border-b border-white/5 bg-[#0a0a0c] flex items-center justify-between z-30 shadow-sm relative">
          <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>
          <div className="flex items-center gap-2.5">
            <div className="p-1.5 bg-purple-500/10 rounded-md border border-purple-500/20">
              <Layers className="w-4 h-4 text-purple-400" />
            </div>
            <h3 className="text-zinc-100 font-bold text-xs uppercase tracking-widest text-shadow-sm">
              Study Execution Protocol
            </h3>
          </div>
          <p className="text-[10px] text-zinc-500 font-mono tracking-wider">
            {trials.length} entries matching current criteria
          </p>
        </div>
        <div className="flex-1 overflow-auto custom-scrollbar bg-[#0f0f12]/50">
          <table className="w-full text-xs text-left whitespace-nowrap border-collapse">
            <thead className="text-zinc-500 sticky top-0 z-20 bg-[#0f0f12]/90 backdrop-blur-md shadow-sm border-b border-white/5">
              <tr>
                <th className="px-6 py-4 font-black uppercase tracking-widest text-[9px] border-b border-white/[0.02]">
                  Trial #
                </th>
                <th className="px-6 py-4 font-black uppercase tracking-widest text-[9px] border-b border-white/[0.02]">
                  Status
                </th>
                <th className="px-6 py-4 font-black uppercase tracking-widest text-[9px] border-b border-white/[0.02]">
                  Values{" "}
                  <span className="text-zinc-600 font-normal lowercase">
                    (HW / Loss)
                  </span>
                </th>
                <th className="px-6 py-4 font-black uppercase tracking-widest text-[9px] border-b border-white/[0.02] w-full">
                  Config Trajectory Space
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.02]">
              {trials
                .slice()
                .reverse()
                .map((t) => (
                  <tr
                    key={t.number}
                    className="hover:bg-white/[0.03] transition-colors group cursor-default"
                  >
                    <td className="px-6 py-4 font-mono text-zinc-500 group-hover:text-zinc-300 transition-colors">
                      {t.number.toString().padStart(4, "0")}
                    </td>
                    <td className="px-6 py-4">
                      <span
                        className={`inline-flex items-center px-2.5 py-1 rounded text-[9px] font-black uppercase tracking-widest border shadow-sm ${
                          t.state === "COMPLETE"
                            ? "bg-blue-500/10 text-blue-400 border-blue-500/20"
                            : t.state === "PRUNED"
                              ? "bg-zinc-500/10 text-zinc-400 border-zinc-500/20"
                              : t.state === "RUNNING"
                                ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30 animate-pulse shadow-[0_0_12px_rgba(16,185,129,0.25)]"
                                : "bg-red-500/10 text-red-400 border-red-500/20"
                        }`}
                      >
                        {t.state}
                      </span>
                    </td>
                    <td className="px-6 py-4 font-mono text-zinc-400">
                      {Array.isArray(t.value) &&
                      t.value.length >= 2 &&
                      t.value[0] != null &&
                      t.value[1] != null ? (
                        <div className="flex gap-2.5">
                          <span className="text-blue-400/90 font-bold">
                            {Number(t.value[0]).toFixed(2)}
                          </span>
                          <span className="text-zinc-700">/</span>
                          <span className="text-purple-400/90 font-bold">
                            {Number(t.value[1]).toFixed(4)}
                          </span>
                        </div>
                      ) : Array.isArray(t.value) &&
                        t.value.length > 0 &&
                        t.value[0] != null ? (
                        <span className="text-zinc-300 font-bold">
                          {Number(t.value[0]).toFixed(4)}
                        </span>
                      ) : t.value != null && !Array.isArray(t.value) ? (
                        <span className="text-zinc-300 font-bold">
                          {Number(t.value).toFixed(4)}
                        </span>
                      ) : (
                        <span className="text-zinc-700 font-bold">-</span>
                      )}
                    </td>
                    <td className="px-6 py-4">
                      <div className="opacity-70 group-hover:opacity-100 transition-opacity">
                        <CompactTrialParams params={t.params || {}} />
                      </div>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </GlassCard>
    </div>
  );
}
