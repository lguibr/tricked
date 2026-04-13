import { useState, useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { VscSettingsGear } from "react-icons/vsc";
import { useAppStore } from "@/store/useAppStore";
import { useOptimizerStudy } from "@/hooks/useOptimizerStudy";
import { GlassCard } from "@/components/dashboard/GlassCard";
import {
  getHistoryOption,
  getImportanceOption,
  getParallelOption,
} from "./chart-options";
import { StudyDiagnosticsUninitialized } from "./StudyDiagnosticsUninitialized";
import { OptimizerStudyHeaderStats } from "./OptimizerStudyHeaderStats";
import { OptimizerStudyTrialsTable } from "./OptimizerStudyTrialsTable";

export function OptimizerStudyDashboard() {
  const selectedRunId = useAppStore((state) => state.selectedRunId);
  const study = useOptimizerStudy(selectedRunId);
  const runs = useAppStore((state) => state.runs);
  const selectedRun = runs.find((r) => r.id === selectedRunId);

  const parsedConfig = useMemo(() => {
    if (!selectedRun?.config) return {};
    try {
      const parentCfg = JSON.parse(selectedRun.config);
      return typeof parentCfg.bounds === "string"
        ? JSON.parse(parentCfg.bounds)
        : parentCfg.bounds || {};
    } catch {
      return {};
    }
  }, [selectedRun]);

  const tuningConfig = parsedConfig;
  const [copiedConfig, setCopiedConfig] = useState(false);

  const trials = study?.trials || [];
  const importance = study?.importance || {};

  const completeTrials = trials.filter((t) => t.state === "COMPLETE");
  const prunedTrials = trials.filter((t) => t.state === "PRUNED");
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
    return <StudyDiagnosticsUninitialized />;
  }

  return (
    <div className="w-full h-full bg-[#050505] flex flex-col p-4 gap-4 overflow-y-auto custom-scrollbar">
      <OptimizerStudyHeaderStats
        trials={trials}
        completeTrials={completeTrials}
        prunedTrials={prunedTrials}
        runningTrials={runningTrials}
        failedTrials={failedTrials}
        bestTrial={bestTrial}
        copiedConfig={copiedConfig}
        handleCopyBestConfig={handleCopyBestConfig}
      />

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

        <GlassCard className="w-full h-full flex flex-col p-4 overflow-y-auto custom-scrollbar bg-[#0a0a0c]">
          <h3 className="text-zinc-100 font-bold text-[11px] uppercase tracking-widest text-shadow-sm mb-4 border-b border-white/5 pb-2">
            Parameter Search Space Goals
          </h3>
          <div className="flex flex-col gap-2.5">
            {Object.entries(tuningConfig).map(([key, rawValue]) => {
              const value = rawValue as { min?: number; max?: number };
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

      <OptimizerStudyTrialsTable trials={trials} />
    </div>
  );
}
