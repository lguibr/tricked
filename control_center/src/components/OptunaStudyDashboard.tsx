import { useEffect, useState, useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { invoke } from "@tauri-apps/api/core";
import {
  Server,
  Cpu,
  Database,
  Network,
  Layers,
  Target,
} from "lucide-react";
import { CompactTrialParams } from "@/components/execution/HydraConfigViewer";
import { VscCopy, VscCheck, VscSettingsGear } from "react-icons/vsc";
import { useAppStore } from "@/store/useAppStore";
import * as echarts from "echarts/core";

interface Trial {
  number: number;
  state: string;
  value: number | number[] | null;
  params: Record<string, string | number>;
  intermediate_values: Record<string, number>;
}

interface StudyData {
  trials: Trial[];
  importance: Record<string, number>;
}

export function OptunaStudyDashboard() {
  const selectedRunId = useAppStore((state) => state.selectedRunId);
  const [study, setStudy] = useState<StudyData | null>(null);
  const [copiedConfig, setCopiedConfig] = useState(false);

  useEffect(() => {
    let active = true;
    const fetchStudy = async () => {
      try {
        if (!selectedRunId) return;
        const jsonStr = await invoke<string>("get_tuning_study", {
          studyId: selectedRunId,
        });
        const data = JSON.parse(jsonStr);
        if (active) {
          if (Array.isArray(data)) {
            setStudy({ trials: data, importance: {} });
          } else if (data.trials) {
            setStudy(data);
          }
        }
      } catch (e) {
        console.error("Failed to fetch optuna study:", e);
      }
    };
    if (selectedRunId) {
      fetchStudy();
    } else {
      setStudy(null);
    }
    const interval = setInterval(() => {
      if (selectedRunId) fetchStudy();
    }, 2000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [selectedRunId]);

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
              Batch: 256-4096<br />Workers: 8-32
            </p>
          </div>
          <div className="bg-black/40 backdrop-blur-md border border-white/5 shadow-2xl rounded-xl p-5 flex flex-col items-center hover:border-amber-500/30 transition-colors">
            <Network className="w-8 h-8 text-amber-500/80 mb-4" />
            <h3 className="text-[11px] font-black text-zinc-200 uppercase tracking-widest mb-1.5">
              Search Space
            </h3>
            <p className="text-[10px] text-zinc-500 text-center uppercase tracking-wider">
              Simulations: 10-200<br />C_puct: 1.0-5.0
            </p>
          </div>
          <div className="bg-black/40 backdrop-blur-md border border-white/5 shadow-2xl rounded-xl p-5 flex flex-col items-center hover:border-purple-500/30 transition-colors">
            <Database className="w-8 h-8 text-purple-500/80 mb-4" />
            <h3 className="text-[11px] font-black text-zinc-200 uppercase tracking-widest mb-1.5">
              Memory Constraints
            </h3>
            <p className="text-[10px] text-zinc-500 text-center uppercase tracking-wider">
              Buffer: 100k<br />Max Gumbel: 16
            </p>
          </div>
        </div>

        <p className="text-[10px] font-mono tracking-widest uppercase bg-black border border-white/10 p-3 rounded-lg text-center max-w-sm mt-12 text-zinc-500 relative z-10">
          Deploy a scan to begin empirical measurement.
        </p>
      </div>
    );
  }

  const trials = study.trials;
  const importance = study.importance || {};

  const completeTrials = trials.filter(
    (t) => t.state === "COMPLETE" && t.value !== null,
  );
  const prunedTrials = trials.filter(
    (t) => t.state === "PRUNED" && t.value !== null,
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

  // 1. Pareto Front / Scatter Plot
  const historyOption = useMemo(() => {
    return isMultiObjective
      ? {
        backgroundColor: "transparent",
        title: {
          text: "Pareto Front",
          textStyle: { color: "#f4f4f5", fontSize: 13, fontFamily: "Inter, sans-serif", fontWeight: 800, letterSpacing: 1 },
          subtext: "Objective 1 vs Objective 2",
          subtextStyle: { color: "#a1a1aa", fontSize: 9, fontFamily: "monospace" },
          top: 15,
          left: 20,
        },
        tooltip: {
          trigger: "item",
          backgroundColor: "rgba(9, 9, 11, 0.9)",
          borderColor: "rgba(255,255,255,0.1)",
          padding: 12,
          textStyle: { color: "#e4e4e7", fontSize: 11, fontFamily: "monospace" },
          formatter: (p: any) =>
            `<div style="font-weight:bold;margin-bottom:4px;color:#3b82f6;">Trial #${p.data[2]}</div>` +
            `Hardware: ${p.data[0].toFixed(3)}<br/>Loss: ${p.data[1].toFixed(4)}`,
        },
        grid: { left: 50, right: 30, top: 70, bottom: 40 },
        xAxis: {
          name: "Hardware Metric",
          nameLocation: "middle",
          nameGap: 25,
          nameTextStyle: { color: "#71717a", fontSize: 9, fontFamily: "monospace", fontWeight: "bold", textTransform: "uppercase" },
          type: "value",
          scale: true,
          splitLine: { lineStyle: { color: "rgba(255,255,255,0.05)", type: "dashed" } },
          axisLabel: { color: "#71717a", fontFamily: "monospace", fontSize: 10 },
        },
        yAxis: {
          name: "Evaluation Loss",
          nameLocation: "middle",
          nameGap: 35,
          nameTextStyle: { color: "#71717a", fontSize: 9, fontFamily: "monospace", fontWeight: "bold", textTransform: "uppercase" },
          type: "value",
          scale: true,
          splitLine: { lineStyle: { color: "rgba(255,255,255,0.05)", type: "dashed" } },
          axisLabel: { color: "#71717a", fontFamily: "monospace", fontSize: 10 },
        },
        series: [
          {
            name: "Complete",
            type: "scatter",
            symbolSize: 10,
            itemStyle: {
              color: new echarts.graphic.RadialGradient(0.4, 0.3, 1, [
                { offset: 0, color: "#60a5fa" },
                { offset: 1, color: "#2563eb" },
              ]),
              shadowBlur: 10,
              shadowColor: "rgba(59, 130, 246, 0.5)",
            },
            data: completeTrials.map((t) => [
              Array.isArray(t.value) ? t.value[0] : t.number,
              Array.isArray(t.value) ? t.value[1] ?? t.value[0] : t.value,
              t.number,
            ]),
          },
          {
            name: "Pruned",
            type: "scatter",
            symbolSize: 6,
            itemStyle: { color: "rgba(113, 113, 122, 0.5)" },
            data: prunedTrials
              .filter((t) => t.value != null)
              .map((t) => [
                Array.isArray(t.value) ? t.value[0] : t.number,
                Array.isArray(t.value) ? t.value[1] ?? t.value[0] : t.value,
                t.number,
              ]),
          },
        ],
      }
      : {
        backgroundColor: "transparent",
        title: {
          text: "Optimization History",
          textStyle: { color: "#f4f4f5", fontSize: 13, fontFamily: "Inter, sans-serif", fontWeight: 800, letterSpacing: 1 },
          top: 15,
          left: 20,
        },
        tooltip: {
          trigger: "item",
          backgroundColor: "rgba(9, 9, 11, 0.9)",
          borderColor: "rgba(255,255,255,0.1)",
          padding: 12,
          textStyle: { color: "#e4e4e7", fontSize: 11, fontFamily: "monospace" },
        },
        grid: { left: 50, right: 30, top: 60, bottom: 40 },
        xAxis: {
          name: "Trial",
          nameLocation: "middle",
          nameGap: 25,
          nameTextStyle: { color: "#71717a", fontSize: 9, fontFamily: "monospace", fontWeight: "bold", textTransform: "uppercase" },
          type: "value",
          minInterval: 1,
          splitLine: { lineStyle: { color: "rgba(255,255,255,0.05)", type: "dashed" } },
          axisLabel: { color: "#71717a", fontFamily: "monospace", fontSize: 10 },
        },
        yAxis: {
          name: "Value",
          nameLocation: "middle",
          nameGap: 35,
          nameTextStyle: { color: "#71717a", fontSize: 9, fontFamily: "monospace", fontWeight: "bold", textTransform: "uppercase" },
          type: "value",
          scale: true,
          splitLine: { lineStyle: { color: "rgba(255,255,255,0.05)", type: "dashed" } },
          axisLabel: { color: "#71717a", fontFamily: "monospace", fontSize: 10 },
        },
        series: [
          {
            name: "Complete",
            type: "scatter",
            symbolSize: 10,
            itemStyle: {
              color: new echarts.graphic.RadialGradient(0.4, 0.3, 1, [
                { offset: 0, color: "#34d399" },
                { offset: 1, color: "#059669" },
              ]),
              shadowBlur: 10,
              shadowColor: "rgba(16, 185, 129, 0.5)",
            },
            data: completeTrials.map((t) => [t.number, t.value]),
          },
          {
            name: "Pruned",
            type: "scatter",
            symbolSize: 6,
            itemStyle: { color: "rgba(113, 113, 122, 0.5)" },
            data: prunedTrials
              .filter((t) => t.value != null)
              .map((t) => [t.number, t.value]),
          },
        ],
      };
  }, [completeTrials, prunedTrials, isMultiObjective]);

  // 2. Importance Option (Bar Chart)
  const importanceOption = useMemo(() => {
    const impEntries = Object.entries(importance)
      // filter out very low importances to keep it clean (e.g. 0.0)
      .filter((e) => e[1] > 0.01)
      .sort((a, b) => a[1] - b[1]);

    return {
      backgroundColor: "transparent",
      title: {
        text: "Parameter Importance",
        textStyle: { color: "#f4f4f5", fontSize: 13, fontFamily: "Inter, sans-serif", fontWeight: 800, letterSpacing: 1 },
        top: 15,
        left: 20,
      },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        backgroundColor: "rgba(9, 9, 11, 0.9)",
        borderColor: "rgba(255,255,255,0.1)",
        padding: 12,
        textStyle: { color: "#e4e4e7", fontSize: 11, fontFamily: "monospace" },
        formatter: (params: any) => {
          const val = params[0].value;
          return `<div style="font-weight:bold;margin-bottom:4px;color:#a855f7;">${params[0].name}</div>Importance: ${(val * 100).toFixed(2)}%`;
        }
      },
      grid: { left: 140, right: 30, top: 50, bottom: 40 },
      xAxis: {
        type: "value",
        max: 1.0,
        splitLine: { lineStyle: { color: "rgba(255,255,255,0.05)", type: "dashed" } },
        axisLabel: { color: "#71717a", fontFamily: "monospace", fontSize: 10 },
      },
      yAxis: {
        type: "category",
        data: impEntries.map((e) => e[0]),
        axisLabel: { color: "#a1a1aa", fontSize: 10, fontFamily: "monospace", width: 120, overflow: "truncate" },
        axisTick: { show: false },
        axisLine: { show: false },
      },
      series: [
        {
          type: "bar",
          data: impEntries.map((e) => e[1]),
          itemStyle: {
            color: new echarts.graphic.LinearGradient(1, 0, 0, 0, [
              { offset: 0, color: "#a855f7" },
              { offset: 1, color: "#6366f1" },
            ]),
            borderRadius: [0, 4, 4, 0],
          },
          barWidth: "40%",
          showBackground: true,
          backgroundStyle: {
            color: "rgba(255, 255, 255, 0.02)",
            borderRadius: [0, 4, 4, 0],
          }
        },
      ],
    };
  }, [importance]);

  // 3. Parallel Coordinates Option
  const parallelOption = useMemo(() => {
    const paramKeys = new Set<string>();
    trials.forEach((trial) => {
      if (trial.params) Object.keys(trial.params).forEach((k) => paramKeys.add(k));
    });

    const dimensions = Array.from(paramKeys).map((key, i) => {
      const isCategorical = trials.some((t) => typeof t.params?.[key] === "string");
      if (isCategorical) {
        const categories = Array.from(new Set(trials.map((t) => String(t.params?.[key] || ""))));
        return { dim: i, name: key, type: "category", data: categories };
      }
      return { dim: i, name: key };
    });

    dimensions.push({ dim: dimensions.length, name: "Loss" } as any);

    const parallelSeriesData = trials
      .filter((t) => t.value !== null)
      .map((trial) => {
        return dimensions.map((dim) => {
          if (dim.name === "Loss") {
            return Array.isArray(trial.value) ? trial.value[1] ?? trial.value[0] : trial.value;
          }
          const val = trial.params?.[dim.name];
          if (dim.type === "category") return dim.data?.indexOf(String(val));
          return val;
        });
      });

    return {
      backgroundColor: "transparent",
      title: {
        text: "Parameters Distribution",
        textStyle: { color: "#f4f4f5", fontSize: 13, fontFamily: "Inter, sans-serif", fontWeight: 800, letterSpacing: 1 },
        top: 15,
        left: 20,
      },
      tooltip: {
        backgroundColor: "rgba(9, 9, 11, 0.9)",
        borderColor: "rgba(255,255,255,0.1)",
        padding: 12,
        textStyle: { color: "#e4e4e7", fontSize: 11, fontFamily: "monospace" },
      },
      parallelAxis: dimensions.map((d) => ({
        ...d,
        nameTextStyle: {
          fontSize: 9,
          color: "#a1a1aa",
          fontFamily: "monospace",
          overflow: "truncate",
          width: 80,
        },
        axisLine: { lineStyle: { color: "rgba(255,255,255,0.1)" } },
        axisTick: { lineStyle: { color: "rgba(255,255,255,0.1)" } },
        axisLabel: { color: "#71717a", fontSize: 9, fontFamily: "monospace" },
      })),
      parallel: {
        left: 50,
        right: 80,
        bottom: 30,
        top: 70,
        parallelAxisDefault: { type: "value" },
      },
      visualMap: {
        show: true,
        min: Math.min(...parallelSeriesData.map(d => d[dimensions.length - 1] as number)) || 0,
        max: Math.max(...parallelSeriesData.map(d => d[dimensions.length - 1] as number)) || 10,
        dimension: dimensions.length - 1,
        inRange: {
          color: ["#10b981", "#3b82f6", "#8b5cf6", "#ec4899", "#ef4444"], // Emerald to Red (Low Loss to High Loss)
        },
        itemWidth: 8,
        itemHeight: 120,
        right: 15,
        top: "center",
        textStyle: { color: "#71717a", fontSize: 9, fontFamily: "monospace" },
      },
      series: [
        {
          name: "Trials",
          type: "parallel",
          lineStyle: { width: 1.5, opacity: 0.4 },
          inactiveOpacity: 0.05,
          activeOpacity: 1,
          data: parallelSeriesData,
        },
      ],
    };
  }, [trials]);


  // Add a nice glass card wrapper
  const GlassCard = ({ children, className = "" }: { children: React.ReactNode, className?: string }) => (
    <div className={`bg-[#0c0c0e]/80 backdrop-blur-xl border border-white/[0.04] rounded-xl shadow-2xl relative overflow-hidden group ${className}`}>
      <div className="absolute inset-0 bg-gradient-to-br from-white/[0.02] to-transparent pointer-events-none" />
      {children}
    </div>
  );

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
                <div className="w-1.5 h-1.5 rounded-full bg-blue-500 shadow-[0_0_5px_rgba(59,130,246,0.8)]"></div> Complete
              </span>
              <span className="text-xl font-black text-zinc-200 leading-none">
                {completeTrials.length}
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] uppercase font-bold text-zinc-500/70 tracking-widest mb-1.5 flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 rounded-full bg-zinc-500"></div> Pruned
              </span>
              <span className="text-xl font-black text-zinc-300 leading-none">
                {prunedTrials.length}
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] uppercase font-bold text-emerald-500/70 tracking-widest mb-1.5 flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.8)]"></div> Running
              </span>
              <span className="text-xl font-black text-zinc-200 leading-none">
                {runningTrials.length}
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] uppercase font-bold text-red-500/70 tracking-widest mb-1.5 flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 rounded-full bg-red-500 shadow-[0_0_5px_rgba(239,68,68,0.8)]"></div> Failed
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
                {Array.isArray(bestTrial.value)
                  ? `[${bestTrial.value[0].toFixed(2)}, ${bestTrial.value[1].toFixed(4)}]`
                  : bestTrial.value?.toFixed(4)}
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
      <div className="grid grid-cols-2 gap-4 h-[360px] shrink-0">
        <GlassCard className="w-full h-full flex flex-col p-1">
          <ReactECharts
            option={historyOption}
            style={{ width: "100%", height: "100%" }}
            className="flex-1 w-full min-h-0"
            notMerge={false}
            lazyUpdate={true}
          />
        </GlassCard>

        <GlassCard className="w-full h-full flex flex-col p-1">
          <ReactECharts
            option={importanceOption}
            style={{ width: "100%", height: "100%" }}
            className="flex-1 w-full min-h-0"
            notMerge={false}
            lazyUpdate={true}
          />
        </GlassCard>
      </div>

      <div className="h-[360px] shrink-0">
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
                <th className="px-6 py-4 font-black uppercase tracking-widest text-[9px] border-b border-white/[0.02]">Trial #</th>
                <th className="px-6 py-4 font-black uppercase tracking-widest text-[9px] border-b border-white/[0.02]">Status</th>
                <th className="px-6 py-4 font-black uppercase tracking-widest text-[9px] border-b border-white/[0.02]">Values <span className="text-zinc-600 font-normal lowercase">(HW / Loss)</span></th>
                <th className="px-6 py-4 font-black uppercase tracking-widest text-[9px] border-b border-white/[0.02] w-full">Config Trajectory Space</th>
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
                      {t.number.toString().padStart(4, '0')}
                    </td>
                    <td className="px-6 py-4">
                      <span
                        className={`inline-flex items-center px-2.5 py-1 rounded text-[9px] font-black uppercase tracking-widest border shadow-sm ${t.state === "COMPLETE"
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
                      {Array.isArray(t.value)
                        ? (
                          <div className="flex gap-2.5">
                            <span className="text-blue-400/90 font-bold">{t.value[0].toFixed(2)}</span>
                            <span className="text-zinc-700">/</span>
                            <span className="text-purple-400/90 font-bold">{t.value[1].toFixed(4)}</span>
                          </div>
                        )
                        : t.value !== null
                          ? <span className="text-zinc-300 font-bold">{(t.value as number).toFixed(4)}</span>
                          : <span className="text-zinc-700 font-bold">-</span>}
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
