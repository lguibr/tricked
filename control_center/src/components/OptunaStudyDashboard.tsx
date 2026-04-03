import { useEffect, useState } from "react";
import ReactECharts from "echarts-for-react";
import { invoke } from "@tauri-apps/api/core";

interface Trial {
  number: number;
  state: string;
  value: number | null;
  params: Record<string, string | number>;
  intermediate_values: Record<string, number>;
}

interface StudyData {
  trials: Trial[];
  importance: Record<string, number>;
}

import { Server, Cpu, Database, Network } from "lucide-react";

export function OptunaStudyDashboard() {
  const [study, setStudy] = useState<StudyData | null>(null);

  useEffect(() => {
    let active = true;
    const fetchStudy = async () => {
      try {
        const jsonStr = await invoke<string>("get_tuning_study");
        const data = JSON.parse(jsonStr);
        if (active) {
          // Backwards compatibility with old array-format or new dict-format
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
    fetchStudy();
    const interval = setInterval(fetchStudy, 2000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  if (!study || !study.trials || study.trials.length === 0) {
    return (
      <div className="p-8 w-full h-full flex flex-col items-center justify-center bg-[#0a0a0a] text-zinc-400">
        <Server className="w-16 h-16 text-zinc-700 mb-6 block" />
        <h2 className="text-xl font-bold text-zinc-200 mb-2 uppercase tracking-widest text-center">Diagnostics Uninitialized</h2>
        <p className="text-sm max-w-md text-center text-zinc-500 mb-8 leading-relaxed">
          Baseline tuning relies on recursive optimization to precisely estimate maximal hardware concurrency limits, memory alignment paths, and deep Monte Carlo search capacities.
        </p>

        <div className="grid grid-cols-3 gap-6 max-w-3xl w-full">
          <div className="bg-[#111] border border-zinc-800/50 rounded-xl p-4 flex flex-col items-center">
            <Cpu className="w-6 h-6 text-emerald-500/80 mb-3" />
            <h3 className="text-xs font-bold text-zinc-300 uppercase tracking-widest mb-1">Compute Estimates</h3>
            <p className="text-[10px] text-zinc-500 text-center">Batch Size: 256-4096<br />Workers: 8-32</p>
          </div>
          <div className="bg-[#111] border border-zinc-800/50 rounded-xl p-4 flex flex-col items-center">
            <Network className="w-6 h-6 text-amber-500/80 mb-3" />
            <h3 className="text-xs font-bold text-zinc-300 uppercase tracking-widest mb-1">Search Estimates</h3>
            <p className="text-[10px] text-zinc-500 text-center">Simulations: 10-200<br />C_puct: 1.0-5.0</p>
          </div>
          <div className="bg-[#111] border border-zinc-800/50 rounded-xl p-4 flex flex-col items-center">
            <Database className="w-6 h-6 text-purple-500/80 mb-3" />
            <h3 className="text-xs font-bold text-zinc-300 uppercase tracking-widest mb-1">Memory Constraints</h3>
            <p className="text-[10px] text-zinc-500 text-center">Buffer: 100k<br />Max Gumbel: 16</p>
          </div>
        </div>

        <p className="text-xs bg-black/40 border border-white/5 p-3 rounded-lg text-center max-w-sm mt-8 text-zinc-600">
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

  // 1. History Option (Scatter Plot)
  const historyOption = {
    title: {
      text: "Optimization History",
      textStyle: { color: "#e4e4e7", fontSize: 13, fontWeight: "bold" },
      top: 10,
      left: 15,
    },
    tooltip: {
      trigger: "item",
      backgroundColor: "#18181b",
      borderColor: "#27272a",
      textStyle: { color: "#e4e4e7" },
    },
    grid: { left: 40, right: 30, top: 50, bottom: 40 },
    xAxis: {
      name: "Trial",
      type: "value",
      minInterval: 1,
      splitLine: { show: true, lineStyle: { color: "#27272a" } },
      axisLabel: { color: "#a1a1aa" },
    },
    yAxis: {
      name: "Objective Value",
      type: "value",
      splitLine: { show: true, lineStyle: { color: "#27272a" } },
      axisLabel: { color: "#a1a1aa" },
    },
    legend: { textStyle: { color: "#a1a1aa" }, top: 10, right: 15 },
    series: [
      {
        name: "Complete",
        type: "scatter",
        symbolSize: 8,
        itemStyle: { color: "#3b82f6", opacity: 0.8 },
        data: completeTrials.map((t) => [t.number, t.value]),
      },
      {
        name: "Pruned",
        type: "scatter",
        symbolSize: 6,
        itemStyle: { color: "#71717a", opacity: 0.5 },
        data: prunedTrials.map((t) => [t.number, t.value]),
      },
    ],
  };

  // 2. Importance Option (Bar Chart)
  const impEntries = Object.entries(importance).sort((a, b) => a[1] - b[1]);
  const importanceOption = {
    title: {
      text: "Hyperparameter Importance",
      textStyle: { color: "#e4e4e7", fontSize: 13, fontWeight: "bold" },
      top: 10,
      left: 15,
    },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      backgroundColor: "#18181b",
      borderColor: "#27272a",
      textStyle: { color: "#e4e4e7" },
    },
    grid: { left: 120, right: 30, top: 40, bottom: 40 },
    xAxis: {
      type: "value",
      splitLine: { show: true, lineStyle: { color: "#27272a" } },
      axisLabel: { color: "#a1a1aa" },
    },
    yAxis: {
      type: "category",
      data: impEntries.map((e) => e[0]),
      axisLabel: { color: "#a1a1aa", width: 100, overflow: "truncate" },
      axisTick: { show: false },
      axisLine: { lineStyle: { color: "#27272a" } },
    },
    series: [
      {
        type: "bar",
        data: impEntries.map((e) => e[1]),
        itemStyle: { color: "#6366f1", borderRadius: [0, 4, 4, 0] },
        barWidth: "60%",
      },
    ],
  };

  // 3. Intermediate Values Option (Line Chart per Trial)
  const intermediateSeries = trials
    .filter(
      (t) =>
        t.intermediate_values && Object.keys(t.intermediate_values).length > 0,
    )
    .map((t) => {
      const steps = Object.keys(t.intermediate_values)
        .map(Number)
        .sort((a, b) => a - b);
      const data = steps.map((s) => [s, t.intermediate_values[String(s)]]);
      return {
        name: `Trial ${t.number}`,
        type: "line",
        data: data,
        showSymbol: false,
        lineStyle: { width: 1.5, opacity: t.state === "COMPLETE" ? 0.7 : 0.3 },
        itemStyle: { color: t.state === "COMPLETE" ? "#f59e0b" : "#71717a" },
      };
    });

  const intermediateOption = {
    title: {
      text: "Intermediate Values",
      textStyle: { color: "#e4e4e7", fontSize: 13, fontWeight: "bold" },
      top: 10,
      left: 15,
    },
    tooltip: {
      trigger: "item",
      backgroundColor: "#18181b",
      borderColor: "#27272a",
      textStyle: { color: "#e4e4e7" },
    },
    grid: { left: 40, right: 30, top: 40, bottom: 40 },
    xAxis: {
      name: "Step",
      type: "value",
      minInterval: 1,
      splitLine: { show: true, lineStyle: { color: "#27272a" } },
      axisLabel: { color: "#a1a1aa" },
    },
    yAxis: {
      name: "Objective Value",
      type: "value",
      splitLine: { show: true, lineStyle: { color: "#27272a" } },
      axisLabel: { color: "#a1a1aa" },
    },
    series:
      intermediateSeries.length > 0
        ? intermediateSeries
        : [{ type: "line", data: [] }], // Fallback if empty
  };

  // 4. Parallel Coordinates Option
  const paramKeys = new Set<string>();
  trials.forEach((trial) => {
    if (trial.params) {
      Object.keys(trial.params).forEach((k) => paramKeys.add(k));
    }
  });

  const dimensions = Array.from(paramKeys).map((key, i) => {
    const isCategorical = trials.some(
      (t) => typeof t.params?.[key] === "string",
    );
    if (isCategorical) {
      const categories = Array.from(
        new Set(trials.map((t) => String(t.params?.[key] || ""))),
      );
      return { dim: i, name: key, type: "category", data: categories };
    }
    return { dim: i, name: key };
  });

  dimensions.push({ dim: dimensions.length, name: "Objective Value" } as any);

  const parallelSeriesData = trials
    .filter((t) => t.value !== null)
    .map((trial) => {
      return dimensions.map((dim) => {
        if (dim.name === "Objective Value") return trial.value;
        const val = trial.params?.[dim.name];
        if (dim.type === "category") return dim.data?.indexOf(String(val));
        return val;
      });
    });

  const parallelOption = {
    title: {
      text: "Parallel Coordinates",
      textStyle: { color: "#e4e4e7", fontSize: 13, fontWeight: "bold" },
      top: 10,
      left: 15,
    },
    tooltip: {
      padding: 10,
      backgroundColor: "#18181b",
      borderColor: "#27272a",
      textStyle: { color: "#e4e4e7" },
    },
    parallelAxis: dimensions.map((d) => ({
      ...d,
      nameTextStyle: {
        fontSize: 10,
        color: "#a1a1aa",
        overflow: "truncate",
        width: 80,
      },
      axisLine: { lineStyle: { color: "#3f3f46" } },
      axisTick: { lineStyle: { color: "#3f3f46" } },
      axisLabel: { color: "#a1a1aa", fontSize: 9 },
    })),
    parallel: {
      left: 40,
      right: 60,
      bottom: 20,
      top: 60,
      parallelAxisDefault: { type: "value" },
    },
    visualMap: {
      show: true,
      min:
        Math.min(
          ...trials
            .filter((t) => t.value !== null)
            .map((t) => t.value as number),
        ) || 0,
      max:
        Math.max(
          ...trials
            .filter((t) => t.value !== null)
            .map((t) => t.value as number),
        ) || 10,
      dimension: dimensions.length - 1,
      inRange: {
        color: ["#3b82f6", "#8b5cf6", "#ec4899", "#f43f5e"].reverse(),
      },
      itemWidth: 10,
      itemHeight: 80,
      right: 0,
      top: "center",
      textStyle: { color: "#a1a1aa", fontSize: 9 },
    },
    series: [
      {
        name: "Trials",
        type: "parallel",
        lineStyle: { width: 2, opacity: 0.6 },
        data: parallelSeriesData,
      },
    ],
  };

  return (
    <div className="w-full h-full bg-[#0a0a0a] flex flex-col p-4 gap-4 overflow-y-auto overflow-x-hidden">
      {/* Top Row: History & Importance */}
      <div className="grid grid-cols-2 gap-4 h-[320px] shrink-0">
        <div className="bg-[#121214] rounded-lg border border-border/10 p-2 shadow-xl">
          <ReactECharts
            option={historyOption}
            style={{ width: "100%", height: "100%" }}
            notMerge={false}
            lazyUpdate={true}
          />
        </div>
        <div className="bg-[#121214] rounded-lg border border-border/10 p-2 shadow-xl">
          <ReactECharts
            option={importanceOption}
            style={{ width: "100%", height: "100%" }}
            notMerge={false}
            lazyUpdate={true}
          />
        </div>
      </div>

      {/* Middle Row: Parallel Coordinates & Intermediate Values */}
      <div
        className={`grid ${intermediateSeries.length > 0 ? "grid-cols-2" : "grid-cols-1"} gap-4 h-[300px] shrink-0`}
      >
        <div className="bg-[#121214] rounded-lg border border-border/10 p-2 shadow-xl">
          <ReactECharts
            option={parallelOption}
            style={{ width: "100%", height: "100%" }}
            notMerge={false}
            lazyUpdate={true}
          />
        </div>
        {intermediateSeries.length > 0 && (
          <div className="bg-[#121214] rounded-lg border border-border/10 p-2 shadow-xl">
            <ReactECharts
              option={intermediateOption}
              style={{ width: "100%", height: "100%" }}
              notMerge={false}
              lazyUpdate={true}
            />
          </div>
        )}
      </div>

      {/* Bottom Row: Trials Table */}
      <div className="bg-[#121214] rounded-lg border border-border/10 shadow-xl overflow-hidden flex flex-col shrink-0 min-h-[300px] mb-4">
        <div className="p-3 border-b border-border/10 bg-[#18181b]">
          <h3 className="text-zinc-200 font-bold text-sm">
            Trials Execution Table
          </h3>
        </div>
        <div className="flex-1 overflow-auto">
          <table className="w-full text-xs text-left whitespace-nowrap">
            <thead className="bg-[#121214] text-zinc-400 sticky top-0 border-b border-border/10">
              <tr>
                <th className="px-4 py-3 font-semibold">Number</th>
                <th className="px-4 py-3 font-semibold">State</th>
                <th className="px-4 py-3 font-semibold">Value</th>
                <th className="px-4 py-3 font-semibold">Parameters</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/5">
              {trials
                .slice()
                .reverse()
                .map((t) => (
                  <tr
                    key={t.number}
                    className="hover:bg-white/5 transition-colors"
                  >
                    <td className="px-4 py-2 text-zinc-300">#{t.number}</td>
                    <td className="px-4 py-2">
                      <span
                        className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold ${t.state === "COMPLETE"
                            ? "bg-blue-500/10 text-blue-400"
                            : t.state === "PRUNED"
                              ? "bg-zinc-500/10 text-zinc-400"
                              : t.state === "RUNNING"
                                ? "bg-emerald-500/10 text-emerald-400 animate-pulse"
                                : "bg-red-500/10 text-red-400"
                          }`}
                      >
                        {t.state}
                      </span>
                    </td>
                    <td className="px-4 py-2 font-mono text-zinc-300">
                      {t.value !== null ? t.value.toFixed(4) : "-"}
                    </td>
                    <td className="px-4 py-2 font-mono text-[10px] text-zinc-500 max-w-md truncate">
                      {Object.entries(t.params || {})
                        .map(([k, v]) => `${k}=${v}`)
                        .join(", ")}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
