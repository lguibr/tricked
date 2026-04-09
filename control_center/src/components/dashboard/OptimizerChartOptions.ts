import * as echarts from "echarts/core";
import { Trial } from "@/hooks/useOptimizerStudy";

export const getHistoryOption = (
  completeTrials: Trial[],
  prunedTrials: Trial[],
  isMultiObjective: boolean,
) => {
  return isMultiObjective
    ? {
        backgroundColor: "transparent",
        title: {
          text: "Pareto Front",
          textStyle: {
            color: "#f4f4f5",
            fontSize: 13,
            fontFamily: "Inter, sans-serif",
            fontWeight: 800,
            letterSpacing: 1,
          },
          subtext: "Objective 1 vs Objective 2",
          subtextStyle: {
            color: "#a1a1aa",
            fontSize: 9,
            fontFamily: "monospace",
          },
          top: 15,
          left: 20,
        },
        tooltip: {
          trigger: "item",
          backgroundColor: "rgba(9, 9, 11, 0.9)",
          borderColor: "rgba(255,255,255,0.1)",
          padding: 12,
          textStyle: {
            color: "#e4e4e7",
            fontSize: 11,
            fontFamily: "monospace",
          },
          formatter: (p: any) => {
            const hw = p.data[0] != null ? Number(p.data[0]).toFixed(3) : "N/A";
            const loss =
              p.data[1] != null ? Number(p.data[1]).toFixed(4) : "N/A";
            return `<div style="font-weight:bold;margin-bottom:4px;color:#3b82f6;">Trial #${p.data[2]}</div>Hardware: ${hw}<br/>Loss: ${loss}`;
          },
        },
        grid: { left: 50, right: 30, top: 70, bottom: 40 },
        xAxis: {
          name: "Hardware Metric",
          nameLocation: "middle",
          nameGap: 25,
          nameTextStyle: {
            color: "#71717a",
            fontSize: 9,
            fontFamily: "monospace",
            fontWeight: "bold",
            textTransform: "uppercase",
          },
          type: "value",
          scale: true,
          splitLine: {
            lineStyle: { color: "rgba(255,255,255,0.05)", type: "dashed" },
          },
          axisLabel: {
            color: "#71717a",
            fontFamily: "monospace",
            fontSize: 10,
          },
        },
        yAxis: {
          name: "Evaluation Loss",
          nameLocation: "middle",
          nameGap: 35,
          nameTextStyle: {
            color: "#71717a",
            fontSize: 9,
            fontFamily: "monospace",
            fontWeight: "bold",
            textTransform: "uppercase",
          },
          type: "value",
          scale: true,
          splitLine: {
            lineStyle: { color: "rgba(255,255,255,0.05)", type: "dashed" },
          },
          axisLabel: {
            color: "#71717a",
            fontFamily: "monospace",
            fontSize: 10,
          },
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
              Array.isArray(t.value) ? (t.value[1] ?? t.value[0]) : t.value,
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
                Array.isArray(t.value) ? (t.value[1] ?? t.value[0]) : t.value,
                t.number,
              ]),
          },
        ],
      }
    : {
        backgroundColor: "transparent",
        title: {
          text: "Optimization History",
          textStyle: {
            color: "#f4f4f5",
            fontSize: 13,
            fontFamily: "Inter, sans-serif",
            fontWeight: 800,
            letterSpacing: 1,
          },
          top: 15,
          left: 20,
        },
        tooltip: {
          trigger: "item",
          backgroundColor: "rgba(9, 9, 11, 0.9)",
          borderColor: "rgba(255,255,255,0.1)",
          padding: 12,
          textStyle: {
            color: "#e4e4e7",
            fontSize: 11,
            fontFamily: "monospace",
          },
        },
        grid: { left: 50, right: 30, top: 60, bottom: 40 },
        xAxis: {
          name: "Trial",
          nameLocation: "middle",
          nameGap: 25,
          nameTextStyle: {
            color: "#71717a",
            fontSize: 9,
            fontFamily: "monospace",
            fontWeight: "bold",
            textTransform: "uppercase",
          },
          type: "value",
          minInterval: 1,
          splitLine: {
            lineStyle: { color: "rgba(255,255,255,0.05)", type: "dashed" },
          },
          axisLabel: {
            color: "#71717a",
            fontFamily: "monospace",
            fontSize: 10,
          },
        },
        yAxis: {
          name: "Value",
          nameLocation: "middle",
          nameGap: 35,
          nameTextStyle: {
            color: "#71717a",
            fontSize: 9,
            fontFamily: "monospace",
            fontWeight: "bold",
            textTransform: "uppercase",
          },
          type: "value",
          scale: true,
          splitLine: {
            lineStyle: { color: "rgba(255,255,255,0.05)", type: "dashed" },
          },
          axisLabel: {
            color: "#71717a",
            fontFamily: "monospace",
            fontSize: 10,
          },
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
            data: completeTrials
              .map((t) => {
                let v = t.value;
                if (Array.isArray(v)) {
                  v = v.length > 0 ? v[v.length - 1] : null;
                }
                return [t.number, v];
              })
              .filter((d) => d[1] != null && !isNaN(d[1] as number)),
          },
          {
            name: "Pruned",
            type: "scatter",
            symbolSize: 6,
            itemStyle: { color: "rgba(113, 113, 122, 0.5)" },
            data: prunedTrials
              .map((t) => {
                let v = t.value;
                if (Array.isArray(v)) {
                  v = v.length > 0 ? v[v.length - 1] : null;
                }
                return [t.number, v];
              })
              .filter((d) => d[1] != null && !isNaN(d[1] as number)),
          },
        ],
      };
};

export const getImportanceOption = (importance: Record<string, number>) => {
  const impEntries = Object.entries(importance)
    .filter((e) => e[1] > 0.01)
    .sort((a, b) => a[1] - b[1]);
  return {
    backgroundColor: "transparent",
    title: {
      text: "Parameter Importance",
      textStyle: {
        color: "#f4f4f5",
        fontSize: 13,
        fontFamily: "Inter, sans-serif",
        fontWeight: 800,
        letterSpacing: 1,
      },
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
      },
    },
    grid: { left: 140, right: 30, top: 50, bottom: 40 },
    xAxis: {
      type: "value",
      max: 1.0,
      splitLine: {
        lineStyle: { color: "rgba(255,255,255,0.05)", type: "dashed" },
      },
      axisLabel: { color: "#71717a", fontFamily: "monospace", fontSize: 10 },
    },
    yAxis: {
      type: "category",
      data: impEntries.map((e) => e[0]),
      axisLabel: {
        color: "#a1a1aa",
        fontSize: 10,
        fontFamily: "monospace",
        width: 120,
        overflow: "truncate",
      },
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
        },
      },
    ],
  };
};

export const getParallelOption = (trials: Trial[]) => {
  const paramKeys = new Set<string>();
  trials.forEach((trial) => {
    if (trial.params)
      Object.keys(trial.params).forEach((k) => paramKeys.add(k));
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
  dimensions.push({ dim: dimensions.length, name: "Loss" } as any);
  const parallelSeriesData = trials
    .filter((t) => {
      if (t.value == null) return false;
      if (Array.isArray(t.value) && t.value.length === 0) return false;
      if (
        Array.isArray(t.value) &&
        (t.value[0] == null || isNaN(Number(t.value[0])))
      )
        return false;
      if (typeof t.value === "number" && isNaN(t.value)) return false;
      return true;
    })
    .map((trial) => {
      return dimensions.map((dim) => {
        if (dim.name === "Loss") {
          const l = Array.isArray(trial.value)
            ? (trial.value[1] ?? trial.value[0])
            : trial.value;
          return isNaN(Number(l)) ? null : l;
        }
        const val = trial.params?.[dim.name];
        if (dim.type === "category") {
          const catIdx = dim.data?.indexOf(String(val));
          return catIdx !== -1 && catIdx !== undefined ? catIdx : null;
        }
        return val ?? null;
      });
    });
  return {
    backgroundColor: "transparent",
    title: {
      text: "Parameters Distribution",
      textStyle: {
        color: "#f4f4f5",
        fontSize: 13,
        fontFamily: "Inter, sans-serif",
        fontWeight: 800,
        letterSpacing: 1,
      },
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
      min: (() => {
        const validLosses = parallelSeriesData
          .map((d) => d[dimensions.length - 1] as number)
          .filter((v) => typeof v === "number" && !isNaN(v) && v < 1e100);
        return validLosses.length > 0 ? Math.min(...validLosses) : 0;
      })(),
      max: (() => {
        const validLosses = parallelSeriesData
          .map((d) => d[dimensions.length - 1] as number)
          .filter((v) => typeof v === "number" && !isNaN(v) && v < 1e100);
        return validLosses.length > 0 ? Math.max(...validLosses) : 10;
      })(),
      dimension: dimensions.length - 1,
      inRange: {
        color: ["#10b981", "#3b82f6", "#8b5cf6", "#ec4899", "#ef4444"],
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
};
