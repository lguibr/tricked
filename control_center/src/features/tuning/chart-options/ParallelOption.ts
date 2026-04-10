import { Trial } from "@/hooks/useOptimizerStudy";

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
          return isNaN(Number(l)) || Number(l) > 1e100 ? null : l;
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
