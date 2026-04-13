import * as echarts from "echarts/core";

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
