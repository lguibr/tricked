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
            progressive: 0,
            symbolSize: 10,
            itemStyle: {
              color: new echarts.graphic.RadialGradient(0.4, 0.3, 1, [
                { offset: 0, color: "#60a5fa" },
                { offset: 1, color: "#2563eb" },
              ]),
              shadowBlur: 10,
              shadowColor: "rgba(59, 130, 246, 0.5)",
            },
            data: completeTrials
              .map((t) => [
                Array.isArray(t.value)
                  ? Number(t.value[0] ?? 0)
                  : Number(t.number ?? 0),
                Array.isArray(t.value)
                  ? Number(t.value[1] ?? t.value[0] ?? 0)
                  : Number(t.value ?? 0),
                Number(t.number ?? 0),
              ])
              .filter(
                (d) =>
                  !isNaN(d[0]) && !isNaN(d[1]) && d[0] < 1e100 && d[1] < 1e100,
              ),
          },
          {
            name: "Pruned",
            type: "scatter",
            progressive: 0,
            symbolSize: 6,
            itemStyle: { color: "rgba(113, 113, 122, 0.5)" },
            data: prunedTrials
              .filter((t) => t.value != null)
              .map((t) => [
                Array.isArray(t.value)
                  ? Number(t.value[0] ?? 0)
                  : Number(t.number ?? 0),
                Array.isArray(t.value)
                  ? Number(t.value[1] ?? t.value[0] ?? 0)
                  : Number(t.value ?? 0),
                Number(t.number ?? 0),
              ])
              .filter(
                (d) =>
                  !isNaN(d[0]) && !isNaN(d[1]) && d[0] < 1e100 && d[1] < 1e100,
              ),
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
            progressive: 0,
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
                let val: number | null = null;
                if (Array.isArray(t.value) && t.value.length > 0) {
                  val = t.value[t.value.length - 1];
                }
                return [Number(t.number ?? 0), Number(val ?? 0)];
              })
              .filter(
                (d) =>
                  d[1] != null &&
                  !isNaN(d[1] as number) &&
                  !isNaN(d[0] as number) &&
                  (d[1] as number) < 1e100 &&
                  (d[0] as number) < 1e100,
              ),
          },
          {
            name: "Pruned",
            type: "scatter",
            progressive: 0,
            symbolSize: 6,
            itemStyle: { color: "rgba(113, 113, 122, 0.5)" },
            data: prunedTrials
              .map((t) => {
                let val: number | null = null;
                if (Array.isArray(t.value) && t.value.length > 0) {
                  val = t.value[t.value.length - 1];
                }
                return [Number(t.number ?? 0), Number(val ?? 0)];
              })
              .filter(
                (d) =>
                  d[1] != null &&
                  !isNaN(d[1] as number) &&
                  !isNaN(d[0] as number) &&
                  (d[1] as number) < 1e100 &&
                  (d[0] as number) < 1e100,
              ),
          },
        ],
      };
};
