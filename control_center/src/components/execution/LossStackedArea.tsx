import { useEffect, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import * as echarts from "echarts";
import { Info } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { Run } from "@/bindings/Run";

interface LossStackedAreaProps {
  xAxisMode: "step" | "relative" | "absolute";
  smoothingWeight: number;
  runs: Run[];
  runIds: string[];
  metricsDataRef: React.MutableRefObject<Record<string, any[]>>;
  runColors: Record<string, string>;
}

export function LossStackedArea({
  runs,
  runIds,
  metricsDataRef,
  runColors,
  xAxisMode,
  smoothingWeight,
}: LossStackedAreaProps) {
  void runColors;
  const chartRef = useRef<ReactECharts>(null);
  const [pinnedTime, setPinnedTime] = useState<number | null>(null);

  // Connect globally for shared interactions
  useEffect(() => {
    echarts.connect("metricsGroup");
    return () => echarts.disconnect("metricsGroup");
  }, []);

  const hexToHSL = (H: string): [number, number, number] => {
    let r = 0,
      g = 0,
      b = 0;
    if (H.length === 4) {
      r = parseInt(H[1] + H[1], 16);
      g = parseInt(H[2] + H[2], 16);
      b = parseInt(H[3] + H[3], 16);
    } else if (H.length === 7) {
      r = parseInt(H.slice(1, 3), 16);
      g = parseInt(H.slice(3, 5), 16);
      b = parseInt(H.slice(5, 7), 16);
    }
    r /= 255;
    g /= 255;
    b /= 255;
    const cmin = Math.min(r, g, b),
      cmax = Math.max(r, g, b),
      delta = cmax - cmin;
    let h = 0,
      s = 0,
      l = 0;
    if (delta === 0) h = 0;
    else if (cmax === r) h = ((g - b) / delta) % 6;
    else if (cmax === g) h = (b - r) / delta + 2;
    else h = (r - g) / delta + 4;
    h = Math.round(h * 60);
    if (h < 0) h += 360;
    l = (cmax + cmin) / 2;
    s = delta === 0 ? 0 : delta / (1 - Math.abs(2 * l - 1));
    return [h, +(s * 100).toFixed(1), +(l * 100).toFixed(1)];
  };

  const getPinnedSnapshots = () => {
    if (pinnedTime === null) return [];

    const snapshots = runIds.map((id) => {
      const rawData = metricsDataRef.current[id] || [];
      const run = runs.find((r) => r.id === id);
      const baseTime = run?.start_time
        ? new Date(run.start_time + "Z").getTime()
        : Date.now();

      let lastPol = 0,
        lastVal = 0,
        lastRew = 0;

      for (const d of rawData) {
        let xVal = 0;
        const elapsedSecs = Number(d.elapsed_time || 0);

        if (xAxisMode === "step") xVal = parseInt(d.step || 0, 10);
        else if (xAxisMode === "absolute") xVal = baseTime + elapsedSecs * 1000;
        else if (xAxisMode === "relative") xVal = elapsedSecs;

        if (xVal > pinnedTime) break;

        const curPol = Number(d.policy_loss) || 0;
        const curVal = Number(d.value_loss) || 0;
        const curRew = Number(d.reward_loss) || 0;

        if (!isNaN(curPol))
          lastPol = lastPol * smoothingWeight + curPol * (1 - smoothingWeight);
        if (!isNaN(curVal))
          lastVal = lastVal * smoothingWeight + curVal * (1 - smoothingWeight);
        if (!isNaN(curRew))
          lastRew = lastRew * smoothingWeight + curRew * (1 - smoothingWeight);
      }
      return {
        id,
        total: lastPol + lastVal + lastRew,
        pol: lastPol,
        val: lastVal,
        rew: lastRew,
        name: id.substring(0, 4),
      };
    });

    snapshots.sort((a, b) => b.total - a.total);
    return snapshots;
  };

  const getXAxisConfig = () => {
    if (pinnedTime !== null) {
      const snaps = getPinnedSnapshots();
      return {
        type: "category",
        data: snaps.map((s) => s.name),
        axisLabel: { fontSize: 9 },
        splitLine: { show: false },
        boundaryGap: true,
      };
    }

    if (xAxisMode === "absolute") {
      return {
        type: "time",
        boundaryGap: false,
        splitLine: { show: false },
        axisLabel: { fontSize: 9 },
        min: "dataMin",
        max: "dataMax",
      };
    } else if (xAxisMode === "relative") {
      return {
        type: "value",
        boundaryGap: false,
        splitLine: { show: false },
        axisLabel: { fontSize: 9, formatter: "{value} s" },
        min: "dataMin",
        max: "dataMax",
      };
    }
    return {
      type: "value",
      boundaryGap: false,
      splitLine: { show: false },
      axisLabel: { fontSize: 9 },
      min: "dataMin",
      max: "dataMax",
    };
  };

  const getSeries = () => {
    if (pinnedTime !== null) {
      const snaps = getPinnedSnapshots();
      // Render 3 Bar Series
      return [
        {
          name: "Policy Loss",
          type: "bar",
          stack: "total",
          emphasis: { focus: "series" },
          label: { show: false },
          data: snaps.map((s) => {
            const c = runColors[s.id] || "#3b82f6";
            const [h, st, l] = hexToHSL(c);
            return {
              value: s.pol,
              itemStyle: {
                color: `hsl(${h}, ${st}%, ${Math.max(10, l - 20)}%)`,
              },
              groupId: s.id,
            };
          }),
        },
        {
          name: "Value Loss",
          type: "bar",
          stack: "total",
          emphasis: { focus: "series" },
          label: { show: false },
          data: snaps.map((s) => {
            const c = runColors[s.id] || "#3b82f6";
            const [h, st, l] = hexToHSL(c);
            return {
              value: s.val,
              itemStyle: { color: `hsl(${h}, ${st}%, ${l}%)` },
              groupId: s.id,
            };
          }),
        },
        {
          name: "Reward Loss",
          type: "bar",
          stack: "total",
          emphasis: { focus: "series" },
          label: { show: false },
          data: snaps.map((s) => {
            const c = runColors[s.id] || "#3b82f6";
            const [h, st, l] = hexToHSL(c);
            return {
              value: s.rew,
              itemStyle: {
                color: `hsl(${h}, ${st}%, ${Math.min(90, l + 20)}%)`,
              },
              groupId: s.id,
            };
          }),
        },
      ];
    }

    return runIds.flatMap((id) => {
      const rawData = metricsDataRef.current[id] || [];
      const baseColor = runColors[id] || "#3b82f6";
      const [h, s, l] = hexToHSL(baseColor);

      // Create 3 different visual shades by interpolating lightness
      const l1 = Math.max(10, l - 20); // Darker
      const l2 = l; // Base
      const l3 = Math.min(90, l + 20); // Lighter

      let lastPol = rawData.length > 0 ? rawData[0].policy_loss || 0 : 0;
      let lastVal = rawData.length > 0 ? rawData[0].value_loss || 0 : 0;
      let lastRew = rawData.length > 0 ? rawData[0].reward_loss || 0 : 0;

      const run = runs.find((r) => r.id === id);
      const baseTime = run?.start_time
        ? new Date(run.start_time + "Z").getTime()
        : Date.now();

      const smoothedData = rawData.map((d) => {
        let xVal = 0;
        const elapsedSecs = Number(d.elapsed_time || 0);

        if (xAxisMode === "step") {
          xVal = parseInt(d.step || 0, 10);
        } else if (xAxisMode === "absolute") {
          xVal = baseTime + elapsedSecs * 1000;
        } else if (xAxisMode === "relative") {
          xVal = elapsedSecs;
        }

        const curPol = Number(d.policy_loss) || 0;
        const curVal = Number(d.value_loss) || 0;
        const curRew = Number(d.reward_loss) || 0;

        if (!isNaN(curPol))
          lastPol = lastPol * smoothingWeight + curPol * (1 - smoothingWeight);
        if (!isNaN(curVal))
          lastVal = lastVal * smoothingWeight + curVal * (1 - smoothingWeight);
        if (!isNaN(curRew))
          lastRew = lastRew * smoothingWeight + curRew * (1 - smoothingWeight);

        return { xVal, pol: lastPol, val: lastVal, rew: lastRew };
      });

      return [
        {
          id: `${id}_policy_loss`,
          name: `Run ${id.substring(0, 4)} Pol`,
          type: "line",
          smooth: 0.5,
          lineStyle: { width: 1 },
          showSymbol: false,
          symbol: "none",
          emphasis: { focus: "series" },
          data: smoothedData
            .map((d) => [d.xVal, d.pol])
            .filter((d) => !isNaN(d[1] as number)),
          itemStyle: { color: `hsl(${h}, ${s}%, ${l1}%)` },
          groupId: id,
        },
        {
          id: `${id}_value_loss`,
          name: `Run ${id.substring(0, 4)} Val`,
          type: "line",
          smooth: 0.5,
          lineStyle: { width: 1 },
          showSymbol: false,
          symbol: "none",
          emphasis: { focus: "series" },
          data: smoothedData
            .map((d) => [d.xVal, d.val])
            .filter((d) => !isNaN(d[1] as number)),
          itemStyle: { color: `hsl(${h}, ${s}%, ${l2}%)` },
          groupId: id,
        },
        {
          id: `${id}_reward_loss`,
          name: `Run ${id.substring(0, 4)} Rew`,
          type: "line",
          smooth: 0.5,
          lineStyle: { width: 1 },
          showSymbol: false,
          symbol: "none",
          emphasis: { focus: "series" },
          data: smoothedData
            .map((d) => [d.xVal, d.rew])
            .filter((d) => !isNaN(d[1] as number)),
          itemStyle: { color: `hsl(${h}, ${s}%, ${l3}%)` },
          groupId: id,
        },
      ];
    });
  };

  useEffect(() => {
    let timeoutId: NodeJS.Timeout;
    let isCancelled = false;
    let lastDataLength = -1;
    let lastRunIds: string[] = [];

    const renderLoop = () => {
      if (isCancelled) return;

      const currentLength = runIds.reduce(
        (sum, id) => sum + (metricsDataRef.current[id]?.length || 0),
        0,
      );

      const runIdsChanged = runIds.join(",") !== lastRunIds.join(",");

      if (
        (currentLength !== lastDataLength || runIdsChanged) &&
        chartRef.current
      ) {
        lastDataLength = currentLength;
        lastRunIds = [...runIds];
        const instance = chartRef.current.getEchartsInstance();
        if (instance && !instance.isDisposed()) {
          instance.group = "metricsGroup";
          instance.setOption(
            {
              xAxis: getXAxisConfig(),
              yAxis: {
                type: "value",
              },
              series: getSeries(),
            },
            { replaceMerge: ["series"] },
          );
        }
      }

      timeoutId = setTimeout(renderLoop, 500);
    };

    renderLoop();

    return () => {
      isCancelled = true;
      clearTimeout(timeoutId);
    };
  }, [runIds, runs, xAxisMode, smoothingWeight, pinnedTime]);

  const initialOptions = {
    backgroundColor: "transparent",
    tooltip: {
      show: false,
      trigger: "axis",
      axisPointer: {
        type: "cross",
        label: {
          backgroundColor: "#27272a",
          fontSize: 10,
        },
      },
    },
    legend: {
      top: 25,
      textStyle: { color: "#a1a1aa", fontSize: 10 },
      type: "scroll",
    },
    grid: {
      left: "3%",
      right: "4%",
      bottom: "3%",
      top: "25%",
      containLabel: true,
    },
    xAxis: getXAxisConfig(),
    yAxis: {
      type: "value",
      splitLine: { lineStyle: { color: "#27272a" } },
      axisLabel: { fontSize: 9 },
    },
    series: [],
  };

  return (
    <div className="bg-background flex flex-col relative w-full h-full overflow-hidden p-1 border rounded-md border-border/20 min-h-[300px]">
      <div className="absolute top-12 left-2 text-[10px] text-red-500 font-mono z-50 pointer-events-none w-full bg-black/80">
        {(() => {
          if (runIds.length === 0) return "NO DEBUG DATA";
          const activeRunId =
            runIds.find((id) => metricsDataRef.current[id]?.length > 0) ||
            runIds[0];
          const data = metricsDataRef.current[activeRunId];
          if (!data || data.length === 0) return "NO DEBUG DATA";
          const last = data[data.length - 1];
          return `DEBUG DUMP: total=${last.total_loss}, pol=${last.policy_loss}, val=${last.value_loss}, rew=${last.reward_loss}, heat=${last.spatial_heatmap ? last.spatial_heatmap.length : "NULL"}`;
        })()}
      </div>
      <div className="flex items-center justify-between z-10 absolute top-2 left-2 right-2 pointer-events-none">
        <span className="text-[10px] font-semibold text-zinc-400 tracking-wider bg-background px-1">
          Loss Composition
        </span>
        <TooltipProvider delayDuration={100}>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="pointer-events-auto cursor-help">
                <Info className="h-3 w-3 text-zinc-500 hover:text-zinc-300 transition-colors" />
              </div>
            </TooltipTrigger>
            <TooltipContent
              side="left"
              className="max-w-[200px] text-xs leading-relaxed z-50"
            >
              <p>
                Stacked area graph showing the distinct components of the
                composite loss function during gradient descent.
              </p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
      <ReactECharts
        ref={chartRef}
        option={initialOptions}
        style={{ width: "100%", height: "100%" }}
        theme="dark"
        onEvents={{
          dblclick: (params: any) => {
            if (
              params.componentType === "series" ||
              params.componentType === "xAxis"
            ) {
              const val = Array.isArray(params.value)
                ? params.value[0]
                : params.value;
              if (typeof val === "number") {
                setPinnedTime(val);
              } else {
                setPinnedTime(null);
              }
            } else {
              setPinnedTime(null);
            }
          },
          mouseover: (params: any) => {
            if (
              params.componentType === "series" &&
              params.data &&
              params.data.groupId
            ) {
              chartRef.current?.getEchartsInstance()?.dispatchAction({
                type: "highlight",
                seriesName: params.data.groupId, // We will dispatch globally, but first locally. Wait, Echarts connects by seriesName usually.
                // We'll see if `focus: 'series'` natively groups by `groupId` when connected, or if it groups locally.
                // Wait, native highlighting via mouse handles it internally if connected groups match exact series indexes.
                // We will manually force a highlight based on matching groupId across ALL graphs.
              });
              // actually in an echarts connected group, focus: group is native inside the same chart using series.groupId in echarts 5.x!
            }
          },
        }}
      />
    </div>
  );
}
