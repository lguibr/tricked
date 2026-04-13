import { useMemo, useCallback, useEffect, useState } from "react";
import uPlot from "uplot";
import { Info } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { Run } from "@/bindings/Run";
import { UPlotReact } from "@/components/ui/UPlotReact";
import { useChartSync } from "@/hooks/useChartSync";

interface MetricChartProps {
  title: string;
  description: string;
  metricKey: string;
  runs: Run[];
  runIds: string[];
  metricsDataRef: React.MutableRefObject<Record<string, any[]>>;
  runColors: Record<string, string>;
  xAxisMode?: "step" | "relative" | "absolute";
  smoothingWeight?: number;
}

export function MetricChart({
  title,
  description,
  metricKey,
  runs,
  runIds,
  metricsDataRef,
  runColors,
  xAxisMode = "step",
  smoothingWeight = 0.9,
}: MetricChartProps) {
  const { registerChart, unregisterChart } = useChartSync();
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let lastRender = 0;
    let rafId: number;

    const handleUpdate = () => {
      const now = performance.now();
      if (now - lastRender > 250) {
        lastRender = now;
        setTick(t => t + 1);
      } else {
        cancelAnimationFrame(rafId);
        rafId = requestAnimationFrame(() => {
          if (performance.now() - lastRender > 250) {
            lastRender = performance.now();
            setTick(t => t + 1);
          }
        });
      }
    };
    window.addEventListener("engine_telemetry_force_update", handleUpdate);
    return () => {
      window.removeEventListener("engine_telemetry_force_update", handleUpdate);
      cancelAnimationFrame(rafId);
    };
  }, []);

  const chartDeps = useMemo(() => {
    if (!runIds.length) {
      return { data: [[0]] as uPlot.AlignedData, options: null };
    }

    const xSet = new Set<number>();
    
    // First pass: Find all unique X coordinates
    runIds.forEach(id => {
      const data = metricsDataRef.current[id] || [];
      const run = runs.find(r => r.id === id);
      const baseTime = run?.start_time ? new Date(run.start_time + "Z").getTime() : Date.now();
      
      data.forEach(d => {
        let xVal = 0;
        const elapsedSecs = Number(d.elapsed_time || 0);
        if (xAxisMode === "step") xVal = parseInt(d.step, 10) || 0;
        else if (xAxisMode === "absolute") xVal = (baseTime + elapsedSecs * 1000) / 1000;
        else if (xAxisMode === "relative") xVal = elapsedSecs;
        
        xSet.add(xVal);
      });
    });

    const xArray = Array.from(xSet).sort((a, b) => a - b);
    if (xArray.length === 0) {
       return { data: [[0]] as uPlot.AlignedData, options: null };
    }

    const xCache = new Map<number, number>();
    xArray.forEach((x, i) => xCache.set(x, i));

    const finalData: (number | null)[][] = [xArray];
    const seriesOptions: uPlot.Series[] = [
      {
         value: (_u, v) => v == null ? "-" : (xAxisMode === "step" ? v.toFixed(0) : v.toFixed(2))
      }
    ];

    runIds.forEach((id) => {
      const data = metricsDataRef.current[id] || [];
      const run = runs.find(r => r.id === id);
      const baseTime = run?.start_time ? new Date(run.start_time + "Z").getTime() : Date.now();
      const baseColor = runColors[id] || "#10b981";
      
      const yRaw: (number | null)[] = new Array(xArray.length).fill(null);
      const ySmooth: (number | null)[] = new Array(xArray.length).fill(null);

      let lastEma: number | null = null;

      data.forEach(d => {
        let xVal = 0;
        const elapsedSecs = Number(d.elapsed_time || 0);
        if (xAxisMode === "step") xVal = parseInt(d.step, 10) || 0;
        else if (xAxisMode === "absolute") xVal = (baseTime + elapsedSecs * 1000) / 1000;
        else if (xAxisMode === "relative") xVal = elapsedSecs;

        const val = Number(d[metricKey]);
        if (isNaN(val)) return;

        const i = xCache.get(xVal)!;
        yRaw[i] = val;

        if (lastEma === null) {
          lastEma = val;
        } else {
          lastEma = lastEma * smoothingWeight + val * (1 - smoothingWeight);
        }
        ySmooth[i] = lastEma;
      });

      // Raw Series
      seriesOptions.push({
        label: `${run?.name || id.substring(0,4)} (Raw)`,
        stroke: `${baseColor}40`,
        width: 1,
        spanGaps: true,
        points: { show: false }
      });
      finalData.push(yRaw);

      // Smooth Series
      seriesOptions.push({
        label: `${run?.name || id.substring(0,4)}`,
        stroke: baseColor,
        width: 1.5,
        spanGaps: true,
        points: { show: false }
      });
      finalData.push(ySmooth);
    });

    const isTime = xAxisMode === "absolute";
    const xAxesOptions: uPlot.Axis = {
      grid: { stroke: "#27272a" },
      stroke: "#a1a1aa",
      font: "10px Inter, sans-serif",
    };
    
    if (!isTime) {
      xAxesOptions.values = (_u, vals) => vals.map(v => {
        if (xAxisMode === "relative") return v.toFixed(0) + "s";
        return v.toFixed(0);
      });
    }

    const tooltipPlugin = (
      mode: string,
      rIds: string[],
      rns: Run[],
      rColors: Record<string, string>
    ): uPlot.Plugin => {
      let tooltip: HTMLDivElement;

      return {
        hooks: {
          init: u => {
            const over = u.root.querySelector(".u-over");
            if (!over) return;
            tooltip = document.createElement("div");
            tooltip.className = "pointer-events-none absolute z-[100] min-w-[120px] bg-black/90 border border-emerald-500/20 text-[10px] p-2 rounded shadow-xl font-mono flex flex-col gap-1";
            tooltip.style.display = "none";
            over.appendChild(tooltip);
          },
          setCursor: u => {
            if (!tooltip) return;
            const { left, top, idx } = u.cursor;
            if (left === undefined || left < 0 || top === undefined || top < 0 || idx === undefined || idx === null) {
              tooltip.style.display = "none";
              return;
            }

            const xRaw = u.data[0][idx];
            let xValStr = String(xRaw);
            if (mode === "absolute") {
              xValStr = new Date(Number(xRaw) * 1000).toLocaleTimeString([], { hour12: false });
            } else if (mode === "relative") {
              xValStr = Number(xRaw).toFixed(1) + "s";
            } else {
               xValStr = "Step " + xValStr;
            }

            let html = `<div class="font-bold text-zinc-400 mb-2 border-b border-white/10 pb-1.5">${xValStr}</div>`;
            let hasValues = false;

            rIds.forEach((id, runIdx) => {
              const run = rns.find(r => r.id === id);
              const name = run?.name || id.substring(0, 4);
              const baseColor = rColors[id] || "#10b981";
              
              const rawDataIdx = runIdx * 2 + 1;
              const smoothDataIdx = runIdx * 2 + 2;

              const rawVal = u.data[rawDataIdx] ? u.data[rawDataIdx][idx] : null;
              const smoothVal = u.data[smoothDataIdx] ? u.data[smoothDataIdx][idx] : null;

              if (rawVal != null || smoothVal != null) {
                const rStr = rawVal != null ? Number(rawVal).toFixed(4) : "-";
                const sStr = smoothVal != null ? Number(smoothVal).toFixed(4) : "-";

                html += `
                  <div class="flex items-center justify-between gap-6 hover:bg-white/5 p-1 rounded px-1.5 -mx-1.5 transition-colors">
                    <div class="flex items-center gap-2">
                       <div class="w-3 h-3 rounded-sm border border-black/50 shadow-sm" style="background-color: ${baseColor}"></div>
                       <span class="text-zinc-200 font-medium truncate max-w-[140px]">${name}</span>
                    </div>
                    <div class="flex items-center gap-3 text-right">
                       <div class="flex flex-col items-end leading-none">
                         <span class="text-[8px] text-zinc-500 uppercase font-semibold">Raw</span>
                         <span class="text-zinc-400">${rStr}</span>
                       </div>
                       <div class="flex flex-col items-end leading-none">
                         <span class="text-[8px] text-zinc-500 uppercase font-semibold">Smoothed</span>
                         <span class="text-white font-bold">${sStr}</span>
                       </div>
                    </div>
                  </div>
                `;
                hasValues = true;
              }
            });

            if (hasValues) {
              tooltip.style.display = "flex";
              tooltip.innerHTML = html;
              
              const bcr = u.over.getBoundingClientRect();
              const tBcr = tooltip.getBoundingClientRect();
              
              let tLeft = left + 15;
              let tTop = top + 15;
              
              if (tLeft + tBcr.width > bcr.width) tLeft = left - tBcr.width - 15;
              if (tTop + tBcr.height > bcr.height) tTop = top - tBcr.height - 15;
              
              tooltip.style.left = tLeft + "px";
              tooltip.style.top = tTop + "px";
            } else {
              tooltip.style.display = "none";
            }
          }
        }
      };
    };

    const opts: uPlot.Options = {
      width: 400,
      height: 200,
      plugins: [tooltipPlugin(xAxisMode, runIds, runs, runColors)],
      scales: {
        x: { time: isTime }
      },
      cursor: {
        y: false,
        sync: { key: "metricsGroup", setSeries: true }
      },
      legend: { show: false },
      axes: [
        xAxesOptions,
        {
          grid: { stroke: "#27272a" },
          stroke: "#a1a1aa",
          font: "10px Inter, sans-serif",
          space: 25,
        }
      ],
      series: seriesOptions
    };

    return { data: finalData as uPlot.AlignedData, options: opts };
  }, [runIds, runs, metricsDataRef, runColors, xAxisMode, smoothingWeight, metricKey, tick]);

  const onInstance = useCallback((u: uPlot) => {
    registerChart(u);
  }, [registerChart]);

  const onUnmount = useCallback((u: uPlot) => {
    unregisterChart(u);
  }, [unregisterChart]);

  return (
    <div className="bg-background flex flex-col relative w-full h-full overflow-hidden p-1 border rounded-md border-border/20">
      <div className="flex items-center justify-center gap-1 z-10 absolute top-2 left-0 right-0 pointer-events-none">
        <span className="text-[10px] font-semibold text-zinc-400 tracking-wider bg-background px-1">
          {title}
        </span>
        <TooltipProvider delayDuration={100}>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="pointer-events-auto cursor-help">
                <Info className="h-3 w-3 text-zinc-500 hover:text-zinc-300 transition-colors" />
              </div>
            </TooltipTrigger>
            <TooltipContent
              side="top"
              className="max-w-[200px] text-xs leading-relaxed text-center z-50"
            >
               <p>{description}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>

      <div className="flex-1 w-full min-h-0 pt-6">
        {chartDeps.options && (
          <UPlotReact 
             key={`${Object.values(runColors).join("-")}-${xAxisMode}-${runIds.length}`}
             options={chartDeps.options} 
             data={chartDeps.data} 
             onInstance={onInstance}
             onUnmount={onUnmount}
          />
        )}
      </div>
    </div>
  );
}
