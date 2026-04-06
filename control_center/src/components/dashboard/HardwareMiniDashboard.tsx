import { useEffect, useRef } from "react";
import ReactECharts from "echarts-for-react";

const isTauri =
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

import type { HardwareMetrics } from "@/bindings/HardwareMetrics";

export function HardwareMiniDashboard() {
  const metricsHistoryRef = useRef<HardwareMetrics[]>([]);
  const latestRef = useRef<HardwareMetrics>({
    cpu_usage: 0,
    ram_usage_pct: 0,
    gpu_util: 0,
    vram_used_mb: 0,
    network_rx_mbps: 0,
    network_tx_mbps: 0,
    disk_read_mbps: 0,
    disk_write_mbps: 0,
    cpu_cores_usage: [],
    ram_used_mb: 0,
    disk_usage_pct: 0,
  });

  const cpuChartRef = useRef<ReactECharts>(null);
  const ramChartRef = useRef<ReactECharts>(null);
  const gpuChartRef = useRef<ReactECharts>(null);
  const vramChartRef = useRef<ReactECharts>(null);
  const netRxChartRef = useRef<ReactECharts>(null);
  const netTxChartRef = useRef<ReactECharts>(null);
  const diskRChartRef = useRef<ReactECharts>(null);
  const diskWChartRef = useRef<ReactECharts>(null);

  const cpuTextRef = useRef<HTMLSpanElement>(null);
  const ramTextRef = useRef<HTMLSpanElement>(null);
  const gpuTextRef = useRef<HTMLSpanElement>(null);
  const vramTextRef = useRef<HTMLSpanElement>(null);
  const netRxTextRef = useRef<HTMLSpanElement>(null);
  const netTxTextRef = useRef<HTMLSpanElement>(null);
  const diskRTextRef = useRef<HTMLSpanElement>(null);
  const diskWTextRef = useRef<HTMLSpanElement>(null);

  const cpuCoresContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let isCancelled = false;

    if (isTauri) {
      import("@tauri-apps/api/event").then(({ listen }) => {
        listen<HardwareMetrics>("hardware_telemetry", (event) => {
          if (isCancelled) return;
          metricsHistoryRef.current.push(event.payload);
          if (metricsHistoryRef.current.length > 60) {
            metricsHistoryRef.current.shift();
          }
          latestRef.current = event.payload;
        }).then((u) => {
          if (isCancelled) u();
          else unlisten = u;
        });
      });
    }

    return () => {
      isCancelled = true;
      if (unlisten) unlisten();
    };
  }, []);

  useEffect(() => {
    let animationFrameId: number;
    let isCancelled = false;

    const renderLoop = () => {
      if (isCancelled) return;
      const history = metricsHistoryRef.current;
      const latest = latestRef.current;

      if (cpuTextRef.current)
        cpuTextRef.current.innerText = `${latest.cpu_usage.toFixed(1)}%`;
      if (ramTextRef.current)
        ramTextRef.current.innerText = `${latest.ram_usage_pct.toFixed(1)}%`;
      if (gpuTextRef.current)
        gpuTextRef.current.innerText = `${latest.gpu_util.toFixed(1)}%`;
      if (vramTextRef.current)
        vramTextRef.current.innerText = `${(latest.vram_used_mb / 1024).toFixed(1)}G`;
      if (netRxTextRef.current)
        netRxTextRef.current.innerText = `${latest.network_rx_mbps.toFixed(1)}M/s`;
      if (netTxTextRef.current)
        netTxTextRef.current.innerText = `${latest.network_tx_mbps.toFixed(1)}M/s`;
      if (diskRTextRef.current)
        diskRTextRef.current.innerText = `${latest.disk_read_mbps.toFixed(1)}M/s`;
      if (diskWTextRef.current)
        diskWTextRef.current.innerText = `${latest.disk_write_mbps.toFixed(1)}M/s`;

      const updateChart = (ref: React.RefObject<any>, data: number[]) => {
        if (ref.current) {
          const instance = ref.current.getEchartsInstance();
          if (instance && !instance.isDisposed()) {
            instance.setOption({ series: [{ data }] });
          }
        }
      };

      if (history.length > 0) {
        updateChart(
          cpuChartRef,
          history.map((m) => m.cpu_usage),
        );
        updateChart(
          ramChartRef,
          history.map((m) => m.ram_usage_pct),
        );
        updateChart(
          gpuChartRef,
          history.map((m) => m.gpu_util),
        );
        updateChart(
          vramChartRef,
          history.map((m) => m.vram_used_mb),
        );
        updateChart(
          netRxChartRef,
          history.map((m) => m.network_rx_mbps),
        );
        updateChart(
          netTxChartRef,
          history.map((m) => m.network_tx_mbps),
        );
        updateChart(
          diskRChartRef,
          history.map((m) => m.disk_read_mbps),
        );
        updateChart(
          diskWChartRef,
          history.map((m) => m.disk_write_mbps),
        );
      }

      if (
        cpuCoresContainerRef.current &&
        latest.cpu_cores_usage &&
        latest.cpu_cores_usage.length > 0
      ) {
        const container = cpuCoresContainerRef.current;
        const cores = latest.cpu_cores_usage;
        if (container.children.length !== cores.length) {
          container.innerHTML = "";
          for (let i = 0; i < cores.length; i++) {
            const childWrapper = document.createElement("div");
            childWrapper.className =
              "flex-1 bg-zinc-800/50 rounded-t-[1px] h-full flex flex-col justify-end overflow-hidden";
            const childInner = document.createElement("div");
            childInner.className =
              "w-full rounded-t-[1px] transition-all duration-300";
            childWrapper.appendChild(childInner);
            container.appendChild(childWrapper);
          }
        }
        for (let i = 0; i < cores.length; i++) {
          const usage = cores[i];
          const color =
            usage > 80 ? "#ef4444" : usage > 50 ? "#f59e0b" : "#3b82f6";
          const inner = container.children[i].firstElementChild as HTMLElement;
          if (inner) {
            inner.style.height = `${Math.max(2, usage)}%`;
            inner.style.backgroundColor = color;
          }
        }
        if (cpuChartRef.current && cpuChartRef.current.ele) {
          cpuChartRef.current.ele.style.display = "none";
        }
      } else {
        if (cpuCoresContainerRef.current)
          cpuCoresContainerRef.current.innerHTML = "";
        if (cpuChartRef.current && cpuChartRef.current.ele) {
          cpuChartRef.current.ele.style.display = "block";
        }
      }

      animationFrameId = requestAnimationFrame(renderLoop);
    };

    animationFrameId = requestAnimationFrame(renderLoop);

    return () => {
      isCancelled = true;
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  const createSparklineOption = (
    data: number[],
    color: string,
    max?: number,
  ) => ({
    grid: { top: 2, bottom: 2, left: 0, right: 0 },
    xAxis: { type: "category", show: false, boundaryGap: false },
    yAxis: {
      type: "value",
      show: false,
      min: 0,
      ...(max !== undefined ? { max } : {}),
    },
    series: [
      {
        type: "line",
        data,
        showSymbol: false,
        lineStyle: { color, width: 1.5 },
        areaStyle: {
          color: {
            type: "linear",
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: color + "60" },
              { offset: 1, color: color + "00" },
            ],
          },
        },
        animation: false,
      },
    ],
    tooltip: { show: false },
  });

  return (
    <div className="flex flex-col gap-3 p-3 bg-zinc-950 border-t border-border/10 shrink-0">
      <div className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest mb-1">
        System Telemetry
      </div>
      <div className="grid grid-cols-2 gap-3">
        {/* CPU */}
        <div className="flex flex-col gap-1">
          <div className="h-8 w-full opacity-60 relative">
            <div
              ref={cpuCoresContainerRef}
              className="absolute inset-0 w-full h-full flex items-end gap-[1px]"
            ></div>
            <ReactECharts
              ref={cpuChartRef}
              option={createSparklineOption([], "#3b82f6", 100)}
              style={{ height: "100%", width: "100%" }}
            />
          </div>
          <div className="flex items-end justify-between">
            <span className="text-[9px] text-zinc-500 uppercase font-bold tracking-widest">
              CPU
            </span>
            <span
              ref={cpuTextRef}
              className="text-[10px] text-zinc-300 font-mono"
            >
              0.0%
            </span>
          </div>
        </div>

        {/* RAM */}
        <div className="flex flex-col gap-1">
          <div className="h-8 w-full opacity-60">
            <ReactECharts
              ref={ramChartRef}
              option={createSparklineOption([], "#10b981", 100)}
              style={{ height: "100%", width: "100%" }}
            />
          </div>
          <div className="flex items-end justify-between">
            <span className="text-[9px] text-zinc-500 uppercase font-bold tracking-widest">
              RAM
            </span>
            <span
              ref={ramTextRef}
              className="text-[10px] text-zinc-300 font-mono"
            >
              0.0%
            </span>
          </div>
        </div>

        {/* GPU */}
        <div className="flex flex-col gap-1">
          <div className="h-8 w-full opacity-60">
            <ReactECharts
              ref={gpuChartRef}
              option={createSparklineOption([], "#f59e0b", 100)}
              style={{ height: "100%", width: "100%" }}
            />
          </div>
          <div className="flex items-end justify-between">
            <span className="text-[9px] text-zinc-500 uppercase font-bold tracking-widest">
              GPU
            </span>
            <span
              ref={gpuTextRef}
              className="text-[10px] text-zinc-300 font-mono"
            >
              0.0%
            </span>
          </div>
        </div>

        {/* VRAM */}
        <div className="flex flex-col gap-1">
          <div className="h-8 w-full opacity-60">
            <ReactECharts
              ref={vramChartRef}
              option={createSparklineOption([], "#8b5cf6", 24000)}
              style={{ height: "100%", width: "100%" }}
            />
          </div>
          <div className="flex items-end justify-between">
            <span className="text-[9px] text-zinc-500 uppercase font-bold tracking-widest">
              VRAM
            </span>
            <span
              ref={vramTextRef}
              className="text-[10px] text-zinc-300 font-mono"
            >
              0.0G
            </span>
          </div>
        </div>

        {/* NET RX */}
        <div className="flex flex-col gap-1">
          <div className="h-8 w-full opacity-60">
            <ReactECharts
              ref={netRxChartRef}
              option={createSparklineOption([], "#0ea5e9")}
              style={{ height: "100%", width: "100%" }}
            />
          </div>
          <div className="flex items-end justify-between">
            <span className="text-[9px] text-zinc-500 uppercase font-bold tracking-widest">
              NET RX
            </span>
            <span
              ref={netRxTextRef}
              className="text-[10px] text-zinc-300 font-mono"
            >
              0.0M/s
            </span>
          </div>
        </div>

        {/* NET TX */}
        <div className="flex flex-col gap-1">
          <div className="h-8 w-full opacity-60">
            <ReactECharts
              ref={netTxChartRef}
              option={createSparklineOption([], "#06b6d4")}
              style={{ height: "100%", width: "100%" }}
            />
          </div>
          <div className="flex items-end justify-between">
            <span className="text-[9px] text-zinc-500 uppercase font-bold tracking-widest">
              NET TX
            </span>
            <span
              ref={netTxTextRef}
              className="text-[10px] text-zinc-300 font-mono"
            >
              0.0M/s
            </span>
          </div>
        </div>

        {/* DISK R */}
        <div className="flex flex-col gap-1">
          <div className="h-8 w-full opacity-60">
            <ReactECharts
              ref={diskRChartRef}
              option={createSparklineOption([], "#ec4899")}
              style={{ height: "100%", width: "100%" }}
            />
          </div>
          <div className="flex items-end justify-between">
            <span className="text-[9px] text-zinc-500 uppercase font-bold tracking-widest">
              DISK R
            </span>
            <span
              ref={diskRTextRef}
              className="text-[10px] text-zinc-300 font-mono"
            >
              0.0M/s
            </span>
          </div>
        </div>

        {/* DISK W */}
        <div className="flex flex-col gap-1">
          <div className="h-8 w-full opacity-60">
            <ReactECharts
              ref={diskWChartRef}
              option={createSparklineOption([], "#e11d48")}
              style={{ height: "100%", width: "100%" }}
            />
          </div>
          <div className="flex items-end justify-between">
            <span className="text-[9px] text-zinc-500 uppercase font-bold tracking-widest">
              DISK W
            </span>
            <span
              ref={diskWTextRef}
              className="text-[10px] text-zinc-300 font-mono"
            >
              0.0M/s
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
