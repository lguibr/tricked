import { useEffect, useState } from "react";
import ReactECharts from "echarts-for-react";

const isTauri =
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

import type { HardwareMetrics } from "@/bindings/HardwareMetrics";

export function HardwareMiniDashboard() {
  const [metricsHistory, setMetricsHistory] = useState<HardwareMetrics[]>([]);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let isCancelled = false;

    if (isTauri) {
      import("@tauri-apps/api/event").then(({ listen }) => {
        listen<HardwareMetrics>("hardware_telemetry", (event) => {
          if (isCancelled) return;
          setMetricsHistory((prev) => {
            const next = [...prev, event.payload];
            if (next.length > 60) next.shift(); // Keep last 60 seconds
            return next;
          });
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

  const latest = metricsHistory[metricsHistory.length - 1] || {
    cpu_usage: 0,
    ram_usage_pct: 0,
    gpu_util: 0,
    vram_used_mb: 0,
    network_rx_mbps: 0,
    network_tx_mbps: 0,
    disk_read_mbps: 0,
    disk_write_mbps: 0,
  };

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
          <div className="h-8 w-full opacity-60">
            {latest.cpu_cores_usage && latest.cpu_cores_usage.length > 0 ? (
              <div className="w-full h-full flex items-end gap-[1px]">
                {latest.cpu_cores_usage.map((usage, i) => {
                  const color =
                    usage > 80 ? "#ef4444" : usage > 50 ? "#f59e0b" : "#3b82f6";
                  return (
                    <div
                      key={i}
                      className="flex-1 bg-zinc-800/50 rounded-t-[1px] h-full flex flex-col justify-end overflow-hidden"
                    >
                      <div
                        className="w-full rounded-t-[1px] transition-all duration-300"
                        style={{
                          height: `${Math.max(2, usage)}%`,
                          backgroundColor: color,
                        }}
                      />
                    </div>
                  );
                })}
              </div>
            ) : (
              <ReactECharts
                option={createSparklineOption(
                  metricsHistory.map((m) => m.cpu_usage),
                  "#3b82f6",
                  100,
                )}
                style={{ height: "100%", width: "100%" }}
              />
            )}
          </div>
          <div className="flex items-end justify-between">
            <span className="text-[9px] text-zinc-500 uppercase font-bold tracking-widest">
              CPU
            </span>
            <span className="text-[10px] text-zinc-300 font-mono">
              {latest.cpu_usage.toFixed(1)}%
            </span>
          </div>
        </div>

        {/* RAM */}
        <div className="flex flex-col gap-1">
          <div className="h-8 w-full opacity-60">
            <ReactECharts
              option={createSparklineOption(
                metricsHistory.map((m) => m.ram_usage_pct),
                "#10b981",
                100,
              )}
              style={{ height: "100%", width: "100%" }}
            />
          </div>
          <div className="flex items-end justify-between">
            <span className="text-[9px] text-zinc-500 uppercase font-bold tracking-widest">
              RAM
            </span>
            <span className="text-[10px] text-zinc-300 font-mono">
              {latest.ram_usage_pct.toFixed(1)}%
            </span>
          </div>
        </div>

        {/* GPU */}
        <div className="flex flex-col gap-1">
          <div className="h-8 w-full opacity-60">
            <ReactECharts
              option={createSparklineOption(
                metricsHistory.map((m) => m.gpu_util),
                "#f59e0b",
                100,
              )}
              style={{ height: "100%", width: "100%" }}
            />
          </div>
          <div className="flex items-end justify-between">
            <span className="text-[9px] text-zinc-500 uppercase font-bold tracking-widest">
              GPU
            </span>
            <span className="text-[10px] text-zinc-300 font-mono">
              {latest.gpu_util.toFixed(1)}%
            </span>
          </div>
        </div>

        {/* VRAM */}
        <div className="flex flex-col gap-1">
          <div className="h-8 w-full opacity-60">
            <ReactECharts
              option={createSparklineOption(
                metricsHistory.map((m) => m.vram_used_mb),
                "#8b5cf6",
                24000,
              )}
              style={{ height: "100%", width: "100%" }}
            />
          </div>
          <div className="flex items-end justify-between">
            <span className="text-[9px] text-zinc-500 uppercase font-bold tracking-widest">
              VRAM
            </span>
            <span className="text-[10px] text-zinc-300 font-mono">
              {(latest.vram_used_mb / 1024).toFixed(1)}G
            </span>
          </div>
        </div>

        {/* NET RX */}
        <div className="flex flex-col gap-1">
          <div className="h-8 w-full opacity-60">
            <ReactECharts
              option={createSparklineOption(
                metricsHistory.map((m) => m.network_rx_mbps),
                "#0ea5e9", // sky-500
              )}
              style={{ height: "100%", width: "100%" }}
            />
          </div>
          <div className="flex items-end justify-between">
            <span className="text-[9px] text-zinc-500 uppercase font-bold tracking-widest">
              NET RX
            </span>
            <span className="text-[10px] text-zinc-300 font-mono">
              {latest.network_rx_mbps.toFixed(1)}M/s
            </span>
          </div>
        </div>

        {/* NET TX */}
        <div className="flex flex-col gap-1">
          <div className="h-8 w-full opacity-60">
            <ReactECharts
              option={createSparklineOption(
                metricsHistory.map((m) => m.network_tx_mbps),
                "#06b6d4", // cyan-500
              )}
              style={{ height: "100%", width: "100%" }}
            />
          </div>
          <div className="flex items-end justify-between">
            <span className="text-[9px] text-zinc-500 uppercase font-bold tracking-widest">
              NET TX
            </span>
            <span className="text-[10px] text-zinc-300 font-mono">
              {latest.network_tx_mbps.toFixed(1)}M/s
            </span>
          </div>
        </div>

        {/* DISK R */}
        <div className="flex flex-col gap-1">
          <div className="h-8 w-full opacity-60">
            <ReactECharts
              option={createSparklineOption(
                metricsHistory.map((m) => m.disk_read_mbps),
                "#ec4899", // rose-500
              )}
              style={{ height: "100%", width: "100%" }}
            />
          </div>
          <div className="flex items-end justify-between">
            <span className="text-[9px] text-zinc-500 uppercase font-bold tracking-widest">
              DISK R
            </span>
            <span className="text-[10px] text-zinc-300 font-mono">
              {latest.disk_read_mbps.toFixed(1)}M/s
            </span>
          </div>
        </div>

        {/* DISK W */}
        <div className="flex flex-col gap-1">
          <div className="h-8 w-full opacity-60">
            <ReactECharts
              option={createSparklineOption(
                metricsHistory.map((m) => m.disk_write_mbps),
                "#e11d48", // rose-600
              )}
              style={{ height: "100%", width: "100%" }}
            />
          </div>
          <div className="flex items-end justify-between">
            <span className="text-[9px] text-zinc-500 uppercase font-bold tracking-widest">
              DISK W
            </span>
            <span className="text-[10px] text-zinc-300 font-mono">
              {latest.disk_write_mbps.toFixed(1)}M/s
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
