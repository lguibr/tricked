import ReactECharts from "echarts-for-react";
import { useHardwareTelemetry } from "./useHardwareTelemetry";
import { createSparklineOption } from "./SparklineOptions";

export function HardwareMiniDashboard() {
  const { latestRef, refs } = useHardwareTelemetry();
  const {
    cpuChartRef,
    ramChartRef,
    gpuChartRef,
    vramChartRef,
    netRxChartRef,
    netTxChartRef,
    diskRChartRef,
    diskWChartRef,
    cpuTextRef,
    ramTextRef,
    gpuTextRef,
    vramTextRef,
    netRxTextRef,
    netTxTextRef,
    diskRTextRef,
    diskWTextRef,
    cpuCoresContainerRef,
  } = refs;

  return (
    <div className="flex flex-col gap-3 p-3 bg-zinc-950 border-t border-border/10 shrink-0">
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest mb-1">
          System Telemetry
        </span>
        <span className="text-[10px] font-bold text-zinc-400">
          {latestRef.current.machineIdentifier}
        </span>
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
