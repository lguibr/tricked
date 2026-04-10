import { useEffect, useRef } from "react";
import ReactECharts from "echarts-for-react";
import type { HardwareMetrics } from "@/bindings/HardwareMetrics";

const isTauri =
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

export function useHardwareTelemetry() {
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
    machine_identifier: "CPU / GPU",
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

      if (cpuTextRef.current) cpuTextRef.current.innerText = `${Math.floor(latest.cpu_usage)}%`;
      if (ramTextRef.current) ramTextRef.current.innerText = `${Math.floor(latest.ram_usage_pct)}%`;
      if (gpuTextRef.current) gpuTextRef.current.innerText = `${Math.floor(latest.gpu_util)}%`;
      if (vramTextRef.current) vramTextRef.current.innerText = `${(latest.vram_used_mb / 1024).toFixed(1)}G`;
      if (netRxTextRef.current) netRxTextRef.current.innerText = `${latest.network_rx_mbps.toFixed(1)}M/s`;
      if (netTxTextRef.current) netTxTextRef.current.innerText = `${latest.network_tx_mbps.toFixed(1)}M/s`;
      if (diskRTextRef.current) diskRTextRef.current.innerText = `${latest.disk_read_mbps.toFixed(1)}M/s`;
      if (diskWTextRef.current) diskWTextRef.current.innerText = `${latest.disk_write_mbps.toFixed(1)}M/s`;

      const updateChart = (ref: React.RefObject<any>, data: number[]) => {
        if (ref.current) {
          const instance = ref.current.getEchartsInstance();
          if (instance && !instance.isDisposed()) {
            instance.setOption({ series: [{ data }] });
          }
        }
      };

      if (history.length > 0) {
        updateChart(cpuChartRef, history.map((m) => m.cpu_usage));
        updateChart(ramChartRef, history.map((m) => m.ram_usage_pct));
        updateChart(gpuChartRef, history.map((m) => m.gpu_util));
        updateChart(vramChartRef, history.map((m) => m.vram_used_mb));
        updateChart(netRxChartRef, history.map((m) => m.network_rx_mbps));
        updateChart(netTxChartRef, history.map((m) => m.network_tx_mbps));
        updateChart(diskRChartRef, history.map((m) => m.disk_read_mbps));
        updateChart(diskWChartRef, history.map((m) => m.disk_write_mbps));
      }

      if (cpuCoresContainerRef.current && latest.cpu_cores_usage && latest.cpu_cores_usage.length > 0) {
        const container = cpuCoresContainerRef.current;
        const cores = latest.cpu_cores_usage;
        if (container.children.length !== cores.length) {
          container.innerHTML = "";
          for (let i = 0; i < cores.length; i++) {
            const childWrapper = document.createElement("div");
            childWrapper.className = "flex-1 bg-zinc-800/50 rounded-t-[1px] h-full flex flex-col justify-end overflow-hidden";
            const childInner = document.createElement("div");
            childInner.className = "w-full rounded-t-[1px] transition-all duration-300";
            childWrapper.appendChild(childInner);
            container.appendChild(childWrapper);
          }
        }
        for (let i = 0; i < cores.length; i++) {
          const usage = cores[i];
          const color = usage > 80 ? "#ef4444" : usage > 50 ? "#f59e0b" : "#3b82f6";
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
        if (cpuCoresContainerRef.current) cpuCoresContainerRef.current.innerHTML = "";
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

  return {
    latestRef,
    refs: {
      cpuChartRef, ramChartRef, gpuChartRef, vramChartRef, netRxChartRef, netTxChartRef, diskRChartRef, diskWChartRef,
      cpuTextRef, ramTextRef, gpuTextRef, vramTextRef, netRxTextRef, netTxTextRef, diskRTextRef, diskWTextRef,
      cpuCoresContainerRef
    }
  };
}
