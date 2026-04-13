import { useEffect, useRef } from "react";
import ReactECharts from "echarts-for-react";
import { HardwareMetrics } from "@/bindings/proto/tricked";

export function useHardwareTelemetry() {
  const metricsHistoryRef = useRef<HardwareMetrics[]>([]);
  const latestRef = useRef<HardwareMetrics>({
    cpuUsage: 0,
    ramUsagePct: 0,
    gpuUtil: 0,
    vramUsedMb: 0,
    networkRxMbps: 0,
    networkTxMbps: 0,
    diskReadMbps: 0,
    diskWriteMbps: 0,
    cpuCoresUsage: [],
    ramUsedMb: 0,
    diskUsagePct: 0,
    machineIdentifier: "CPU / GPU",
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
    let ws: WebSocket;

    const connectWS = () => {
      ws = new WebSocket("ws://127.0.0.1:8000/api/ws/hardware");
      ws.binaryType = "arraybuffer";
      ws.onmessage = (e) => {
        if (isCancelled) return;
        try {
          const payload = HardwareMetrics.fromBinary(new Uint8Array(e.data));
          metricsHistoryRef.current.push(payload);
          if (metricsHistoryRef.current.length > 60) {
            metricsHistoryRef.current.shift();
          }
          latestRef.current = payload;
        } catch (err) {}
      };
      ws.onclose = () => {
        if (!isCancelled) setTimeout(connectWS, 2000);
      };
    };

    connectWS();

    return () => {
      isCancelled = true;
      if (ws) ws.close();
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
        cpuTextRef.current.innerText = `${Math.floor(latest.cpuUsage)}%`;
      if (ramTextRef.current)
        ramTextRef.current.innerText = `${Math.floor(latest.ramUsagePct)}%`;
      if (gpuTextRef.current)
        gpuTextRef.current.innerText = `${Math.floor(latest.gpuUtil)}%`;
      if (vramTextRef.current)
        vramTextRef.current.innerText = `${(latest.vramUsedMb / 1024).toFixed(1)}G`;
      if (netRxTextRef.current)
        netRxTextRef.current.innerText = `${latest.networkRxMbps.toFixed(1)}M/s`;
      if (netTxTextRef.current)
        netTxTextRef.current.innerText = `${latest.networkTxMbps.toFixed(1)}M/s`;
      if (diskRTextRef.current)
        diskRTextRef.current.innerText = `${latest.diskReadMbps.toFixed(1)}M/s`;
      if (diskWTextRef.current)
        diskWTextRef.current.innerText = `${latest.diskWriteMbps.toFixed(1)}M/s`;

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
          history.map((m) => m.cpuUsage),
        );
        updateChart(
          ramChartRef,
          history.map((m) => m.ramUsagePct),
        );
        updateChart(
          gpuChartRef,
          history.map((m) => m.gpuUtil),
        );
        updateChart(
          vramChartRef,
          history.map((m) => m.vramUsedMb),
        );
        updateChart(
          netRxChartRef,
          history.map((m) => m.networkRxMbps),
        );
        updateChart(
          netTxChartRef,
          history.map((m) => m.networkTxMbps),
        );
        updateChart(
          diskRChartRef,
          history.map((m) => m.diskReadMbps),
        );
        updateChart(
          diskWChartRef,
          history.map((m) => m.diskWriteMbps),
        );
      }

      if (
        cpuCoresContainerRef.current &&
        latest.cpuCoresUsage &&
        latest.cpuCoresUsage.length > 0
      ) {
        const container = cpuCoresContainerRef.current;
        const cores = latest.cpuCoresUsage;
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

  return {
    latestRef,
    refs: {
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
    },
  };
}
