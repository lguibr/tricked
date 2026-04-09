import React, { useState, useEffect, useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { VscLayout } from "react-icons/vsc";
import type { ProcessInfo } from "@/bindings/ProcessInfo";
import { getProcessColorVariation } from "@/lib/utils";
import { useAppStore } from "@/store/useAppStore";

export function CpuSunburstChart() {
  const jobs = useAppStore((state) => state.activeJobs);
  const runColors = useAppStore((state) => state.runColors);
  const [isSunburst, setIsSunburst] = useState(false);

  const [throttledJobs, setThrottledJobs] = useState(jobs);
  const jobsRef = React.useRef(jobs);

  useEffect(() => {
    jobsRef.current = jobs;
  }, [jobs]);

  useEffect(() => {
    // Throttle the topography updates to once every 2.5 seconds
    // to prevent aggressive screen flashing caused by notMerge={true}
    const interval = setInterval(() => setThrottledJobs(jobsRef.current), 2500);
    return () => clearInterval(interval);
  }, []);

  const data = useMemo(() => {
    const idSet = new Set<string>();
    return throttledJobs
      .map((job) => {
        const color = runColors[job.id] || "#3b82f6";

        const mapProcess = (proc: ProcessInfo, depth: number): any => {
          const selfCpu = Math.max(0.1, Number(proc.cpu_usage) || 0);
          const baseName = proc.name || "Unknown";
          const nodeName = `${baseName} [${proc.pid}]`;
          const nodeColor = getProcessColorVariation(color, baseName);

          let currentId = `${job.id}-${proc.pid}`;
          let counter = 1;
          while (idSet.has(currentId)) {
            currentId = `${job.id}-${proc.pid}-${counter}`;
            counter++;
          }
          idSet.add(currentId);

          const children =
            proc.children
              ?.filter((c) => c.cpu_usage > 0.1 || c.children.length > 0)
              .map((c) => mapProcess(c, depth + 1)) || [];

          if (children.length === 0) {
            return {
              id: currentId,
              name: nodeName,
              value: selfCpu,
              itemStyle: { color: nodeColor },
            };
          }

          const selfNodeId = `${currentId}-self`;
          idSet.add(selfNodeId);

          const selfNode = {
            id: selfNodeId,
            name: "Self",
            value: selfCpu,
            itemStyle: { color: getProcessColorVariation(color, "self") },
          };

          return {
            id: currentId,
            name: nodeName,
            itemStyle: { color: nodeColor },
            children: [selfNode, ...children],
          };
        };

        return job.root_process ? mapProcess(job.root_process, 0) : null;
      })
      .filter(
        (d) => d && (d.value > 0 || (d.children && d.children.length > 0)),
      );
  }, [JSON.stringify(throttledJobs)]);

  const getLevelOption = () => {
    return [
      {
        itemStyle: {
          borderColor: "#777",
          borderWidth: 0,
          gapWidth: 1,
        },
        upperLabel: {
          show: false,
        },
      },
      {
        itemStyle: {
          borderColor: "#333",
          borderWidth: 2,
          gapWidth: 1,
        },
        emphasis: {
          itemStyle: {
            borderColor: "#666",
          },
        },
      },
      {
        colorSaturation: [0.35, 0.5],
        itemStyle: {
          borderWidth: 2,
          gapWidth: 1,
          borderColorSaturation: 0.6,
        },
      },
    ];
  };

  const treemapSeries = {
    name: "CPU Topography",
    type: "treemap",
    progressive: 0,
    visibleMin: 0.1,
    universalTransition: true,
    animationDurationUpdate: 200,
    roam: false,
    label: {
      show: true,
      formatter: "{b}",
      fontSize: 9,
      fontWeight: "bold",
    },
    upperLabel: {
      show: true,
      height: 16,
      color: "#fff",
      fontSize: 9,
      fontWeight: "bold",
    },
    itemStyle: {
      borderColor: "#020202",
    },
    levels: getLevelOption(),
    data: data,
  };

  const sunburstSeries = {
    type: "sunburst",
    universalTransition: true,
    animationDurationUpdate: 200,
    data: data,
    radius: ["15%", "90%"],
    sort: null,
    itemStyle: {
      borderRadius: 2,
      borderWidth: 1,
      borderColor: "#020202",
    },
    label: {
      show: true,
      rotate: "radial",
      color: "#fff",
      fontSize: 9,
      fontWeight: "bold",
      minAngle: 12,
      formatter: (params: any) => {
        if (params.name === "Self") return "";
        return params.name;
      },
    },
  };

  const option = {
    backgroundColor: "transparent",
    tooltip: {
      trigger: "item",
      formatter: (info: any) => {
        const value = Number(info.value || 0).toFixed(1);
        const name = info.name === "Self" ? "Self" : info.name;
        return [
          '<div class="font-black tracking-widest text-[9px] uppercase mb-1 text-zinc-400">CPU Usage</div>',
          `<span class="text-white text-[10px] font-bold">${name}</span>: <span class="text-primary font-mono ml-1 text-[10px]">${value}%</span>`,
        ].join("");
      },
      backgroundColor: "rgba(5,5,5,0.9)",
      borderColor: "rgba(255,255,255,0.1)",
      textStyle: { color: "#fff", fontSize: 10 },
      padding: [4, 8],
    },
    series: [isSunburst ? sunburstSeries : treemapSeries],
  };

  return (
    <div className="w-full h-full flex flex-col bg-[#050505]">
      <div className="flex items-center justify-between px-2 py-1.5 border-b border-white/5 bg-[#080808] shrink-0">
        <span className="text-[9px] font-black uppercase tracking-widest text-zinc-400">
          CPU Topography
        </span>
        <button
          onClick={() => setIsSunburst(!isSunburst)}
          className="text-zinc-500 hover:text-zinc-300 transition-colors p-1 bg-white/5 rounded-sm hover:bg-white/10"
          title="Toggle Sunburst/Treemap View"
        >
          <VscLayout className="w-3.5 h-3.5" />
        </button>
      </div>
      <div className="flex-1 w-full relative overflow-hidden p-1">
        {data.length === 0 ? (
          <div className="absolute inset-0 flex items-center justify-center text-zinc-600 text-[10px] font-black uppercase tracking-widest">
            No Telemetry
          </div>
        ) : (
          <ReactECharts
            option={option}
            style={{ height: "100%", width: "100%" }}
            opts={{ renderer: "canvas" }}
            notMerge={true}
          />
        )}
      </div>
    </div>
  );
}
