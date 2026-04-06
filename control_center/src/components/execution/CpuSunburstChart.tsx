import { useState } from "react";
import ReactECharts from "echarts-for-react";
import { LayoutGrid } from "lucide-react";
import type { ActiveJob } from "@/bindings/ActiveJob";
import type { ProcessInfo } from "@/bindings/ProcessInfo";

interface CpuSunburstChartProps {
  jobs: ActiveJob[];
  runColors?: Record<string, string>;
}

export function CpuSunburstChart({
  jobs,
  runColors = {},
}: CpuSunburstChartProps) {
  const [isSunburst, setIsSunburst] = useState(false);

  const data = jobs
    .map((job) => {
      const color = runColors[job.id] || "#3b82f6";

      const mapProcess = (proc: ProcessInfo, depth: number): any => {
        const selfCpu = Math.max(0.1, proc.cpu_usage);
        const children =
          proc.children
            ?.filter((c) => c.cpu_usage > 0.1 || c.children.length > 0)
            .map((c) => mapProcess(c, depth + 1)) || [];

        if (children.length === 0) {
          return {
            id: proc.pid.toString(),
            name: proc.name,
            value: selfCpu,
            itemStyle: { color: color },
          };
        }

        const selfNode = {
          id: `${proc.pid}-self`,
          name: "Self",
          value: selfCpu,
          itemStyle: { color: color },
        };

        return {
          id: proc.pid.toString(),
          name: proc.name,
          itemStyle: { color: color },
          children: [selfNode, ...children],
        };
      };

      return job.root_process
        ? mapProcess(job.root_process, 0)
        : null;
    })
    .filter((d) => d && (d.value > 0 || (d.children && d.children.length > 0)));

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
    visibleMin: 0.1,
    universalTransition: true,
    animationDurationUpdate: 500,
    roam: false,
    label: {
      show: true,
      formatter: "{b}",
    },
    upperLabel: {
      show: true,
      height: 20,
      color: "#fff",
    },
    itemStyle: {
      borderColor: "#080808",
    },
    levels: getLevelOption(),
    data: data,
  };

  const sunburstSeries = {
    type: "sunburst",
    universalTransition: true,
    animationDurationUpdate: 500,
    data: data,
    radius: ["15%", "90%"],
    sort: null,
    itemStyle: {
      borderRadius: 3,
      borderWidth: 1.5,
      borderColor: "#080808",
    },
    label: {
      show: true,
      rotate: "radial",
      color: "#fff",
      fontSize: 10,
      fontWeight: 600,
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
        // Treemap tooltip styling:
        return [
          '<div class="font-bold text-xs mb-1 text-zinc-300">CPU Usage</div>',
          `<span class="text-white">${name}</span>: <span class="text-primary font-mono ml-1">${value}%</span>`,
        ].join("");
      },
      backgroundColor: "rgba(0,0,0,0.8)",
      borderColor: "#222",
      textStyle: { color: "#fff", fontSize: 12 },
    },
    series: [isSunburst ? sunburstSeries : treemapSeries],
  };

  return (
    <div className="w-full h-full flex flex-col bg-[#0a0a0a] border border-border/20 rounded-md">
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-zinc-950 shrink-0">
        <span className="text-[10px] font-bold uppercase tracking-widest text-zinc-400">
          CPU Topography
        </span>
        <button
          onClick={() => setIsSunburst(!isSunburst)}
          className="text-zinc-500 hover:text-zinc-300 transition-colors p-1 bg-zinc-900 rounded hover:bg-zinc-800"
          title="Toggle Sunburst/Treemap View"
        >
          <LayoutGrid className="w-3 h-3" />
        </button>
      </div>
      <div className="flex-1 w-full relative overflow-hidden">
        {data.length === 0 ? (
          <div className="absolute inset-0 flex items-center justify-center text-zinc-600 text-xs font-medium uppercase tracking-wider">
            No Telemetry
          </div>
        ) : (
          <ReactECharts
            option={option}
            style={{ height: "100%", width: "100%" }}
            opts={{ renderer: "canvas" }}
          />
        )}
      </div>
    </div>
  );
}
