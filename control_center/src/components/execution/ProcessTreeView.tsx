import type { ActiveJob } from "@/bindings/ActiveJob";
import type { ProcessInfo } from "@/bindings/ProcessInfo";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useState } from "react";
import {
  Server,
  Activity,
  Cpu,
  HardDrive,
  ChevronDown,
  ChevronRight,
  MonitorPlay,
  Settings,
  Database,
} from "lucide-react";
import { getProcessColorVariation } from "@/lib/utils";

interface ProcessTreeViewProps {
  jobs: ActiveJob[];
  runColors?: Record<string, string>;
}

const ProcessNode = ({
  process,
  runColor,
  depth = 0,
}: {
  process: ProcessInfo;
  runColor: string;
  depth?: number;
}) => {
  const [expanded, setExpanded] = useState(true);
  const isZombie = process.status.toLowerCase() === "zombie";
  const isRunning = process.status === "Running";
  const statusColor = isZombie
    ? "text-red-500 bg-red-500/10 border-red-500/20"
    : isRunning
      ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/20"
      : "text-zinc-500 bg-zinc-800 border-zinc-700/50";

  const nodeColor = getProcessColorVariation(runColor, process.name);
  const nameL = process.name.toLowerCase();

  let DynamicIcon = Cpu;
  if (nameL.includes('inference')) DynamicIcon = MonitorPlay;
  else if (nameL.includes('reanalyze')) DynamicIcon = Database;
  else if (nameL.includes('mcts')) DynamicIcon = Activity;
  else if (nameL.includes('prefetch')) DynamicIcon = HardDrive;
  else DynamicIcon = Settings;

  return (
    <div className="flex flex-col">
      <div
        className={`flex items-center group py-1.5 px-2 mb-[1px] rounded-sm transition-colors text-[11px] font-mono border-l-[3px] hover:bg-white/5 ${isRunning ? 'bg-white/[0.02]' : ''}`}
        style={{ paddingLeft: `${depth * 14 + 8}px`, borderLeftColor: depth === 0 ? nodeColor : 'transparent' }}
      >
        <div className="flex-1 flex items-center min-w-0 pr-2 overflow-hidden text-ellipsis whitespace-nowrap">
          {depth > 0 && (
            <span className="mr-2 opacity-50" style={{ color: runColor }}>
              └─
            </span>
          )}
          {process.children && process.children.length > 0 && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="mr-1.5 p-0.5 rounded hover:bg-white/10 transition-colors"
              style={{ color: nodeColor }}
            >
              {expanded ? (
                <ChevronDown className="w-3.5 h-3.5" />
              ) : (
                <ChevronRight className="w-3.5 h-3.5" />
              )}
            </button>
          )}
          <DynamicIcon className="w-3.5 h-3.5 mr-2 opacity-80" style={{ color: nodeColor }} />
          <span className="font-semibold truncate mr-2 tracking-wide" style={{ color: isRunning ? nodeColor : '#a1a1aa' }}>
            {process.name}
          </span>
          <span className="text-zinc-600 text-[10px] truncate bg-black/40 px-1 rounded">
            PID:{process.pid}
          </span>
        </div>

        <div className="flex items-center space-x-4 shrink-0 font-medium">
          <div className="flex items-center w-16" style={{ color: process.cpu_usage > 10 ? nodeColor : '#71717a' }}>
            <span>{process.cpu_usage.toFixed(1)}%</span>
            <span className="text-[9px] ml-1 opacity-50 uppercase">CPU</span>
          </div>
          <div className="flex items-center w-[72px] text-zinc-400">
            <span>{process.memory_mb.toFixed(0)}</span>
            <span className="text-[9px] ml-1 opacity-50 uppercase">MB</span>
          </div>
          <div
            className={`px-2 py-0.5 rounded border ${isZombie ? "font-bold" : ""} ${statusColor} text-[9px] uppercase min-w-[55px] text-center tracking-widest`}
          >
            {process.status}
          </div>
        </div>
      </div>

      {expanded && process.children && process.children.length > 0 && (
        <div
          className="flex flex-col border-l ml-2 mt-0.5 relative"
          style={{ borderColor: `${runColor}40` }}
        >
          {process.children.map((child) => (
            <ProcessNode
              key={child.pid}
              process={child}
              runColor={runColor}
              depth={depth + 1}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export function ProcessTreeView({
  jobs,
  runColors = {},
}: ProcessTreeViewProps) {
  if (!jobs || jobs.length === 0) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center text-zinc-600 gap-2 bg-[#080808]">
        <Server className="w-8 h-8 opacity-20" />
        <span className="text-xs font-medium uppercase tracking-wider">
          No Active Processes
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full w-full bg-[#080808] border-r border-border/20">
      <div className="flex items-center px-3 py-1.5 border-b border-white/10 bg-zinc-950/80 shrink-0">
        <Activity className="w-3 h-3 text-primary mr-2" />
        <span className="text-[10px] font-bold uppercase tracking-widest text-zinc-400">
          Process Tree
        </span>
      </div>

      <ScrollArea className="flex-1 w-full bg-black/40">
        <div className="p-2 space-y-3">
          {[...jobs]
            .sort((a, b) => {
              const aRunning = a.root_process?.status === "Running";
              const bRunning = b.root_process?.status === "Running";
              if (aRunning && !bRunning) return -1;
              if (!aRunning && bRunning) return 1;
              return b.id.localeCompare(a.id);
            })
            .map((job) => {
              const runColor = runColors[job.id] || "#3b82f6";
              return (
                <div
                  key={job.id}
                  className="border border-white/5 rounded bg-black/50 overflow-hidden"
                  style={{ borderLeft: `2px solid ${runColor}` }}
                >
                  <div className="flex items-center justify-between px-2 py-1.5 bg-zinc-900 border-b border-white/5">
                    <div className="flex items-center space-x-2 truncate">
                      <span
                        className="w-2 h-2 rounded-full animate-pulse"
                        style={{ backgroundColor: runColor }}
                      />
                      <span className="text-[11px] font-bold text-white truncate">
                        {job.name}
                      </span>
                      <span className="text-[9px] px-1 py-0.5 rounded bg-white/10 text-zinc-400 uppercase tracking-wider">
                        {job.job_type}
                      </span>
                    </div>
                  </div>

                  <div className="p-1.5 pt-0 bg-[#0c0c0e]">
                    {job.root_process ? (
                      <ProcessNode
                        process={job.root_process}
                        runColor={runColor}
                      />
                    ) : (
                      <div className="py-4 text-center text-[10px] text-zinc-600 uppercase tracking-widest border border-dashed border-white/5 rounded mt-1.5">
                        Process Initializing...
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
        </div>
      </ScrollArea>
    </div>
  );
}
