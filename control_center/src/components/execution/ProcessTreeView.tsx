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
} from "lucide-react";

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
  const statusColor = isZombie
    ? "text-red-500 bg-red-500/10"
    : process.status === "Running"
      ? "text-green-500 bg-green-500/10"
      : "text-zinc-400 bg-zinc-800";

  return (
    <div className="flex flex-col">
      <div
        className="flex items-center group hover:bg-white/5 py-1 px-2 rounded-sm transition-colors text-[10px] font-mono"
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
      >
        <div className="flex-1 flex items-center min-w-0 pr-2 overflow-hidden text-ellipsis whitespace-nowrap">
          {depth > 0 && <span className="mr-2 opacity-80" style={{ color: runColor }}>└─</span>}
          {process.children && process.children.length > 0 && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="mr-1 p-0.5 rounded hover:bg-white/10"
              style={{ color: runColor }}
            >
              {expanded ? (
                <ChevronDown className="w-3 h-3" />
              ) : (
                <ChevronRight className="w-3 h-3" />
              )}
            </button>
          )}
          <span className="text-zinc-300 font-medium truncate mr-2">
            {process.name}
          </span>
          <span className="text-zinc-600 text-[9px] truncate">
            [{process.pid}]
          </span>
        </div>

        <div className="flex items-center space-x-3 shrink-0">
          <div className="flex items-center w-16 text-zinc-400">
            <Cpu className="w-3 h-3 mr-1 opacity-50" />
            <span>{process.cpu_usage.toFixed(1)}%</span>
          </div>
          <div className="flex items-center w-16 text-zinc-400">
            <HardDrive className="w-3 h-3 mr-1 opacity-50" />
            <span>{process.memory_mb.toFixed(0)} MB</span>
          </div>
          <div
            className={`px-1.5 py-0.5 rounded border border-transparent ${isZombie ? "border-red-500/20 font-bold" : ""} ${statusColor} text-[9px] uppercase min-w-[50px] text-center`}
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
            <ProcessNode key={child.pid} process={child} runColor={runColor} depth={depth + 1} />
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
          {jobs.map((job) => {
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

                <div className="p-1">
                  {job.root_process ? (
                    <ProcessNode process={job.root_process} runColor={runColor} />
                  ) : (
                    <div className="py-2 text-center text-[10px] text-zinc-600 italic">
                      Process initializing or unreachable...
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
