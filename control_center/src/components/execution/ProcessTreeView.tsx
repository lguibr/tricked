import type { ProcessInfo } from "@/bindings/ProcessInfo";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useState } from "react";
import {
  VscServer,
  VscPulse,
  VscCircuitBoard,
  VscDatabase,
  VscChevronDown,
  VscChevronRight,
  VscPlayCircle,
  VscSettingsGear,
} from "react-icons/vsc";
import { getProcessColorVariation } from "@/lib/utils";
import { useAppStore } from "@/store/useAppStore";

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

  let DynamicIcon = VscCircuitBoard;
  if (nameL.includes("inference")) DynamicIcon = VscPlayCircle;
  else if (nameL.includes("reanalyze")) DynamicIcon = VscDatabase;
  else if (nameL.includes("mcts")) DynamicIcon = VscPulse;
  else if (nameL.includes("prefetch")) DynamicIcon = VscServer;
  else DynamicIcon = VscSettingsGear;

  return (
    <div className="flex flex-col">
      <div
        className={`flex items-center group py-1 px-1.5 mb-[1px] rounded-sm transition-colors text-[10px] font-mono border-l-2 hover:bg-white/5 ${isRunning ? "bg-white/[0.02]" : ""}`}
        style={{
          paddingLeft: `${depth * 10 + 4}px`,
          borderLeftColor: depth === 0 ? nodeColor : "transparent",
        }}
      >
        <div className="flex-1 flex items-center min-w-0 pr-2 overflow-hidden text-ellipsis whitespace-nowrap">
          {depth > 0 && (
            <span
              className="mr-1.5 opacity-40 font-bold"
              style={{ color: runColor }}
            >
              └─
            </span>
          )}
          {process.children && process.children.length > 0 && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="mr-1 p-0.5 rounded hover:bg-white/10 transition-colors"
              style={{ color: nodeColor }}
            >
              {expanded ? (
                <VscChevronDown className="w-3 h-3" />
              ) : (
                <VscChevronRight className="w-3 h-3" />
              )}
            </button>
          )}
          <DynamicIcon
            className="w-3 h-3 mr-1.5 opacity-80"
            style={{ color: nodeColor }}
          />
          <span
            className="font-bold truncate mr-1.5 tracking-wide"
            style={{ color: isRunning ? nodeColor : "#a1a1aa" }}
          >
            {process.name}
          </span>
          <span className="text-zinc-500 text-[8.5px] truncate bg-black/60 px-1 rounded-sm border border-white/5">
            PID:{process.pid}
          </span>
        </div>

        <div className="flex items-center space-x-3 shrink-0 font-medium">
          <div
            className="flex items-center w-14"
            style={{ color: process.cpu_usage > 10 ? nodeColor : "#71717a" }}
          >
            <span className="font-bold">{process.cpu_usage.toFixed(1)}%</span>
            <span className="text-[7.5px] ml-0.5 opacity-50 uppercase font-black">
              CPU
            </span>
          </div>
          <div className="flex items-center w-[60px] text-zinc-400">
            <span className="font-bold">{process.memory_mb.toFixed(0)}</span>
            <span className="text-[7.5px] ml-0.5 opacity-50 uppercase font-black">
              MB
            </span>
          </div>
          <div
            className={`px-1.5 py-[1px] rounded-[3px] border ${isZombie ? "font-bold" : ""} ${statusColor} text-[8px] font-black uppercase min-w-[48px] text-center tracking-widest`}
          >
            {process.status}
          </div>
        </div>
      </div>

      {expanded && process.children && process.children.length > 0 && (
        <div
          className="flex flex-col border-l border-dashed ml-1.5 mt-0.5 relative"
          style={{ borderColor: `${runColor}30` }}
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

export function ProcessTreeView() {
  const jobs = useAppStore((state) => state.activeJobs);
  const runColors = useAppStore((state) => state.runColors);
  if (!jobs || jobs.length === 0) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center text-zinc-600 gap-2 bg-[#050505] border-r border-white/5">
        <VscServer className="w-8 h-8 opacity-20" />
        <span className="text-[10px] font-black uppercase tracking-widest">
          No Active Processes
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full w-full bg-[#020202] border-r border-white/5">
      <div className="flex items-center px-2 py-1.5 border-b border-white/5 bg-[#050505] shrink-0">
        <VscPulse className="w-3.5 h-3.5 text-primary mr-1.5" />
        <span className="text-[9px] font-black uppercase tracking-widest text-zinc-400">
          Process Tree
        </span>
      </div>

      <ScrollArea className="flex-1 w-full bg-[#020202]">
        <div className="p-1.5 space-y-2">
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
                  className="border border-white/5 rounded-sm bg-black/80 overflow-hidden shadow-sm"
                  style={{ borderLeft: `2px solid ${runColor}` }}
                >
                  <div className="flex items-center justify-between px-2 py-1 bg-[#080808] border-b border-white/5">
                    <div className="flex items-center space-x-2 truncate">
                      <span
                        className="w-1.5 h-1.5 rounded-full animate-pulse shadow-[0_0_5px_rgba(255,255,255,0.2)]"
                        style={{ backgroundColor: runColor }}
                      />
                      <span className="text-[10px] font-black tracking-widest uppercase text-white truncate">
                        {job.name}
                      </span>
                      <span className="text-[8px] font-black px-1 py-0.5 rounded-sm bg-white/10 text-zinc-400 uppercase tracking-widest">
                        {job.job_type}
                      </span>
                    </div>
                  </div>

                  <div className="p-1 pt-0 bg-[#050505]">
                    {job.root_process ? (
                      <ProcessNode
                        process={job.root_process}
                        runColor={runColor}
                      />
                    ) : (
                      <div className="py-2 text-center text-[9px] font-black text-zinc-600 uppercase tracking-widest border border-dashed border-white/5 rounded-sm mt-1">
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
