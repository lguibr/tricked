import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Toggle } from "@/components/ui/toggle";
import {
  Edit2,
  Eraser,
  Trash2,
  Play,
  Square,
  Pause,
  Copy,
  ChevronDown,
  ChevronRight,
  Activity,
  Cpu,
  HardDrive,
  Zap,
} from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { HydraConfigViewer } from "@/components/execution/HydraConfigViewer";
import type { Run } from "@/bindings/Run";
import type { MetricRow } from "@/bindings/MetricRow";

interface RunsSidebarListProps {
  runs: Run[];
  selectedRunId: string | null;
  setSelectedRunId: (id: string) => void;
  selectedDashboardRuns: string[];
  setSelectedDashboardRuns: React.Dispatch<React.SetStateAction<string[]>>;
  toggleDashboardRun: (id: string, pressed: boolean) => void;
  runColors: Record<string, string>;
  setRunColors: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  defaultColors: string[];
  setRunToRename: (id: string) => void;
  setNewName: (name: string) => void;
  setRunToFlush: (id: string) => void;
  setRunToDelete: (id: string) => void;
  handleEngineCmd: (runId: string, cmd: string, force?: boolean) => void;
  handleClone: (run: any) => void;
}

export function RunsSidebarList({
  runs,
  selectedRunId,
  setSelectedRunId,
  selectedDashboardRuns,
  setSelectedDashboardRuns,
  toggleDashboardRun,
  runColors,
  setRunColors,
  defaultColors,
  setRunToRename,
  setNewName,
  setRunToFlush,
  setRunToDelete,
  handleEngineCmd,
  handleClone,
}: RunsSidebarListProps) {
  const [expandedRunId, setExpandedRunId] = useState<string | null>(null);
  const [liveMetrics, setLiveMetrics] = useState<MetricRow | null>(null);

  // Fetch metrics for expanded run if it's running
  useEffect(() => {
    if (!expandedRunId) return;
    const run = runs.find((r) => r.id === expandedRunId);
    if (run?.status !== "RUNNING") {
      setLiveMetrics(null);
      return;
    }

    let active = true;
    const fetchStats = async () => {
      try {
        const data = await invoke<MetricRow[]>("get_run_metrics", {
          id: expandedRunId,
        });
        if (active && data.length > 0) {
          setLiveMetrics(data[data.length - 1]); // get latest
        }
      } catch (e) { }
    };
    fetchStats();
    const interval = setInterval(fetchStats, 2000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [expandedRunId, runs]);

  return (
    <ScrollArea className="flex-1 p-0">
      <div className="flex flex-col">
        {runs.map((run, idx) => {
          const runColor =
            runColors[run.id] || defaultColors[idx % defaultColors.length];
          const isSelected = selectedRunId === run.id;
          const isDashboardVisible = selectedDashboardRuns.includes(run.id);

          return (
            <div
              key={run.id}
              className={`border-b border-border/30 relative flex flex-col`}
            >
              <div
                className={`px-3 py-2 group cursor-pointer transition-colors flex flex-col gap-2 ${isSelected ? "bg-primary/5 border-l-2 border-l-primary" : "border-l-2 border-l-transparent hover:bg-accent"}`}
                onClick={() => {
                  setSelectedRunId(run.id);
                  setExpandedRunId(expandedRunId === run.id ? null : run.id);
                }}
              >
                <div className="flex items-start justify-between">
                  <div className="flex flex-col gap-1 overflow-hidden pr-2">
                    <div className="flex items-center gap-2">
                      {expandedRunId === run.id ? (
                        <ChevronDown className="w-3 h-3 text-zinc-500 shrink-0" />
                      ) : (
                        <ChevronRight className="w-3 h-3 text-zinc-500 shrink-0" />
                      )}
                      <h3
                        className={`font-medium text-xs leading-tight truncate ${isSelected ? "text-primary" : ""}`}
                      >
                        {run.name}
                      </h3>
                    </div>
                    <div className="flex items-center gap-2 pl-5">
                      <p
                        className={`text-[10px] font-mono ${run.status === "RUNNING" ? "text-green-500 animate-pulse" : "text-muted-foreground"}`}
                      >
                        {run.type.substring(0, 1)} · {run.status}
                      </p>
                      {run.tag && (
                        <span
                          className="px-1.5 py-0.5 rounded bg-secondary text-secondary-foreground hover:bg-primary hover:text-white cursor-pointer transition-colors text-[8px] font-bold tracking-wider relative z-10"
                          onClick={(e) => {
                            e.stopPropagation();
                            const ids = runs
                              .filter((r) => r.tag === run.tag)
                              .map((r) => r.id);
                            setSelectedDashboardRuns(ids);
                          }}
                        >
                          {run.tag}
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    <input
                      type="color"
                      value={runColor}
                      onChange={(e) =>
                        setRunColors((prev) => ({
                          ...prev,
                          [run.id]: e.target.value,
                        }))
                      }
                      className="w-5 h-5 p-0 border-0 rounded cursor-pointer bg-transparent"
                      onClick={(e) => e.stopPropagation()}
                    />
                    <Toggle
                      pressed={isDashboardVisible}
                      onPressedChange={(p) => toggleDashboardRun(run.id, p)}
                      size="sm"
                      className="h-5 px-1.5 text-[9px] data-[state=on]:bg-primary/20 data-[state=on]:text-primary border border-zinc-800 data-[state=on]:border-primary/50 text-zinc-400"
                      onClick={(e) => e.stopPropagation()}
                    >
                      Graph Match
                    </Toggle>
                  </div>
                </div>
              </div>

              {expandedRunId === run.id && (
                <div className="bg-black/50 border-t border-border/10 p-3 flex flex-col gap-3 text-xs animate-in slide-in-from-top-2 duration-200">
                  <div className="flex flex-wrap gap-2">
                    <Button
                      variant="default"
                      size="sm"
                      className="h-7 text-[10px]"
                      disabled={run.status === "RUNNING"}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleEngineCmd(run.id, "start");
                      }}
                    >
                      <Play className="w-3 h-3 mr-1" /> Start
                    </Button>
                    {run.status === "RUNNING" && (
                      <>
                        <Button
                          variant="secondary"
                          size="sm"
                          className="h-7 text-[10px] bg-zinc-800"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleEngineCmd(run.id, "stop", false);
                          }}
                        >
                          <Pause className="w-3 h-3 mr-1" /> Stop
                        </Button>
                        <Button
                          variant="destructive"
                          size="sm"
                          className="h-7 text-[10px]"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleEngineCmd(run.id, "stop", true);
                          }}
                        >
                          <Square className="w-3 h-3 mr-1" /> Kill
                        </Button>
                      </>
                    )}
                    {run.status !== "RUNNING" && (
                      <>
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-7 text-[10px]"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleClone(run);
                          }}
                        >
                          <Copy className="w-3 h-3 mr-1" /> Clone
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-7 text-[10px]"
                          onClick={(e) => {
                            e.stopPropagation();
                            setRunToRename(run.id);
                            setNewName(run.name);
                          }}
                        >
                          <Edit2 className="w-3 h-3 mr-1" /> Rename
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-7 text-[10px] text-yellow-500"
                          onClick={(e) => {
                            e.stopPropagation();
                            setRunToFlush(run.id);
                          }}
                        >
                          <Eraser className="w-3 h-3 mr-1" /> Flush
                        </Button>
                        <Button
                          variant="destructive"
                          size="sm"
                          className="h-7 text-[10px]"
                          onClick={(e) => {
                            e.stopPropagation();
                            setRunToDelete(run.id);
                          }}
                        >
                          <Trash2 className="w-3 h-3 mr-1" /> Delete
                        </Button>
                      </>
                    )}
                  </div>

                  {run.status === "RUNNING" && liveMetrics && (
                    <TooltipProvider>
                      <div className="grid grid-cols-2 gap-2 p-2 bg-zinc-950 rounded-md border border-border/20 text-[10px] font-mono text-zinc-400">
                        <Tooltip delayDuration={300}>
                          <TooltipTrigger asChild>
                            <div className="flex items-center justify-between gap-1 cursor-help hover:text-zinc-300">
                              <span className="flex items-center gap-1"><Cpu className="w-3 h-3 text-blue-400" /> CPU:</span>
                              <span className="font-bold text-zinc-300">{Number(liveMetrics.cpu_usage_pct || 0).toFixed(1)}%</span>
                            </div>
                          </TooltipTrigger>
                          <TooltipContent className="text-xs bg-zinc-900 border-border/50">CPU execution usage for all parallel workers</TooltipContent>
                        </Tooltip>

                        <Tooltip delayDuration={300}>
                          <TooltipTrigger asChild>
                            <div className="flex items-center justify-between gap-1 cursor-help hover:text-zinc-300">
                              <span className="flex items-center gap-1"><Zap className="w-3 h-3 text-green-400" /> GPU:</span>
                              <span className="font-bold text-zinc-300">{Number(liveMetrics.gpu_usage_pct || 0).toFixed(1)}%</span>
                            </div>
                          </TooltipTrigger>
                          <TooltipContent className="text-xs bg-zinc-900 border-border/50">GPU tensor saturation for the PyTorch core</TooltipContent>
                        </Tooltip>

                        <Tooltip delayDuration={300}>
                          <TooltipTrigger asChild>
                            <div className="flex items-center justify-between gap-1 cursor-help hover:text-zinc-300">
                              <span className="flex items-center gap-1"><HardDrive className="w-3 h-3 text-purple-400" /> RAM:</span>
                              <span className="font-bold text-zinc-300">{Number(liveMetrics.ram_usage_mb || 0).toFixed(0)}MB</span>
                            </div>
                          </TooltipTrigger>
                          <TooltipContent className="text-xs bg-zinc-900 border-border/50">Total buffer memory and process footprint</TooltipContent>
                        </Tooltip>

                        <Tooltip delayDuration={300}>
                          <TooltipTrigger asChild>
                            <div className="flex items-center justify-between gap-1 cursor-help hover:text-zinc-300">
                              <span className="flex items-center gap-1"><Activity className="w-3 h-3 text-orange-400" /> GL:</span>
                              <span className="font-bold text-zinc-300">{Number(liveMetrics.mcts_depth_mean || 0).toFixed(1)}</span>
                            </div>
                          </TooltipTrigger>
                          <TooltipContent className="text-xs bg-zinc-900 border-border/50">Mean Search Depth mapping trajectory horizons</TooltipContent>
                        </Tooltip>
                      </div>
                    </TooltipProvider>
                  )}

                  {run.config && <HydraConfigViewer configStr={run.config} />}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </ScrollArea>
  );
}
