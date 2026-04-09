import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Toggle } from "@/components/ui/toggle";
import {
  VscEdit,
  VscClearAll,
  VscTrash,
  VscPlay,
  VscDebugStop,
  VscDebugPause,
  VscCopy,
  VscChevronDown,
  VscChevronRight,
  VscPulse,
  VscCircuitBoard,
  VscServer,
  VscFlame,
  VscCheckAll,
} from "react-icons/vsc";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { HydraConfigViewer } from "@/components/execution/HydraConfigViewer";
import { EditableConfigViewer } from "@/components/execution/EditableConfigViewer";
import type { MetricRow } from "@/bindings/MetricRow";
import { useAppStore } from "@/store/useAppStore";

export function RunsSidebarList({
  filterType,
}: {
  filterType?: "SINGLE" | "TUNING_TRIAL" | "STUDY";
}) {
  const allRuns = useAppStore((state) => state.runs);
  const runs = filterType
    ? allRuns.filter((r) => r.type === filterType)
    : allRuns.filter((r) => r.type !== "STUDY");

  const isStudies = filterType === "STUDY";
  const selectedRunId = useAppStore((state) => state.selectedRunId);
  const setSelectedRunId = useAppStore((state) => state.setSelectedRunId);
  const selectedDashboardRuns = useAppStore(
    (state) => state.selectedDashboardRuns,
  );
  const setSelectedDashboardRuns = useAppStore(
    (state) => state.setSelectedDashboardRuns,
  );
  const toggleDashboardRun = useAppStore((state) => state.toggleDashboardRun);
  const runColors = useAppStore((state) => state.runColors);
  const setRunColors = useAppStore((state) => state.setRunColors);
  const setRunToRename = useAppStore((state) => state.setRunToRename);
  const setNewName = useAppStore((state) => state.setNewName);
  const setRunToFlush = useAppStore((state) => state.setRunToFlush);
  const setRunToDelete = useAppStore((state) => state.setRunToDelete);
  const handleEngineCmd = useAppStore((state) => state.handleEngineCmd);
  const handleClone = useAppStore((state) => state.handleClone);

  const defaultColors = [
    "#10b981",
    "#3b82f6",
    "#f59e0b",
    "#8b5cf6",
    "#ec4899",
    "#ef4444",
    "#14b8a6",
  ];

  const [expandedRunId, setExpandedRunId] = useState<string | null>(null);
  const [liveMetrics, setLiveMetrics] = useState<MetricRow | null>(null);

  // Toggle All Logic
  const visibleRunIds = runs.map((r) => r.id);
  const allSelected =
    visibleRunIds.length > 0 &&
    visibleRunIds.every((id) => selectedDashboardRuns.includes(id));

  const handleToggleAll = () => {
    if (allSelected) {
      setSelectedDashboardRuns(
        selectedDashboardRuns.filter((id) => !visibleRunIds.includes(id)),
      );
    } else {
      const newSelected = [
        ...new Set([...selectedDashboardRuns, ...visibleRunIds]),
      ];
      setSelectedDashboardRuns(newSelected);
    }
  };

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
          run_id: expandedRunId,
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
    <div className="flex flex-col flex-1 overflow-hidden">
      {/* Tools / Toggle Bar */}
      <div className="flex items-center justify-between px-3 py-1 bg-[#0a0a0a] border-b border-white/5 shrink-0 select-none">
        <span className="text-[8px] font-bold text-zinc-500 uppercase tracking-widest">
          {runs.length} Active {isStudies ? "Studies" : "Runs"}
        </span>
        {runs.length > 0 && (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleToggleAll}
            className="h-5 px-1.5 py-0 text-[8px] text-zinc-400 hover:text-white uppercase tracking-widest hover:bg-white/5"
          >
            {allSelected ? (
              <VscClearAll className="w-3.5 h-3.5 mr-1" />
            ) : (
              <VscCheckAll className="w-3.5 h-3.5 mr-1" />
            )}
            {allSelected ? "Deselect All" : "Select All"}
          </Button>
        )}
      </div>

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
                className="border-b border-white/5 relative flex flex-col bg-[#050505]"
              >
                {/* Dense List Item */}
                <div
                  className={`px-2 py-1.5 group cursor-pointer transition-colors flex flex-col gap-1 ${isSelected ? "bg-primary/5" : "hover:bg-white/[0.02]"}`}
                  style={{
                    borderLeftWidth: "2px",
                    borderLeftColor: isSelected ? runColor : "transparent",
                  }}
                  onClick={() => {
                    setSelectedRunId(run.id);
                    setExpandedRunId(expandedRunId === run.id ? null : run.id);
                  }}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5 overflow-hidden flex-1 pr-2">
                      {expandedRunId === run.id ? (
                        <VscChevronDown className="w-3.5 h-3.5 text-zinc-500 shrink-0" />
                      ) : (
                        <VscChevronRight className="w-3.5 h-3.5 text-zinc-500 shrink-0" />
                      )}
                      <div className="flex flex-col flex-1 min-w-0">
                        <div className="flex items-center gap-1.5">
                          <h3
                            className="font-bold text-[10px] uppercase tracking-wider truncate"
                            style={{ color: isSelected ? runColor : "#e4e4e7" }}
                          >
                            {run.name}
                          </h3>
                          {run.tag && (
                            <span
                              className="px-1 py-[1px] rounded-sm bg-white/10 text-zinc-300 hover:bg-white/20 cursor-pointer text-[7px] font-black tracking-widest"
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
                        <p
                          className={`text-[8.5px] font-mono font-bold tracking-widest ${run.status === "RUNNING" ? "text-emerald-400" : "text-zinc-600"}`}
                        >
                          {run.type === "TUNING_TRIAL"
                            ? "TRIAL"
                            : isStudies
                              ? "STUDY"
                              : run.type}{" "}
                          · {run.status}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-1.5 shrink-0">
                      <input
                        type="color"
                        value={runColor}
                        onChange={(e) =>
                          setRunColors((prev) => ({
                            ...prev,
                            [run.id]: e.target.value,
                          }))
                        }
                        className="w-3.5 h-3.5 p-0 border-0 rounded cursor-pointer bg-transparent"
                        onClick={(e) => e.stopPropagation()}
                      />
                      <Toggle
                        pressed={isDashboardVisible}
                        onPressedChange={(p) => toggleDashboardRun(run.id, p)}
                        size="sm"
                        className="h-5 px-1.5 text-[8.5px] font-bold uppercase tracking-wider border rounded-sm"
                        style={{
                          backgroundColor: isDashboardVisible
                            ? `${runColor}20`
                            : "transparent",
                          color: isDashboardVisible ? runColor : "#52525b",
                          borderColor: isDashboardVisible
                            ? `${runColor}50`
                            : "rgba(255,255,255,0.1)",
                        }}
                        onClick={(e) => e.stopPropagation()}
                      >
                        MATCH
                      </Toggle>
                    </div>
                  </div>
                </div>

                {expandedRunId === run.id && (
                  <div className="bg-[#020202] border-t border-white/5 p-2 flex flex-col gap-2 animate-in slide-in-from-top-1 duration-150">
                    {/* Action Buttons */}
                    <div className="grid grid-cols-4 gap-1">
                      {run.status === "RUNNING" ? (
                        <>
                          <Button
                            variant="secondary"
                            size="sm"
                            className="h-6 text-[8px] bg-zinc-800 hover:bg-zinc-700 px-1 font-bold tracking-widest uppercase col-span-2"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleEngineCmd(run.id, "stop", false);
                            }}
                          >
                            <VscDebugPause className="w-3 h-3 mr-1" /> Pause
                          </Button>
                          <Button
                            variant="destructive"
                            size="sm"
                            className="h-6 text-[8px] px-1 font-bold tracking-widest uppercase col-span-2"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleEngineCmd(run.id, "stop", true);
                            }}
                          >
                            <VscDebugStop className="w-3 h-3 mr-1" /> Kill
                          </Button>
                        </>
                      ) : (
                        <>
                          <Button
                            variant="default"
                            size="sm"
                            className="h-6 text-[8px] px-1 font-bold tracking-widest uppercase col-span-4"
                            style={{
                              backgroundColor: `${runColor}30`,
                              color: runColor,
                              borderColor: `${runColor}50`,
                              borderWidth: "1px",
                            }}
                            onClick={(e) => {
                              e.stopPropagation();
                              handleEngineCmd(run.id, "start");
                            }}
                          >
                            <VscPlay className="w-3 h-3 mr-1" /> Start Run
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-6 text-[8px] px-1 font-bold tracking-widest uppercase border-white/10 hover:bg-white/5"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleClone(run);
                            }}
                          >
                            <VscCopy className="w-3 h-3" />
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-6 text-[8px] px-1 font-bold tracking-widest uppercase border-white/10 hover:bg-white/5"
                            onClick={(e) => {
                              e.stopPropagation();
                              setRunToRename(run.id);
                              setNewName(run.name);
                            }}
                          >
                            <VscEdit className="w-3 h-3" />
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-6 text-[8px] px-1 font-bold tracking-widest uppercase text-yellow-500 border-white/10 hover:bg-yellow-500/10 hover:border-yellow-500/30"
                            onClick={(e) => {
                              e.stopPropagation();
                              setRunToFlush(run.id);
                            }}
                          >
                            <VscClearAll className="w-3 h-3" />
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-6 text-[8px] px-1 font-bold tracking-widest uppercase text-red-500 border-white/10 hover:bg-red-500/10 hover:border-red-500/30"
                            onClick={(e) => {
                              e.stopPropagation();
                              setRunToDelete(run.id);
                            }}
                          >
                            <VscTrash className="w-3 h-3" />
                          </Button>
                        </>
                      )}
                    </div>

                    {run.status === "RUNNING" && liveMetrics && (
                      <TooltipProvider>
                        <div className="grid grid-cols-2 gap-1 p-1.5 bg-black rounded shadow-inner border border-white/5 text-[9px] font-mono text-zinc-400">
                          <Tooltip delayDuration={300}>
                            <TooltipTrigger asChild>
                              <div className="flex items-center justify-between gap-1 cursor-help hover:text-zinc-200">
                                <span className="flex items-center gap-1 opacity-70">
                                  <VscCircuitBoard className="w-3 h-3 text-blue-400" />{" "}
                                  CPU
                                </span>
                                <span className="font-bold text-zinc-300">
                                  {Number(liveMetrics.cpu_usage_pct || 0).toFixed(
                                    1,
                                  )}
                                  %
                                </span>
                              </div>
                            </TooltipTrigger>
                            <TooltipContent className="text-[10px] bg-zinc-900 border-white/10 font-sans">
                              CPU execution usage for all parallel workers
                            </TooltipContent>
                          </Tooltip>

                          <Tooltip delayDuration={300}>
                            <TooltipTrigger asChild>
                              <div className="flex items-center justify-between gap-1 cursor-help hover:text-zinc-200">
                                <span className="flex items-center gap-1 opacity-70">
                                  <VscFlame className="w-3 h-3 text-orange-500" />{" "}
                                  GPU
                                </span>
                                <span className="font-bold text-zinc-300">
                                  {Number(liveMetrics.gpu_usage_pct || 0).toFixed(
                                    1,
                                  )}
                                  %
                                </span>
                              </div>
                            </TooltipTrigger>
                            <TooltipContent className="text-[10px] bg-zinc-900 border-white/10 font-sans">
                              GPU tensor saturation for the PyTorch core
                            </TooltipContent>
                          </Tooltip>

                          <Tooltip delayDuration={300}>
                            <TooltipTrigger asChild>
                              <div className="flex items-center justify-between gap-1 cursor-help hover:text-zinc-200">
                                <span className="flex items-center gap-1 opacity-70">
                                  <VscServer className="w-3 h-3 text-purple-400" />{" "}
                                  RAM
                                </span>
                                <span className="font-bold text-zinc-300">
                                  {Number(liveMetrics.ram_usage_mb || 0).toFixed(
                                    0,
                                  )}{" "}
                                  MB
                                </span>
                              </div>
                            </TooltipTrigger>
                            <TooltipContent className="text-[10px] bg-zinc-900 border-white/10 font-sans">
                              Total buffer memory and process footprint
                            </TooltipContent>
                          </Tooltip>

                          <Tooltip delayDuration={300}>
                            <TooltipTrigger asChild>
                              <div className="flex items-center justify-between gap-1 cursor-help hover:text-zinc-200">
                                <span className="flex items-center gap-1 opacity-70">
                                  <VscPulse className="w-3 h-3 text-emerald-400" />{" "}
                                  GL
                                </span>
                                <span className="font-bold text-zinc-300">
                                  {Number(
                                    liveMetrics.mcts_depth_mean || 0,
                                  ).toFixed(1)}
                                </span>
                              </div>
                            </TooltipTrigger>
                            <TooltipContent className="text-[10px] bg-zinc-900 border-white/10 font-sans">
                              Mean Search Depth mapping trajectory horizons
                            </TooltipContent>
                          </Tooltip>
                        </div>
                      </TooltipProvider>
                    )}

                    {run.config && (
                      <div className="mt-1">
                        {run.status === "WAITING" ? (
                          <EditableConfigViewer run={run} />
                        ) : (
                          <HydraConfigViewer configStr={run.config} />
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );
}
