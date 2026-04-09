import React, { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  ResizablePanel,
  ResizablePanelGroup,
  ResizableHandle,
} from "@/components/ui/resizable";
import { VscTerminal, VscCopy, VscCheck, VscClearAll } from "react-icons/vsc";

interface LiveLogsViewerProps {
  runs: any[];
  runLogs: Record<string, string[]>;
  handleCopyLogs: (id: string, logs: string) => void;
  copiedLogId: string | null;
  logsEndRef: React.MutableRefObject<Record<string, HTMLDivElement | null>>;
  runColors?: Record<string, string>;
}

export function LiveLogsViewer({
  runs,
  runLogs,
  handleCopyLogs,
  copiedLogId,
  logsEndRef,
  runColors = {},
}: LiveLogsViewerProps) {
  const [forceCollapsed, setForceCollapsed] = useState<string[]>([]);
  const [forceExpanded, setForceExpanded] = useState<string[]>([]);

  const activeRuns = runs.filter((r) => r.status !== "WAITING");

  const expandedRuns = activeRuns.filter((r) => {
    if (forceCollapsed.includes(r.id)) return false;
    if (forceExpanded.includes(r.id)) return true;
    return (runLogs[r.id]?.length || 0) > 0;
  });

  const collapsedRuns = activeRuns.filter((r) => !expandedRuns.includes(r));

  const handleExpand = (id: string) => {
    setForceCollapsed((prev) => prev.filter((x) => x !== id));
    setForceExpanded((prev) => [...new Set([...prev, id])]);
  };

  const handleCollapse = (id: string) => {
    setForceExpanded((prev) => prev.filter((x) => x !== id));
    setForceCollapsed((prev) => [...new Set([...prev, id])]);
  };

  return (
    <Card className="h-full w-full flex flex-col rounded-none shadow-none border-0 overflow-hidden shrink-0 bg-[#050505]">
      {/* Terminal Header */}
      <div className="flex items-center justify-between px-2 py-1 border-b border-border/20 bg-[#0a0a0a] shrink-0">
        <div className="flex items-center text-[9px] font-bold uppercase tracking-widest text-zinc-500">
          <VscTerminal className="w-3.5 h-3.5 text-primary mr-1.5" />
          <span>Stdout / Logs</span>
          {expandedRuns.length > 0 && (
            <>
              <div className="w-[1px] h-3 bg-white/10 mx-2" />
              <Button
                variant="ghost"
                size="sm"
                className="h-5 px-1.5 text-[9px] text-zinc-400 hover:text-white bg-white/5 hover:bg-white/10 rounded-sm uppercase tracking-wider"
                onClick={() => {
                  const allLogs = expandedRuns
                    .map((r) => {
                      return `--- ${r.name} ---\n${(runLogs[r.id] || []).join("\n")}`;
                    })
                    .join("\n\n");
                  handleCopyLogs("all", allLogs);
                }}
              >
                {copiedLogId === "all" ? (
                  <VscCheck className="w-3 h-3 text-green-500 mr-1" />
                ) : (
                  <VscCopy className="w-3 h-3 mr-1" />
                )}
                {copiedLogId === "all" ? "COPIED" : "COPY VISIBLE"}
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Terminal Body */}
      <div className="flex-1 flex overflow-x-auto overflow-y-hidden bg-[#030303] text-zinc-300 selection:bg-primary/30 selection:text-white">
        {expandedRuns.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center text-zinc-700 text-[10px] font-mono tracking-widest uppercase opacity-50 gap-2">
            <VscClearAll className="w-8 h-8" />
            NO LOGS YET
          </div>
        ) : (
          <ResizablePanelGroup direction="horizontal" className="flex-1">
            {expandedRuns.map((run, idx) => {
              const runId = run.id;
              const color = runColors[runId] || "#3b82f6";
              const lines = runLogs[runId] || [];

              return (
                <React.Fragment key={runId}>
                  <ResizablePanel
                    defaultSize={100 / expandedRuns.length}
                    minSize={10}
                  >
                    <div
                      className="h-full flex flex-col font-mono text-[9.5px] relative"
                      style={{ backgroundColor: `${color}05` }}
                    >
                      {/* Embedded Run Header */}
                      <div
                        className="sticky top-0 backdrop-blur-md px-2 py-1 mb-1 z-10 border-b flex items-center shadow-sm select-none group/header"
                        style={{
                          backgroundColor: `${color}15`,
                          borderColor: `${color}30`,
                          borderBottomWidth: "1px",
                        }}
                      >
                        <VscTerminal
                          className="w-3 h-3 mr-1.5"
                          style={{
                            color,
                            filter: `drop-shadow(0 0 2px ${color})`,
                          }}
                        />
                        <span
                          style={{ color, textShadow: `0 0 4px ${color}80` }}
                          className="font-bold tracking-widest uppercase flex-1 truncate"
                        >
                          {run.name}
                        </span>
                        <div className="flex items-center space-x-1 opacity-0 group-hover/header:opacity-100 transition-opacity">
                          <button
                            onClick={() =>
                              handleCopyLogs(runId, lines.join("\n"))
                            }
                            className="p-1 hover:bg-white/10 rounded"
                            title="Copy Terminal"
                          >
                            {copiedLogId === runId ? (
                              <VscCheck className="w-3 h-3 text-green-500" />
                            ) : (
                              <VscCopy className="w-3 h-3 text-zinc-400 hover:text-white" />
                            )}
                          </button>
                          <button
                            onClick={() => handleCollapse(runId)}
                            className="p-1 hover:bg-white/10 rounded group"
                            title="Collapse Terminal"
                          >
                            <div className="w-3 h-3 flex items-center justify-center">
                              <div className="w-2.5 h-[2px] bg-zinc-400 group-hover:bg-white" />
                            </div>
                          </button>
                        </div>
                      </div>

                      {/* Logs Area */}
                      <div className="flex-1 overflow-y-auto overflow-x-hidden pb-12 custom-scrollbar">
                        {lines.length === 0 ? (
                          <div className="px-3 py-4 flex flex-col gap-2 text-zinc-600 italic text-[10px] items-start">
                            <div className="flex items-center gap-2">
                              <div className="w-1.5 h-1.5 rounded-full bg-zinc-600 animate-pulse" />
                              <span>
                                [SYS] Awaiting stdout for {run.name}...
                              </span>
                            </div>
                          </div>
                        ) : (
                          lines.map((line, lIdx) => {
                            let lvlBadge = null;
                            let contentClass = "opacity-80";
                            let lineContent = line;

                            if (line.includes("[WARN]")) {
                              lvlBadge = (
                                <span className="text-yellow-400 bg-yellow-500/10 px-1 mr-1.5 border border-yellow-500/20 font-bold rounded-sm shrink-0">
                                  W
                                </span>
                              );
                              contentClass = "text-yellow-300/90";
                              lineContent = line.replace("[WARN]", "").trim();
                            } else if (line.includes("[ERR]")) {
                              lvlBadge = (
                                <span className="text-red-400 bg-red-500/10 px-1 mr-1.5 border border-red-500/20 font-bold rounded-sm shrink-0 shadow-[0_0_5px_rgba(239,68,68,0.2)]">
                                  E
                                </span>
                              );
                              contentClass = "text-red-300 font-semibold";
                              lineContent = line.replace("[ERR]", "").trim();
                            } else if (line.includes("[INFO]")) {
                              lvlBadge = (
                                <span className="text-sky-400 bg-sky-500/10 px-1 mr-1.5 border border-sky-500/20 rounded-sm shrink-0">
                                  I
                                </span>
                              );
                              contentClass = "text-zinc-300";
                              lineContent = line.replace("[INFO]", "").trim();
                            } else if (
                              line.includes("[SYS]") ||
                              line.includes("[DEBUG]")
                            ) {
                              lvlBadge = (
                                <span className="text-emerald-400 bg-emerald-500/10 px-1 mr-1.5 border border-emerald-500/20 rounded-sm shrink-0">
                                  S
                                </span>
                              );
                              contentClass = "text-emerald-200/80";
                              lineContent = line
                                .replace("[SYS]", "")
                                .replace("[DEBUG]", "")
                                .trim();
                            }

                            return (
                              <div
                                key={lIdx}
                                className="flex font-mono text-[9px] leading-tight hover:bg-white/[0.03] px-1 py-[1.5px] group/line w-full"
                              >
                                <span
                                  className="w-8 shrink-0 text-right pr-2 select-none opacity-30 group-hover/line:opacity-60 transition-opacity border-r border-white/5 mr-1.5 tabular-nums inline-block"
                                  style={{ color }}
                                >
                                  {lIdx + 1}
                                </span>
                                <span className="flex-1 flex overflow-hidden break-all whitespace-pre-wrap">
                                  {lvlBadge}
                                  <span className={contentClass}>
                                    {lineContent}
                                  </span>
                                </span>
                              </div>
                            );
                          })
                        )}
                        <div
                          ref={(el) => {
                            if (el) logsEndRef.current[runId] = el;
                          }}
                        />
                      </div>
                    </div>
                  </ResizablePanel>
                  {idx < expandedRuns.length - 1 && (
                    <ResizableHandle className="w-1 bg-white/5 hover:bg-primary/50 transition-colors z-50 cursor-col-resize" />
                  )}
                </React.Fragment>
              );
            })}
          </ResizablePanelGroup>
        )}

        {/* Collapsed bars on the right */}
        {collapsedRuns.length > 0 && (
          <div className="flex shrink-0 border-l border-white/5 bg-[#050505]">
            {collapsedRuns.map((r) => {
              const color = runColors[r.id] || "#3b82f6";
              return (
                <div
                  key={r.id}
                  onClick={() => handleExpand(r.id)}
                  className="w-10 border-r border-white/5 flex flex-col items-center py-3 cursor-pointer hover:brightness-150 transition-all select-none group/collapsed"
                  style={{ backgroundColor: `${color}05` }}
                  title={`Expand ${r.name}`}
                >
                  <div
                    style={{
                      backgroundColor: color,
                      boxShadow: `0 0 8px ${color}80`,
                    }}
                    className="w-2 h-2 rounded-full mb-6 shrink-0"
                  />
                  <span
                    className="[writing-mode:vertical-lr] text-[10px] tracking-[0.2em] font-black opacity-60 group-hover/collapsed:opacity-100 transition-opacity flex-1 flex justify-start pt-2 items-center"
                    style={{ color, textShadow: `0 0 4px ${color}40` }}
                  >
                    {r.name}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </Card>
  );
}
