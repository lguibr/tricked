import { Card } from "@/components/ui/card";
import { Toggle } from "@/components/ui/toggle";
import { Button } from "@/components/ui/button";
import {
  VscTerminal,
  VscCopy,
  VscCheck,
  VscClearAll,
  VscClose,
} from "react-icons/vsc";

interface LiveLogsViewerProps {
  runs: any[];
  runLogs: Record<string, string[]>;
  selectedLogRunIds: string[];
  toggleLogRun: (id: string, pressed: boolean) => void;
  handleCopyLogs: (id: string, logs: string) => void;
  copiedLogId: string | null;
  logsEndRef: React.MutableRefObject<Record<string, HTMLDivElement | null>>;
  runColors?: Record<string, string>;
}

export function LiveLogsViewer({
  runs,
  runLogs,
  selectedLogRunIds,
  toggleLogRun,
  handleCopyLogs,
  copiedLogId,
  logsEndRef,
  runColors = {},
}: LiveLogsViewerProps) {
  return (
    <Card className="h-full w-full flex flex-col rounded-none shadow-none border-0 overflow-hidden shrink-0 bg-[#050505]">
      {/* Terminal Header */}
      <div className="flex items-center justify-between px-2 py-1 border-b border-border/20 bg-[#0a0a0a] shrink-0">
        <div className="flex items-center text-[9px] font-bold uppercase tracking-widest text-zinc-500">
          <VscTerminal className="w-3.5 h-3.5 text-primary mr-1.5" />
          <span>Stdout</span>
          {selectedLogRunIds.length > 0 && (
            <>
              <div className="w-[1px] h-3 bg-white/10 mx-2" />
              <Button
                variant="ghost"
                size="sm"
                className="h-5 px-1.5 text-[9px] text-zinc-400 hover:text-white bg-white/5 hover:bg-white/10 rounded-sm uppercase tracking-wider"
                onClick={() => {
                  const allLogs = selectedLogRunIds
                    .map((id) => {
                      const name = runs.find((r) => r.id === id)?.name || id;
                      return `--- ${name} ---\n${(runLogs[id] || []).join("\n")}`;
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
                {copiedLogId === "all" ? "COPIED" : "COPY"}
              </Button>
            </>
          )}
        </div>
        <div className="flex items-center gap-1">
          {runs
            .filter((r) => r.status !== "WAITING")
            .map((r) => {
              const c = runColors[r.id] || "#3b82f6";
              const isSelected = selectedLogRunIds.includes(r.id);
              return (
                <Toggle
                  key={r.id}
                  size="sm"
                  pressed={isSelected}
                  onPressedChange={(p) => toggleLogRun(r.id, p)}
                  className="h-5 px-2 text-[9px] border transition-colors uppercase font-bold tracking-wider"
                  style={{
                    backgroundColor: isSelected ? `${c}20` : "transparent",
                    color: isSelected ? c : "#71717a",
                    borderColor: isSelected ? `${c}50` : "transparent",
                  }}
                >
                  {r.name}
                </Toggle>
              );
            })}
        </div>
      </div>

      {/* Terminal Body */}
      <div className="flex-1 flex overflow-x-auto overflow-y-hidden bg-[#030303] text-zinc-300 selection:bg-primary/30 selection:text-white">
        {selectedLogRunIds.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center text-zinc-700 text-[10px] font-mono tracking-widest uppercase opacity-50 gap-2">
            <VscClearAll className="w-8 h-8" />
            NO TTY ATTACHED
          </div>
        ) : (
          selectedLogRunIds.map((runId) => {
            const run = runs.find((r) => r.id === runId);
            const color = runColors[runId] || "#3b82f6";
            const lines = runLogs[runId] || [];

            return (
              <div
                key={runId}
                className={`${selectedLogRunIds.length === 1 ? "flex-1 min-w-0" : "w-[650px] shrink-0 min-w-0"} h-full flex flex-col font-mono text-[9.5px] border-r last:border-r-0 relative`}
                style={{
                  borderRightColor: `${color}40`,
                  backgroundColor: `${color}05`,
                }}
              >
                {/* Embedded Run Header */}
                <div
                  className="sticky top-0 backdrop-blur-md px-2 py-1 mb-1 z-10 border-b flex items-center shadow-sm select-none"
                  style={{
                    backgroundColor: `${color}15`,
                    borderColor: `${color}30`,
                    borderBottomWidth: "1px",
                  }}
                >
                  <VscTerminal
                    className="w-3 h-3 mr-1.5"
                    style={{ color, filter: `drop-shadow(0 0 2px ${color})` }}
                  />
                  <span
                    style={{ color, textShadow: `0 0 4px ${color}80` }}
                    className="font-bold tracking-widest uppercase flex-1"
                  >
                    {run?.name}
                  </span>
                  <button
                    onClick={() => toggleLogRun(runId, false)}
                    className="opacity-50 hover:opacity-100 transition-opacity ml-2 pointer-events-auto"
                    style={{ color }}
                  >
                    <VscClose className="w-3.5 h-3.5" />
                  </button>
                </div>

                {/* Logs Area */}
                <div className="flex-1 overflow-y-auto overflow-x-hidden pb-12 custom-scrollbar">
                  {lines.length === 0 ? (
                    <div className="px-3 py-4 flex flex-col gap-2 text-zinc-600 italic text-[10px] items-start">
                      <div className="flex items-center gap-2">
                        <div className="w-1.5 h-1.5 rounded-full bg-zinc-600 animate-pulse" />
                        <span>[SYS] Awaiting stdout for {run?.name}...</span>
                      </div>
                    </div>
                  ) : (
                    lines.map((line, idx) => {
                      // Ultra-dense styling parsing
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
                          key={idx}
                          className="flex font-mono text-[9px] leading-tight hover:bg-white/[0.03] px-1 py-[1.5px] group w-full"
                        >
                          <span
                            className="w-8 shrink-0 text-right pr-2 select-none opacity-30 group-hover:opacity-60 transition-opacity border-r border-white/5 mr-1.5 tabular-nums inline-block"
                            style={{ color }}
                          >
                            {idx + 1}
                          </span>
                          <span className="flex-1 flex overflow-hidden break-all whitespace-pre-wrap">
                            {lvlBadge}
                            <span className={contentClass}>{lineContent}</span>
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
            );
          })
        )}
      </div>
    </Card>
  );
}
