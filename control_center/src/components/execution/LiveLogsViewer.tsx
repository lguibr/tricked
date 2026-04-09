import React from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { VscTerminal, VscCopy, VscCheck, VscClearAll } from "react-icons/vsc";

interface LiveLogsViewerProps {
  runs: any[];
  globalLogs: { runId: string; line: string }[];
  handleCopyLogs: (id: string, logs: string) => void;
  copiedLogId: string | null;
  logsEndRef: React.MutableRefObject<Record<string, HTMLDivElement | null>>;
  runColors?: Record<string, string>;
}

export function LiveLogsViewer({
  runs,
  globalLogs,
  handleCopyLogs,
  copiedLogId,
  logsEndRef,
  runColors = {},
}: LiveLogsViewerProps) {
  React.useEffect(() => {
    if (logsEndRef.current["global"]) {
      logsEndRef.current["global"].scrollIntoView({ behavior: "instant" });
    }
  }, [globalLogs]);

  return (
    <Card className="h-full w-full flex flex-col rounded-none shadow-none border-0 overflow-hidden shrink-0 bg-[#050505]">
      {/* Terminal Header */}
      <div className="flex items-center justify-between px-2 py-1 border-b border-border/20 bg-[#0a0a0a] shrink-0">
        <div className="flex items-center text-[9px] font-bold uppercase tracking-widest text-zinc-500">
          <VscTerminal className="w-3.5 h-3.5 text-primary mr-1.5" />
          <span>Stdout / Logs</span>
          {globalLogs.length > 0 && (
            <>
              <div className="w-[1px] h-3 bg-white/10 mx-2" />
              <Button
                variant="ghost"
                size="sm"
                className="h-5 px-1.5 text-[9px] text-zinc-400 hover:text-white bg-white/5 hover:bg-white/10 rounded-sm uppercase tracking-wider"
                onClick={() => {
                  const allLogs = globalLogs
                    .map((l) => `[${l.runId}] ${l.line}`)
                    .join("\n");
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
        {globalLogs.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center text-zinc-700 text-[10px] font-mono tracking-widest uppercase opacity-50 gap-2">
            <VscClearAll className="w-8 h-8" />
            NO LOGS YET
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto overflow-x-hidden pb-12 custom-scrollbar">
            {globalLogs.map((logEntry, lIdx) => {
              const runId = logEntry.runId;
              const line = logEntry.line;
              const color = runColors[runId] || "#3b82f6";

              // Find the run name if possible
              const runName =
                runs.find((r) => r.id === runId)?.name || runId.substring(0, 8);

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
              } else if (line.includes("[SYS]") || line.includes("[DEBUG]")) {
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
                  <span className="w-10 shrink-0 text-right pr-2 select-none opacity-30 group-hover/line:opacity-60 transition-opacity border-r border-white/5 mr-1.5 tabular-nums inline-block">
                    {lIdx + 1}
                  </span>

                  <span
                    className="w-24 shrink-0 text-left pr-2 select-none overflow-hidden truncate font-bold tracking-widest uppercase inline-block border-r border-white/5 mr-1.5"
                    style={{ color }}
                  >
                    {runName}
                  </span>

                  <span className="flex-1 flex overflow-hidden break-all whitespace-pre-wrap">
                    {lvlBadge}
                    <span className={contentClass}>{lineContent}</span>
                  </span>
                </div>
              );
            })}
            <div
              ref={(el) => {
                if (el) logsEndRef.current["global"] = el;
              }}
            />
          </div>
        )}
      </div>
    </Card>
  );
}
