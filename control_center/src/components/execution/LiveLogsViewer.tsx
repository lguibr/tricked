import React from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { VscTerminal, VscCopy, VscCheck } from "react-icons/vsc";



export function LiveLogsViewer({
  runs,
  handleCopyLogs,
  copiedLogId,
}: {
  runs: any[];
  handleCopyLogs: (id: string, logs: string) => void;
  copiedLogId: string | null;
}) {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const logsStateRef = React.useRef<{ runId: string; line: string }[]>([]);

  React.useEffect(() => {
    const handleLogBatch = (e: Event) => {
      const customEvent = e as CustomEvent<any[]>;
      const batchedLogs = customEvent.detail;
      if (!batchedLogs || batchedLogs.length === 0) return;

      const container = containerRef.current;
      if (!container) return;

      const fragment = document.createDocumentFragment();

      for (let i = 0; i < batchedLogs.length; i++) {
        const log = batchedLogs[i];
        logsStateRef.current.push({ runId: log.run_id, line: log.line });

        const lineDiv = document.createElement("div");
        lineDiv.className =
          "flex font-mono text-[9px] leading-tight hover:bg-white/[0.03] px-1 py-[1.5px] group/line w-full";

        // basic number
        const numSpan = document.createElement("span");
        numSpan.className =
          "w-10 shrink-0 text-right pr-2 select-none opacity-30 group-hover/line:opacity-60 transition-opacity border-r border-white/5 mr-1.5 tabular-nums inline-block";
        numSpan.innerText = String(logsStateRef.current.length);
        lineDiv.appendChild(numSpan);

        // run name
        const nameSpan = document.createElement("span");
        nameSpan.className =
          "w-24 shrink-0 text-left pr-2 select-none overflow-hidden truncate font-bold tracking-widest uppercase inline-block border-r border-white/5 mr-1.5 text-blue-500";
        nameSpan.innerText = log.run_id.substring(0, 8);
        lineDiv.appendChild(nameSpan);

        // content
        const contentSpan = document.createElement("span");
        contentSpan.className =
          "flex-1 flex overflow-hidden break-all whitespace-pre-wrap opacity-80 text-zinc-300";
        contentSpan.innerText = log.line;
        lineDiv.appendChild(contentSpan);

        fragment.appendChild(lineDiv);
      }

      container.appendChild(fragment);

      // keep only last 1000 nodes
      while (container.childNodes.length > 1000) {
        container.removeChild(container.firstChild as Node);
      }

      if (logsStateRef.current.length > 1000) {
        logsStateRef.current = logsStateRef.current.slice(-1000);
      }

      container.scrollTop = container.scrollHeight;
    };

    window.addEventListener("engine_log_batch", handleLogBatch);
    return () => {
      window.removeEventListener("engine_log_batch", handleLogBatch);
    };
  }, [runs]);

  return (
    <Card className="h-full w-full flex flex-col rounded-none shadow-none border-0 overflow-hidden shrink-0 bg-[#050505]">
      {/* Terminal Header */}
      <div className="flex items-center justify-between px-2 py-1 border-b border-border/20 bg-[#0a0a0a] shrink-0">
        <div className="flex items-center text-[9px] font-bold uppercase tracking-widest text-zinc-500">
          <VscTerminal className="w-3.5 h-3.5 text-primary mr-1.5" />
          <span>Stdout / Logs</span>
          <div className="w-[1px] h-3 bg-white/10 mx-2" />
          <Button
            variant="ghost"
            size="sm"
            className="h-5 px-1.5 text-[9px] text-zinc-400 hover:text-white bg-white/5 hover:bg-white/10 rounded-sm uppercase tracking-wider"
            onClick={() => {
              const allLogs = logsStateRef.current
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
        </div>
      </div>

      {/* Terminal Body */}
      <div className="flex-1 flex overflow-x-auto overflow-y-hidden bg-[#030303] text-zinc-300 selection:bg-primary/30 selection:text-white">
        <div
          ref={containerRef}
          className="flex-1 overflow-y-auto overflow-x-hidden pb-12 custom-scrollbar"
        ></div>
      </div>
    </Card>
  );
}
