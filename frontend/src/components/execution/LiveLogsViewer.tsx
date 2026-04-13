import React from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { VscTerminal, VscCopy, VscCheck } from "react-icons/vsc";
import { useAppStore } from "@/store/useAppStore";

export function LiveLogsViewer({
  handleCopyLogs,
  copiedLogId,
}: {
  runs?: any[];
  handleCopyLogs: (id: string, logs: string) => void;
  copiedLogId: string | null;
}) {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const selectedRunId = useAppStore((s) => s.selectedRunId);
  const [logs, setLogs] = React.useState<string[]>([]);
  const lastScrollTopRef = React.useRef<number>(0);

  // Stream logs dynamically via WebSockets
  React.useEffect(() => {
    if (!selectedRunId) {
      setLogs([]);
      return;
    }

    let active = true;
    let ws: WebSocket;

    setLogs([]); // Reset terminal on run switch

    const connectWS = () => {
      ws = new WebSocket(
        `ws://127.0.0.1:8000/api/ws/runs/${selectedRunId}/logs`,
      );
      
      ws.onopen = () => {
        setLogs([]);
      };

      ws.onmessage = (e) => {
        if (!active) return;
        try {
          const incoming = JSON.parse(e.data) as string[];
          setLogs((prev) => {
            const next = [...prev, ...incoming];
            return next.length > 500 ? next.slice(-500) : next;
          });
        } catch (err) {}
      };
      ws.onclose = () => {
        if (active) setTimeout(connectWS, 2000);
      };
    };

    connectWS();

    return () => {
      active = false;
      if (ws) ws.close();
    };
  }, [selectedRunId]);

  // Autoscroll to bottom always
  React.useEffect(() => {
    const el = containerRef.current;
    if (el) {
      // Unconditionally scroll to bottom so the user always sees the latest logs
      el.scrollTop = el.scrollHeight;
    }
  }, [logs]);

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
              const allLogs = logs.join("\n");
              handleCopyLogs(selectedRunId || "all", allLogs);
            }}
          >
            {copiedLogId === (selectedRunId || "all") ? (
              <VscCheck className="w-3 h-3 text-green-500 mr-1" />
            ) : (
              <VscCopy className="w-3 h-3 mr-1" />
            )}
            {copiedLogId === (selectedRunId || "all")
              ? "COPIED"
              : "COPY VISIBLE"}
          </Button>
        </div>
      </div>

      {/* Terminal Body */}
      <div className="flex-1 flex overflow-x-auto overflow-y-hidden bg-[#030303] text-zinc-300 selection:bg-primary/30 selection:text-white">
        <div
          ref={containerRef}
          onScroll={(e) => {
            lastScrollTopRef.current = e.currentTarget.scrollTop;
          }}
          className="flex-1 overflow-y-auto overflow-x-hidden pb-12 custom-scrollbar"
        >
          {logs.length === 0 ? (
            <div className="flex w-full px-2 py-2 text-zinc-600 text-xs font-mono select-none">
              No output captured. (Make sure this run is actively executing).
            </div>
          ) : (
            logs.map((line, i) => (
              <div
                key={i}
                className="flex font-mono text-[9px] leading-tight hover:bg-white/[0.03] px-1 py-[1.5px] group/line w-full"
              >
                <span className="w-10 shrink-0 text-right pr-2 select-none opacity-30 group-hover/line:opacity-60 transition-opacity border-r border-white/5 mr-1.5 tabular-nums inline-block">
                  {i + 1}
                </span>
                <span className="flex-1 overflow-hidden break-all whitespace-pre-wrap opacity-80 text-zinc-300">
                  {line}
                </span>
              </div>
            ))
          )}
        </div>
      </div>
    </Card>
  );
}
