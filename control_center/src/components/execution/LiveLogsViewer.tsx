import { Card } from "@/components/ui/card";
import { Toggle } from "@/components/ui/toggle";
import { Button } from "@/components/ui/button";
import { TerminalSquare, Copy, Check } from "lucide-react";

interface LiveLogsViewerProps {
  runs: any[];
  runLogs: Record<string, string[]>;
  selectedLogRunIds: string[];
  toggleLogRun: (id: string, pressed: boolean) => void;
  handleCopyLogs: (id: string, logs: string) => void;
  copiedLogId: string | null;
  logsEndRef: React.MutableRefObject<Record<string, HTMLDivElement | null>>;
}

export function LiveLogsViewer({
  runs,
  runLogs,
  selectedLogRunIds,
  toggleLogRun,
  handleCopyLogs,
  copiedLogId,
  logsEndRef,
}: LiveLogsViewerProps) {
  return (
    <Card className="h-64 flex flex-col rounded-none shadow-none border-0 overflow-hidden shrink-0 bg-[#0c0c0c]">
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-black/50 bg-zinc-900 shadow-sm">
        <div className="flex items-center space-x-2 text-muted-foreground text-[10px] font-mono">
          <TerminalSquare className="w-3 h-3" />
          <span>LIVE LOGROUTER</span>
        </div>
        <div className="flex items-center gap-2">
          {runs
            .filter((r) => r.status !== "WAITING")
            .map((r) => (
              <Toggle
                key={r.id}
                size="sm"
                pressed={selectedLogRunIds.includes(r.id)}
                onPressedChange={(p) => toggleLogRun(r.id, p)}
                className="h-6 px-2 text-[10px] data-[state=on]:bg-primary/20 data-[state=on]:text-primary border border-transparent data-[state=on]:border-primary/50"
              >
                {r.name}
              </Toggle>
            ))}
        </div>
      </div>
      <div className="flex-1 flex overflow-x-auto overflow-y-hidden bg-black text-green-500 selection:bg-green-900 selection:text-white">
        {selectedLogRunIds.length === 0 ? (
          <div className="flex-1 flex items-center justify-center text-zinc-600 text-xs font-mono">
            Select a run to view logs
          </div>
        ) : (
          selectedLogRunIds.map((runId) => {
            const run = runs.find((r) => r.id === runId);
            const color =
              runId === "1"
                ? "border-orange-500/50"
                : runId === "2"
                  ? "border-purple-500/50"
                  : "border-blue-500/50";
            const lines = runLogs[runId] || [];
            const logsText = lines.join("\n");

            return (
              <div
                key={runId}
                className={`${selectedLogRunIds.length === 1 ? "flex-1" : "w-[600px] shrink-0"} h-full p-2 font-mono text-[10px] leading-relaxed border-r ${color} last:border-r-0 pb-12 relative overflow-y-auto`}
              >
                <div className="sticky top-0 bg-black/90 backdrop-blur-sm py-1 mb-2 font-semibold text-zinc-300 z-10 border-b border-border/20 flex justify-between items-center group/header">
                  <div>
                    <span className="opacity-50">#</span> {run?.name}
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-5 w-5 opacity-0 group-hover/header:opacity-100 text-muted-foreground hover:text-white"
                    onClick={() => handleCopyLogs(runId, logsText)}
                  >
                    {copiedLogId === runId ? (
                      <Check className="h-3 w-3 text-green-500" />
                    ) : (
                      <Copy className="h-3 w-3" />
                    )}
                  </Button>
                </div>
                <div className="space-y-0.5 whitespace-pre-wrap">
                  {lines.length === 0 ? (
                    <span className="text-zinc-600 italic">
                      Waiting for connection...
                    </span>
                  ) : (
                    lines.map((line, idx) => (
                      <div key={idx}>
                        {line.includes("[WARN]") ? (
                          <span className="text-yellow-500">{line}</span>
                        ) : line.includes("[ERR]") ? (
                          <span className="text-red-500">{line}</span>
                        ) : line.includes("[INFO]") ? (
                          <span>
                            <span className="text-blue-400">[INFO]</span>
                            {line.split("[INFO]")[1]}
                          </span>
                        ) : (
                          line
                        )}
                      </div>
                    ))
                  )}
                  <div
                    ref={(el) => {
                      logsEndRef.current[runId] = el;
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
