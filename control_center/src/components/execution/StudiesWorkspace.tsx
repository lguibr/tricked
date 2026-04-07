import { useEffect, useRef } from "react";
import { VscServerProcess } from "react-icons/vsc";
import { LiveLogsViewer } from "./LiveLogsViewer";
import { OptunaStudyDashboard } from "@/components/OptunaStudyDashboard";
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from "@/components/ui/resizable";
import { useAppStore } from "@/store/useAppStore";
import { useTuningStore } from "@/store/useTuningStore";

export function StudiesWorkspace() {
  const runLogs = useAppStore((state) => state.runLogs);
  const isActive = useTuningStore((state) => state.isActive);
  const checkActive = useTuningStore((state) => state.checkActive);

  const logsRef = useRef<Record<string, HTMLDivElement | null>>({});

  useEffect(() => {
    checkActive();
    const int = setInterval(checkActive, 2000);
    return () => clearInterval(int);
  }, []);

  return (
    <div className="flex h-full w-full bg-[#0a0a0a] text-zinc-200">
      {/* Dashboard & Terminal View */}
      <div className="flex-1 flex flex-col bg-[#020202] overflow-hidden">
        <ResizablePanelGroup direction="vertical">
          {/* Top: Optuna Dashboard */}
          <ResizablePanel defaultSize={70} minSize={30}>
            <OptunaStudyDashboard />
          </ResizablePanel>

          <ResizableHandle className="h-0.5 bg-white/10 hover:bg-emerald-500/50 transition-colors z-50 cursor-row-resize shadow-[0_0_5px_rgba(255,255,255,0.1)]" />

          {/* Bottom: Terminal */}
          <ResizablePanel defaultSize={30} minSize={10}>
            <div className="flex flex-col h-full">
              <div className="h-6 px-3 flex items-center border-b border-white/5 bg-[#050505] shrink-0">
                <span className="text-[8.5px] font-black uppercase tracking-widest text-zinc-500 flex items-center gap-1.5">
                  <VscServerProcess className="text-zinc-600" />
                  Live Process Diagnostics
                </span>
              </div>
              <div className="flex-1 relative overflow-hidden bg-[#020202]">
                <LiveLogsViewer
                  runs={[
                    {
                      id: "STUDY",
                      name: "Tuning Engine",
                      type: "STUDY",
                      status: isActive ? "RUNNING" : "STOPPED",
                      config: "{}",
                    },
                  ]}
                  runLogs={runLogs}
                  selectedLogRunIds={["STUDY"]}
                  toggleLogRun={() => {}}
                  handleCopyLogs={(_id, logs) =>
                    navigator.clipboard.writeText(logs)
                  }
                  copiedLogId={null}
                  logsEndRef={logsRef}
                  runColors={{ STUDY: "#10b981" }}
                />
              </div>
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>
    </div>
  );
}
