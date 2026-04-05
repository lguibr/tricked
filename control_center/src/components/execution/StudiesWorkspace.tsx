import { useState, useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Brain,
  Play,
  Square,
  Loader2,
  Trash2,
  CheckCircle2,
} from "lucide-react";
import { LiveLogsViewer } from "./LiveLogsViewer";
import { OptunaStudyDashboard } from "@/components/OptunaStudyDashboard";
import { CreateTuningModal } from "./CreateTuningModal";
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from "@/components/ui/resizable";

interface StudiesWorkspaceProps {
  runLogs: Record<string, string[]>;
}

export function StudiesWorkspace({ runLogs }: StudiesWorkspaceProps) {
  const [isActive, setIsActive] = useState(false);
  const [isTuningModalOpen, setIsTuningModalOpen] = useState(false);
  const [tuneComplete, setTuneComplete] = useState(false);

  const logsRef = useRef<Record<string, HTMLDivElement | null>>({});

  const refreshStatus = async () => {
    try {
      setTuneComplete(
        await invoke<boolean>("get_study_status", { studyType: "UNIFIED" }),
      );
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    const checkActive = async () => {
      try {
        const active = await invoke<boolean>("get_active_study");
        setIsActive(active);
        refreshStatus();
      } catch (e) {
        console.error(e);
      }
    };
    checkActive();
    const int = setInterval(checkActive, 2000);
    return () => clearInterval(int);
  }, []);

  const openTuningModal = () => {
    setIsTuningModalOpen(true);
  };

  const handleStop = async () => {
    try {
      await invoke("stop_study", { force: false });
      setIsActive(false);
    } catch (e) {
      console.error(e);
    }
  };

  const handleFlush = async () => {
    if (!confirm(`Are you sure you want to wipe the global study database?`))
      return;
    try {
      await invoke("flush_study", { studyType: "UNIFIED" });
      await refreshStatus();
    } catch (e) {
      console.error(e);
      alert(e);
    }
  };

  return (
    <div className="flex h-full w-full bg-[#050505] text-zinc-200">
      {/* Configuration Column */}
      <div className="w-[350px] border-r border-border/20 p-6 flex flex-col gap-5 overflow-y-auto shrink-0 bg-[#0a0a0a] custom-scrollbar">
        <div className="flex flex-col gap-1 shrink-0">
          <h2 className="text-xl font-black tracking-widest uppercase text-zinc-100">
            Tuning Lab
          </h2>
          <span className="text-[10px] text-zinc-500 uppercase tracking-widest font-semibold flex items-center justify-between">
            Optuna Hub Integrations
          </span>
        </div>

        <Card
          className={`bg-[#101010] border p-5 flex flex-col gap-4 shrink-0 transition-colors ${tuneComplete ? "border-emerald-500/50" : "border-border/10"}`}
        >
          <div className="flex items-center justify-between text-emerald-400">
            <div className="flex items-center gap-3">
              <Brain className="w-5 h-5" />
              <h3 className="font-bold text-sm uppercase tracking-wider">
                Holistic Tuning
              </h3>
            </div>
            {tuneComplete && (
              <CheckCircle2 className="w-4 h-4 text-emerald-500" />
            )}
          </div>
          <p className="text-xs text-zinc-400 leading-relaxed">
            Multi-objective optimization mapping hardware throughput limits and
            training convergence potential simultaneously.
          </p>
          <div className="flex gap-2">
            <Button
              disabled={isActive}
              onClick={() => openTuningModal()}
              className="flex-1 bg-emerald-600/10 text-emerald-400 hover:bg-emerald-600/30 border border-emerald-500/20 shadow-none text-xs"
            >
              {isActive ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Play className="w-3 h-3 mr-2" />
              )}
              {tuneComplete ? "Restart Scan" : "Start Scan"}
            </Button>
            {tuneComplete && (
              <Button
                variant="outline"
                size="icon"
                onClick={() => handleFlush()}
                className="bg-transparent border-red-500/20 text-red-500 hover:bg-red-500/10"
              >
                <Trash2 className="w-4 h-4" />
              </Button>
            )}
          </div>
        </Card>

        {isActive && (
          <Button
            onClick={handleStop}
            variant="destructive"
            className="w-full shrink-0 text-xs mt-2"
          >
            <Square className="w-3 h-3 mr-2" /> Kill Active Process
          </Button>
        )}
      </div>

      {/* Dashboard & Terminal View */}
      <div className="flex-1 flex flex-col bg-[#030303] overflow-hidden">
        <ResizablePanelGroup direction="vertical">
          {/* Top: Optuna Dashboard */}
          <ResizablePanel defaultSize={70} minSize={30}>
            <OptunaStudyDashboard />
          </ResizablePanel>

          <ResizableHandle className="h-1 bg-border/20 hover:bg-primary/50 transition-colors z-50 cursor-row-resize" />

          {/* Bottom: Terminal */}
          <ResizablePanel defaultSize={30} minSize={10}>
            <div className="flex flex-col h-full">
              <div className="h-8 px-4 flex items-center border-b border-border/20 bg-zinc-950/80 shrink-0">
                <span className="text-[10px] font-bold uppercase tracking-widest text-zinc-500">
                  Live Process Diagnostics
                </span>
              </div>
              <div className="flex-1 relative overflow-hidden bg-black">
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
                />
              </div>
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>

      {/* Create Tuning Modal Injection */}
      <CreateTuningModal
        isOpen={isTuningModalOpen}
        setIsOpen={setIsTuningModalOpen}
        runsViewRefresh={refreshStatus}
      />
    </div>
  );
}
