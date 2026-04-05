import { useState, useEffect, useRef } from "react";
import { invoke as tauriInvoke } from "@tauri-apps/api/core";
import { BarChart2 } from "lucide-react";
import { MetricsDashboard } from "@/components/MetricsDashboard";
import { CreateSimpleRunModal } from "@/components/execution/CreateSimpleRunModal";
import { StudiesWorkspace } from "@/components/execution/StudiesWorkspace";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogDescription,
} from "@/components/ui/dialog";
import { AppSidebar } from "@/components/app-sidebar";
import { Input } from "@/components/ui/input";
import { Field, FieldLabel, FieldSet } from "@/components/ui/field";
import { Button } from "@/components/ui/button";
import {
  ResizablePanel,
  ResizablePanelGroup,
  ResizableHandle,
} from "@/components/ui/resizable";

import type { Run } from "@/bindings/Run";
import type { ActiveJob } from "@/bindings/ProcessInfo";
import { ProcessManagerWorkspace } from "@/components/execution/ProcessManagerWorkspace";

const isTauri =
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const invoke = async <T,>(
  cmd: string,
  args?: Record<string, any>,
): Promise<T> => {
  if (isTauri) {
    return tauriInvoke<T>(cmd, args);
  }
  console.warn("Mocking Tauri invoke for browser env:", cmd, args);
  if (cmd === "list_runs") return [] as T;
  return null as T;
};

export default function App() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [selectedDashboardRuns, setSelectedDashboardRuns] = useState<string[]>(
    [],
  );
  const [runLogs, setRunLogs] = useState<Record<string, string[]>>({});
  const [runColors, setRunColors] = useState<Record<string, string>>({});
  const dashboardLogsEndRef = useRef<Record<string, HTMLDivElement | null>>({});

  const [isSimpleModalOpen, setIsSimpleModalOpen] = useState(false);
  const [viewMode, setViewMode] = useState<"runs" | "studies">("runs");
  const [activeJobs, setActiveJobs] = useState<ActiveJob[]>([]);

  // Dialog state
  const [runToRename, setRunToRename] = useState<string | null>(null);
  const [newName, setNewName] = useState("");
  const [runToDelete, setRunToDelete] = useState<string | null>(null);
  const [runToFlush, setRunToFlush] = useState<string | null>(null);

  const DEFAULT_COLORS = [
    "#10b981",
    "#3b82f6",
    "#f59e0b",
    "#8b5cf6",
    "#ec4899",
    "#ef4444",
    "#14b8a6",
  ];

  const loadRuns = async () => {
    try {
      const list = await invoke<Run[]>("list_runs");
      setRuns(list);
    } catch (e) {
      console.error(e);
    }
  };

  // Auto-select the first run on initial load or if the selected run is deleted
  useEffect(() => {
    if (runs.length > 0) {
      if (!selectedRunId || !runs.find((r) => r.id === selectedRunId)) {
        setSelectedRunId(runs[0].id);
      }
      if (selectedDashboardRuns.length === 0) {
        setSelectedDashboardRuns([runs[0].id]);
      }
    }
  }, [runs, selectedRunId, selectedDashboardRuns.length]);

  useEffect(() => {
    loadRuns();
    const interval = setInterval(loadRuns, 3000);

    let unlisten: (() => void) | undefined;
    let isCancelled = false;

    import("@tauri-apps/api/event").then(({ listen }) => {
      if (!isTauri) {
        console.warn("Skipping Tauri listen in browser env");
        return;
      }
      let logBuffer: Record<string, string[]> = {};
      let flushTimeout: number | null = null;

      const flushLogs = () => {
        setRunLogs((prev) => {
          let hasChanges = false;
          const newState = { ...prev };
          for (const [run_id, lines] of Object.entries(logBuffer)) {
            if (lines.length > 0) {
              hasChanges = true;
              newState[run_id] = [...(newState[run_id] || []), ...lines].slice(
                -500,
              );
            }
          }
          logBuffer = {};
          flushTimeout = null;
          return hasChanges ? newState : prev;
        });
      };

      listen("log_event", (event: any) => {
        const { run_id, line } = event.payload;
        if (!logBuffer[run_id]) logBuffer[run_id] = [];
        logBuffer[run_id].push(line);

        if (!flushTimeout) {
          flushTimeout = window.setTimeout(flushLogs, 100);
        }
      }).then((u) => {
        if (isCancelled) {
          u();
        } else {
          unlisten = u;
        }
      });

      listen("process_telemetry", (event: any) => {
        setActiveJobs(event.payload as ActiveJob[]);
      });
    });

    return () => {
      isCancelled = true;
      clearInterval(interval);
      if (unlisten) unlisten();
    };
  }, []);

  useEffect(() => {
    selectedDashboardRuns.forEach((runId) => {
      const ref = dashboardLogsEndRef.current[runId];
      if (ref) ref.scrollIntoView({ behavior: "auto" });
    });
  }, [runLogs, selectedDashboardRuns]);

  const toggleDashboardRun = (id: string, pressed: boolean) => {
    if (pressed) setSelectedDashboardRuns((prev) => [...prev, id]);
    else setSelectedDashboardRuns((prev) => prev.filter((r) => r !== id));
  };

  // Handlers for runs

  const handleRename = async () => {
    if (!runToRename || !newName.trim()) return;
    try {
      await invoke("rename_run", { id: runToRename, newName });
      setRunToRename(null);
      loadRuns();
    } catch (e) {
      console.error(e);
    }
  };
  const handleDelete = async () => {
    if (!runToDelete) return;
    try {
      await invoke("delete_run", { id: runToDelete });
      setRunToDelete(null);
      if (selectedRunId === runToDelete) setSelectedRunId(null);
      setSelectedDashboardRuns((prev) =>
        prev.filter((id) => id !== runToDelete),
      );
      loadRuns();
    } catch (e) {
      console.error(e);
    }
  };
  const handleFlush = async () => {
    if (!runToFlush) return;
    try {
      await invoke("flush_run", { id: runToFlush });
      setRunToFlush(null);
      setRunLogs((prev) => ({ ...prev, [runToFlush]: [] }));
      loadRuns();
    } catch (e) {
      console.error(e);
    }
  };
  const handleEngineCmd = async (
    runId: string,
    cmd: string,
    force?: boolean,
  ) => {
    try {
      if (cmd === "start") {
        setRunLogs((prev) => ({ ...prev, [runId]: [] }));
        if (!selectedDashboardRuns.includes(runId))
          setSelectedDashboardRuns((prev) => [...prev, runId]);
        await invoke("start_run", { id: runId });
      } else if (cmd === "stop") {
        await invoke("stop_run", { id: runId, force });
      }
      loadRuns();
    } catch (e) {
      console.error(e);
    }
  };
  const handleClone = async (run: Run) => {
    try {
      const createdRun = await invoke<Run>("create_run", {
        name: run.name + "_clone",
        type: run.type,
        preset: "default",
      });
      await invoke("update_run_config", {
        id: createdRun.id,
        config: run.config,
      });
      loadRuns();
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="h-screen w-screen overflow-hidden bg-background text-foreground flex">
      <ResizablePanelGroup direction="horizontal" className="h-full w-full">
        <ResizablePanel defaultSize={20} minSize={15} maxSize={40}>
          <AppSidebar
            runs={runs}
            selectedRunId={selectedRunId}
            setSelectedRunId={setSelectedRunId}
            selectedDashboardRuns={selectedDashboardRuns}
            setSelectedDashboardRuns={setSelectedDashboardRuns}
            toggleDashboardRun={toggleDashboardRun}
            runColors={runColors}
            setRunColors={setRunColors}
            defaultColors={DEFAULT_COLORS}
            setRunToRename={setRunToRename}
            setNewName={setNewName}
            setRunToFlush={setRunToFlush}
            setRunToDelete={setRunToDelete}
            handleEngineCmd={handleEngineCmd}
            handleClone={handleClone}
            setIsSimpleModalOpen={setIsSimpleModalOpen}
            viewMode={viewMode}
            setViewMode={setViewMode}
          />
        </ResizablePanel>

        <ResizableHandle className="w-1 bg-border/20 hover:bg-primary/50 transition-colors cursor-col-resize z-50" />

        <ResizablePanel defaultSize={80}>
          <ResizablePanelGroup direction="vertical">
            <ResizablePanel defaultSize={75} minSize={30}>
              <div className="w-full h-full overflow-hidden bg-black animate-in fade-in duration-300">
                {viewMode === "studies" ? (
                  <StudiesWorkspace runLogs={runLogs} />
                ) : selectedDashboardRuns.length > 0 ? (
                  <MetricsDashboard
                    runs={runs}
                    runIds={selectedDashboardRuns}
                    runColors={runColors}
                  />
                ) : (
                  <div className="flex flex-col items-center justify-center w-full h-full text-zinc-600 gap-4 bg-[#050505]">
                    <BarChart2 className="w-12 h-12 opacity-20" />
                    <p className="text-sm font-medium">
                      Toggle "Graph Match" on runs in the Sidebar to project
                      them here.
                    </p>
                  </div>
                )}
              </div>
            </ResizablePanel>

            <ResizableHandle className="h-1 bg-border/20 hover:bg-primary/50 transition-colors cursor-row-resize z-50" />

            <ResizablePanel defaultSize={25} minSize={15}>
              <ProcessManagerWorkspace
                runs={runs}
                runLogs={runLogs}
                activeJobs={activeJobs}
                selectedDashboardRuns={selectedDashboardRuns}
                toggleDashboardRun={toggleDashboardRun}
                logsEndRef={dashboardLogsEndRef}
              />
            </ResizablePanel>
          </ResizablePanelGroup>
        </ResizablePanel>
      </ResizablePanelGroup>

      {/* Run Generation Modals */}
      <CreateSimpleRunModal
        isOpen={isSimpleModalOpen}
        setIsOpen={setIsSimpleModalOpen}
        loadRuns={loadRuns}
      />

      {/* Editing Dialogs */}
      <Dialog
        open={!!runToRename}
        onOpenChange={(open) => !open && setRunToRename(null)}
      >
        <DialogContent className="sm:max-w-[350px] border-border/20 bg-[#0a0a0a]">
          <DialogHeader>
            <DialogTitle>Rename Run</DialogTitle>
          </DialogHeader>
          <FieldSet>
            <Field>
              <FieldLabel>New Name</FieldLabel>
              <Input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                className="bg-zinc-900 border-border/30"
              />
            </Field>
          </FieldSet>
          <DialogFooter>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setRunToRename(null)}
            >
              Cancel
            </Button>
            <Button size="sm" onClick={handleRename}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={!!runToDelete}
        onOpenChange={(open) => !open && setRunToDelete(null)}
      >
        <DialogContent className="sm:max-w-[400px] border-border/20 bg-[#0a0a0a]">
          <DialogHeader>
            <DialogTitle>Delete Config</DialogTitle>
            <DialogDescription>
              This deletes the configuration entirely. Cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setRunToDelete(null)}
            >
              Cancel
            </Button>
            <Button variant="destructive" size="sm" onClick={handleDelete}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={!!runToFlush}
        onOpenChange={(open) => !open && setRunToFlush(null)}
      >
        <DialogContent className="sm:max-w-[400px] border-border/20 bg-[#0a0a0a]">
          <DialogHeader>
            <DialogTitle>Flush Data</DialogTitle>
            <DialogDescription>
              Clears all metrics, checkpoints, and logs for this run but keeps
              the config.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setRunToFlush(null)}
            >
              Cancel
            </Button>
            <Button variant="destructive" size="sm" onClick={handleFlush}>
              Flush
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
