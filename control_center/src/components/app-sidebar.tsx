import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { RunsSidebarList } from "@/components/execution/RunsSidebarList";
import logoUrl from "@/assets/logo.svg";
import { HardwareMiniDashboard } from "@/components/dashboard/HardwareMiniDashboard";
import { CreateSimpleRunSidebar } from "@/components/execution/CreateSimpleRunSidebar";

import type { Run } from "@/bindings/Run";

export function AppSidebar({
  runs,
  selectedRunId,
  setSelectedRunId,
  selectedDashboardRuns,
  setSelectedDashboardRuns,
  toggleDashboardRun,
  runColors,
  setRunColors,
  defaultColors,
  setRunToRename,
  setNewName,
  setRunToFlush,
  setRunToDelete,
  handleEngineCmd,
  handleClone,
  isCreatingRun,
  setIsCreatingRun,
  loadRuns,
  viewMode,
  setViewMode,
}: {
  runs: Run[];
  selectedRunId: string | null;
  setSelectedRunId: (id: string) => void;
  selectedDashboardRuns: string[];
  setSelectedDashboardRuns: React.Dispatch<React.SetStateAction<string[]>>;
  toggleDashboardRun: (id: string, pressed: boolean) => void;
  runColors: Record<string, string>;
  setRunColors: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  defaultColors: string[];
  setRunToRename: (id: string) => void;
  setNewName: (name: string) => void;
  setRunToFlush: (id: string) => void;
  setRunToDelete: (id: string) => void;
  handleEngineCmd: (runId: string, cmd: string, force?: boolean) => void;
  handleClone: (run: Run) => void;
  isCreatingRun: boolean;
  setIsCreatingRun: (v: boolean) => void;
  loadRuns: () => void;
  viewMode: "runs" | "studies" | "playground";
  setViewMode: (v: "runs" | "studies" | "playground") => void;
}) {
  return (
    <div className="flex flex-col h-full w-full border-r border-border/20 bg-[#09090b]">
      <div className="border-b border-border/10 pb-4 pt-4 px-4 flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-primary/10 text-primary border border-primary/20 shrink-0">
            <img src={logoUrl} alt="Logo" className="size-5" />
          </div>
          <div className="grid flex-1 text-left text-sm leading-tight">
            <span className="truncate font-bold tracking-wider text-zinc-100">
              TRICKED AI
            </span>
            <span className="truncate text-[10px] text-zinc-500 uppercase tracking-widest font-semibold">
              Control Center
            </span>
          </div>
        </div>
        <div className="flex bg-black/40 p-1 rounded-lg border border-border/10">
          <button
            onClick={() => setViewMode("runs")}
            className={`flex-1 text-[11px] uppercase tracking-widest font-bold py-1.5 rounded-md transition-colors ${viewMode === "runs" ? "bg-zinc-800 text-zinc-100 shadow-sm" : "text-zinc-500 hover:text-zinc-300"}`}
          >
            Experiments
          </button>
          <button
            onClick={() => setViewMode("studies")}
            className={`flex-1 text-[11px] uppercase tracking-widest font-bold py-1.5 rounded-md transition-colors ${viewMode === "studies" ? "bg-zinc-800 text-zinc-100 shadow-sm" : "text-zinc-500 hover:text-zinc-300"}`}
          >
            Tuning Lab
          </button>
          <button
            onClick={() => setViewMode("playground")}
            className={`flex-1 text-[11px] uppercase tracking-widest font-bold py-1.5 rounded-md transition-colors ${viewMode === "playground" ? "bg-zinc-800 text-zinc-100 shadow-sm" : "text-zinc-500 hover:text-zinc-300"}`}
          >
            Playground
          </button>
        </div>
        <div className="flex flex-col gap-2">
          <Button
            size="sm"
            variant="default"
            className="w-full text-xs font-medium shadow-md shadow-primary/20"
            onClick={() => setIsCreatingRun(true)}
          >
            <Plus className="w-3 h-3 mr-1" /> New Simple Run
          </Button>
        </div>
      </div>
      <div className="flex min-h-0 flex-1 flex-col overflow-auto">
        <div className="relative flex w-full min-w-0 flex-col p-2 flex-1 overflow-hidden">
          <div className="px-5 py-2 text-[10px] font-bold text-zinc-500 uppercase tracking-widest border-b border-border/10 mb-1">
            Experiment Library
          </div>
          <div className="flex-1 overflow-hidden flex flex-col">
            {isCreatingRun ? (
              <CreateSimpleRunSidebar onClose={() => setIsCreatingRun(false)} loadRuns={loadRuns} />
            ) : (
              <RunsSidebarList
                runs={runs}
                selectedRunId={selectedRunId}
                setSelectedRunId={setSelectedRunId}
                selectedDashboardRuns={selectedDashboardRuns}
                setSelectedDashboardRuns={setSelectedDashboardRuns}
                toggleDashboardRun={toggleDashboardRun}
                runColors={runColors}
                setRunColors={setRunColors}
                defaultColors={defaultColors}
                setRunToRename={setRunToRename}
                setNewName={setNewName}
                setRunToFlush={setRunToFlush}
                setRunToDelete={setRunToDelete}
                handleEngineCmd={handleEngineCmd}
                handleClone={handleClone}
              />
            )}
          </div>
        </div>
      </div>
      <HardwareMiniDashboard />
    </div>
  );
}
