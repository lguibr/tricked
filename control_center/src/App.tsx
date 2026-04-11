import { useEffect } from "react";
import { BarChart2 } from "lucide-react";
import { MetricsDashboard } from "@/features/dashboard/metrics/MetricsDashboard";
import { StudiesWorkspace } from "@/components/execution/StudiesWorkspace";
import { VaultWorkspace } from "@/components/execution/VaultWorkspace";
import { TrickedPlayground } from "@/features/playground/TrickedPlayground";
import { EvaluationWorkspace } from "@/features/evaluation/EvaluationWorkspace";
import { AppSidebar } from "@/components/app-sidebar";
import { AppDialogs } from "@/features/app/AppDialogs";
import {
  ResizablePanel,
  ResizablePanelGroup,
  ResizableHandle,
} from "@/components/ui/resizable";

import type { ActiveJob } from "@/bindings/ActiveJob";
import { ProcessManagerWorkspace } from "@/components/execution/ProcessManagerWorkspace";
import { HardwareMiniDashboard } from "@/features/dashboard/hardware/HardwareMiniDashboard";
import { CpuSunburstChart } from "@/components/execution/CpuSunburstChart";
import { ProcessTreeView } from "@/components/execution/ProcessTreeView";

import { useAppStore } from "@/store/useAppStore";

const isTauri =
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

export default function App() {
  const viewMode = useAppStore((state) => state.viewMode);
  const loadRuns = useAppStore((state) => state.loadRuns);

  const selectedDashboardRunsLength = useAppStore(
    (state) => state.selectedDashboardRuns.length,
  );

  useEffect(() => {
    loadRuns();
    const interval = setInterval(loadRuns, 3000);

    let unlistenLog: (() => void) | undefined;
    let unlistenTelemetry: (() => void) | undefined;
    let unlistenEngineTelemetry: (() => void) | undefined;
    let isCancelled = false;

    import("@tauri-apps/api/event").then(({ listen }) => {
      if (!isTauri) {
        console.warn("Skipping Tauri listen in browser env");
        return;
      }
      // 1. Listen to the BATCHED log event and emit Native Event to bypass React VDOM DDOS
      listen("log_event_batch", (event: any) => {
        const batchedLogs = event.payload; // Array of logs
        if (batchedLogs.length > 0) {
           window.dispatchEvent(
             new CustomEvent("engine_log_batch", { detail: batchedLogs }),
           );
        }
      }).then((u) => {
        if (isCancelled) {
          u();
        } else {
          unlistenLog = u;
        }
      });

      // 2. FIX THE EVENT NAME: Backend emits "engine_telemetry", not "live_metric"
      listen("engine_telemetry", (event: any) => {
        const metric = event.payload;
        const state = useAppStore.getState();
        if (state.selectedDashboardRuns.includes(metric.run_id)) {
          // We dispatch a custom DOM event so MetricsDashboard can pick it up
          // WITHOUT triggering a global Zustand re-render.
          window.dispatchEvent(
            new CustomEvent("engine_telemetry_update", { detail: metric }),
          );
        }
      }).then((u) => {
        if (isCancelled) {
          u();
        } else {
          unlistenEngineTelemetry = u;
        }
      });

      // Throttle process telemetry to 500ms to avoid React re-render DDOS
      let lastProcessTelemetry = 0;
      listen("process_telemetry", (event: any) => {
        const now = Date.now();
        if (now - lastProcessTelemetry > 500) {
          useAppStore.getState().setActiveJobs(event.payload as ActiveJob[]);
          lastProcessTelemetry = now;
        }
      }).then((u) => {
        if (isCancelled) {
          u();
        } else {
          unlistenTelemetry = u;
        }
      });
    });

    return () => {
      isCancelled = true;
      clearInterval(interval);
      if (unlistenLog) unlistenLog();
      if (unlistenTelemetry) unlistenTelemetry();
      if (unlistenEngineTelemetry) unlistenEngineTelemetry();
    };
  }, [loadRuns]);

  return (
    <div className="h-screen w-screen overflow-hidden bg-background text-foreground flex">
      <ResizablePanelGroup direction="horizontal" className="h-full w-full">
        <ResizablePanel defaultSize={20} minSize={15} maxSize={40}>
          <AppSidebar />
        </ResizablePanel>

        <ResizableHandle className="w-1 bg-border/20 hover:bg-primary/50 transition-colors cursor-col-resize z-50" />

        <ResizablePanel defaultSize={80}>
          <ResizablePanelGroup direction="vertical">
            <ResizablePanel defaultSize={75} minSize={30}>
              <div className="w-full h-full overflow-hidden bg-black animate-in fade-in duration-300">
                {viewMode === "playground" ? (
                  <TrickedPlayground />
                ) : viewMode === "vault" ? (
                  <VaultWorkspace />
                ) : viewMode === "studies" ? (
                  <StudiesWorkspace />
                ) : viewMode === "evaluation" ? (
                  <EvaluationWorkspace />
                ) : selectedDashboardRunsLength > 0 ? (
                  <MetricsDashboard />
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
              <ProcessManagerWorkspace />
            </ResizablePanel>
          </ResizablePanelGroup>
        </ResizablePanel>

        <ResizableHandle className="w-1 bg-border/20 hover:bg-primary/50 transition-colors cursor-col-resize z-50" />

        <ResizablePanel defaultSize={20} minSize={15} maxSize={30}>
          <div className="flex flex-col h-full bg-[#050505] border-l border-border/20">
            <div className="flex-1 min-h-0 flex flex-col">
              <div className="flex-1 min-h-0 border-b border-border/10">
                <CpuSunburstChart />
              </div>
              <div className="flex-1 min-h-0">
                <ProcessTreeView />
              </div>
            </div>
            <HardwareMiniDashboard />
          </div>
        </ResizablePanel>
      </ResizablePanelGroup>

      <AppDialogs />
    </div>
  );
}
