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

import { ProcessManagerWorkspace } from "@/components/execution/ProcessManagerWorkspace";
import { HardwareMiniDashboard } from "@/features/dashboard/hardware/HardwareMiniDashboard";
import { ProcessTreeView } from "@/components/execution/ProcessTreeView";

import { useAppStore } from "@/store/useAppStore";

export default function App() {
  const viewMode = useAppStore((state) => state.viewMode);
  const loadRuns = useAppStore((state) => state.loadRuns);
  const setActiveJobs = useAppStore((state) => state.setActiveJobs);

  const selectedDashboardRunsLength = useAppStore(
    (state) => state.selectedDashboardRuns.length,
  );

  useEffect(() => {
    // Initial fetch to load the data synchronously for fast paint
    loadRuns();

    let ws: WebSocket;
    let wsJobs: WebSocket;
    let reconnectTimeout: ReturnType<typeof setTimeout>;
    let reconnectJobsTimeout: ReturnType<typeof setTimeout>;

    const connectRuns = () => {
      ws = new WebSocket("ws://127.0.0.1:8000/api/ws/runs");

      ws.onmessage = (event) => {
        try {
          const runs = JSON.parse(event.data);
          loadRuns(runs);
        } catch (e) {
          console.error("Failed to parse runs ws message", e);
        }
      };

      ws.onclose = () => {
        reconnectTimeout = setTimeout(connectRuns, 2000);
      };

      ws.onerror = () => ws.close();
    };

    const connectJobs = () => {
      wsJobs = new WebSocket("ws://127.0.0.1:8000/api/ws/jobs");

      wsJobs.onmessage = (event) => {
        try {
          const jobs = JSON.parse(event.data);
          setActiveJobs(jobs);
        } catch (e) {
          console.error("Failed to parse jobs ws message", e);
        }
      };

      wsJobs.onclose = () => {
        reconnectJobsTimeout = setTimeout(connectJobs, 2000);
      };

      wsJobs.onerror = () => wsJobs.close();
    };

    connectRuns();
    connectJobs();

    return () => {
      clearTimeout(reconnectTimeout);
      clearTimeout(reconnectJobsTimeout);
      if (ws) {
        ws.onclose = null;
        ws.close();
      }
      if (wsJobs) {
        wsJobs.onclose = null;
        wsJobs.close();
      }
    };
  }, [loadRuns]);

  return (
    <div className="h-screen w-screen overflow-hidden bg-background text-foreground flex">
      <ResizablePanelGroup direction="horizontal" className="h-full w-full">
        <ResizablePanel defaultSize={20} minSize={15} maxSize={40}>
          <AppSidebar />
        </ResizablePanel>

        <ResizableHandle className="w-1 bg-border/20 hover:bg-primary/50 transition-colors cursor-col-resize z-50" />

        <ResizablePanel defaultSize={60}>
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
