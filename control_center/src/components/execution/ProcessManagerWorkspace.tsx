import React, { useState } from "react";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { ActiveJob } from "@/bindings/ProcessInfo";
import { Run } from "@/bindings/Run";
import { ProcessTreeView } from "./ProcessTreeView";
import { LiveLogsViewer } from "./LiveLogsViewer";

interface ProcessManagerWorkspaceProps {
  runs: Run[];
  runLogs: Record<string, string[]>;
  activeJobs: ActiveJob[];
  selectedDashboardRuns: string[];
  toggleDashboardRun: (id: string, pressed: boolean) => void;
  logsEndRef: React.MutableRefObject<Record<string, HTMLDivElement | null>>;
}

export function ProcessManagerWorkspace({
  runs,
  runLogs,
  activeJobs,
  selectedDashboardRuns,
  toggleDashboardRun,
  logsEndRef,
}: ProcessManagerWorkspaceProps) {
  const [copiedLogId, setCopiedLogId] = useState<string | null>(null);

  const handleCopyLogs = (id: string, logs: string) => {
    navigator.clipboard.writeText(logs);
    setCopiedLogId(id);
    setTimeout(() => setCopiedLogId(null), 2000);
  };

  return (
    <div className="flex flex-col w-full h-full bg-black border-t border-border/20 relative group">
      <ResizablePanelGroup direction="horizontal">
        <ResizablePanel defaultSize={30} minSize={20}>
          <ProcessTreeView jobs={activeJobs} />
        </ResizablePanel>

        <ResizableHandle className="w-1 bg-border/20 hover:bg-primary/50 transition-colors" />

        <ResizablePanel defaultSize={70} minSize={30}>
          <div className="h-full w-full relative">
            <LiveLogsViewer
              runs={runs}
              runLogs={runLogs}
              selectedLogRunIds={selectedDashboardRuns}
              toggleLogRun={toggleDashboardRun}
              handleCopyLogs={handleCopyLogs}
              copiedLogId={copiedLogId}
              logsEndRef={logsEndRef}
            />
          </div>
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
}
