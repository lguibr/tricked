import React, { useState } from "react";
import { ActiveJob } from "@/bindings/ActiveJob";
import { Run } from "@/bindings/Run";
import { LiveLogsViewer } from "./LiveLogsViewer";

interface ProcessManagerWorkspaceProps {
  runs: Run[];
  runLogs: Record<string, string[]>;
  activeJobs?: ActiveJob[];
  selectedDashboardRuns: string[];
  toggleDashboardRun: (id: string, pressed: boolean) => void;
  logsEndRef: React.MutableRefObject<Record<string, HTMLDivElement | null>>;
  runColors: Record<string, string>;
}

export function ProcessManagerWorkspace({
  runs,
  runLogs,
  selectedDashboardRuns,
  toggleDashboardRun,
  logsEndRef,
  runColors,
}: ProcessManagerWorkspaceProps) {
  const [copiedLogId, setCopiedLogId] = useState<string | null>(null);

  const handleCopyLogs = (id: string, logs: string) => {
    navigator.clipboard.writeText(logs);
    setCopiedLogId(id);
    setTimeout(() => setCopiedLogId(null), 2000);
  };

  return (
    <div className="flex flex-col w-full h-full bg-black relative group">
      <div className="h-full w-full relative">
        <LiveLogsViewer
          runs={runs}
          runLogs={runLogs}
          selectedLogRunIds={selectedDashboardRuns}
          toggleLogRun={toggleDashboardRun}
          handleCopyLogs={handleCopyLogs}
          copiedLogId={copiedLogId}
          logsEndRef={logsEndRef}
          runColors={runColors}
        />
      </div>
    </div>
  );
}
