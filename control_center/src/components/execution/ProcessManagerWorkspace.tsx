import { useState, useRef } from "react";
import { LiveLogsViewer } from "./LiveLogsViewer";
import { useAppStore } from "@/store/useAppStore";

export function ProcessManagerWorkspace() {
  const runs = useAppStore((state) => state.runs);
  const runLogs = useAppStore((state) => state.runLogs);
  const selectedDashboardRuns = useAppStore(
    (state) => state.selectedDashboardRuns,
  );
  const toggleDashboardRun = useAppStore((state) => state.toggleDashboardRun);
  const runColors = useAppStore((state) => state.runColors);
  const logsEndRef = useRef<Record<string, HTMLDivElement | null>>({});
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
