import { useState } from "react";
import { LiveLogsViewer } from "./LiveLogsViewer";
import { useAppStore } from "@/store/useAppStore";

export function ProcessManagerWorkspace() {
  const runs = useAppStore((state) => state.runs);
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
          handleCopyLogs={handleCopyLogs}
          copiedLogId={copiedLogId}
        />
      </div>
    </div>
  );
}
