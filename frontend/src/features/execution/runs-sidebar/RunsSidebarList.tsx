import { useState } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { VscClearAll, VscCheckAll } from "react-icons/vsc";
import { useAppStore } from "@/store/useAppStore";
import { RunsSidebarListItem } from "./RunsSidebarListItem";

export function RunsSidebarList({
  filterType,
}: {
  filterType?: "SINGLE" | "TUNING_TRIAL" | "STUDY";
}) {
  const allRuns = useAppStore((state) => state.runs);
  const runs = filterType
    ? allRuns.filter((r) => r.type === filterType)
    : allRuns.filter((r) => r.type !== "STUDY");

  const isStudies = filterType === "STUDY";
  const selectedDashboardRuns = useAppStore(
    (state) => state.selectedDashboardRuns,
  );
  const setSelectedDashboardRuns = useAppStore(
    (state) => state.setSelectedDashboardRuns,
  );

  const defaultColors = [
    "#10b981",
    "#3b82f6",
    "#f59e0b",
    "#8b5cf6",
    "#ec4899",
    "#ef4444",
    "#14b8a6",
  ];

  const [expandedRunId, setExpandedRunId] = useState<string | null>(null);

  const visibleRunIds = runs.map((r) => r.id);
  const allSelected =
    visibleRunIds.length > 0 &&
    visibleRunIds.every((id) => selectedDashboardRuns.includes(id));

  const handleToggleAll = () => {
    if (allSelected) {
      setSelectedDashboardRuns(
        selectedDashboardRuns.filter((id) => !visibleRunIds.includes(id)),
      );
    } else {
      const newSelected = [
        ...new Set([...selectedDashboardRuns, ...visibleRunIds]),
      ];
      setSelectedDashboardRuns(newSelected);
    }
  };

  return (
    <div className="flex flex-col flex-1 overflow-hidden">
      <div className="flex items-center justify-between px-3 py-1 bg-[#0a0a0a] border-b border-white/5 shrink-0 select-none">
        <span className="text-[8px] font-bold text-zinc-500 uppercase tracking-widest">
          {runs.length} Active {isStudies ? "Studies" : "Runs"}
        </span>
        {runs.length > 0 && (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleToggleAll}
            className="h-5 px-1.5 py-0 text-[8px] text-zinc-400 hover:text-white uppercase tracking-widest hover:bg-white/5"
          >
            {allSelected ? (
              <VscClearAll className="w-3.5 h-3.5 mr-1" />
            ) : (
              <VscCheckAll className="w-3.5 h-3.5 mr-1" />
            )}
            {allSelected ? "Deselect All" : "Select All"}
          </Button>
        )}
      </div>

      <ScrollArea className="flex-1 p-0">
        <div className="flex flex-col">
          {runs.map((run, idx) => (
            <RunsSidebarListItem
              key={run.id}
              run={run}
              idx={idx}
              isStudies={isStudies}
              expandedRunId={expandedRunId}
              setExpandedRunId={setExpandedRunId}
              defaultColors={defaultColors}
            />
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}
