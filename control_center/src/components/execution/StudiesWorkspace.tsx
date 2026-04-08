import { OptunaStudyDashboard } from "@/components/OptunaStudyDashboard";

export function StudiesWorkspace() {
  return (
    <div className="flex h-full w-full bg-[#0a0a0a] text-zinc-200">
      <div className="flex-1 flex flex-col bg-[#020202] overflow-hidden">
        <OptunaStudyDashboard />
      </div>
    </div>
  );
}
