import { Layers } from "lucide-react";
import { GlassCard } from "@/components/dashboard/GlassCard";
import { CompactTrialParams } from "@/components/execution/HydraConfigViewer";
import { Trial } from "@/hooks/useOptimizerStudy";

export function OptimizerStudyTrialsTable({ trials }: { trials: Trial[] }) {
  return (
    <GlassCard className="flex flex-col shrink-0 min-h-[450px] mb-4">
      <div className="p-4 border-b border-white/5 bg-[#0a0a0c] flex items-center justify-between z-30 shadow-sm relative">
        <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>
        <div className="flex items-center gap-2.5">
          <div className="p-1.5 bg-purple-500/10 rounded-md border border-purple-500/20">
            <Layers className="w-4 h-4 text-purple-400" />
          </div>
          <h3 className="text-zinc-100 font-bold text-xs uppercase tracking-widest text-shadow-sm">
            Study Execution Protocol
          </h3>
        </div>
        <p className="text-[10px] text-zinc-500 font-mono tracking-wider">
          {trials.length} entries matching current criteria
        </p>
      </div>
      <div className="flex-1 overflow-auto custom-scrollbar bg-[#0f0f12]/50">
        <table className="w-full text-xs text-left whitespace-nowrap border-collapse">
          <thead className="text-zinc-500 sticky top-0 z-20 bg-[#0f0f12]/90 backdrop-blur-md shadow-sm border-b border-white/5">
            <tr>
              <th className="px-6 py-4 font-black uppercase tracking-widest text-[9px] border-b border-white/[0.02]">
                Trial #
              </th>
              <th className="px-6 py-4 font-black uppercase tracking-widest text-[9px] border-b border-white/[0.02]">
                Status
              </th>
              <th className="px-6 py-4 font-black uppercase tracking-widest text-[9px] border-b border-white/[0.02]">
                Values{" "}
                <span className="text-zinc-600 font-normal lowercase">
                  (HW / Loss)
                </span>
              </th>
              <th className="px-6 py-4 font-black uppercase tracking-widest text-[9px] border-b border-white/[0.02] w-full">
                Config Trajectory Space
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/[0.02]">
            {trials
              .slice()
              .reverse()
              .map((t) => (
                <tr
                  key={t.number}
                  className="hover:bg-white/[0.03] transition-colors group cursor-default"
                >
                  <td className="px-6 py-4 font-mono text-zinc-500 group-hover:text-zinc-300 transition-colors">
                    {t.number.toString().padStart(4, "0")}
                  </td>
                  <td className="px-6 py-4">
                    <span
                      className={`inline-flex items-center px-2.5 py-1 rounded text-[9px] font-black uppercase tracking-widest border shadow-sm ${
                        t.state === "COMPLETE"
                          ? "bg-blue-500/10 text-blue-400 border-blue-500/20"
                          : t.state === "PRUNED"
                            ? "bg-zinc-500/10 text-zinc-400 border-zinc-500/20"
                            : t.state === "RUNNING"
                              ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30 animate-pulse shadow-[0_0_12px_rgba(16,185,129,0.25)]"
                              : "bg-red-500/10 text-red-400 border-red-500/20"
                      }`}
                    >
                      {t.state}
                    </span>
                  </td>
                  <td className="px-6 py-4 font-mono text-zinc-400">
                    {Array.isArray(t.value) &&
                    t.value.length >= 2 &&
                    t.value[0] != null &&
                    t.value[1] != null ? (
                      <div className="flex gap-2.5">
                        <span className="text-blue-400/90 font-bold">
                          {Number(t.value[0]).toFixed(2)}
                        </span>
                        <span className="text-zinc-700">/</span>
                        <span className="text-purple-400/90 font-bold">
                          {Number(t.value[1]) > 1e100 ? (
                            <span className="text-red-500 font-bold tracking-widest text-[9px] bg-red-500/10 px-1 py-0.5 rounded">
                              DIVERGED (NaN)
                            </span>
                          ) : (
                            Number(t.value[1]).toFixed(4)
                          )}
                        </span>
                      </div>
                    ) : Array.isArray(t.value) &&
                      t.value.length > 0 &&
                      t.value[0] != null ? (
                      <span className="text-zinc-300 font-bold">
                        {Number(t.value[0]) > 1e100
                          ? "INFINITY"
                          : Number(t.value[0]).toFixed(4)}
                      </span>
                    ) : t.value != null && !Array.isArray(t.value) ? (
                      <span className="text-zinc-300 font-bold">
                        {Number(t.value) > 1e100
                          ? "INFINITY"
                          : Number(t.value).toFixed(4)}
                      </span>
                    ) : (
                      <span className="text-zinc-700 font-bold">-</span>
                    )}
                  </td>
                  <td className="px-6 py-4">
                    <div className="opacity-70 group-hover:opacity-100 transition-opacity">
                      <CompactTrialParams params={t.params || {}} />
                    </div>
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </GlassCard>
  );
}
