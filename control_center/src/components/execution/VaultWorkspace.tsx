import { VscWorkspaceTrusted } from "react-icons/vsc";
import { useAppStore } from "@/store/useAppStore";
import { useVaultStore } from "@/store/useVaultStore";

export function VaultWorkspace() {
  const runId = useAppStore((state) => state.selectedRunId);
  const games = useVaultStore((state) => state.games);

  if (!runId) {
    return (
      <div className="flex h-full w-full items-center justify-center bg-[#050505] text-zinc-600 text-[10px] font-black uppercase tracking-widest">
        Select a run in the sidebar to view its Vault
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-[#020202] text-zinc-100 p-6 gap-4 items-center justify-center">
      <VscWorkspaceTrusted className="w-16 h-16 text-emerald-500/20" />
      <h2 className="text-xl font-black tracking-widest uppercase text-zinc-400">
        Top 100 Games Vault
      </h2>
      <p className="text-sm font-mono text-zinc-500">
        {games.length} games retrieved.
      </p>
      <div className="bg-zinc-900 border border-zinc-800 rounded p-4 text-center max-w-sm mt-4">
        <p className="text-xs text-zinc-400">
          The game list has been moved to the sidebar. Select a game from the
          sidebar to view detailed telemetry and playback (coming soon).
        </p>
      </div>
    </div>
  );
}
