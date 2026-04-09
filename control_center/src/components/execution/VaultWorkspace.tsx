import { VscWorkspaceTrusted } from "react-icons/vsc";
import { useVaultStore } from "@/store/useVaultStore";
import { VaultReplayPlayer } from "./VaultReplayPlayer";

export function VaultWorkspace() {
  const selectedGameIndex = useVaultStore((state) => state.selectedGameIndex);

  if (selectedGameIndex !== null) {
    return <VaultReplayPlayer />;
  }

  return (
    <div className="flex flex-col h-full w-full bg-[#020202] text-zinc-100 p-6 gap-4 items-center justify-center border-l border-white/5">
      <VscWorkspaceTrusted className="w-16 h-16 text-emerald-500/20" />
      <h2 className="text-xl font-black tracking-widest uppercase text-zinc-400">
        Top 100 Games Vault
      </h2>
      <div className="bg-[#080808] border border-white/5 rounded p-4 text-center max-w-sm mt-4">
        <p className="text-xs text-zinc-400 uppercase tracking-widest font-black">
          Select a game from the sidebar to view its step-by-step playback.
        </p>
      </div>
    </div>
  );
}
