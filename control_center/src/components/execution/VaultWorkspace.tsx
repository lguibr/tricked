import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { ScrollArea } from "@/components/ui/scroll-area";
import { VscRefresh, VscWorkspaceTrusted, VscFileCode } from "react-icons/vsc";
import { Button } from "@/components/ui/button";
import { useAppStore } from "@/store/useAppStore";

export function VaultWorkspace() {
  const runId = useAppStore((state) => state.selectedRunId);
  const [games, setGames] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchVault = async () => {
    if (!runId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await invoke<any[]>("get_vault_games", { runId });
      setGames(data);
    } catch (e: any) {
      setError(e.toString());
      setGames([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchVault();
  }, [runId]);

  if (!runId) {
    return (
      <div className="flex h-full w-full items-center justify-center bg-[#050505] text-zinc-600 text-[10px] font-black uppercase tracking-widest">
        Select a run in the sidebar to view its Vault
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-[#020202] text-zinc-100 p-2 gap-2">
      <div className="flex justify-between items-center border-b border-white/5 pb-2 px-1">
        <div className="flex items-center gap-1.5">
          <VscWorkspaceTrusted className="w-4 h-4 text-emerald-500" />
          <h2 className="text-[10px] font-black tracking-widest uppercase text-zinc-200">
            Top 100 Games Vault
          </h2>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={fetchVault}
          disabled={loading}
          className="h-6 px-2 py-0 text-[9px] font-black tracking-widest uppercase gap-1.5 bg-white/5 border-white/10 hover:bg-white/10 text-zinc-300"
        >
          <VscRefresh
            className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`}
          />
          Refresh
        </Button>
      </div>

      <div className="flex-1 overflow-hidden">
        {error ? (
          <div className="flex flex-col items-center justify-center w-full h-full text-red-500/50 gap-2">
            <VscFileCode className="w-8 h-8 opacity-50" />
            <p className="text-[10px] font-bold uppercase tracking-widest">
              {error}
            </p>
          </div>
        ) : games.length === 0 ? (
          <div className="flex flex-col items-center justify-center w-full h-full text-zinc-600 gap-2">
            <VscWorkspaceTrusted className="w-8 h-8 opacity-20" />
            <p className="text-[10px] font-bold uppercase tracking-widest">
              No games in vault yet
            </p>
          </div>
        ) : (
          <ScrollArea className="h-full pr-2">
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6 gap-1.5">
              {games.map((g, i) => (
                <div
                  key={i}
                  className="bg-[#080808] border border-white/5 rounded-sm p-2 flex flex-col gap-1.5 hover:border-emerald-500/30 transition-colors group"
                >
                  <div className="flex justify-between items-center border-b border-white/5 pb-1">
                    <span className="font-black text-[12px] text-emerald-400">
                      {g.episode_score.toFixed(0)}{" "}
                      <span className="text-[8px] text-zinc-500 uppercase tracking-widest">
                        Pts
                      </span>
                    </span>
                    <span className="text-[8px] font-black text-zinc-500 bg-black px-1 py-0.5 rounded-sm uppercase tracking-widest border border-white/5">
                      D{g.difficulty_setting}
                    </span>
                  </div>

                  <div className="grid grid-cols-1 gap-y-0.5 mt-0.5 text-[9px] text-zinc-400 font-mono">
                    <div className="flex justify-between hover:bg-white/5 px-1 py-0.5 rounded-sm">
                      <span className="text-zinc-500">LINES</span>
                      <span className="font-bold text-zinc-200">
                        {g.lines_cleared}
                      </span>
                    </div>
                    <div className="flex justify-between hover:bg-white/5 px-1 py-0.5 rounded-sm">
                      <span className="text-zinc-500">STEPS</span>
                      <span className="font-bold text-zinc-200">
                        {g.steps.length}
                      </span>
                    </div>
                    <div className="flex justify-between hover:bg-white/5 px-1 py-0.5 rounded-sm">
                      <span className="text-zinc-500">DEPTH</span>
                      <span className="font-bold text-zinc-200">
                        {g.mcts_depth_mean.toFixed(1)}
                      </span>
                    </div>
                    <div className="flex justify-between hover:bg-white/5 px-1 py-0.5 rounded-sm">
                      <span className="text-zinc-500">TIME</span>
                      <span className="font-bold text-amber-400/80">
                        {g.mcts_search_time_mean.toFixed(1)}ms
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        )}
      </div>
    </div>
  );
}
