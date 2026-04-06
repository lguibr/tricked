import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { ScrollArea } from "@/components/ui/scroll-area";
import { RefreshCw, Trophy, FileJson } from "lucide-react";
import { Button } from "@/components/ui/button";

export function VaultWorkspace({ runId }: { runId: string | null }) {
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
      <div className="flex h-full w-full items-center justify-center bg-[#050505] text-zinc-500">
        Select a run in the sidebar to view its Vault.
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-[#0a0a0a] text-zinc-100 p-4 gap-4">
      <div className="flex justify-between items-center border-b border-border/10 pb-4">
        <div className="flex items-center gap-2">
          <Trophy className="w-5 h-5 text-yellow-500" />
          <h2 className="text-lg font-bold tracking-wider uppercase text-zinc-200">
            Top 100 Games Vault
          </h2>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={fetchVault}
          disabled={loading}
          className="gap-2"
        >
          <RefreshCw className={`w-3 h-3 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      <div className="flex-1 overflow-hidden">
        {error ? (
          <div className="flex flex-col items-center justify-center w-full h-full text-zinc-500 gap-2">
            <FileJson className="w-8 h-8 opacity-20" />
            <p className="text-sm">{error}</p>
          </div>
        ) : games.length === 0 ? (
          <div className="flex flex-col items-center justify-center w-full h-full text-zinc-600 gap-2">
            <Trophy className="w-8 h-8 opacity-20" />
            <p className="text-sm">No games in vault yet.</p>
          </div>
        ) : (
          <ScrollArea className="h-full pr-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
              {games.map((g, i) => (
                <div
                  key={i}
                  className="bg-black/40 border border-border/10 rounded-lg p-4 flex flex-col gap-2 hover:border-primary/50 transition-colors"
                >
                  <div className="flex justify-between items-center">
                    <span className="font-bold text-lg text-primary">
                      {g.episode_score.toFixed(0)} Pts
                    </span>
                    <span className="text-xs text-zinc-500 bg-zinc-900 px-2 py-0.5 rounded uppercase tracking-wider">
                      Diff: {g.difficulty_setting}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-x-4 gap-y-2 mt-2 text-xs text-zinc-400">
                    <div className="flex justify-between">
                      <span>Lines Cleared</span>
                      <span className="text-zinc-200">{g.lines_cleared}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Total Steps</span>
                      <span className="text-zinc-200">{g.steps.length}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Avg Depth</span>
                      <span className="text-zinc-200">
                        {g.mcts_depth_mean.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Avg Time</span>
                      <span className="text-zinc-200">
                        {g.mcts_search_time_mean.toFixed(2)}ms
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
