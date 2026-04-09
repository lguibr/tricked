import { useEffect } from "react";
import {
  VscRefresh,
  VscFileCode,
  VscWorkspaceTrusted,
  VscArrowUp,
  VscArrowDown,
} from "react-icons/vsc";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { useVaultStore } from "@/store/useVaultStore";

export function VaultSidebar() {
  const games = useVaultStore((state) => state.games);
  const loading = useVaultStore((state) => state.loading);
  const error = useVaultStore((state) => state.error);
  const sortBy = useVaultStore((state) => state.sortBy);
  const sortDirection = useVaultStore((state) => state.sortDirection);
  const selectedGameIndex = useVaultStore((state) => state.selectedGameIndex);

  const fetchVault = useVaultStore((state) => state.fetchVault);
  const setSortBy = useVaultStore((state) => state.setSortBy);
  const setSelectedGameIndex = useVaultStore((state) => state.setSelectedGameIndex);

  useEffect(() => {
    fetchVault();
  }, []);

  const sortedGamesWithIndex = games.map((g, originalIndex) => ({ g, originalIndex }))
    .sort((a, b) => {
      let diff = 0;
      if (sortBy === "score") diff = a.g.episode_score - b.g.episode_score;
      else if (sortBy === "lines") diff = a.g.lines_cleared - b.g.lines_cleared;
      else if (sortBy === "depth") diff = a.g.mcts_depth_mean - b.g.mcts_depth_mean;
      else if (sortBy === "time") diff = a.g.mcts_search_time_mean - b.g.mcts_search_time_mean;

      return sortDirection === "desc" ? -diff : diff;
    });

  const sortBtn = (field: typeof sortBy, label: string) => (
    <Button
      variant="ghost"
      size="sm"
      onClick={() => setSortBy(field)}
      className={`h-6 px-2 py-0 text-[9px] font-black tracking-widest uppercase gap-1 ${
        sortBy === field
          ? "text-emerald-400 bg-white/10"
          : "text-zinc-500 hover:bg-white/5 hover:text-zinc-300"
      }`}
    >
      {label}
      {sortBy === field &&
        (sortDirection === "asc" ? (
          <VscArrowUp className="w-3 h-3" />
        ) : (
          <VscArrowDown className="w-3 h-3" />
        ))}
    </Button>
  );

  return (
    <div className="flex flex-col h-full bg-[#0a0a0a] text-foreground overflow-hidden">
      <div className="px-3 py-2 border-b border-border/10 flex flex-col gap-2 bg-black/20 shrink-0">
        <div className="flex justify-between items-center">
          <h3 className="text-xs font-bold text-zinc-100 uppercase tracking-widest flex items-center gap-1.5">
            <VscWorkspaceTrusted className="w-4 h-4 text-emerald-500" />
            Top 100 Games Vault
          </h3>
          <Button
            variant="outline"
            size="icon"
            onClick={() => fetchVault()}
            disabled={loading}
            className="h-6 w-6 bg-white/5 border-white/10 hover:bg-white/10 text-zinc-300"
          >
            <VscRefresh
              className={`w-3 h-3 ${loading ? "animate-spin" : ""}`}
            />
          </Button>
        </div>
        <div className="flex gap-1 overflow-x-auto custom-scrollbar pb-1">
          {sortBtn("score", "Score")}
          {sortBtn("lines", "Lines")}
          {sortBtn("depth", "Depth")}
          {sortBtn("time", "Time")}
        </div>
      </div>

      <div className="flex-1 overflow-hidden p-2 bg-[#030303]">
        {error ? (
          <div className="flex flex-col items-center justify-center w-full h-full text-red-500/50 gap-2">
            <VscFileCode className="w-8 h-8 opacity-50" />
            <p className="text-[10px] font-bold uppercase tracking-widest text-center">
              {error}
            </p>
          </div>
        ) : games.length === 0 ? (
          <div className="flex flex-col items-center justify-center w-full h-full text-zinc-600 gap-2">
            <VscWorkspaceTrusted className="w-8 h-8 opacity-20" />
            <p className="text-[10px] font-bold uppercase tracking-widest text-center">
              {loading ? "Loading..." : "No games in vault yet"}
            </p>
          </div>
        ) : (
          <ScrollArea className="h-full pr-2">
            <div className="flex flex-col gap-1.5">
              {sortedGamesWithIndex.map(({ g, originalIndex }, i) => (
                <div
                  key={i}
                  onClick={() => setSelectedGameIndex(originalIndex)}
                  className={`border rounded-sm p-2 flex flex-col gap-1.5 transition-colors group cursor-pointer ${
                    selectedGameIndex === originalIndex
                      ? "bg-emerald-500/10 border-emerald-500/50 shadow-[0_0_10px_rgba(16,185,129,0.1)]"
                      : "bg-[#080808] border-white/5 hover:border-emerald-500/30"
                  }`}
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

                  <div className="flex items-center justify-between text-[8px] font-mono mt-1">
                     <span className="text-zinc-300 font-bold max-w-[120px] truncate" title={g.source_run_name}>
                       {g.source_run_name}
                     </span>
                     <span className="text-zinc-500 bg-white/5 px-2 py-0.5 rounded uppercase tracking-widest font-black">
                       {g.run_type}
                     </span>
                  </div>

                  <div className="grid grid-cols-2 gap-x-2 gap-y-0.5 mt-1 text-[9px] text-zinc-400 font-mono">
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
                        {parseFloat(g.mcts_depth_mean as any).toFixed(1)}
                      </span>
                    </div>
                    <div className="flex justify-between hover:bg-white/5 px-1 py-0.5 rounded-sm">
                      <span className="text-zinc-500">TIME</span>
                      <span className="font-bold text-zinc-200">
                        {parseFloat(g.mcts_search_time_mean as any).toFixed(1)}ms
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
