import { usePlaygroundStore } from "@/store/usePlaygroundStore";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export function PlaygroundSidebar() {
  const gameState = usePlaygroundStore((state) => state.gameState);
  const difficulty = usePlaygroundStore((state) => state.difficulty);
  const highScore = usePlaygroundStore((state) => state.highScore);
  const setDifficulty = usePlaygroundStore((state) => state.setDifficulty);
  const boardMask = gameState
    ? BigInt.asUintN(64, BigInt(gameState.board_low)) |
      (BigInt.asUintN(64, BigInt(gameState.board_high)) << 64n)
    : 0n;

  return (
    <div className="flex flex-col h-full bg-[#0a0a0a] text-foreground overflow-hidden">
      <div className="px-5 py-3 border-b border-border/10 flex justify-between items-center bg-black/20 shrink-0">
        <h3 className="text-xs font-bold text-zinc-100 uppercase tracking-widest">
          Playground Stats
        </h3>
      </div>

      <div className="flex-1 overflow-y-auto p-4 custom-scrollbar flex flex-col gap-4">
        {(!gameState || gameState.terminal) && (
          <div className="flex flex-col gap-2">
            <span className="text-xs text-zinc-500 uppercase tracking-widest font-bold">
              Difficulty Setting
            </span>
            <Select value={difficulty} onValueChange={setDifficulty}>
              <SelectTrigger className="h-8 w-full bg-zinc-900 border border-zinc-800 text-white">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-black border-zinc-800 text-white">
                {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((d) => (
                  <SelectItem key={d} value={d.toString()}>
                    Level {d}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}

        <div className="bg-zinc-950 border border-zinc-800/80 rounded-xl p-4 shadow-sm">
          <p className="text-xs text-zinc-500 mb-1 tracking-widest uppercase">
            Score
          </p>
          <div className="text-4xl font-mono font-bold text-white mb-4">
            {gameState?.score || 0}
          </div>

          <p className="text-xs text-zinc-500 mb-1 tracking-widest uppercase">
            High Score (Lvl {difficulty})
          </p>
          <div className="text-xl font-mono text-zinc-400 mb-4">
            {highScore}
          </div>

          <p className="text-xs text-zinc-500 mb-1 tracking-widest uppercase">
            Lines Cleared
          </p>
          <div className="text-xl font-mono text-primary/80">
            {gameState?.lines_cleared || 0}
          </div>
        </div>

        <div className="bg-zinc-950 border border-zinc-800/80 rounded-xl p-4 shadow-sm flex-1">
          <p className="text-xs text-zinc-500 mb-3 tracking-widest uppercase">
            Tricked Environment
          </p>
          <div className="space-y-2 text-sm text-zinc-400 font-mono">
            <div className="flex justify-between">
              <span>Status</span>{" "}
              <span
                className={
                  gameState?.terminal ? "text-red-500" : "text-green-500"
                }
              >
                {gameState?.terminal ? "TERMINAL" : "ACTIVE"}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Difficulty</span>{" "}
              <span>Level {gameState?.difficulty || 0}</span>
            </div>
            <div className="flex justify-between">
              <span>Pieces L</span> <span>{gameState?.pieces_left || 0}</span>
            </div>
            <div className="pt-4 border-t border-zinc-800/50 mt-4">
              <span className="text-[10px] text-zinc-600 block mb-2 whitespace-normal break-all">
                Bits: {boardMask.toString()}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
