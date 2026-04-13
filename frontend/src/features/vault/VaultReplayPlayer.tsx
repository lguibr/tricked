import { useState, useMemo, useEffect } from "react";
import gridCoords from "@/lib/game/gridCoords.json";
import { useVaultStore } from "@/store/useVaultStore";
import { CellCoord, getPieceMask, getGridBits } from "./VaultReplayHelpers";
import { CellTriangle } from "./VaultReplayCell";
import { MiniPieceItem } from "./VaultReplayMiniPiece";
import { VaultReplayControls } from "./VaultReplayControls";

export function VaultReplayPlayer() {
  const games = useVaultStore((state) => state.games);
  const selectedGameIndex = useVaultStore((state) => state.selectedGameIndex);

  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const game = selectedGameIndex !== null ? games[selectedGameIndex] : null;

  // Reset step to 0 when selecting a new game
  useEffect(() => {
    setCurrentStep(0);
    setIsPlaying(false);
  }, [selectedGameIndex]);

  // Autoplay effect
  useEffect(() => {
    if (!isPlaying || !game) return;

    const interval = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev >= game.steps.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 150);

    return () => clearInterval(interval);
  }, [isPlaying, game]);

  if (!game) {
    return (
      <div className="flex h-full w-full items-center justify-center p-4 bg-[#050505] text-zinc-600 text-[10px] font-black uppercase tracking-widest text-center border-l border-white/5">
        Select a game from the Vault to playback its Replay Engine
      </div>
    );
  }

  const stepData = game.steps[currentStep];

  const boardMask = useMemo(() => {
    if (!stepData) return 0n;
    const low = stepData.board_low ?? (stepData as any).board_state?.[0] ?? "0";
    const high = stepData.board_high ?? (stepData as any).board_state?.[1] ?? "0";
    return (
      BigInt.asUintN(64, BigInt(low)) |
      (BigInt.asUintN(64, BigInt(high)) << 64n)
    );
  }, [stepData]);

  const activeBoardSet = useMemo(
    () => new Set(getGridBits(boardMask)),
    [boardMask],
  );

  // The piece selected in this step
  const placementMask = useMemo(() => {
    if (!stepData || stepData.piece_identifier === -1) return null;
    return getPieceMask(stepData.piece_identifier, stepData.action_taken);
  }, [stepData]);

  const activePlacementSet = useMemo(() => {
    if (!placementMask) return new Set();
    return new Set(getGridBits(placementMask));
  }, [placementMask]);

  return (
    <div className="flex flex-col h-full bg-[#0a0a0a] border-l border-white/5">
      {/* Header Panel */}
      <div className="p-4 border-b border-white/5 flex items-center justify-between bg-black/40">
        <div>
          <h2 className="text-sm font-black text-white tracking-widest uppercase mb-1">
            Vault Game Replay
          </h2>
          <div className="text-[10px] text-zinc-500 uppercase font-mono flex items-center gap-2">
            <span className="text-emerald-400 font-bold">
              {game.episode_score} PTS
            </span>
            <span>•</span>
            <span>{game.source_run_name}</span>
            <span>•</span>
            <span className="bg-white/10 px-1 rounded text-white">
              {game.run_type}
            </span>
          </div>
        </div>
        <div className="text-right">
          <div className="text-[10px] font-black tracking-widest text-zinc-600 mb-1">
            LINES CLEARED
          </div>
          <div className="text-lg font-bold text-zinc-200">
            {game.lines_cleared}
          </div>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Playback Board Area */}
        <div className="flex-1 flex flex-col items-center justify-center relative p-8">
          <div className="shadow-[0_0_120px_rgba(0,0,0,0.5)] rounded-full p-4 bg-black/40 border border-white/5 flex items-center justify-center">
            <svg
              width="400"
              height="400"
              viewBox="-80 -80 160 160"
              className="overflow-visible filter drop-shadow-2xl translate-y-4"
            >
              {(gridCoords as CellCoord[]).map((c) => {
                const isActive = activeBoardSet.has(c.id);
                const isPlacedHere = activePlacementSet.has(c.id);

                let fillClass = "fill-[#1c1c24] stroke-white/5";
                if (isActive) fillClass = "fill-zinc-700 stroke-zinc-900";
                if (isPlacedHere)
                  fillClass =
                    "fill-emerald-400 stroke-emerald-900 drop-shadow-[0_0_8px_rgba(16,185,129,0.5)] z-10 relative";

                return (
                  <g key={c.id}>
                    <CellTriangle c={c} fillClass={fillClass} />
                  </g>
                );
              })}
            </svg>
          </div>
        </div>

        {/* Action Panel */}
        <div className="w-48 border-l border-white/5 bg-[#080808] flex flex-col">
          <div className="p-4 border-b border-white/5 text-[9px] font-black uppercase text-zinc-500 tracking-widest">
            Available Pieces{" "}
            <span className="float-right">
              {currentStep + 1} / {game.steps.length}
            </span>
          </div>
          <div className="flex flex-col gap-6 p-4 justify-center flex-1">
            {(stepData?.available || (stepData as any)?.available_pieces || []).map((pid: number, idx: number) => (
              <div key={idx}>
                <MiniPieceItem
                  pieceId={pid}
                  isBeingPlaced={
                    pid !== -1 && pid === stepData.piece_identifier
                  }
                />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Media Controls */}
      <VaultReplayControls
        currentStep={currentStep}
        setCurrentStep={setCurrentStep}
        isPlaying={isPlaying}
        setIsPlaying={setIsPlaying}
        totalSteps={game.steps.length}
      />
    </div>
  );
}
