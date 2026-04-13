import { useState, useMemo } from "react";
import { RotateCw, RotateCcw, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import gridCoords from "@/lib/game/gridCoords.json";
import { usePlaygroundStore } from "@/store/usePlaygroundStore";
import {
  getPieceMask,
  getVectorRotatedMask,
  getGridBits,
  renderTriangle,
  u64Low,
  u64High,
} from "./PlaygroundMath";
import { PlaygroundDeployQueue } from "./PlaygroundDeployQueue";

interface CellCoord {
  id: number;
  row: number;
  col: number;
  x: number;
  y: number;
  up: boolean;
}

export function TrickedPlayground() {
  const gameState = usePlaygroundStore((state) => state.gameState);
  const applyMoveToStore = usePlaygroundStore((state) => state.applyMove);
  const startGame = usePlaygroundStore((state) => state.startGame);

  const [selectedSlot, setSelectedSlot] = useState<number | null>(null);
  const [hoverCell, setHoverCell] = useState<number | null>(null);
  const [boardRotation, setBoardRotation] = useState(0);
  const [pieceRotations, setPieceRotations] = useState<
    [number, number, number]
  >([0, 0, 0]);

  const applyMove = async (slot: number) => {
    if (!gameState || previewMask === null) return;
    const pid = gameState.available[slot];
    let actionTaken = -1;
    for (let i = 0; i < 96; i++) {
      if (getPieceMask(pid, i) === previewMask) {
        actionTaken = i;
        break;
      }
    }
    if (actionTaken === -1) actionTaken = hoverCell || 0;
    const ok = await applyMoveToStore(
      slot,
      u64Low(previewMask),
      u64High(previewMask),
      actionTaken,
      pid,
    );
    if (ok) {
      setSelectedSlot(null);
      setPieceRotations([0, 0, 0]);
    }
  };

  const boardMask = useMemo(() => {
    if (!gameState) return 0n;
    return (
      BigInt.asUintN(64, BigInt(gameState.board_low)) |
      (BigInt.asUintN(64, BigInt(gameState.board_high)) << 64n)
    );
  }, [gameState]);

  const activeBoardCells = useMemo(() => getGridBits(boardMask), [boardMask]);

  const previewMask = useMemo(() => {
    if (selectedSlot !== null && hoverCell !== null && gameState) {
      const pid = gameState.available[selectedSlot];
      if (pid === -1) return null;
      const visualBoardRot = Math.round((((boardRotation / 60) % 6) + 6) % 6);
      const totalRot = pieceRotations[selectedSlot] - visualBoardRot;
      const mask = getVectorRotatedMask(pid, hoverCell, totalRot);
      if (mask !== null && (boardMask & mask) === 0n) return mask;
    }
    return null;
  }, [
    selectedSlot,
    hoverCell,
    gameState,
    boardMask,
    pieceRotations,
    boardRotation,
  ]);

  const activePreviewCells = useMemo(() => {
    if (previewMask === null) return [];
    return getGridBits(previewMask);
  }, [previewMask]);

  return (
    <div className="flex flex-col h-full w-full bg-[#0a0a0a] text-foreground p-6 overflow-hidden">
      <div className="flex flex-1 overflow-hidden">
        <div className="flex-1 flex flex-col items-center justify-center relative">
          <div className="absolute top-4 right-4 flex gap-2 z-50">
            <Button
              variant="outline"
              size="icon"
              className="bg-zinc-950 border-zinc-800 shadow-xl"
              onClick={() => setBoardRotation((r) => r - 60)}
            >
              <RotateCcw className="w-5 h-5 text-zinc-400" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              className="bg-zinc-950 border-zinc-800 shadow-xl"
              onClick={() => setBoardRotation((r) => r + 60)}
            >
              <RotateCw className="w-5 h-5 text-zinc-400" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              className="bg-zinc-950 border-zinc-800 shadow-xl ml-4"
              onClick={() => setBoardRotation(0)}
              title="Reset Board Rotation"
            >
              <RefreshCw className="w-5 h-5 text-zinc-400" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="bg-red-500/10 border-red-500/30 text-red-400 hover:bg-red-500/20 hover:text-red-300 shadow-xl ml-4 font-black tracking-widest uppercase"
              onClick={() => startGame()}
            >
              Restart
            </Button>
          </div>

          {!gameState && (
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50 flex flex-col items-center justify-center p-12 bg-black/80 rounded-2xl border border-white/10 backdrop-blur-sm shadow-2xl">
              <h1 className="text-3xl font-black text-emerald-400 mb-6 tracking-widest uppercase">
                Play
              </h1>
              <Button
                variant="default"
                size="lg"
                onClick={() => startGame()}
                className="bg-emerald-500 hover:bg-emerald-400 text-black font-black uppercase tracking-widest px-12 py-6 text-lg hover:scale-105 transition-transform"
              >
                Start Game
              </Button>
            </div>
          )}

          {gameState && gameState.terminal && (
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50 bg-black/80 backdrop-blur-md px-8 py-6 rounded-2xl border border-red-500/50 shadow-[0_0_50px_rgba(239,68,68,0.2)] text-center">
              <h1 className="text-4xl font-bold text-red-500 mb-2">
                Game Over
              </h1>
              <p className="text-zinc-300">Final Score: {gameState.score}</p>
            </div>
          )}

          <div
            className="transition-transform duration-500 ease-out relative shadow-[0_0_120px_rgba(0,0,0,0.5)] rounded-full p-12 bg-black/40 border border-white/5"
            style={{ transform: `scale(1.6) rotate(${boardRotation}deg)` }}
            onMouseLeave={() => setHoverCell(null)}
          >
            <svg
              width="350"
              height="350"
              viewBox="-80 -70 160 140"
              className="overflow-visible filter drop-shadow-2xl"
            >
              {(gridCoords as CellCoord[]).map((c) => {
                const isActive = activeBoardCells.includes(c.id);
                const isPreview = activePreviewCells.includes(c.id);
                let fillClass =
                  "fill-[#1c1c24] hover:fill-[#2d2d3a] stroke-black/60";
                if (isActive)
                  fillClass =
                    "fill-zinc-800 stroke-zinc-900 drop-shadow-[0_0_5px_rgba(255,255,255,0.1)] hover:fill-zinc-700";
                if (isPreview)
                  fillClass =
                    "fill-emerald-500/80 stroke-emerald-400 drop-shadow-[0_0_10px_rgba(16,185,129,0.5)] cursor-pointer";
                return (
                  <g key={c.id}>
                    {renderTriangle(
                      c,
                      fillClass,
                      () => {
                        if (selectedSlot !== null && previewMask !== null)
                          applyMove(selectedSlot);
                      },
                      () => {
                        if (selectedSlot !== null) setHoverCell(c.id);
                      },
                    )}
                  </g>
                );
              })}
            </svg>
          </div>
        </div>

        <PlaygroundDeployQueue
          available={gameState?.available || []}
          selectedSlot={selectedSlot}
          setSelectedSlot={setSelectedSlot}
          pieceRotations={pieceRotations}
          setPieceRotations={setPieceRotations}
        />
      </div>
    </div>
  );
}
