import { useState, useMemo, useEffect, memo } from "react";
import {
  VscPlay,
  VscDebugPause,
  VscTriangleLeft,
  VscTriangleRight,
} from "react-icons/vsc";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import gridCoords from "@/lib/game/gridCoords.json";
import masksData from "@/lib/game/masks.json";
import { useVaultStore } from "@/store/useVaultStore";

interface CellCoord {
  id: number;
  row: number;
  col: number;
  x: number;
  y: number;
  up: boolean;
}

function getPieceMask(pieceId: number, cellIndex: number): bigint | null {
  if (pieceId < 0 || pieceId >= masksData.standard.length) return null;
  const p = masksData.standard[pieceId];
  if (!p || cellIndex >= p.length) return null;
  const [m0, m1] = p[cellIndex];
  if (m0 === 0 && m1 === 0) return null; // Mask is 0
  return (
    BigInt.asUintN(64, BigInt(m0)) | (BigInt.asUintN(64, BigInt(m1)) << 64n)
  );
}

function getGridBits(mask: bigint): number[] {
  const bits: number[] = [];
  for (let i = 0n; i < 96n; i++) {
    if ((mask & (1n << i)) !== 0n) bits.push(Number(i));
  }
  return bits;
}

const CellTriangle = memo(
  ({ c, fillClass }: { c: CellCoord; fillClass: string }) => {
    const s = 20;
    const h = 17.32;
    let path = "";
    if (!c.up) {
      path = `M${c.x},${c.y - h / 2} L${c.x + s / 2},${c.y + h / 2} L${c.x - s / 2},${c.y + h / 2} Z`;
    } else {
      path = `M${c.x - s / 2},${c.y - h / 2} L${c.x + s / 2},${c.y - h / 2} L${c.x},${c.y + h / 2} Z`;
    }

    return (
      <path
        d={path}
        className={`stroke-black/60 stroke-[1px] transition-colors duration-150 ${fillClass}`}
      />
    );
  },
);

const MiniPieceItem = memo(
  ({ pieceId, isBeingPlaced }: { pieceId: number; isBeingPlaced: boolean }) => {
    const { cx, cy, repBits } = useMemo(() => {
      if (pieceId === -1) return { cx: 0, cy: 0, repBits: [] };

      const validMasks = [];
      for (let i = 0; i < 96; i++) {
        const m = getPieceMask(pieceId, i);
        if (m) validMasks.push({ id: i, mask: m, bits: getGridBits(m) });
      }
      const rep = validMasks[Math.floor(validMasks.length / 2)];
      if (!rep) return { cx: 0, cy: 0, repBits: [] };

      let sumX = 0,
        sumY = 0;
      rep.bits.forEach((b) => {
        sumX += (gridCoords as CellCoord[])[b].x;
        sumY += (gridCoords as CellCoord[])[b].y;
      });
      return {
        cx: sumX / rep.bits.length,
        cy: sumY / rep.bits.length,
        repBits: rep.bits,
      };
    }, [pieceId]);

    if (pieceId === -1 || repBits.length === 0) {
      return (
        <div
          className={`w-16 h-16 rounded-xl border border-dashed mx-auto ${pieceId === -1 ? "border-white/5" : "border-red-500/20"}`}
        />
      );
    }

    return (
      <div
        className={`w-16 h-16 rounded-xl mx-auto overflow-hidden ${isBeingPlaced ? "bg-emerald-500/20 border border-emerald-500/50 scale-110 shadow-[0_0_15px_rgba(16,185,129,0.3)] transition-all" : "bg-[#080808] border border-white/5 opacity-50 transition-all"}`}
      >
        <svg viewBox="-40 -40 80 80" className="w-full h-full">
          <g transform={`translate(${-cx}, ${-cy})`}>
            {repBits.map((b) => (
              <CellTriangle
                key={b}
                c={(gridCoords as CellCoord[])[b]}
                fillClass="fill-emerald-500/80 stroke-emerald-950"
              />
            ))}
          </g>
        </svg>
      </div>
    );
  },
);

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
    return (
      BigInt.asUintN(64, BigInt(stepData.board_low)) |
      (BigInt.asUintN(64, BigInt(stepData.board_high)) << 64n)
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
                if (isActive) {
                  fillClass = "fill-zinc-700 stroke-zinc-900";
                }
                if (isPlacedHere) {
                  fillClass =
                    "fill-emerald-400 stroke-emerald-900 drop-shadow-[0_0_8px_rgba(16,185,129,0.5)] z-10 relative";
                }

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
            {stepData?.available.map((pid: number, idx: number) => (
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
      <div className="h-24 border-t border-white/5 bg-black p-4 flex flex-col gap-3 shrink-0">
        <div className="flex-1 flex items-center gap-4">
          <div className="flex gap-1 shrink-0">
            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8 bg-white/5 border-white/10"
              onClick={() => setCurrentStep((prev) => Math.max(0, prev - 1))}
              disabled={currentStep === 0}
            >
              <VscTriangleLeft className="text-zinc-400 w-4 h-4" />
            </Button>

            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8 bg-emerald-500/10 border-emerald-500/20 hover:bg-emerald-500/20"
              onClick={() => setIsPlaying(!isPlaying)}
            >
              {isPlaying ? (
                <VscDebugPause className="text-emerald-400 w-4 h-4" />
              ) : (
                <VscPlay className="text-emerald-400 w-4 h-4" />
              )}
            </Button>

            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8 bg-white/5 border-white/10"
              onClick={() =>
                setCurrentStep((prev) =>
                  Math.min(game.steps.length - 1, prev + 1),
                )
              }
              disabled={currentStep === game.steps.length - 1}
            >
              <VscTriangleRight className="text-zinc-400 w-4 h-4" />
            </Button>
          </div>

          <div className="flex-1 px-4">
            <Slider
              value={[currentStep]}
              min={0}
              max={game.steps.length - 1}
              step={1}
              onValueChange={(v) => {
                setIsPlaying(false);
                setCurrentStep(v[0]);
              }}
              className="cursor-pointer"
            />
          </div>
          <div className="text-[10px] uppercase font-mono font-bold text-zinc-500 w-12 text-right">
            {currentStep + 1}
            <span className="opacity-50">/{game.steps.length}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
