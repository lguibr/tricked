import { RotateCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import gridCoords from "@/lib/game/gridCoords.json";
import { getVectorRotatedMask, getGridBits, renderTriangle } from "./PlaygroundMath";

interface CellCoord {
  id: number;
  row: number;
  col: number;
  x: number;
  y: number;
  up: boolean;
}

export function PlaygroundDeployQueue({
  available,
  selectedSlot,
  setSelectedSlot,
  pieceRotations,
  setPieceRotations,
}: {
  available: number[];
  selectedSlot: number | null;
  setSelectedSlot: (s: number | null) => void;
  pieceRotations: [number, number, number];
  setPieceRotations: React.Dispatch<React.SetStateAction<[number, number, number]>>;
}) {
  const renderMiniPiece = (
    pieceId: number,
    isSelected: boolean,
    slotIndex: number,
  ) => {
    if (pieceId === -1) {
      return <div className="w-24 h-24 rounded-xl border border-dashed border-zinc-800/50 mx-auto" />;
    }
    const rot = pieceRotations[slotIndex];
    const validMasks = [];
    for (let i = 0; i < 96; i++) {
      const m = getVectorRotatedMask(pieceId, i, rot);
      if (m) validMasks.push({ id: i, mask: m, bits: getGridBits(m) });
    }
    const rep = validMasks[Math.floor(validMasks.length / 2)];
    if (!rep) return null;

    let sumX = 0, sumY = 0;
    rep.bits.forEach((b) => {
      sumX += (gridCoords as CellCoord[])[b].x;
      sumY += (gridCoords as CellCoord[])[b].y;
    });
    const cx = sumX / rep.bits.length;
    const cy = sumY / rep.bits.length;

    return (
      <div
        className={`w-24 h-24 rounded-xl cursor-pointer relative overflow-hidden transition-all mx-auto ${isSelected ? "ring-2 ring-primary bg-primary/10" : "bg-black hover:bg-zinc-900 border border-zinc-800"}`}
        onClick={() => setSelectedSlot(selectedSlot === slotIndex ? null : slotIndex)}
      >
        <svg viewBox="-40 -40 80 80" className="w-full h-full pointer-events-none">
          <g transform={`translate(${-cx}, ${-cy})`}>
            {rep.bits.map((b) => renderTriangle((gridCoords as CellCoord[])[b], "fill-primary/60 stroke-primary/30"))}
          </g>
        </svg>
        <div className="absolute bottom-1 right-1 flex gap-1 z-10 pointer-events-auto">
          <Button
            variant="ghost"
            size="icon"
            className="w-6 h-6 bg-black/60 hover:bg-black/90 rounded text-zinc-400 hover:text-white"
            onClick={(e) => {
              e.stopPropagation();
              setPieceRotations((prev) => {
                const next = [...prev] as [number, number, number];
                next[slotIndex] = (next[slotIndex] + 1) % 6;
                return next;
              });
            }}
          >
            <RotateCw className="w-3 h-3" />
          </Button>
        </div>
      </div>
    );
  };

  return (
    <div className="w-80 border-l border-border/10 pl-6 flex flex-col justify-center gap-6">
      <h3 className="text-zinc-500 uppercase tracking-widest text-sm mb-4 font-semibold">
        Deploy Queue
      </h3>
      {available ? (
        <div className="flex flex-col gap-6">
          {available.map((pid, idx) => (
            <div key={idx} className="flex justify-center transition-all hover:scale-105">
              {renderMiniPiece(pid, selectedSlot === idx, idx)}
            </div>
          ))}
        </div>
      ) : (
        <div className="flex flex-col gap-6 opacity-20 pointer-events-none">
          <div className="w-24 h-24 rounded-xl border border-dashed border-zinc-800/50 mx-auto" />
          <div className="w-24 h-24 rounded-xl border border-dashed border-zinc-800/50 mx-auto" />
          <div className="w-24 h-24 rounded-xl border border-dashed border-zinc-800/50 mx-auto" />
        </div>
      )}
    </div>
  );
}
