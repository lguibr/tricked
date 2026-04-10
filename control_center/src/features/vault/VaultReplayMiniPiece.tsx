import { memo, useMemo } from "react";
import gridCoords from "@/lib/game/gridCoords.json";
import { CellCoord, getPieceMask, getGridBits } from "./VaultReplayHelpers";
import { CellTriangle } from "./VaultReplayCell";

export const MiniPieceItem = memo(({ pieceId, isBeingPlaced }: { pieceId: number; isBeingPlaced: boolean }) => {
  const { cx, cy, repBits } = useMemo(() => {
    if (pieceId === -1) return { cx: 0, cy: 0, repBits: [] };

    const validMasks = [];
    for (let i = 0; i < 96; i++) {
      const m = getPieceMask(pieceId, i);
      if (m) validMasks.push({ id: i, mask: m, bits: getGridBits(m) });
    }
    const rep = validMasks[Math.floor(validMasks.length / 2)];
    if (!rep) return { cx: 0, cy: 0, repBits: [] };

    let sumX = 0, sumY = 0;
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
    return <div className={`w-16 h-16 rounded-xl border border-dashed mx-auto ${pieceId === -1 ? "border-white/5" : "border-red-500/20"}`} />;
  }

  return (
    <div className={`w-16 h-16 rounded-xl mx-auto overflow-hidden ${isBeingPlaced ? "bg-emerald-500/20 border border-emerald-500/50 scale-110 shadow-[0_0_15px_rgba(16,185,129,0.3)] transition-all" : "bg-[#080808] border border-white/5 opacity-50 transition-all"}`}>
      <svg viewBox="-40 -40 80 80" className="w-full h-full">
        <g transform={`translate(${-cx}, ${-cy})`}>
          {repBits.map((b) => (
            <CellTriangle key={b} c={(gridCoords as CellCoord[])[b]} fillClass="fill-emerald-500/80 stroke-emerald-950" />
          ))}
        </g>
      </svg>
    </div>
  );
});
