import gridCoords from "@/lib/game/gridCoords.json";
import {
  getVectorRotatedMask,
  getGridBits,
  renderTriangle,
} from "../playground/PlaygroundMath";

interface CellCoord {
  id: number;
  row: number;
  col: number;
  x: number;
  y: number;
  up: boolean;
}

export function EvaluationDeployQueue({ available }: { available: number[] }) {
  const renderMiniPiece = (pieceId: number) => {
    if (pieceId === -1) {
      return (
        <div className="w-20 h-20 rounded-xl border border-dashed border-zinc-800/50 mx-auto" />
      );
    }
    const validMasks = [];
    for (let i = 0; i < 96; i++) {
      const m = getVectorRotatedMask(pieceId, i, 0);
      if (m) validMasks.push({ id: i, mask: m, bits: getGridBits(m) });
    }
    const rep = validMasks[Math.floor(validMasks.length / 2)];
    if (!rep) return null;

    let sumX = 0,
      sumY = 0;
    rep.bits.forEach((b) => {
      sumX += (gridCoords as CellCoord[])[b].x;
      sumY += (gridCoords as CellCoord[])[b].y;
    });
    const cx = sumX / rep.bits.length;
    const cy = sumY / rep.bits.length;

    return (
      <div
        className={`w-20 h-20 rounded-xl relative overflow-hidden mx-auto bg-black border border-emerald-900/40 opacity-80 backdrop-blur-sm`}
      >
        <svg
          viewBox="-40 -40 80 80"
          className="w-full h-full pointer-events-none drop-shadow-[0_0_8px_rgba(16,185,129,0.5)]"
        >
          <g transform={`translate(${-cx}, ${-cy})`}>
            {rep.bits.map((b) => (
              <g key={b}>
                {renderTriangle(
                  (gridCoords as CellCoord[])[b],
                  "fill-emerald-500/40 stroke-emerald-400/80",
                )}
              </g>
            ))}
          </g>
        </svg>
      </div>
    );
  };

  return (
    <div className="absolute right-6 top-1/2 -translate-y-1/2 w-32 border-l border-white/5 pl-4 flex flex-col justify-center gap-4">
      <h3
        className="text-zinc-600 uppercase tracking-widest text-[9px] mb-2 font-black rotate-180"
        style={{ writingMode: "vertical-rl" }}
      >
        Agent Buffer
      </h3>
      {available ? (
        <div className="flex flex-col gap-4">
          {available.map((pid, idx) => (
            <div key={idx} className="flex justify-center flex-1">
              {renderMiniPiece(pid)}
            </div>
          ))}
        </div>
      ) : (
        <div className="flex flex-col gap-4 opacity-10 pointer-events-none">
          <div className="w-20 h-20 rounded-xl border border-dashed border-zinc-800/50 mx-auto" />
          <div className="w-20 h-20 rounded-xl border border-dashed border-zinc-800/50 mx-auto" />
          <div className="w-20 h-20 rounded-xl border border-dashed border-zinc-800/50 mx-auto" />
        </div>
      )}
    </div>
  );
}
