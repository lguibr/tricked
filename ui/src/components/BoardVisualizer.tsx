import React, { useMemo } from 'react';
import { getRowCol, isUp, getPoints, TOTAL_TRIANGLES, getBoardBit } from '@/lib/math';
import { useEngineStore } from '@/store/useEngineStore';
import { cn } from '@/lib/utils';

interface TriangleProps {
  idx: number;
  isFilled: boolean;
  policyProb?: number;
  holeLogit?: number;
  showPolicy: boolean;
  showHoles: boolean;
  onClick?: (idx: number) => void;
}

const HexTriangle = React.memo(({ idx, isFilled, policyProb, holeLogit, showPolicy, showHoles, onClick }: TriangleProps) => {
  const [r, c] = getRowCol(idx);
  const up = isUp(r, c);
  const points = getPoints(r, c, up);

  let fillClass = 'fill-slate-800/80 stroke-slate-700/50';
  let opacity = 1;

  if (isFilled) {
    fillClass = 'fill-primary stroke-primary drop-shadow-[0_0_8px_#00fbfb]';
  } else if (showPolicy && policyProb !== undefined && policyProb > 0.01) {
    fillClass = 'fill-cyan-500 stroke-cyan-400';
    opacity = policyProb;
  } else if (showHoles && holeLogit !== undefined && holeLogit > 0.5) {
    fillClass = 'fill-red-500 stroke-red-500 animate-pulse drop-shadow-[0_0_8px_#ef4444]';
    opacity = holeLogit;
  }

  return (
    <polygon
      points={points}
      className={cn(
        'transition-all duration-300 stroke-[1.5px] hover:stroke-white hover:fill-slate-600 cursor-pointer',
        fillClass,
      )}
      style={{ opacity: isFilled ? 1 : Math.max(0.1, opacity) }}
      onClick={() => onClick && onClick(idx)}
    />
  );
});
HexTriangle.displayName = 'HexTriangle';

export function BoardVisualizer({
  gameStateOverride,
  showPolicy = false,
  showHoles = false,
  onPlayMove,
}: {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  gameStateOverride?: any;
  showPolicy?: boolean;
  showHoles?: boolean;
  onPlayMove?: (idx: number) => void;
}) {
  const liveGameState = useEngineStore((state) => state.gameState);
  const gameState = gameStateOverride || liveGameState;

  const triangles = useMemo(() => {
    return Array.from({ length: TOTAL_TRIANGLES }).map((_, idx) => {
      let isFilled = false;
      const policyProb = gameState?.policy_probs ? gameState.policy_probs[idx] : 0;
      const holeLogit = gameState?.hole_logits ? gameState.hole_logits[idx] : 0;

      if (gameState?.features) {
        // Features array index 0..95 contains the board state.
        // And mathematical spatial indexing in Rust maps hex_idx to r * 16 + c.
        const [r, c] = getRowCol(idx);
        const spatialIdx = r * 16 + c;
        isFilled = gameState.features[spatialIdx] > 0.5;
      } else {
        isFilled = gameState ? getBoardBit(gameState.board || '0', idx) : false;
      }

      return (
        <HexTriangle
          key={idx}
          idx={idx}
          isFilled={isFilled}
          policyProb={policyProb}
          holeLogit={holeLogit}
          showPolicy={showPolicy}
          showHoles={showHoles}
          onClick={onPlayMove}
        />
      );
    });
  }, [gameState, showPolicy, showHoles]);

  return (
    <div className="relative w-full aspect-square max-w-3xl mx-auto flex items-center justify-center p-6 bg-background rounded-none border border-white/10 shadow-[0_0_40px_rgba(0,0,0,0.5)] overflow-hidden">
      <div className="absolute inset-0 transition-opacity duration-1000 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:32px_32px] pointer-events-none" />
      <svg viewBox="-200 -180 700 400" className="w-full h-full relative z-10">
        {triangles}
      </svg>
    </div>
  );
}
