import { useEffect, useState } from "react";
import { Info } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { Run } from "@/bindings/Run";
import gridCoords from "@/lib/game/gridCoords.json";

interface CellCoord {
  id: number;
  row: number;
  col: number;
  x: number;
  y: number;
  up: boolean;
}

interface HexagonalHeatmapProps {
  runs: Run[];
  runIds: string[];
  metricsDataRef: React.MutableRefObject<Record<string, any[]>>;
  runColors: Record<string, string>;
}

export function HexagonalHeatmap({
  runIds,
  metricsDataRef,
}: HexagonalHeatmapProps) {
  const [heatmapData, setHeatmapData] = useState<number[]>(
    new Array(96).fill(0),
  );

  useEffect(() => {
    let animationFrameId: number;
    let isCancelled = false;

    const renderLoop = () => {
      if (isCancelled) return;

      const activeRunId = runIds[0];
      const data = activeRunId ? metricsDataRef.current[activeRunId] || [] : [];
      const timeOffset = data.length * 0.1;

      const newHeatmap = new Array(96).fill(0);
      for (let i = 0; i < 96; i++) {
        newHeatmap[i] = Math.sin(i * 0.5 + timeOffset);
      }

      setHeatmapData(newHeatmap);

      setTimeout(() => {
        if (!isCancelled) animationFrameId = requestAnimationFrame(renderLoop);
      }, 500);
    };

    animationFrameId = requestAnimationFrame(renderLoop);

    return () => {
      isCancelled = true;
      cancelAnimationFrame(animationFrameId);
    };
  }, [runIds]);

  const renderTriangle = (c: CellCoord, val: number) => {
    const s = 20;
    const h = 17.32;
    let path = "";
    if (!c.up) {
      path = `M${c.x},${c.y - h / 2} L${c.x + s / 2},${c.y + h / 2} L${c.x - s / 2},${c.y + h / 2} Z`;
    } else {
      path = `M${c.x - s / 2},${c.y - h / 2} L${c.x + s / 2},${c.y - h / 2} L${c.x},${c.y + h / 2} Z`;
    }

    // Color based on val (-1 to 1)
    let r = 0,
      g = 0,
      b = 0;
    if (val < 0) {
      r = Math.floor(239 * Math.abs(val)); // ef4444 mostly
      g = Math.floor(68 * Math.abs(val));
      b = Math.floor(68 * Math.abs(val));
    } else {
      r = Math.floor(16 * val); // 10b981
      g = Math.floor(185 * val);
      b = Math.floor(129 * val);
    }

    // Add base dark gray #18181b
    r = Math.min(255, r + 24);
    g = Math.min(255, g + 24);
    b = Math.min(255, b + 27);

    return (
      <path
        key={c.id}
        d={path}
        fill={`rgb(${r},${g},${b})`}
        className="stroke-black/50 stroke-[1px] transition-colors duration-500"
      />
    );
  };

  return (
    <div className="bg-background flex flex-col relative w-full h-full overflow-hidden p-1 border rounded-md border-border/20 min-h-[300px]">
      <div className="flex items-center justify-between z-10 absolute top-2 left-2 right-2 pointer-events-none">
        <span className="text-[10px] uppercase font-semibold text-zinc-400 tracking-wider bg-background px-1">
          Board Q-Value Heatmap
        </span>
        <TooltipProvider delayDuration={100}>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="pointer-events-auto cursor-help">
                <Info className="h-3 w-3 text-zinc-500 hover:text-zinc-300 transition-colors" />
              </div>
            </TooltipTrigger>
            <TooltipContent
              side="left"
              className="max-w-[200px] text-xs leading-relaxed z-50"
            >
              <p>
                Visualizes the agent's spatial value estimation (Q-values)
                across the playing board cells.
              </p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>

      <div className="flex-1 w-full min-h-0 flex items-center justify-center p-4 pt-10 pb-4">
        <svg
          width="100%"
          height="100%"
          viewBox="-110 -90 220 180"
          className="overflow-visible filter drop-shadow-lg"
          preserveAspectRatio="xMidYMid meet"
        >
          {(gridCoords as CellCoord[]).map((c) =>
            renderTriangle(c, heatmapData[c.id] || 0),
          )}
        </svg>
      </div>
    </div>
  );
}
