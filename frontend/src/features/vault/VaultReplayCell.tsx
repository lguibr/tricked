import { memo } from "react";
import { CellCoord } from "./VaultReplayHelpers";

export const CellTriangle = memo(
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
