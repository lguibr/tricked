import gridCoords from "@/lib/game/gridCoords.json";
import masksData from "@/lib/game/masks.json";

interface CellCoord {
  id: number;
  row: number;
  col: number;
  x: number;
  y: number;
  up: boolean;
}

export function getPieceMask(pieceId: number, cellIndex: number): bigint | null {
  if (pieceId < 0 || pieceId >= masksData.standard.length) return null;
  const p = masksData.standard[pieceId];
  if (!p || cellIndex >= p.length) return null;
  const [m0, m1] = p[cellIndex];
  if (m0 === 0 && m1 === 0) return null; // Mask is 0 (invalid translation)
  return (
    BigInt.asUintN(64, BigInt(m0)) | (BigInt.asUintN(64, BigInt(m1)) << 64n)
  );
}

export const u64Low = (mask: bigint) => BigInt.asUintN(64, mask).toString();
export const u64High = (mask: bigint) => BigInt.asUintN(64, mask >> 64n).toString();

export function getVectorRotatedMask(
  pieceId: number,
  cellIndex: number,
  rotations: number,
): bigint | null {
  if (rotations % 6 === 0) return getPieceMask(pieceId, cellIndex);

  const masks = masksData.standard[pieceId];
  if (!masks) return null;

  let baseMask = 0n;
  let baseAnchor = -1;
  // Pick the first valid mask to sample relative geometry
  for (let i = 0; i < 96; i++) {
    const [m0, m1] = masks[i];
    if (m0 !== 0 || m1 !== 0) {
      baseMask =
        BigInt.asUintN(64, BigInt(m0)) |
        (BigInt.asUintN(64, BigInt(m1)) << 64n);
      baseAnchor = i;
      break;
    }
  }
  if (baseAnchor === -1) return null;

  const bits = getGridBits(baseMask);
  const coords = gridCoords as CellCoord[];

  const h = 17.320508;
  const getTrueCentroid = (c: CellCoord) => ({
    x: c.x,
    y: c.up ? c.y - h / 6 : c.y + h / 6,
  });

  const pAnchor = getTrueCentroid(coords[baseAnchor]);
  const targetAnchor = getTrueCentroid(coords[cellIndex]);

  let cRot = 1,
    sRot = 0;
  const rotNorm = ((rotations % 6) + 6) % 6;
  if (rotNorm === 1) {
    cRot = 0.5;
    sRot = Math.sqrt(3) / 2;
  } else if (rotNorm === 2) {
    cRot = -0.5;
    sRot = Math.sqrt(3) / 2;
  } else if (rotNorm === 3) {
    cRot = -1;
    sRot = 0;
  } else if (rotNorm === 4) {
    cRot = -0.5;
    sRot = -Math.sqrt(3) / 2;
  } else if (rotNorm === 5) {
    cRot = 0.5;
    sRot = -Math.sqrt(3) / 2;
  }

  let resultMask = 0n;

  for (const b of bits) {
    const pCell = getTrueCentroid(coords[b]);
    const dx = pCell.x - pAnchor.x;
    const dy = pCell.y - pAnchor.y;

    const rx = dx * cRot - dy * sRot;
    const ry = dx * sRot + dy * cRot;

    const tx = targetAnchor.x + rx;
    const ty = targetAnchor.y + ry;

    let bestId = -1;
    let bestDist = Infinity;
    for (const c2 of coords) {
      const p2 = getTrueCentroid(c2);
      const d = (tx - p2.x) ** 2 + (ty - p2.y) ** 2;
      if (d < bestDist) {
        bestDist = d;
        bestId = c2.id;
      }
    }

    if (bestDist > 2.0) return null; // Fell off the board or Parity violation
    resultMask |= 1n << BigInt(bestId);
  }

  return resultMask;
}

export function getGridBits(mask: bigint): number[] {
  const bits: number[] = [];
  for (let i = 0n; i < 96n; i++) {
    if ((mask & (1n << i)) !== 0n) bits.push(Number(i));
  }
  return bits;
}

export function renderTriangle(
  c: CellCoord,
  fillClass: string,
  onClick?: () => void,
  onHover?: () => void,
) {
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
      className={`stroke-zinc-800/20 stroke-[1px] transition-colors cursor-pointer ${fillClass}`}
      onClick={onClick}
      onMouseEnter={onHover}
    />
  );
}
