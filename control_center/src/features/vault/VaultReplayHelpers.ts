import masksData from "@/lib/game/masks.json";

export interface CellCoord {
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
  if (m0 === 0 && m1 === 0) return null; // Mask is 0
  return BigInt.asUintN(64, BigInt(m0)) | (BigInt.asUintN(64, BigInt(m1)) << 64n);
}

export function getGridBits(mask: bigint): number[] {
  const bits: number[] = [];
  for (let i = 0n; i < 96n; i++) {
    if ((mask & (1n << i)) !== 0n) bits.push(Number(i));
  }
  return bits;
}
