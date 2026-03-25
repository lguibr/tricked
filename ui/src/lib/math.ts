export const ROW_LENGTHS = [9, 11, 13, 15, 15, 13, 11, 9];
export const TOTAL_TRIANGLES = 96;
export const TRI_SIDE = 40;
export const TRI_HEIGHT = 34.64;

export function getRowCol(idx: number): [number, number] {
  let rem = idx;
  for (let r = 0; r < 8; r++) {
    if (rem < ROW_LENGTHS[r]) return [r, rem];
    rem -= ROW_LENGTHS[r];
  }
  return [-1, -1];
}

export function isUp(r: number, c: number): boolean {
  if (r < 4) return c % 2 === 0;
  return c % 2 === 1;
}

/**
 * Transposes Flat-Topped Triango coordinate layouts mathematically into standardized SVG Polygons.
 *
 * Specifically converts 1D array indexes into precise `{x,y}` dimensions leveraging
 * absolute Triango spacing: $y = r \times \sin(60^\circ) \times 40$.
 *
 * @param r - The geometric Matrix row projection.
 * @param c - The bounding-box column extraction.
 * @param isUpTri - Boolean evaluating symmetrical triangle orientation point limits.
 * @returns SVG `<polygon>` point string mapping.
 */
export function getPoints(r: number, c: number, isUpTri: boolean): string {
  const rowOffset = (15 - ROW_LENGTHS[r]) * (TRI_SIDE / 4);
  const x = c * (TRI_SIDE / 2) + rowOffset - 140;
  const y = r * TRI_HEIGHT - 130;

  if (isUpTri) {
    return `${x},${y + TRI_HEIGHT} ${x + TRI_SIDE / 2},${y} ${x + TRI_SIDE},${y + TRI_HEIGHT}`;
  } else {
    return `${x},${y} ${x + TRI_SIDE / 2},${y + TRI_HEIGHT} ${x + TRI_SIDE},${y}`;
  }
}

export function getBoardBit(boardStr: string, idx: number): boolean {
  const board = BigInt(boardStr);
  const mask = 1n << BigInt(idx);
  return (board & mask) !== 0n;
}

export function getMaskBit(maskStr: string, idx: number): boolean {
  const mask = BigInt(maskStr);
  const b = 1n << BigInt(idx);
  return (mask & b) !== 0n;
}

/**
 * Statically evaluates if a 96-tile boolean fragment visually collides natively
 * against the persistent `u128` BigInt environment mask natively utilizing bitwise logic.
 *
 * @param p_id - Internal fragment signature.
 * @param anchorIdx - Extrapolated UI drop-zone index.
 * @param piece_masks - Raw string mathematical permutations representing geometry.
 * @param boardStateStr - Active Triango map serialized state.
 * @returns The strict rotation index array location validly placed or -1 if intersecting.
 */
export function findValidPlacementIndex(
  p_id: number,
  anchorIdx: number,
  piece_masks: Record<string, string[]>,
  boardStateStr: string,
): number {
  if (p_id === -1 || !piece_masks) return -1;
  const masks = piece_masks[p_id];
  for (let idx = 0; idx < 96; idx++) {
    const mStr = masks[idx];
    if (mStr === '0') continue;

    if (getMaskBit(mStr, anchorIdx)) {
      const currentBoard = BigInt(boardStateStr);
      const moveMask = BigInt(mStr);
      if ((currentBoard & moveMask) === 0n) {
        return idx;
      }
    }
  }
  return -1;
}

export function getPieceData(p_id: number, piece_masks: Record<string, string[]>) {
  if (p_id === -1 || !piece_masks) return null;
  let mStr = '0';
  for (let idx = 0; idx < 96; idx++) {
    if (piece_masks[p_id][idx] !== '0') {
      mStr = piece_masks[p_id][idx];
      break;
    }
  }
  if (mStr === '0') return null;

  const polys: string[] = [];
  let minX = 999,
    minY = 999,
    maxX = -999,
    maxY = -999;

  for (let i = 0; i < TOTAL_TRIANGLES; i++) {
    if (getMaskBit(mStr, i)) {
      const [r, c] = getRowCol(i);
      const up = isUp(r, c);
      const pts = getPoints(r, c, up);

      const rowOffset = (15 - ROW_LENGTHS[r]) * (TRI_SIDE / 4);
      const x = c * (TRI_SIDE / 2) + rowOffset - 140;
      const y = r * TRI_HEIGHT - 130;

      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;

      polys.push(pts);
    }
  }
  const cx = minX + (maxX - minX + TRI_SIDE) / 2;
  const cy = minY + (maxY - minY + TRI_HEIGHT) / 2;
  return { polys, viewBox: `${cx - 60} ${cy - 60} 120 120` };
}
