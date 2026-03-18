import { describe, it, expect } from 'vitest';
import { getRowCol, isUp, getBoardBit, getMaskBit, getPoints, findValidPlacementIndex, getPieceData } from './math';

describe('Tricked Math Engine JS', () => {
	it('Validates Coordinate Mappings', () => {
		expect(getRowCol(0)).toEqual([0, 0]);
		expect(isUp(0, 0)).toBe(true);
		expect(isUp(0, 1)).toBe(false);
	});

	it('Validates Bitwise Operations', () => {
		expect(getBoardBit('1', 0)).toBe(true);
		expect(getBoardBit('1', 1)).toBe(false);
		expect(getMaskBit('2', 1)).toBe(true);
		expect(getMaskBit('2', 0)).toBe(false);
	});

	it('Validates Geometry Projections', () => {
		const pts = getPoints(0, 0, true);
		expect(pts.length).toBeGreaterThan(0);
		expect(pts).toContain(',');
	});

	it('Validates Valid Placement Indexing & Piece Vector Extraction', () => {
		const arr = new Array(96).fill('0');
		arr[5] = '32';
		const mockMasks = { '0': arr } as Record<string, string[]>;

		const idx = findValidPlacementIndex(0, 5, mockMasks, '0');
		expect(idx).toBe(5);

		const idxCollide = findValidPlacementIndex(0, 5, mockMasks, '32');
		expect(idxCollide).toBe(-1);

		const pieceVec = getPieceData(0, mockMasks);
		expect(pieceVec?.polys.length).toBeGreaterThan(0);
		expect(pieceVec?.viewBox).toContain('120 120');
	});
});
