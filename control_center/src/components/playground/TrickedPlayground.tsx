import { useState, useEffect, useMemo } from "react";
import { invoke } from "@tauri-apps/api/core";
import { RotateCw, RotateCcw, Play, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import gridCoords from "@/lib/game/gridCoords.json";
import masksData from "@/lib/game/masks.json";

const isTauri =
    typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

interface CellCoord {
    id: number;
    row: number;
    col: number;
    x: number;
    y: number;
    up: boolean;
}

interface PlaygroundState {
    board_low: string;
    board_high: string;
    available: [number, number, number];
    score: number;
    pieces_left: number;
    terminal: boolean;
    difficulty: number;
    lines_cleared: number;
}

// Convert i64 arrays from Rust to BigInt u128
function getPieceMask(pieceId: number, cellIndex: number): bigint | null {
    if (pieceId < 0 || pieceId >= masksData.standard.length) return null;
    const p = masksData.standard[pieceId];
    if (!p || cellIndex >= p.length) return null;
    const [m0, m1] = p[cellIndex];
    if (m0 === 0 && m1 === 0) return null; // Mask is 0 (invalid translation)
    return (
        BigInt.asUintN(64, BigInt(m0)) | (BigInt.asUintN(64, BigInt(m1)) << 64n)
    );
}

const u64Low = (mask: bigint) => BigInt.asUintN(64, mask).toString();
const u64High = (mask: bigint) => BigInt.asUintN(64, mask >> 64n).toString();

function getVectorRotatedMask(
    pieceId: number,
    cellIndex: number,
    rotations: number
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
            baseMask = BigInt.asUintN(64, BigInt(m0)) | (BigInt.asUintN(64, BigInt(m1)) << 64n);
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

    let cRot = 1, sRot = 0;
    const rotNorm = ((rotations % 6) + 6) % 6;
    if (rotNorm === 1) { cRot = 0.5; sRot = Math.sqrt(3) / 2; }
    else if (rotNorm === 2) { cRot = -0.5; sRot = Math.sqrt(3) / 2; }
    else if (rotNorm === 3) { cRot = -1; sRot = 0; }
    else if (rotNorm === 4) { cRot = -0.5; sRot = -Math.sqrt(3) / 2; }
    else if (rotNorm === 5) { cRot = 0.5; sRot = -Math.sqrt(3) / 2; }

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
        resultMask |= (1n << BigInt(bestId));
    }

    return resultMask;
}

function getGridBits(mask: bigint): number[] {
    const bits: number[] = [];
    for (let i = 0n; i < 96n; i++) {
        if ((mask & (1n << i)) !== 0n) bits.push(Number(i));
    }
    return bits;
}

export function TrickedPlayground() {
    const [gameState, setGameState] = useState<PlaygroundState | null>(null);
    const [difficulty, setDifficulty] = useState("6");
    const [clutter] = useState("0");
    const [highScore, setHighScore] = useState(0);

    const [selectedSlot, setSelectedSlot] = useState<number | null>(null);
    const [hoverCell, setHoverCell] = useState<number | null>(null);
    const [boardRotation, setBoardRotation] = useState(0);
    const [pieceRotations, setPieceRotations] = useState<[number, number, number]>([0, 0, 0]);

    // Load High Score on mount
    useEffect(() => {
        const saved = localStorage.getItem(`tricked_high_score_${difficulty}`);
        if (saved) setHighScore(parseInt(saved, 10));
        else setHighScore(0);
    }, [difficulty]);

    useEffect(() => {
        if (gameState && gameState.score > highScore) {
            setHighScore(gameState.score);
            localStorage.setItem(
                `tricked_high_score_${gameState.difficulty}`,
                gameState.score.toString(),
            );
        }
    }, [gameState, highScore]);

    const startGame = async () => {
        if (!isTauri) return;
        try {
            const state = await invoke<PlaygroundState>("playground_start_game", {
                difficulty: parseInt(difficulty, 10),
                clutter: parseInt(clutter, 10),
            });
            setGameState(state);
            setSelectedSlot(null);
            setHoverCell(null);
            setPieceRotations([0, 0, 0]);
        } catch (e) {
            console.error(e);
        }
    };

    const applyMove = async (slot: number) => {
        if (!isTauri || !gameState || previewMask === null) return;
        try {
            const nextState = await invoke<PlaygroundState | null>(
                "playground_apply_move",
                {
                    boardLow: gameState.board_low,
                    boardHigh: gameState.board_high,
                    available: gameState.available,
                    score: gameState.score,
                    slot: slot,
                    pieceMaskLow: u64Low(previewMask),
                    pieceMaskHigh: u64High(previewMask),
                    difficulty: gameState.difficulty,
                    linesCleared: gameState.lines_cleared,
                },
            );

            if (nextState) {
                setGameState(nextState);
                setSelectedSlot(null);
                setPieceRotations([0, 0, 0]);
            }
        } catch (e) {
            console.error(e);
        }
    };

    const boardMask = useMemo(() => {
        if (!gameState) return 0n;
        return (
            BigInt.asUintN(64, BigInt(gameState.board_low)) |
            (BigInt.asUintN(64, BigInt(gameState.board_high)) << 64n)
        );
    }, [gameState]);

    const activeBoardCells = useMemo(() => getGridBits(boardMask), [boardMask]);

    // Derived preview mask for hovering a selected piece
    const previewMask = useMemo(() => {
        if (selectedSlot !== null && hoverCell !== null && gameState) {
            const pid = gameState.available[selectedSlot];
            if (pid === -1) return null;
            // Offset the piece mathematically so it aligns correctly against the visual board rotation!
            const visualBoardRot = Math.round((((boardRotation / 60) % 6) + 6) % 6);
            const totalRot = pieceRotations[selectedSlot] - visualBoardRot;

            const mask = getVectorRotatedMask(pid, hoverCell, totalRot);
            if (mask !== null && (boardMask & mask) === 0n) return mask;
        }
        return null;
    }, [selectedSlot, hoverCell, gameState, boardMask, pieceRotations, boardRotation]);

    const activePreviewCells = useMemo(() => {
        if (previewMask === null) return [];
        return getGridBits(previewMask);
    }, [previewMask]);

    const renderTriangle = (
        c: CellCoord,
        fillClass: string,
        onClick?: () => void,
        onHover?: () => void,
    ) => {
        // Calculate 3 points of the triangle based on coordinate array (side 20, height 17.32)
        // For "up", flat is bottom. For "down" (!up), flat is top.
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
                key={c.id}
                d={path}
                className={`stroke-zinc-800/20 stroke-[1px] transition-colors cursor-pointer ${fillClass}`}
                onClick={onClick}
                onMouseEnter={onHover}
            />
        );
    };

    const renderMiniPiece = (
        pieceId: number,
        isSelected: boolean,
        slotIndex: number,
    ) => {
        if (pieceId === -1) {
            return (
                <div className="w-24 h-24 rounded-xl border border-dashed border-zinc-800/50" />
            );
        }
        const rot = pieceRotations[slotIndex];
        // Find the first valid mask (offset roughly in the middle for rendering)
        const validMasks = [];
        for (let i = 0; i < 96; i++) {
            const m = getVectorRotatedMask(pieceId, i, rot);
            if (m) validMasks.push({ id: i, mask: m, bits: getGridBits(m) });
        }
        // pick one in the middle of the valid translations
        const rep = validMasks[Math.floor(validMasks.length / 2)];
        if (!rep) return null;

        // determine bounding box of bits
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
                className={`w-24 h-24 rounded-xl cursor-pointer relative overflow-hidden transition-all ${isSelected ? "ring-2 ring-primary bg-primary/10" : "bg-black hover:bg-zinc-900 border border-zinc-800"}`}
                onClick={() =>
                    setSelectedSlot(selectedSlot === slotIndex ? null : slotIndex)
                }
            >
                <svg viewBox="-40 -40 80 80" className="w-full h-full pointer-events-none">
                    <g transform={`translate(${-cx}, ${-cy})`}>
                        {rep.bits.map((b) =>
                            renderTriangle(
                                (gridCoords as CellCoord[])[b],
                                "fill-primary/60 stroke-primary/30",
                            ),
                        )}
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
        <div className="flex flex-col h-full w-full bg-[#0a0a0a] text-foreground p-6 overflow-hidden">
            <div className="flex items-center justify-between mb-8 border-b border-border/20 pb-4">
                <div>
                    <h2 className="text-2xl font-bold font-mono tracking-tight text-primary">
                        Playground
                    </h2>
                    <p className="text-sm text-zinc-400">
                        Play Tricked environment natively via Rust MCTS Bindings.
                    </p>
                </div>

                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2 bg-black px-3 py-1.5 rounded-md border border-zinc-800">
                        <span className="text-xs text-zinc-500 uppercase tracking-widest">
                            Difficulty
                        </span>
                        <Select value={difficulty} onValueChange={setDifficulty}>
                            <SelectTrigger className="h-7 w-[80px] bg-transparent border-0 shadow-none focus:ring-0 px-1 py-0">
                                <SelectValue />
                            </SelectTrigger>
                            <SelectContent className="bg-black border-zinc-800 text-white">
                                {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((d) => (
                                    <SelectItem key={d} value={d.toString()}>
                                        Level {d}
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    </div>
                    <Button
                        onClick={startGame}
                        size="sm"
                        variant="default"
                        className="gap-2"
                    >
                        <Play className="w-4 h-4" /> New Game
                    </Button>
                </div>
            </div>

            <div className="flex flex-1 overflow-hidden">
                {/* Left Stats Panes */}
                <div className="w-64 flex flex-col gap-4 pr-6 border-r border-border/10">
                    <div className="bg-zinc-950 border border-zinc-800/80 rounded-xl p-4 shadow-sm">
                        <p className="text-xs text-zinc-500 mb-1 tracking-widest uppercase">
                            Score
                        </p>
                        <div className="text-4xl font-mono font-bold text-white mb-4">
                            {gameState?.score || 0}
                        </div>

                        <p className="text-xs text-zinc-500 mb-1 tracking-widest uppercase">
                            High Score (Lvl {difficulty})
                        </p>
                        <div className="text-xl font-mono text-zinc-400 mb-4">
                            {highScore}
                        </div>

                        <p className="text-xs text-zinc-500 mb-1 tracking-widest uppercase">
                            Lines Cleared
                        </p>
                        <div className="text-xl font-mono text-primary/80">
                            {gameState?.lines_cleared || 0}
                        </div>
                    </div>

                    <div className="bg-zinc-950 border border-zinc-800/80 rounded-xl p-4 shadow-sm flex-1">
                        <p className="text-xs text-zinc-500 mb-3 tracking-widest uppercase">
                            Tricked Environment
                        </p>
                        <div className="space-y-2 text-sm text-zinc-400 font-mono">
                            <div className="flex justify-between">
                                <span>Status</span>{" "}
                                <span
                                    className={
                                        gameState?.terminal ? "text-red-500" : "text-green-500"
                                    }
                                >
                                    {gameState?.terminal ? "TERMINAL" : "ACTIVE"}
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span>Difficulty</span>{" "}
                                <span>Level {gameState?.difficulty || 0}</span>
                            </div>
                            <div className="flex justify-between">
                                <span>Pieces L</span> <span>{gameState?.pieces_left || 0}</span>
                            </div>
                            <div className="pt-4 border-t border-zinc-800/50 mt-4">
                                <span className="text-xs text-zinc-600 block mb-2">
                                    Bits: {boardMask.toString()}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Center Board SVG */}
                <div className="flex-1 flex flex-col items-center justify-center relative">
                    <div className="absolute top-4 right-4 flex gap-2">
                        <Button
                            variant="outline"
                            size="icon"
                            className="bg-zinc-950 border-zinc-800 shadow-xl"
                            onClick={() => setBoardRotation((r) => r - 60)}
                        >
                            <RotateCcw className="w-5 h-5 text-zinc-400" />
                        </Button>
                        <Button
                            variant="outline"
                            size="icon"
                            className="bg-zinc-950 border-zinc-800 shadow-xl"
                            onClick={() => setBoardRotation((r) => r + 60)}
                        >
                            <RotateCw className="w-5 h-5 text-zinc-400" />
                        </Button>
                        <Button
                            variant="outline"
                            size="icon"
                            className="bg-zinc-950 border-zinc-800 shadow-xl ml-4"
                            onClick={() => setBoardRotation(0)}
                        >
                            <RefreshCw className="w-5 h-5 text-zinc-400" />
                        </Button>
                    </div>

                    {gameState && gameState.terminal && (
                        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50 bg-black/80 backdrop-blur-md px-8 py-6 rounded-2xl border border-red-500/50 shadow-[0_0_50px_rgba(239,68,68,0.2)] text-center">
                            <h1 className="text-4xl font-bold text-red-500 mb-2">
                                Game Over
                            </h1>
                            <p className="text-zinc-300">Final Score: {gameState.score}</p>
                        </div>
                    )}

                    <div
                        className="transition-transform duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)] relative shadow-[0_0_120px_rgba(0,0,0,0.5)] rounded-full p-12 bg-black/40 border border-white/5"
                        style={{ transform: `scale(1.2) rotate(${boardRotation}deg)` }}
                        onMouseLeave={() => setHoverCell(null)}
                    >
                        <svg
                            width="250"
                            height="250"
                            viewBox="-80 -70 160 140"
                            className="overflow-visible filter drop-shadow-2xl"
                        >
                            {(gridCoords as CellCoord[]).map((c) => {
                                const isActive = activeBoardCells.includes(c.id);
                                const isPreview = activePreviewCells.includes(c.id);

                                let fillClass = "fill-[#121215] hover:fill-zinc-800";
                                if (isActive)
                                    fillClass =
                                        "fill-zinc-400 stroke-zinc-500 drop-shadow-[0_0_5px_rgba(255,255,255,0.3)]";
                                if (isPreview)
                                    fillClass =
                                        "fill-emerald-500/80 stroke-emerald-400 drop-shadow-[0_0_10px_rgba(16,185,129,0.5)] cursor-pointer";

                                return renderTriangle(
                                    c,
                                    fillClass,
                                    () => {
                                        if (selectedSlot !== null && previewMask !== null) {
                                            applyMove(selectedSlot);
                                        }
                                    },
                                    () => {
                                        if (selectedSlot !== null) {
                                            setHoverCell(c.id);
                                        }
                                    },
                                );
                            })}
                        </svg>
                    </div>
                </div>

                {/* Right Tray Pane */}
                <div className="w-80 border-l border-border/10 pl-6 flex flex-col justify-center gap-6">
                    <h3 className="text-zinc-500 uppercase tracking-widest text-sm mb-4 font-semibold">
                        Deploy Queue
                    </h3>
                    {gameState ? (
                        <div className="flex flex-col gap-6">
                            {gameState.available.map((pid, idx) => (
                                <div
                                    key={idx}
                                    className="flex justify-center transition-all hover:scale-105"
                                >
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
            </div>
        </div>
    );
}
