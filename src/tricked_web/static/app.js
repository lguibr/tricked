let gameState = null;
let selectedSlot = -1;

const ROW_LENGTHS = [9, 11, 13, 15, 15, 13, 11, 9];
const TOTAL_TRIANGLES = 96;
const TRI_SIDE = 40;
const TRI_HEIGHT = 34.64; // 40 * sin(60)

function getRowCol(idx) {
    let rem = idx;
    for (let r = 0; r < 8; r++) {
        if (rem < ROW_LENGTHS[r]) return [r, rem];
        rem -= ROW_LENGTHS[r];
    }
    return [-1, -1];
}

function isUp(r, c) {
    if (r < 4) return c % 2 === 0;
    return c % 2 === 1;
}

// Convert board coordinate string (u128) into boolean array
function getBoardBit(boardStr, idx) {
    // BigInt bitwise check
    const board = BigInt(boardStr);
    const mask = 1n << BigInt(idx);
    return (board & mask) !== 0n;
}

function getMaskBit(maskStr, idx) {
    const mask = BigInt(maskStr);
    const b = 1n << BigInt(idx);
    return (mask & b) !== 0n;
}

function createTrianglePolygon(r, c, isUpTri, klass = "fill-board stroke-slate-700 stroke-[1px]") {
    // Calculate center-ish
    // row 0 has length 9.
    // X offset shifts slightly per row to interlock
    const rowOffset = (15 - ROW_LENGTHS[r]) * (TRI_SIDE / 4);
    const x = c * (TRI_SIDE / 2) + rowOffset - 140; 
    const y = r * TRI_HEIGHT - 130;

    const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    
    let pts = "";
    if (isUpTri) {
        pts = `${x},${y + TRI_HEIGHT} ${x + TRI_SIDE/2},${y} ${x + TRI_SIDE},${y + TRI_HEIGHT}`;
    } else {
        pts = `${x},${y} ${x + TRI_SIDE/2},${y + TRI_HEIGHT} ${x + TRI_SIDE},${y}`;
    }
    
    poly.setAttribute("points", pts);
    poly.setAttribute("class", klass);
    return { poly, x, y };
}

function initBoard() {
    const grid = document.getElementById("grid-layer");
    grid.innerHTML = "";
    
    for (let i = 0; i < TOTAL_TRIANGLES; i++) {
        const [r, c] = getRowCol(i);
        const up = isUp(r, c);
        
        const { poly } = createTrianglePolygon(r, c, up, "fill-board stroke-slate-700 stroke-[1px]");
        
        poly.dataset.idx = i;
        poly.id = `poly-${i}`;
        
        poly.addEventListener("mouseenter", () => handleHover(i, true));
        poly.addEventListener("mouseleave", () => handleHover(i, false));
        poly.addEventListener("click", () => handleClick(i));
        
        grid.appendChild(poly);
    }
}

function renderBoard() {
    if (!gameState) return;
    // Just update classes, dont recreate elements
    for (let i = 0; i < TOTAL_TRIANGLES; i++) {
        const poly = document.getElementById(`poly-${i}`);
        if (!poly) continue;
        
        const isPlaced = getBoardBit(gameState.board, i);
        if (isPlaced) {
            poly.setAttribute("class", "fill-placed stroke-slate-900 stroke-[2px]");
        } else {
            poly.setAttribute("class", "fill-board hover:fill-slate-700 stroke-slate-700 stroke-[1px]");
        }
    }
}

// Determines if a placement index matches the shape mask anchored natively at `anchorIdx`
function findValidPlacementIndex(p_id, anchorIdx) {
    if (!gameState || p_id === -1) return -1;
    
    const masks = gameState.piece_masks[p_id];
    for (let idx = 0; idx < 96; idx++) {
        const mStr = masks[idx];
        if (mStr === "0") continue;
        
        // We consider this placement `idx` valid if it intersects our `anchorIdx` mouse pointer
        // This is a naive heuristic (snaps shape to mouse).
        if (getMaskBit(mStr, anchorIdx)) {
            // Also check physical collision
            const currentBoard = BigInt(gameState.board);
            const moveMask = BigInt(mStr);
            if ((currentBoard & moveMask) === 0n) {
                return idx;
            }
        }
    }
    return -1;
}

function handleHover(anchorIdx, isEnter) {
    if (selectedSlot === -1 || !gameState) return;
    const p_id = gameState.available[selectedSlot];
    if (p_id === -1) return;

    // Reset all highlights smoothly
    renderBoard();

    if (!isEnter) return;

    const validIdx = findValidPlacementIndex(p_id, anchorIdx);
    if (validIdx !== -1) {
        const moveMaskStr = gameState.piece_masks[p_id][validIdx];
        
        // Highlight grid in-place
        for (let i = 0; i < TOTAL_TRIANGLES; i++) {
            if (getMaskBit(moveMaskStr, i)) {
                const poly = document.getElementById(`poly-${i}`);
                if (poly) {
                    poly.setAttribute("class", "fill-highlight opacity-80 stroke-white stroke-[2px]");
                }
            }
        }
    }
}

async function handleClick(anchorIdx) {
    if (selectedSlot === -1 || !gameState) return;
    const p_id = gameState.available[selectedSlot];
    if (p_id === -1) return;

    const validIdx = findValidPlacementIndex(p_id, anchorIdx);
    if (validIdx !== -1) {
        // Issue API Call
        const res = await fetch("/api/move", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ slot: selectedSlot, idx: validIdx })
        });
        
        if (res.ok) {
            gameState = await res.json();
            selectedSlot = -1; // Deselect
            updateUI();
        } else {
            console.error("Move rejected by math engine.");
        }
    }
}

function renderSlots() {
    for (let s = 0; s < 3; s++) {
        const p_id = gameState.available[s];
        const svg = document.getElementById(`svg-slot-${s}`);
        const container = document.getElementById(`slot-${s}`);
        svg.innerHTML = "";
        
        if (s === selectedSlot) {
            container.classList.add("ring-2", "ring-cyan-400", "scale-105");
        } else {
            container.classList.remove("ring-2", "ring-cyan-400", "scale-105");
        }

        if (p_id === -1) continue;
        
        // Find first valid mask to render its standalone shape
        let mStr = "0";
        for(let idx=0; idx<96; idx++) {
            if (gameState.piece_masks[p_id][idx] !== "0") {
                mStr = gameState.piece_masks[p_id][idx];
                break;
            }
        }
        
        if (mStr === "0") continue;

        // Render standalone shape centered
        // Group points
        let minX = 999, minY = 999, maxX = -999, maxY = -999;
        const polys = [];
        
        for (let i = 0; i < TOTAL_TRIANGLES; i++) {
            if (getMaskBit(mStr, i)) {
                const [r, c] = getRowCol(i);
                const { poly, x, y } = createTrianglePolygon(r, c, isUp(r, c), "fill-cyan-500 stroke-cyan-200 stroke-[2px]");
                
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
                
                polys.push(poly);
            }
        }
        
        // Center the camera loosely
        const cx = minX + (maxX - minX + TRI_SIDE)/2;
        const cy = minY + (maxY - minY + TRI_HEIGHT)/2;
        
        svg.setAttribute("viewBox", `${cx - 60} ${cy - 60} 120 120`);
        polys.forEach(p => svg.appendChild(p));
    }
}

function selectSlot(s) {
    if (!gameState || gameState.available[s] === -1) return;
    selectedSlot = selectedSlot === s ? -1 : s;
    renderSlots();
}

async function rotateSlot(s) {
    if (!gameState || gameState.available[s] === -1) return;
    const res = await fetch("/api/rotate", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ slot: s })
    });
    if (res.ok) {
        gameState = await res.json();
        updateUI();
    }
}

function updateUI() {
    document.getElementById("score").innerText = gameState.score;
    document.getElementById("pieces-left").innerText = gameState.pieces_left;
    
    if (gameState.terminal) {
        document.getElementById("game-over").classList.remove("hidden");
        document.getElementById("final-score").innerText = gameState.score;
    } else {
        document.getElementById("game-over").classList.add("hidden");
    }
    
    renderBoard();
    renderSlots();
}

async function fetchState() {
    const res = await fetch("/api/state");
    gameState = await res.json();
    updateUI();
}

async function resetGame(diff = 6) {
    const res = await fetch("/api/reset", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ difficulty: diff })
    });
    gameState = await res.json();
    selectedSlot = -1;
    updateUI();
}

let spectatorInterval = null;

function toggleSpectator() {
    const btn = document.getElementById("spectator-btn");
    if (spectatorInterval) {
        clearInterval(spectatorInterval);
        spectatorInterval = null;
        btn.innerText = "Start Spectator Mode";
        btn.classList.remove("bg-rose-600", "hover:bg-rose-500", "animate-pulse");
        btn.classList.add("bg-indigo-600", "hover:bg-indigo-500");
        fetchState(); // Restore local manual playing state
    } else {
        btn.innerText = "🛑 Stop Spectator Live View";
        btn.classList.remove("bg-indigo-600", "hover:bg-indigo-500");
        btn.classList.add("bg-rose-600", "hover:bg-rose-500", "animate-pulse");
        
        spectatorInterval = setInterval(async () => {
            try {
                const res = await fetch("/api/spectator");
                if (res.ok) {
                    gameState = await res.json();
                    selectedSlot = -1;
                    updateUI();
                }
            } catch (e) {
                console.warn("Spectator tick failed");
            }
        }, 500);
    }
}

// Initial boot
initBoard();
fetchState();
