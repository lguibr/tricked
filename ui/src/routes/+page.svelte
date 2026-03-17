<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  
  // Game Constants
  const ROW_LENGTHS = [9, 11, 13, 15, 15, 13, 11, 9];
  const TOTAL_TRIANGLES = 96;
  const TRI_SIDE = 40;
  const TRI_HEIGHT = 34.64; // 40 * sin(60)

  interface GameStateData {
      board: string;
      piece_masks: Record<string, string[]>;
      available: number[];
      score: number;
      pieces_left: number;
      terminal: boolean;
  }

  // Reactive State
  let gameState = $state<GameStateData | null>(null);
  let selectedSlot = $state(-1);
  let hoveredIdx = $state(-1);
  let isSpectator = $state(false);
  let spectatorInterval: ReturnType<typeof setInterval> | null = null;
  let loading = $state(true);

  // Math Utilities
  function getRowCol(idx: number): [number, number] {
      let rem = idx;
      for (let r = 0; r < 8; r++) {
          if (rem < ROW_LENGTHS[r]) return [r, rem];
          rem -= ROW_LENGTHS[r];
      }
      return [-1, -1];
  }

  function isUp(r: number, c: number): boolean {
      if (r < 4) return c % 2 === 0;
      return c % 2 === 1;
  }

  function getPoints(r: number, c: number, isUpTri: boolean): string {
      const rowOffset = (15 - ROW_LENGTHS[r]) * (TRI_SIDE / 4);
      const x = c * (TRI_SIDE / 2) + rowOffset - 140; 
      const y = r * TRI_HEIGHT - 130;
      
      if (isUpTri) {
          return `${x},${y + TRI_HEIGHT} ${x + TRI_SIDE/2},${y} ${x + TRI_SIDE},${y + TRI_HEIGHT}`;
      } else {
          return `${x},${y} ${x + TRI_SIDE/2},${y + TRI_HEIGHT} ${x + TRI_SIDE},${y}`;
      }
  }

  function getBoardBit(boardStr: string, idx: number): boolean {
      const board = BigInt(boardStr);
      const mask = 1n << BigInt(idx);
      return (board & mask) !== 0n;
  }

  function getMaskBit(maskStr: string, idx: number): boolean {
      const mask = BigInt(maskStr);
      const b = 1n << BigInt(idx);
      return (mask & b) !== 0n;
  }

  function findValidPlacementIndex(p_id: number, anchorIdx: number): number {
      if (!gameState || p_id === -1) return -1;
      
      const masks = gameState.piece_masks[p_id];
      for (let idx = 0; idx < 96; idx++) {
          const mStr = masks[idx];
          if (mStr === "0") continue;
          
          if (getMaskBit(mStr, anchorIdx)) {
              const currentBoard = BigInt(gameState.board);
              const moveMask = BigInt(mStr);
              if ((currentBoard & moveMask) === 0n) {
                  return idx;
              }
          }
      }
      return -1;
  }

  // Derived Reactive Data
  let activeMaskStr = $derived.by(() => {
      if (selectedSlot === -1 || !gameState || hoveredIdx === -1) return null;
      const p_id = gameState.available[selectedSlot];
      if (p_id === -1) return null;
      const validIdx = findValidPlacementIndex(p_id, hoveredIdx);
      if (validIdx !== -1) {
          return gameState.piece_masks[p_id][validIdx];
      }
      return null;
  });

  // Tray Piece Calculation
  function getPieceData(p_id: number) {
      if (p_id === -1 || !gameState) return null;
      let mStr = "0";
      for(let idx=0; idx<96; idx++) {
          if (gameState.piece_masks[p_id][idx] !== "0") {
              mStr = gameState.piece_masks[p_id][idx];
              break;
          }
      }
      if (mStr === "0") return null;
      
      let polys: string[] = [];
      let minX = 999, minY = 999, maxX = -999, maxY = -999;
      
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
      const cx = minX + (maxX - minX + TRI_SIDE)/2;
      const cy = minY + (maxY - minY + TRI_HEIGHT)/2;
      return { polys, viewBox: `${cx - 60} ${cy - 60} 120 120` };
  }

  // API Interactions
  const API_BASE = "http://127.0.0.1:8080/api";

  async function fetchState() {
      try {
          const res = await fetch(`${API_BASE}/state`);
          if (res.ok) gameState = await res.json();
      } catch(e) { console.error("Could not reach API"); }
      loading = false;
  }

  async function resetGame(diff = 6) {
      const res = await fetch(`${API_BASE}/reset`, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({ difficulty: diff })
      });
      if (res.ok) {
          gameState = await res.json();
          selectedSlot = -1;
      }
  }

  async function rotateSlot(s: number) {
      if (!gameState || gameState.available[s] === -1 || isSpectator) return;
      const res = await fetch(`${API_BASE}/rotate`, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({ slot: s })
      });
      if (res.ok) gameState = await res.json();
  }

  async function handleClick(anchorIdx: number) {
      if (selectedSlot === -1 || !gameState || isSpectator) return;
      const p_id = gameState.available[selectedSlot];
      if (p_id === -1) return;

      const validIdx = findValidPlacementIndex(p_id, anchorIdx);
      if (validIdx !== -1) {
          const res = await fetch(`${API_BASE}/move`, {
              method: "POST",
              headers: {"Content-Type": "application/json"},
              body: JSON.stringify({ slot: selectedSlot, idx: validIdx })
          });
          if (res.ok) {
              gameState = await res.json();
              selectedSlot = -1;
          }
      }
  }

  function toggleSpectator() {
      if (isSpectator) {
          if (spectatorInterval) clearInterval(spectatorInterval);
          spectatorInterval = null;
          isSpectator = false;
          fetchState(); // Restore local playing state
      } else {
          isSpectator = true;
          spectatorInterval = setInterval(async () => {
              try {
                  const res = await fetch(`${API_BASE}/spectator`);
                  if (res.ok) {
                      gameState = await res.json();
                      selectedSlot = -1; // Deselect during spec
                  }
              } catch (e) { }
          }, 500);
      }
  }

  onMount(() => {
      fetchState();
  });
  
  onDestroy(() => {
      if (spectatorInterval) clearInterval(spectatorInterval);
  });
</script>

<svelte:head>
  <title>Tricked: AI Engine</title>
</svelte:head>

<main class="min-h-screen flex flex-col items-center justify-center p-8 font-sans">
  
  {#if loading}
      <div class="text-emerald-500 animate-pulse text-2xl font-bold">Booting Svelte Neural Interface...</div>
  {:else}
      <!-- Header -->
      <div class="mb-6 text-center">
          <h1 class="text-5xl font-black tracking-tight text-transparent bg-clip-text bg-gradient-to-br from-emerald-400 to-cyan-500 mb-2 drop-shadow-sm">Tricked</h1>
          <p class="text-zinc-500 font-medium tracking-wide">120-Degree Mathematical Engine</p>
      </div>

      <!-- Controls -->
      <div class="flex flex-col sm:flex-row gap-4 mb-8 bg-zinc-900/80 backdrop-blur border border-zinc-800 p-3 rounded-2xl items-center shadow-2xl w-full max-w-3xl justify-between">
          <div class="flex items-center gap-2">
              <span class="text-xs font-bold uppercase tracking-widest text-zinc-500 ml-2">Complexity:</span>
              <button onclick={() => resetGame(1)} disabled={isSpectator} class="px-4 py-1.5 text-sm font-medium rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 transition-colors disabled:opacity-50">1 (Easy)</button>
              <button onclick={() => resetGame(3)} disabled={isSpectator} class="px-4 py-1.5 text-sm font-medium rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 transition-colors disabled:opacity-50">3 (Normal)</button>
              <button onclick={() => resetGame(6)} disabled={isSpectator} class="px-4 py-1.5 text-sm font-bold rounded-lg bg-emerald-600/20 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-600/30 transition-all disabled:opacity-50">6 (Master)</button>
          </div>
          <button onclick={toggleSpectator} class="px-6 py-2 rounded-xl font-bold transition-all shadow-lg {isSpectator ? 'bg-rose-600/20 text-rose-400 border border-rose-500/30 animate-pulse' : 'bg-indigo-600/20 text-indigo-400 border border-indigo-500/30 hover:bg-indigo-600/30'}">
              {isSpectator ? '🛑 Halt Spectator View' : 'Start Spectator Mode'}
          </button>
      </div>

      <!-- Board Area -->
      <div class="relative w-full max-w-3xl aspect-square bg-zinc-900 rounded-3xl shadow-2xl flex items-center justify-center border border-zinc-800 mb-8 overflow-hidden group">
          
          <svg class="w-[90%] h-[90%] filter drop-shadow-[0_0_25px_rgba(16,185,129,0.15)] transition-all duration-500" viewBox="-300 -300 600 600">
              <g>
                  {#each Array(96) as _, i}
                      {@const [r, c] = getRowCol(i)}
                      {@const up = isUp(r, c)}
                      {@const points = getPoints(r, c, up)}
                      {@const isPlaced = gameState ? getBoardBit(gameState.board, i) : false}
                      {@const isHighlight = activeMaskStr ? getMaskBit(activeMaskStr, i) : false}
                      
                      <!-- svelte-ignore a11y_click_events_have_key_events -->
                      <!-- svelte-ignore a11y_no_static_element_interactions -->
                      <polygon 
                          {points}
                          onclick={() => handleClick(i)}
                          onmouseenter={() => hoveredIdx = i}
                          onmouseleave={() => { if (hoveredIdx === i) hoveredIdx = -1; }}
                          class="cursor-pointer outline-none transition-all duration-200 transform-origin-center
                              {isHighlight ? 'fill-emerald-400 opacity-90 stroke-emerald-200 stroke-[2px] z-10 scale-105' 
                              : isPlaced ? 'fill-zinc-700 stroke-zinc-950 stroke-[2px]' 
                              : 'fill-zinc-800 hover:fill-zinc-700 stroke-zinc-700 stroke-[1px]'}"
                      />
                  {/each}
              </g>
          </svg>

          <!-- Game Over Overlay -->
          {#if gameState?.terminal}
              <div class="absolute inset-0 bg-zinc-950/80 backdrop-blur-md flex flex-col items-center justify-center z-20">
                  <h2 class="text-6xl font-black text-rose-500 mb-4 drop-shadow-2xl">SYSTEM HALTED</h2>
                  <p class="text-2xl text-zinc-300 mb-8">Final Efficiency: <span class="font-bold text-white tracking-widest">{gameState.score}</span></p>
                  <button onclick={() => resetGame()} class="px-10 py-4 bg-emerald-500 text-zinc-950 rounded-full font-black text-lg shadow-[0_0_30px_rgba(16,185,129,0.4)] hover:scale-105 transition-transform">Re-Initialize Simulation</button>
              </div>
          {/if}
      </div>

      <!-- Footer Stats -->
      <div class="w-full max-w-3xl flex justify-between items-end mb-6 px-4">
          <div class="text-3xl font-bold tracking-tight text-zinc-200">Efficiency: <span class="text-emerald-400">{gameState?.score || 0}</span></div>
          <div class="text-zinc-500 font-medium">Entities Remaining: <span class="text-zinc-300">{gameState?.pieces_left || 0}</span></div>
      </div>

      <!-- Piece Trays -->
      <div class="grid grid-cols-3 gap-6 w-full max-w-3xl">
          {#each [0, 1, 2] as s}
              {@const p_id = gameState ? gameState.available[s] : -1}
              {@const isSelected = selectedSlot === s}
              {@const pieceData = getPieceData(p_id)}
              
              <!-- svelte-ignore a11y_click_events_have_key_events -->
              <!-- svelte-ignore a11y_no_static_element_interactions -->
              <div 
                  class="bg-zinc-900/50 rounded-2xl border {isSelected ? 'border-emerald-500 bg-emerald-500/5 scale-105' : 'border-zinc-800'} p-4 shadow-inner min-h-[160px] flex items-center justify-center flex-col relative cursor-pointer transition-all duration-200 hover:border-zinc-600"
                  onclick={() => { if(!isSpectator && p_id!==-1) selectedSlot = isSelected ? -1 : s; }}
                  oncontextmenu={(e) => { e.preventDefault(); rotateSlot(s); }}
              >
                  <span class="absolute top-3 left-3 text-[10px] font-bold uppercase tracking-widest text-zinc-600">Buffer {s}</span>
                  {#if pieceData}
                      <!-- Piece Render -->
                      <svg class="w-full h-full drop-shadow-[0_0_12px_rgba(16,185,129,0.3)]" viewBox={pieceData.viewBox}>
                          {#each pieceData.polys as pts}
                              <polygon points={pts} class="fill-emerald-500 stroke-emerald-300 stroke-[2px]" />
                          {/each}
                      </svg>
                  {/if}
              </div>
          {/each}
      </div>

      <div class="text-center mt-6 text-zinc-500 text-xs font-semibold tracking-widest uppercase">
          Right-Click to mathematically rotate buffer entities 60°
      </div>
  {/if}
</main>
