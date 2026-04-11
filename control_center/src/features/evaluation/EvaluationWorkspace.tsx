import { useEffect, useMemo } from "react";
import { Play, Square, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import gridCoords from "@/lib/game/gridCoords.json";
import { useEvaluationStore, EvaluationStepData } from "@/store/useEvaluationStore";
import { useAppStore } from "@/store/useAppStore";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { getGridBits, renderTriangle } from "../playground/PlaygroundMath";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

interface CellCoord {
  id: number;
  row: number;
  col: number;
  x: number;
  y: number;
  up: boolean;
}

export function EvaluationWorkspace() {
  const {
    isEvaluating,
    selectedRunId,
    selectedCheckpoint,
    checkpoints,
    stepData,
    setIsEvaluating,
    setSelectedRunId,
    setSelectedCheckpoint,
    setCheckpoints,
    setStepData,
  } = useEvaluationStore();

  const runs = useAppStore((state) => state.runs);

  // Load checkpoints when a run is selected
  useEffect(() => {
    if (selectedRunId) {
      invoke<string[]>("list_checkpoints", { id: selectedRunId })
        .then((cps) => {
          setCheckpoints(cps);
          if (cps.length > 0 && !cps.includes(selectedCheckpoint || "")) {
            setSelectedCheckpoint(cps[cps.length - 1]);
          } else if (cps.length === 0) {
            setSelectedCheckpoint(null);
          }
        })
        .catch(console.error);
    } else {
      setCheckpoints([]);
      setSelectedCheckpoint(null);
    }
  }, [selectedRunId]);

  // Handle Evaluator events
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    listen<EvaluationStepData>("evaluation_state_update", (event) => {
      setStepData(event.payload);
    }).then((u) => { unlisten = u; });

    return () => {
      if (unlisten) unlisten();
      // Cleanup when unmounting
      if (isEvaluating) {
        invoke("stop_evaluation").catch(console.error);
        setIsEvaluating(false);
      }
    };
  }, [isEvaluating]);

  const toggleEvaluation = async () => {
    if (isEvaluating) {
      await invoke("stop_evaluation");
      setIsEvaluating(false);
    } else {
      if (!selectedRunId || !selectedCheckpoint) return;
      setStepData(null);
      await invoke("start_evaluation", {
        id: selectedRunId,
        checkpointPath: selectedCheckpoint,
      });
      setIsEvaluating(true);
    }
  };

  const boardMask = useMemo(() => {
    if (!stepData) return 0n;
    return BigInt.asUintN(64, BigInt(stepData.board_low)) | (BigInt.asUintN(64, BigInt(stepData.board_high)) << 64n);
  }, [stepData]);

  const activeBoardCells = useMemo(() => getGridBits(boardMask), [boardMask]);

  const previewMask = useMemo(() => {
    if (
      !stepData ||
      stepData.selected_action === -1 ||
      stepData.selected_piece_id === -1
    ) {
      return null;
    }
    const cellIdx = stepData.selected_action % 96;
    // Fast mock visualization snippet (since computing exact vector rot mask without piece defs is tough, we can just glow the target cell for now)
    // Actually getGridBits on just the target cell is cool enough or glow it.
    return cellIdx;
  }, [stepData]);

  return (
    <div className="flex flex-col h-full w-full bg-[#0a0a0a] text-foreground p-6 overflow-hidden relative">
      {/* Header controls */}
      <div className="absolute top-6 left-6 z-50 flex flex-col gap-4 bg-black/60 p-4 rounded-xl border border-white/10 backdrop-blur-md">
        <div className="flex flex-col gap-1">
          <label className="text-xs font-semibold text-zinc-500 uppercase tracking-widest">
            Select Model Run
          </label>
          <Select
            value={selectedRunId || ""}
            onValueChange={setSelectedRunId}
            disabled={isEvaluating}
          >
            <SelectTrigger className="w-64 bg-zinc-950 border-zinc-800">
              <SelectValue placeholder="Choose a run..." />
            </SelectTrigger>
            <SelectContent>
              {runs.map((r) => (
                <SelectItem key={r.id} value={r.id}>
                  {r.name.substring(0, 24)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {selectedRunId && checkpoints.length > 0 && (
          <div className="flex flex-col gap-1">
            <label className="text-xs font-semibold text-zinc-500 uppercase tracking-widest">
              Checkpoint
            </label>
            <Select
              value={selectedCheckpoint || ""}
              onValueChange={setSelectedCheckpoint}
              disabled={isEvaluating}
            >
              <SelectTrigger className="w-64 bg-zinc-950 border-zinc-800 text-xs">
                <SelectValue placeholder="Choose checkpoint..." />
              </SelectTrigger>
              <SelectContent>
                {checkpoints.map((cp) => {
                  const name = cp.split("/").pop() || cp;
                  return (
                    <SelectItem key={cp} value={cp} className="text-xs font-mono">
                      {name}
                    </SelectItem>
                  );
                })}
              </SelectContent>
            </Select>
          </div>
        )}

        <Button
          variant="default"
          onClick={toggleEvaluation}
          disabled={!selectedRunId || (!selectedCheckpoint && checkpoints.length > 0)}
          className={`mt-2 font-bold uppercase tracking-widest shadow-xl transition-all ${isEvaluating
            ? "bg-red-500/20 text-red-500 border border-red-500/50 hover:bg-red-500/30"
            : "bg-emerald-500 hover:bg-emerald-400 text-black"
            }`}
        >
          {isEvaluating ? (
            <>
              <Square className="w-4 h-4 mr-2" /> Stop Agent
            </>
          ) : (
            <>
              <Play className="w-4 h-4 mr-2" /> Start Evaluation
            </>
          )}
        </Button>
      </div>

      {isEvaluating && !stepData && (
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50 flex flex-col items-center gap-4 text-emerald-500">
          <Loader2 className="w-12 h-12 animate-spin opacity-50" />
          <p className="font-mono text-sm uppercase tracking-widest animate-pulse">
            Booting Neural Evaluator...
          </p>
        </div>
      )}

      {stepData?.terminal && (
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50 bg-black/80 backdrop-blur-md px-8 py-6 rounded-2xl border border-red-500/50 shadow-[0_0_50px_rgba(239,68,68,0.2)] text-center">
          <h1 className="text-4xl font-bold text-red-500 mb-2">Terminal State</h1>
          <p className="text-zinc-300">Final Score: {stepData.score}</p>
        </div>
      )}

      <div className="flex-1 flex flex-col items-center justify-center relative">
        <div
          className="transition-transform duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)] relative shadow-[0_0_120px_rgba(0,0,0,0.5)] rounded-full p-12 bg-black/40 border border-white/5"
          style={{ transform: `scale(1.2)` }}
        >
          <svg width="350" height="350" viewBox="-80 -70 160 140" className="overflow-visible filter drop-shadow-2xl">
            {(gridCoords as CellCoord[]).map((c) => {
              const isActive = activeBoardCells.includes(c.id);
              const isTargetMove = previewMask === c.id;

              let fillClass = "fill-[#1c1c24] stroke-black/60";
              if (isActive)
                fillClass = "fill-zinc-800 stroke-zinc-900 drop-shadow-[0_0_5px_rgba(255,255,255,0.1)]";
              if (isTargetMove)
                fillClass = "fill-cyan-500/80 stroke-cyan-400 drop-shadow-[0_0_15px_rgba(6,182,212,0.8)]";

              return (
                <g key={c.id}>
                  {renderTriangle(c, fillClass, () => { }, () => { })}
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      {stepData && (
        <div className="absolute bottom-6 right-6 flex flex-col gap-2 bg-black/60 p-4 rounded-xl border border-white/10 backdrop-blur-md font-mono text-xs uppercase tracking-widest text-zinc-400">
          <div className="flex justify-between gap-8">
            <span>Score:</span>
            <span className="text-white">{stepData.score}</span>
          </div>
          <div className="flex justify-between gap-8">
            <span>Lines Cleared:</span>
            <span className="text-white">{stepData.lines_cleared}</span>
          </div>
          <div className="flex justify-between gap-8">
            <span>Available Trays:</span>
            <span className="text-white">
              {stepData.available.filter((p) => p !== -1).length} / 3
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
