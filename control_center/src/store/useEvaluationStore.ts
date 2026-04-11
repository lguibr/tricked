import { create } from "zustand";

export interface EvaluationStepData {
  board_low: number | bigint;
  board_high: number | bigint;
  score: number;
  lines_cleared: number;
  available: number[];
  terminal: boolean;
  pieces_left: number;
  selected_piece_id: number;
  selected_action: number;
}

interface EvaluationState {
  isEvaluating: boolean;
  selectedRunId: string | null;
  selectedCheckpoint: string | null;
  checkpoints: string[];
  stepData: EvaluationStepData | null;
  
  setIsEvaluating: (isEvaluating: boolean) => void;
  setSelectedRunId: (id: string | null) => void;
  setSelectedCheckpoint: (cp: string | null) => void;
  setCheckpoints: (cps: string[]) => void;
  setStepData: (data: EvaluationStepData | null) => void;
}

export const useEvaluationStore = create<EvaluationState>()((set) => ({
  isEvaluating: false,
  selectedRunId: null,
  selectedCheckpoint: null,
  checkpoints: [],
  stepData: null,

  setIsEvaluating: (isEvaluating) => set({ isEvaluating }),
  setSelectedRunId: (selectedRunId) => set({ selectedRunId }),
  setSelectedCheckpoint: (selectedCheckpoint) => set({ selectedCheckpoint }),
  setCheckpoints: (checkpoints) => set({ checkpoints }),
  setStepData: (stepData) => set({ stepData }),
}));
