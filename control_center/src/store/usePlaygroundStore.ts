import { create } from "zustand";
import { invoke as tauriInvoke } from "@tauri-apps/api/core";

const isTauri =
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const invoke = async <T>(
  cmd: string,
  args?: Record<string, any>,
): Promise<T> => {
  if (isTauri) {
    return tauriInvoke<T>(cmd, args);
  }
  return null as T;
};

import { type PlaygroundState } from "../bindings/PlaygroundState";

interface PlaygroundStore {
  gameState: PlaygroundState | null;
  difficulty: string;
  stepHistory: any[];
  clutter: string;
  highScore: number;

  setGameState: (state: PlaygroundState | null) => void;
  setDifficulty: (difficulty: string) => void;
  setClutter: (clutter: string) => void;
  setHighScore: (score: number) => void;

  loadHighScore: () => void;
  updateHighScoreIfBetter: () => void;

  startGame: () => Promise<void>;
  applyMove: (
    slot: number,
    pieceMaskLow: string,
    pieceMaskHigh: string,
    actionTaken: number,
    pieceIdentifier: number,
  ) => Promise<boolean>;
}

export const usePlaygroundStore = create<PlaygroundStore>((set, get) => ({
  gameState: null,
  difficulty: "6",
  clutter: "0",
  stepHistory: [],
  highScore: 0,

  setGameState: (gameState) => set({ gameState }),
  setDifficulty: (difficulty) => {
    set({ difficulty });
    get().loadHighScore();
  },
  setClutter: (clutter) => set({ clutter }),
  setHighScore: (highScore) => set({ highScore }),

  loadHighScore: () => {
    const { difficulty } = get();
    const saved = localStorage.getItem(`tricked_high_score_${difficulty}`);
    if (saved) set({ highScore: parseInt(saved, 10) });
    else set({ highScore: 0 });
  },

  updateHighScoreIfBetter: () => {
    const { gameState, highScore, difficulty } = get();
    if (gameState && gameState.score > highScore) {
      set({ highScore: gameState.score });
      localStorage.setItem(
        `tricked_high_score_${difficulty}`,
        gameState.score.toString(),
      );
    }
  },

  startGame: async () => {
    const { difficulty, clutter } = get();
    try {
      const state = await invoke<PlaygroundState>("playground_start_game", {
        difficulty: parseInt(difficulty, 10),
        clutter: parseInt(clutter, 10),
      });
      set({ gameState: state, stepHistory: [] });
    } catch (e) {
      console.error(e);
    }
  },

  applyMove: async (slot, pieceMaskLow, pieceMaskHigh, actionTaken, pieceIdentifier) => {
    const { gameState, stepHistory } = get();
    if (!gameState) return false;

    // Record the step *before* we apply, representing the choice
    const stepRecord = {
      board_low: gameState.board_low,
      board_high: gameState.board_high,
      available: [...gameState.available],
      action_taken: actionTaken,
      piece_identifier: pieceIdentifier,
    };

    try {
      const nextState = await invoke<PlaygroundState | null>(
        "playground_apply_move",
        {
          boardLow: gameState.board_low,
          boardHigh: gameState.board_high,
          available: gameState.available,
          score: gameState.score,
          slot,
          pieceMaskLow,
          pieceMaskHigh,
          difficulty: gameState.difficulty,
          linesCleared: gameState.lines_cleared,
        },
      );

      if (nextState) {
        const newHistory = [...stepHistory, stepRecord];
        set({ gameState: nextState, stepHistory: newHistory });
        get().updateHighScoreIfBetter();

        if (nextState.terminal) {
          await invoke("playground_commit_to_vault", {
             steps: newHistory,
             score: nextState.score,
             difficulty: gameState.difficulty,
             linesCleared: nextState.lines_cleared,
          }).catch(console.error);
        }

        return true;
      }
    } catch (e) {
      console.error(e);
    }
    return false;
  },
}));
