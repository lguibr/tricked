import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { EvaluationWorkspace } from "../EvaluationWorkspace";
import { useEvaluationStore } from "@/store/useEvaluationStore";
import React from "react";

vi.mock("@/store/useEvaluationStore", () => ({
  useEvaluationStore: vi.fn(),
}));

describe("EvaluationWorkspace Component", () => {
  let mockStore: any;

  beforeEach(() => {
    mockStore = {
      targetRunId: null,
      selectedCheckpoint: null,
      difficulty: 1,
      temperature: 1.0,
      mctsSimulations: 50,
      isEvaluating: false,
      isLooping: false,
      progress: 0,
      stepData: null,
      checkpoints: [],
      setTargetRunId: vi.fn(),
      setSelectedCheckpoint: vi.fn(),
      setDifficulty: vi.fn(),
      setTemperature: vi.fn(),
      setMctsSimulations: vi.fn(),
      setIsEvaluating: vi.fn(),
      setIsLooping: vi.fn(),
      setProgress: vi.fn(),
      setStepData: vi.fn(),
      addStepData: vi.fn(),
      setCheckpoints: vi.fn(),
      startEvaluation: vi.fn(),
      stopEvaluation: vi.fn(),
    };
    vi.clearAllMocks();
  });

  it("renders target selector when no target is picked", () => {
    (useEvaluationStore as any).mockImplementation((selector: any) => {
      return selector ? selector(mockStore) : mockStore;
    });

    render(<EvaluationWorkspace />);
    expect(screen.getByText(/Select Model Run/i)).toBeInTheDocument();
  });

  it("renders progress and loop options when target is active", () => {
    (useEvaluationStore as any).mockImplementation((selector: any) => {
      const state = {
        ...mockStore,
        targetRunPath: "mock/path/model.safetensors",
        isEvaluating: true,
        evaluationLoopActive: true,
        stepData: {
          score: 50,
          lines_cleared: 0,
          board_low: "1",
          board_high: "0",
          available: [1, 2, -1],
          terminal: false,
        },
        setCheckpoints: vi.fn(),
        setSelectedCheckpoint: vi.fn(),
      };
      return selector ? selector(state) : state;
    });

    render(<EvaluationWorkspace />);
    expect(screen.getByText(/50/i)).toBeInTheDocument();

    // Some button to stop
    expect(
      screen.getByRole("button", { name: /Stop|Pause/i }),
    ).toBeInTheDocument();
  });
});
