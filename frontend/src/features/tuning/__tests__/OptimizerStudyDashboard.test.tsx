import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { OptimizerStudyDashboard } from "../OptimizerStudyDashboard";
import { useOptimizerStudy } from "@/hooks/useOptimizerStudy";
import { useAppStore } from "@/store/useAppStore";
import React from "react";

vi.mock("@/hooks/useOptimizerStudy", () => ({
  useOptimizerStudy: vi.fn(),
}));

vi.mock("@/store/useAppStore", () => ({
  useAppStore: vi.fn(),
}));

describe("OptimizerStudyDashboard Dashboard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders gracefully when no study is active", () => {
    (useOptimizerStudy as any).mockReturnValue(null);

    (useAppStore as any).mockImplementation((selector: any) => {
      return selector
        ? selector({ selectedRunId: null, runs: [] })
        : { selectedRunId: null, runs: [] };
    });

    render(<OptimizerStudyDashboard />);
    expect(screen.getByText(/No Study Selected/i)).toBeInTheDocument();
  });

  it("renders trial table and metrics when study is active", () => {
    (useOptimizerStudy as any).mockReturnValue({
      trials: [
        {
          number: 1,
          id: 1,
          params: { lr: 0.01 },
          value: 1.5,
          state: "COMPLETE",
        },
      ],
      importance: {},
    });

    (useAppStore as any).mockImplementation((selector: any) => {
      const state = {
        selectedRunId: "test_study_1",
        runs: [{ id: "test_study_1", name: "Test Study", config: "{}" }],
      };
      return selector ? selector(state) : state;
    });

    render(<OptimizerStudyDashboard />);
    expect(
      screen.getByText(/Parameter Search Space Goals/i),
    ).toBeInTheDocument();
  });

  it("handles empty trials array without throwing", () => {
    (useOptimizerStudy as any).mockReturnValue({
      trials: [],
      importance: {},
    });

    (useAppStore as any).mockImplementation((selector: any) => {
      const state = {
        selectedRunId: "test_empty_study",
        runs: [{ id: "test_empty_study", name: "Test Study", config: "{}" }],
      };
      return selector ? selector(state) : state;
    });

    render(<OptimizerStudyDashboard />);
    expect(screen.getByText(/Diagnostics Uninitialized/i)).toBeInTheDocument();
  });
});
