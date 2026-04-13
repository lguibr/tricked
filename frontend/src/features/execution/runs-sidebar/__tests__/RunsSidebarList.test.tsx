import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { RunsSidebarList } from "../RunsSidebarList";
import { useAppStore } from "@/store/useAppStore";
import React from "react";

vi.mock("@/store/useAppStore", () => ({
  useAppStore: vi.fn(),
}));

describe("RunsSidebarList Dashboard Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders correctly with an empty list", () => {
    (useAppStore as any).mockImplementation((selector: any) => {
      const state = {
        runs: [],
        runColors: {},
        selectedDashboardRuns: [],
        setSelectedDashboardRuns: vi.fn(),
      };
      return selector ? selector(state) : state;
    });

    render(<RunsSidebarList />);
    expect(screen.getByText(/0 Active Runs/i)).toBeInTheDocument();
  });

  it("renders runs and handles select all", () => {
    const mockSetSelected = vi.fn();
    (useAppStore as any).mockImplementation((selector: any) => {
      const state = {
        runs: [{ id: "1", name: "Test Run", type: "TRAIN", status: "STOPPED" }],
        runColors: {},
        selectedDashboardRuns: [],
        setSelectedDashboardRuns: mockSetSelected,
      };
      return selector ? selector(state) : state;
    });

    render(<RunsSidebarList />);

    expect(screen.getByText(/1 Active Runs/i)).toBeInTheDocument();

    const selectAllBtn = screen.getByRole("button", { name: /Select All/i });
    fireEvent.click(selectAllBtn);
    expect(mockSetSelected).toHaveBeenCalledWith(["1"]);
  });

  it("handles deselect all gracefully", () => {
    const mockSetSelected = vi.fn();
    (useAppStore as any).mockImplementation((selector: any) => {
      const state = {
        runs: [{ id: "1", name: "Test Run", type: "TRAIN", status: "STOPPED" }],
        runColors: {},
        selectedDashboardRuns: ["1"], // already selected
        setSelectedDashboardRuns: mockSetSelected,
      };
      return selector ? selector(state) : state;
    });

    render(<RunsSidebarList />);

    const deselectBtn = screen.getByRole("button", { name: /Deselect All/i });
    fireEvent.click(deselectBtn);
    expect(mockSetSelected).toHaveBeenCalledWith([]);
  });
});
