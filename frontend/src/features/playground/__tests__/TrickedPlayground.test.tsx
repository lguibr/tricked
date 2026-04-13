import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { TrickedPlayground } from "../TrickedPlayground";
import { usePlaygroundStore } from "@/store/usePlaygroundStore";
import React from "react";

vi.mock("@/store/usePlaygroundStore", () => ({
  usePlaygroundStore: vi.fn(),
}));

describe("TrickedPlayground Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders initial start controls when no game is active", () => {
    (usePlaygroundStore as any).mockImplementation((selector: any) => {
      const state = {
        gameState: null,
        startGame: vi.fn(),
        applyMove: vi.fn(),
      };
      return selector ? selector(state) : state;
    });

    render(<TrickedPlayground />);
    expect(
      screen.getByRole("button", { name: /Start Game/i }),
    ).toBeInTheDocument();
  });

  it("renders game over state when terminal is true", () => {
    (usePlaygroundStore as any).mockImplementation((selector: any) => {
      const state = {
        gameState: {
          terminal: true,
          score: 10,
          board_low: 1,
          board_high: 0,
          available: [1, 2, 3],
        },
        startGame: vi.fn(),
        applyMove: vi.fn(),
      };
      return selector ? selector(state) : state;
    });

    render(<TrickedPlayground />);
    expect(screen.getByText(/Final Score: 10/i)).toBeInTheDocument();
    expect(screen.getByText(/Game Over/i)).toBeInTheDocument();
  });

  it("calls start game action on button click", () => {
    const mockStart = vi.fn();
    (usePlaygroundStore as any).mockImplementation((selector: any) => {
      const state = {
        gameState: null,
        startGame: mockStart,
        applyMove: vi.fn(),
      };
      return selector ? selector(state) : state;
    });

    render(<TrickedPlayground />);
    fireEvent.click(screen.getByRole("button", { name: /Start Game/i }));
    expect(mockStart).toHaveBeenCalled();
  });
});
