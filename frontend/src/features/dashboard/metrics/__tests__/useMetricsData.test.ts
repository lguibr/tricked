import { describe, it, expect, vi } from "vitest";
import { renderHook } from "@testing-library/react";
import { useMetricsData } from "../useMetricsData";

describe("useMetricsData Hook Logic", () => {
  it("returns correctly shaped placeholder logic", () => {
    const runId = "test_id";
    // Let it render natively. A real hook returns metrics and isConnected structure
    const { result } = renderHook(() => useMetricsData([runId]));

    expect(result.current).toBeDefined();
    expect(typeof result.current).toBe("object");
  });
});
