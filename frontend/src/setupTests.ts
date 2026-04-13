import "@testing-library/jest-dom";

global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};

import { vi } from "vitest";

vi.mock("@/lib/apiBridge", () => ({
  invoke: vi.fn(() => Promise.resolve([])),
}));
