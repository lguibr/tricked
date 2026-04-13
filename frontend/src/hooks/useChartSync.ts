import uPlot from "uplot";

export function useChartSync() {
  const registerChart = (_u: uPlot) => {};
  const unregisterChart = (_u: uPlot) => {};
  return { registerChart, unregisterChart };
}
