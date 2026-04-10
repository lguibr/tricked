export const createSparklineOption = (
  data: number[],
  color: string,
  max?: number,
) => ({
  grid: { top: 2, bottom: 2, left: 0, right: 0 },
  xAxis: { type: "category", show: false, boundaryGap: false },
  yAxis: {
    type: "value",
    show: false,
    min: 0,
    ...(max !== undefined ? { max } : {}),
  },
  series: [
    {
      type: "line",
      data,
      showSymbol: false,
      lineStyle: { color, width: 1.5 },
      areaStyle: {
        color: {
          type: "linear",
          x: 0,
          y: 0,
          x2: 0,
          y2: 1,
          colorStops: [
            { offset: 0, color: color + "60" },
            { offset: 1, color: color + "00" },
          ],
        },
      },
      animation: false,
    },
  ],
  tooltip: { show: false },
});
