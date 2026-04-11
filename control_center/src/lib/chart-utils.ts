import type { Run } from "@/bindings/Run";

export function hexToHSL(H: string): [number, number, number] {
  let r = 0, g = 0, b = 0;
  if (H.length === 4) {
    r = parseInt(H[1] + H[1], 16);
    g = parseInt(H[2] + H[2], 16);
    b = parseInt(H[3] + H[3], 16);
  } else if (H.length === 7) {
    r = parseInt(H.slice(1, 3), 16);
    g = parseInt(H.slice(3, 5), 16);
    b = parseInt(H.slice(5, 7), 16);
  }
  r /= 255;
  g /= 255;
  b /= 255;
  const cmin = Math.min(r, g, b),
    cmax = Math.max(r, g, b),
    delta = cmax - cmin;
  let h = 0, s = 0, l = 0;
  if (delta === 0) h = 0;
  else if (cmax === r) h = ((g - b) / delta) % 6;
  else if (cmax === g) h = (b - r) / delta + 2;
  else h = (r - g) / delta + 4;
  h = Math.round(h * 60);
  if (h < 0) h += 360;
  l = (cmax + cmin) / 2;
  s = delta === 0 ? 0 : delta / (1 - Math.abs(2 * l - 1));
  return [h, +(s * 100).toFixed(1), +(l * 100).toFixed(1)];
}

export interface SnapshotBase {
  id: string;
  name: string;
  total: number;
  [key: string]: string | number;
}

export function computePinnedSnapshots(
  runIds: string[],
  runs: Run[],
  metricsDataRef: React.MutableRefObject<Record<string, any[]>>,
  xAxisMode: "step" | "relative" | "absolute",
  pinnedTime: number,
  metricKeys: string[],
  smoothingWeight: number
): SnapshotBase[] {
  const snapshots = runIds.map((id) => {
    const rawData = metricsDataRef.current[id] || [];
    const run = runs.find((r) => r.id === id);
    const baseTime = run?.start_time ? new Date(run.start_time + "Z").getTime() : Date.now();

    const result: SnapshotBase = { id, name: id.substring(0, 4), total: 0 };
    const lasts: Record<string, number> = {};
    metricKeys.forEach((k) => (lasts[k] = 0));

    for (const d of rawData) {
      let xVal = 0;
      const elapsedSecs = Number(d.elapsed_time || 0);

      if (xAxisMode === "step") xVal = parseInt(d.step || 0, 10);
      else if (xAxisMode === "absolute") xVal = baseTime + elapsedSecs * 1000;
      else if (xAxisMode === "relative") xVal = elapsedSecs;

      if (xVal > pinnedTime) break;

      metricKeys.forEach((k) => {
        const cur = Number(d[k]) || 0;
        if (!isNaN(cur)) {
          lasts[k] = lasts[k] * smoothingWeight + cur * (1 - smoothingWeight);
        }
      });
    }

    metricKeys.forEach((k) => {
      result[k] = lasts[k];
      result.total += lasts[k];
    });
    return result;
  });

  snapshots.sort((a, b) => b.total - a.total);
  return snapshots;
}

export function getSharedXAxisConfig(
  xAxisMode: "step" | "relative" | "absolute",
  pinnedTime: number | null,
  snaps: SnapshotBase[]
) {
  if (pinnedTime !== null) {
    return {
      type: "category",
      data: snaps.map((s) => s.name),
      axisLabel: { fontSize: 9 },
      splitLine: { show: false },
      boundaryGap: true,
    };
  }

  if (xAxisMode === "absolute") {
    return {
      type: "time",
      boundaryGap: false,
      splitLine: { show: false },
      axisLabel: { fontSize: 9 },
      min: "dataMin",
      max: "dataMax",
    };
  } else if (xAxisMode === "relative") {
    return {
      type: "value",
      boundaryGap: false,
      splitLine: { show: false },
      axisLabel: { fontSize: 9, formatter: "{value} s" },
      min: "dataMin",
      max: "dataMax",
    };
  }
  return {
    type: "value",
    boundaryGap: false,
    splitLine: { show: false },
    axisLabel: { fontSize: 9 },
    min: "dataMin",
    max: "dataMax",
  };
}
