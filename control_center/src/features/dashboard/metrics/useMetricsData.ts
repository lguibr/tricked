import { useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import * as echarts from "echarts";

export function useMetricsData(runIds: string[]) {
  const metricsDataRef = useRef<Record<string, any[]>>({});

  useEffect(() => {
    echarts.connect("metricsGroup");
    return () => echarts.disconnect("metricsGroup");
  }, []);

  useEffect(() => {
    let active = true;

    const fetchMetrics = async () => {
      const data: Record<string, any[]> = {};
      for (const id of runIds) {
        try {
          const runMetrics = await invoke<any[]>("get_run_metrics", {
            runId: id,
          });
          const existing = metricsDataRef.current[id] || [];
          const merged = new Map<number, any>();
          for (const m of existing) merged.set(m.step, m);
          for (const m of runMetrics) merged.set(m.step, m);

          let finalArray = Array.from(merged.values()).sort(
            (a, b) => a.step - b.step,
          );
          if (finalArray.length > 5000) {
            finalArray = finalArray.slice(finalArray.length - 5000);
          }
          data[id] = finalArray;
          console.warn(
            `[DEBUG] fetchMetrics for ${id}: runMetrics items = ${runMetrics?.length}, final merged size = ${finalArray.length}`,
          );
        } catch (e) {
          console.error(`Failed to fetch metrics for ${id}:`, e);
        }
      }
      if (active) {
        metricsDataRef.current = { ...metricsDataRef.current, ...data };
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000);

    const handleTelemetry = (e: Event) => {
      if (!active) return;
      const metric = (e as CustomEvent).detail;

      if (runIds.includes(metric.run_id)) {
        const currentArr = metricsDataRef.current[metric.run_id] || [];
        if (!currentArr.some((m) => m.step === metric.step)) {
          const newArr = [...currentArr, metric];
          if (newArr.length > 2000) {
            newArr.splice(0, newArr.length - 2000);
          }
          metricsDataRef.current[metric.run_id] = newArr;
        }
      }
    };

    window.addEventListener("engine_telemetry_update", handleTelemetry);

    return () => {
      active = false;
      clearInterval(interval);
      window.removeEventListener("engine_telemetry_update", handleTelemetry);
    };
  }, [runIds]);

  return metricsDataRef;
}
