import { useEffect, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import * as echarts from "echarts";

export function useChartSync(
  runIds: string[],
  metricsDataRef: React.MutableRefObject<Record<string, any[]>>,
  getSeriesCb: (pinnedTime: number | null) => any[],
  getXAxisConfigCb: (pinnedTime: number | null) => any
) {
  const chartRef = useRef<ReactECharts>(null);
  const [pinnedTime, setPinnedTime] = useState<number | null>(null);

  useEffect(() => {
    echarts.connect("metricsGroup");
    return () => echarts.disconnect("metricsGroup");
  }, []);

  useEffect(() => {
    let timeoutId: NodeJS.Timeout;
    let isCancelled = false;
    let lastDataLength = -1;
    let lastRunIds: string[] = [];

    const renderLoop = () => {
      if (isCancelled) return;

      const currentLength = runIds.reduce(
        (sum, id) => sum + (metricsDataRef.current[id]?.length || 0),
        0
      );

      const runIdsChanged = runIds.join(",") !== lastRunIds.join(",");

      if ((currentLength !== lastDataLength || runIdsChanged) && chartRef.current) {
        lastDataLength = currentLength;
        lastRunIds = [...runIds];
        const instance = chartRef.current.getEchartsInstance();
        if (instance && !instance.isDisposed()) {
          instance.group = "metricsGroup";
          instance.setOption(
            {
              xAxis: getXAxisConfigCb(pinnedTime),
              yAxis: { type: "value" },
              series: getSeriesCb(pinnedTime),
            },
            { replaceMerge: ["series"] }
          );
        }
      }

      timeoutId = setTimeout(renderLoop, 500);
    };

    renderLoop();

    return () => {
      isCancelled = true;
      clearTimeout(timeoutId);
    };
  }, [runIds, pinnedTime, getSeriesCb, getXAxisConfigCb, metricsDataRef]);

  const chartEvents = {
    dblclick: (params: any) => {
      if (params.componentType === "series" || params.componentType === "xAxis") {
        const val = Array.isArray(params.value) ? params.value[0] : params.value;
        if (typeof val === "number") {
          setPinnedTime(val);
        } else {
          setPinnedTime(null);
        }
      } else {
        setPinnedTime(null);
      }
    },
    mouseover: (params: any) => {
      if (params.componentType === "series" && params.data && params.data.groupId) {
        chartRef.current?.getEchartsInstance()?.dispatchAction({
          type: "highlight",
          seriesName: params.data.groupId,
        });
      }
    },
  };

  return { chartRef, pinnedTime, setPinnedTime, chartEvents };
}
