import { useEffect, useRef } from "react";
import * as echarts from "echarts";
import { MetricHistory } from "@/bindings/proto/tricked";

export function useMetricsData(runIds: string[]) {
  const metricsDataRef = useRef<Record<string, any[]>>({});

  useEffect(() => {
    echarts.connect("metricsGroup");
    return () => echarts.disconnect("metricsGroup");
  }, []);

  useEffect(() => {
    let active = true;
    const wsConnections: Record<string, WebSocket> = {};

    const fetchMetricsInitial = async () => {
      for (const id of runIds) {
        if (!id || id === "undefined") {
          console.warn("useMetricsData: Skipping undefined runId");
          continue;
        }

        const ws = new WebSocket(
          `ws://127.0.0.1:8000/api/ws/runs/${id}/metrics`,
        );
        ws.binaryType = "arraybuffer";
        wsConnections[id] = ws;

        ws.onmessage = (event) => {
          if (!active) return;
          try {
            const currentArr = metricsDataRef.current[id] || [];
            const buf = new Uint8Array(event.data);
            const history = MetricHistory.fromBinary(buf);

            if (history.metrics.length > 0) {
              const incrementalMerged = new Map<number, any>();
              for (const m of currentArr) incrementalMerged.set(m.step, m);

              for (const r of history.metrics) {
                incrementalMerged.set(r.step, {
                  ...r,
                  total_loss: r.totalLoss,
                  policy_loss: r.policyLoss,
                  value_loss: r.valueLoss,
                  reward_loss: r.rewardLoss,
                  lr: r.lr,
                  game_score_min: r.gameScoreMin,
                  game_score_max: r.gameScoreMax,
                  game_score_med: r.gameScoreMed,
                  game_score_mean: r.gameScoreMean,
                  win_rate: r.winRate,
                  game_lines_cleared: r.gameLinesCleared,
                  game_count: r.gameCount,
                  ram_usage_mb: r.ramUsageMb,
                  gpu_usage_pct: r.gpuUsagePct,
                  cpu_usage_pct: r.cpuUsagePct,
                  disk_usage_pct: r.diskUsagePct,
                  vram_usage_mb: r.vramUsageMb,
                  mcts_depth_mean: r.mctsDepthMean,
                  mcts_search_time_mean: r.mctsSearchTimeMean,
                  elapsed_time: r.elapsedTime,
                  network_tx_mbps: r.networkTxMbps,
                  network_rx_mbps: r.networkRxMbps,
                  disk_read_mbps: r.diskReadMbps,
                  disk_write_mbps: r.diskWriteMbps,
                  policy_entropy: r.policyEntropy,
                  gradient_norm: r.gradientNorm,
                  representation_drift: r.representationDrift,
                  mean_td_error: r.meanTdError,
                  queue_saturation_ratio: r.queueSaturationRatio,
                  sps_vs_tps: r.spsVsTps,
                  queue_latency_us: r.queueLatencyUs,
                  sumtree_contention_us: r.sumtreeContentionUs,
                  action_space_entropy: r.actionSpaceEntropy,
                  layer_gradient_norms: r.layerGradientNorms,
                  spatial_heatmap: r.spatialHeatmap,
                  difficulty: r.difficulty,
                });
              }

              let nextArray = Array.from(incrementalMerged.values()).sort(
                (a, b) => a.step - b.step,
              );
              if (nextArray.length > 5000) {
                nextArray = nextArray.slice(nextArray.length - 5000);
              }

              metricsDataRef.current = {
                ...metricsDataRef.current,
                [id]: nextArray,
              };
              window.dispatchEvent(
                new CustomEvent("engine_telemetry_force_update"),
              );
            }
          } catch (err) {
            console.error(`WebSocket metric decode error: `, err);
          }
        };

        wsConnections[id] = ws;
      }
    };

    fetchMetricsInitial();

    return () => {
      active = false;
      Object.values(wsConnections).forEach((ws) => {
        try {
          ws.close();
        } catch (e) {}
      });
    };
  }, [runIds]);

  return metricsDataRef;
}
