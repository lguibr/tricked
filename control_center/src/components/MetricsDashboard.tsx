import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { MetricChart } from './dashboard/MetricChart';

interface MetricsDashboardProps {
    runIds: string[];
    runColors: Record<string, string>;
}

export function MetricsDashboard({ runIds, runColors }: MetricsDashboardProps) {
    const [metricsData, setMetricsData] = useState<Record<string, any[]>>({});

    useEffect(() => {
        let active = true;
        const fetchMetrics = async () => {
            const data: Record<string, any[]> = {};
            for (const id of runIds) {
                try {
                    const runMetrics = await invoke<any[]>('get_run_metrics', { id });
                    data[id] = runMetrics;
                } catch (e) {
                    console.error(`Failed to fetch metrics for ${id}:`, e);
                }
            }
            if (active) {
                setMetricsData(data);
            }
        };

        fetchMetrics();
        const interval = setInterval(fetchMetrics, 2000);
        return () => {
            active = false;
            clearInterval(interval);
        };
    }, [runIds]);

    const charts = [
        { key: 'total_loss', title: 'TOTAL LOSS', description: 'Overall loss. Should trend downwards over time.' },
        { key: 'policy_loss', title: 'POLICY LOSS', description: 'Error predicting the best move probabilities. Lower is better.' },
        { key: 'value_loss', title: 'VALUE LOSS', description: 'Error evaluating the game state advantages. Lower is better.' },
        { key: 'value_prefix_loss', title: 'VALUE PREFIX LOSS', description: 'Error predicting sequential game rewards. Lower is better.' },
        { key: 'lr', title: 'LEARNING RATE', description: 'Step size for the optimizer. Decays via cosine annealing.' },
        { key: 'game_score_mean', title: 'SCORE MEAN', description: 'Average game score attained in self-play. Higher is better.' },
        { key: 'game_lines_cleared', title: 'LINES CLEARED', description: 'Average clear count in the environment. Higher is better.' },
        { key: 'mcts_depth_mean', title: 'MCTS DEPTH MEAN', description: 'Mean depth of the search tree. Indicates tactical horizon.' },
        { key: 'mcts_search_time_mean', title: 'MCTS TIME (ms)', description: 'Search iteration time. Indicates GPU/CPU bottlenecks.' },
        { key: 'gpu_usage_pct', title: 'GPU USAGE %', description: 'GPU compute saturation. Goal is >95%.' },
        { key: 'cpu_usage_pct', title: 'CPU USAGE %', description: 'Total CPU core saturation across workers.' },
        { key: 'vram_usage_mb', title: 'VRAM USAGE (MB)', description: 'GPU memory consumption.' },
        { key: 'ram_usage_mb', title: 'RAM USAGE (MB)', description: 'System memory consumption.' },
        { key: 'disk_usage_pct', title: 'DISK USAGE %', description: 'Current disk saturation for checkpoints.' },
    ];

    return (
        <div className="grid grid-cols-4 grid-rows-4 h-full w-full gap-[1px] bg-border/50 overflow-y-auto">
            {charts.map(chart => (
                <MetricChart
                    key={chart.key}
                    title={chart.title}
                    description={chart.description}
                    metricKey={chart.key}
                    runIds={runIds}
                    metricsData={metricsData}
                    runColors={runColors}
                />
            ))}
        </div>
    );
}
