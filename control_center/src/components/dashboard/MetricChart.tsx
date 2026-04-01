import ReactECharts from 'echarts-for-react';
import { Info } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

interface MetricChartProps {
    title: string;
    description: string;
    metricKey: string;
    runIds: string[];
    metricsData: Record<string, any[]>;
    runColors: Record<string, string>;
}

export function MetricChart({ title, description, metricKey, runIds, metricsData, runColors }: MetricChartProps) {
    const series = runIds.map((id) => {
        const data = metricsData[id] || [];
        return {
            name: `Run ${id.substring(0, 4)}`,
            type: 'line',
            showSymbol: false,
            smooth: true,
            itemStyle: { color: runColors[id] || '#10b981' },
            data: data.map(d => [parseInt(d.step, 10) || 0, parseFloat(d[metricKey]) || 0]).filter(d => !isNaN(d[1])),
        };
    });

    const options = {
        title: { show: false },
        tooltip: { trigger: 'axis' },
        grid: { left: '10%', right: '5%', bottom: '15%', top: '20%' },
        xAxis: { type: 'value', splitLine: { show: false }, axisLabel: { fontSize: 9 } },
        yAxis: { type: 'value', splitLine: { lineStyle: { color: '#27272a' } }, axisLabel: { fontSize: 9 } },
        series,
        backgroundColor: 'transparent',
    };

    return (
        <div className="bg-background flex flex-col relative w-full h-full overflow-hidden p-1 border rounded-md border-border/20">
            <div className="flex items-center justify-center gap-1 z-10 absolute top-2 left-0 right-0 pointer-events-none">
                <span className="text-[10px] uppercase font-semibold text-zinc-400 tracking-wider bg-background px-1">
                    {title}
                </span>
                <TooltipProvider delayDuration={100}>
                    <Tooltip>
                        <TooltipTrigger asChild>
                            <div className="pointer-events-auto cursor-help">
                                <Info className="h-3 w-3 text-zinc-500 hover:text-zinc-300 transition-colors" />
                            </div>
                        </TooltipTrigger>
                        <TooltipContent side="top" className="max-w-[200px] text-xs leading-relaxed text-center z-50">
                            <p>{description}</p>
                        </TooltipContent>
                    </Tooltip>
                </TooltipProvider>
            </div>
            <ReactECharts
                option={options}
                style={{ width: '100%', height: '100%' }}
                theme="dark"
            />
        </div>
    );
}
