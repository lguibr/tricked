import { useEffect, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { invoke } from '@tauri-apps/api/core';

export function OptunaStudyDashboard() {
    const [studyData, setStudyData] = useState<any[]>([]);

    useEffect(() => {
        let active = true;
        const fetchStudy = async () => {
            try {
                const jsonStr = await invoke<string>('get_tuning_study');
                const data = JSON.parse(jsonStr);
                if (active) {
                    setStudyData(data);
                }
            } catch (e) {
                console.error('Failed to fetch optuna study:', e);
            }
        };
        fetchStudy();
        const interval = setInterval(fetchStudy, 5000);
        return () => {
            active = false;
            clearInterval(interval);
        };
    }, []);

    if (studyData.length === 0) {
        return <div className="p-4 text-xs text-muted-foreground w-full h-full flex items-center justify-center">No Optuna Study data available.</div>;
    }

    // Extract dimensions dynamically from params
    const paramKeys = new Set<string>();
    studyData.forEach(trial => {
        if (trial.params) {
            Object.keys(trial.params).forEach(k => paramKeys.add(k));
        }
    });

    const dimensions = Array.from(paramKeys).map((key, i) => {
        // Check if categorical (string) or numeric
        const isCategorical = studyData.some(t => typeof t.params?.[key] === 'string');
        if (isCategorical) {
            const categories = Array.from(new Set(studyData.map(t => String(t.params?.[key] || ''))));
            return { dim: i, name: key, type: 'category', data: categories };
        }
        return { dim: i, name: key };
    });

    // Add the outcome dimension (value)
    dimensions.push({ dim: dimensions.length, name: 'Value (Loss)' } as any);

    const seriesData = studyData.map(trial => {
        const dataPoint = dimensions.map(dim => {
            if (dim.name === 'Value (Loss)') return trial.value;
            const val = trial.params?.[dim.name];
            if (dim.type === 'category') {
                return dim.data?.indexOf(String(val));
            }
            return val;
        });
        return dataPoint;
    });

    const option = {
        title: { text: 'HYPERPARAMETER TUNING (OPTUNA)', textStyle: { fontSize: 10, color: '#a1a1aa' }, left: 'center', top: 10 },
        tooltip: { padding: 10, backgroundColor: '#222', borderColor: '#777', borderWidth: 1 },
        parallelAxis: dimensions,
        parallel: {
            left: '5%', right: '13%', bottom: '15%', top: '25%',
            parallelAxisDefault: {
                type: 'value',
                nameLocation: 'end',
                nameGap: 20,
                nameTextStyle: { fontSize: 10, color: '#a1a1aa' },
                axisLine: { lineStyle: { color: '#555' } },
                axisTick: { lineStyle: { color: '#555' } },
                splitLine: { show: false },
                axisLabel: { color: '#a1a1aa' }
            }
        },
        visualMap: {
            show: true,
            min: Math.min(...studyData.map(t => t.value || 0)) || 0,
            max: Math.max(...studyData.map(t => t.value || 0)) || 10,
            dimension: dimensions.length - 1,
            inRange: { color: ['#d94e5d', '#eac736', '#50a3ba'].reverse() },
            right: 10,
            top: 'center',
        },
        series: [
            {
                name: 'Optuna Trials',
                type: 'parallel',
                lineStyle: { width: 2, opacity: 0.5 },
                data: seriesData
            }
        ],
        backgroundColor: 'transparent'
    };

    return (
        <div className="w-full h-full bg-background relative overflow-hidden">
            <ReactECharts option={option} style={{ width: '100%', height: '100%' }} theme="dark" />
        </div>
    );
}
