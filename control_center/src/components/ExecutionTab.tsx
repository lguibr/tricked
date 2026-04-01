import { useState, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import ReactECharts from 'echarts-for-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Field, FieldLabel, FieldSet } from '@/components/ui/field';
import { ExecutionParameters } from './execution/ExecutionParameters';
import { RunsSidebarList } from './execution/RunsSidebarList';
import { CreateRunDialog } from './execution/CreateRunDialog';
import { ConfigEditor } from './execution/ConfigEditor';
import { EngineControls } from './execution/EngineControls';
import { LiveLogsViewer } from './execution/LiveLogsViewer';
import logoUrl from '@/assets/logo.png';

type RunStatus = 'WAITING' | 'RUNNING' | 'COMPLETED';
type RunType = 'SINGLE' | 'TUNING';

interface Run {
    id: string;
    name: string;
    status: RunStatus;
    type: RunType;
    config: string;
}

export function ExecutionTab() {
    const [runs, setRuns] = useState<Run[]>([]);
    const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

    const selectedRun = runs.find(r => r.id === selectedRunId);

    // States for deleting & flushing
    const [runToDelete, setRunToDelete] = useState<string | null>(null);
    const [runToFlush, setRunToFlush] = useState<string | null>(null);
    const [runToRename, setRunToRename] = useState<string | null>(null);
    const [newName, setNewName] = useState('');

    // States for creating new run
    const [isCreateOpen, setIsCreateOpen] = useState(false);
    const [newRunType, setNewRunType] = useState<RunType>('SINGLE');
    const [newRunName, setNewRunName] = useState('');
    const [newRunPreset, setNewRunPreset] = useState<string>('default');

    // State for viewing logs
    const [selectedLogRunIds, setSelectedLogRunIds] = useState<string[]>([]);
    const [copiedLogId, setCopiedLogId] = useState<string | null>(null);
    const [runLogs, setRunLogs] = useState<Record<string, string[]>>({});
    const [localConfig, setLocalConfig] = useState<string>('');
    const [tuningData, setTuningData] = useState<{ number: number; value: number; params: any }[]>([]);
    const logsEndRef = useRef<{ [key: string]: HTMLDivElement | null }>({});

    useEffect(() => {
        selectedLogRunIds.forEach(runId => {
            if (logsEndRef.current[runId]) {
                logsEndRef.current[runId]?.scrollIntoView({ behavior: 'auto' });
            }
        });
    }, [runLogs, selectedLogRunIds]);

    const updateConfigKey = async (key: string, value: any) => {
        try {
            const parsed = JSON.parse(localConfig);
            parsed[key] = value;
            const newConfig = JSON.stringify(parsed, null, 2);
            setLocalConfig(newConfig);
            if (selectedRunId) {
                await invoke('save_config', { id: selectedRunId, config: newConfig });
            }
        } catch (e) {
            console.error("Failed to parse config", e);
        }
    };

    const getConfigValue = (key: string, fallback: any) => {
        try {
            const parsed = JSON.parse(localConfig);
            return parsed[key] ?? fallback;
        } catch (e) {
            return fallback;
        }
    };

    useEffect(() => {
        if (selectedRun) {
            setLocalConfig(selectedRun.config);
        }
    }, [selectedRun?.id]);

    const loadTuningData = async () => {
        try {
            const rawTuning: string = await invoke('get_tuning_study');
            let parsed = JSON.parse(rawTuning);
            if (!Array.isArray(parsed)) parsed = [];
            setTuningData(parsed);
        } catch (e) {
            console.error('Failed to parse tuning study', e);
        }
    };

    useEffect(() => {
        if (selectedRun?.type === 'TUNING') {
            loadTuningData();
            if (selectedRun.status === 'RUNNING') {
                const interval = setInterval(loadTuningData, 3000);
                return () => clearInterval(interval);
            }
        }
    }, [selectedRunId, runs]);

    const handleCopyLogs = (id: string, logs: string) => {
        navigator.clipboard.writeText(logs);
        setCopiedLogId(id);
        setTimeout(() => setCopiedLogId(null), 2000);
    };

    useEffect(() => {
        loadRuns();
        const interval = setInterval(loadRuns, 3000);

        let unlisten: (() => void) | undefined;
        listen('log_event', (event: any) => {
            const { run_id, line } = event.payload;
            setRunLogs(prev => {
                const updated = [...(prev[run_id] || []), line].slice(-500);
                return { ...prev, [run_id]: updated };
            });
        }).then(u => unlisten = u);

        return () => {
            clearInterval(interval);
            if (unlisten) unlisten();
        };
    }, []);

    const loadRuns = async () => {
        try {
            const result = await invoke<Run[]>('list_runs');
            setRuns(result);
            if (result.length > 0 && !selectedRunId) setSelectedRunId(result[0].id);
        } catch (e) {
            console.error("Failed to load runs", e);
        }
    };

    const toggleLogRun = (id: string, pressed: boolean) => {
        if (pressed) setSelectedLogRunIds([...selectedLogRunIds, id]);
        else setSelectedLogRunIds(selectedLogRunIds.filter(r => r !== id));
    };

    const handleDelete = async () => {
        if (runToDelete) {
            await invoke('delete_run', { id: runToDelete });
            setRuns(runs.filter(r => r.id !== runToDelete));
            if (selectedRunId === runToDelete) setSelectedRunId(null);
            setSelectedLogRunIds(selectedLogRunIds.filter(id => id !== runToDelete));
            setRunToDelete(null);
        }
    };

    const handleFlush = () => {
        if (runToFlush) setRunToFlush(null);
    };

    const handleRename = async () => {
        if (runToRename && newName.trim()) {
            await invoke('rename_run', { id: runToRename, newName });
            loadRuns();
            setRunToRename(null);
            setNewName('');
        }
    };

    const handleCreateNew = async () => {
        if (newRunName.trim()) {
            const presetArg = newRunPreset === 'default' ? null : newRunPreset;
            const newRun = await invoke<Run>('create_run', { name: newRunName, type: newRunType, preset: presetArg });
            setRuns([...runs, newRun]);
            setIsCreateOpen(false);
            setNewRunName('');
            setNewRunType('SINGLE');
            setNewRunPreset('default');
            setSelectedRunId(newRun.id);
        }
    };

    const handleSaveConfig = async () => {
        if (selectedRunId) {
            await invoke('save_config', { id: selectedRunId, config: localConfig });
            loadRuns();
        }
    };

    const calculateEma = (data: number[], weight: number = 0.8) => {
        if (data.length === 0) return [];
        let ema = [data[0]];
        for (let i = 1; i < data.length; i++) {
            ema.push(ema[i - 1] * weight + data[i] * (1 - weight));
        }
        return ema;
    };

    const getChartData = (runId: string | null) => {
        if (!runId) return { steps: [], loss: [], score: [], smoothedLoss: [], smoothedScore: [] };
        const lines = runLogs[runId] || [];
        const lossData: number[] = [];
        const scoreData: number[] = [];
        const steps: string[] = [];

        lines.forEach(line => {
            let parsedJson = false;
            try {
                if (line.includes('"step":') && line.startsWith('{')) {
                    const data = JSON.parse(line);
                    if (data.step !== undefined) {
                        steps.push(data.step.toString());
                        lossData.push(data.total_loss || 0);
                        scoreData.push(data.game_score_mean || 0);
                        parsedJson = true;
                    }
                }
            } catch (e) {
                // Ignore parse errors
            }

            if (!parsedJson && line.includes('Loss:')) {
                const stepMatch = line.match(/Step (\d+)/);
                const lossMatch = line.match(/Loss: ([\d.]+)/);
                if (stepMatch && lossMatch) {
                    if (steps.length === 0 || steps[steps.length - 1] !== stepMatch[1]) {
                        steps.push(stepMatch[1]);
                        lossData.push(parseFloat(lossMatch[1]));
                        scoreData.push(0);
                    }
                }
            }
        });

        if (lossData.length === 0) {
            return {
                steps: [],
                loss: [],
                score: [],
                smoothedLoss: [],
                smoothedScore: []
            };
        }

        return {
            steps,
            loss: lossData,
            score: scoreData,
            smoothedLoss: calculateEma(lossData),
            smoothedScore: calculateEma(scoreData)
        };
    };

    const generateParallelChart = () => {
        if (tuningData.length === 0 || !tuningData[0].params) return null;
        const keys = Object.keys(tuningData[0].params);

        const parallelAxis = keys.map((k, i) => ({
            dim: i,
            name: k,
            nameTextStyle: { color: '#888', fontSize: 9 },
            axisLabel: { color: '#888', fontSize: 8 },
            axisLine: { lineStyle: { color: '#333' } }
        }));

        parallelAxis.push({
            dim: keys.length,
            name: "Loss",
            nameTextStyle: { color: '#ffffff', fontSize: 9 },
            axisLabel: { color: '#888', fontSize: 8 },
            axisLine: { lineStyle: { color: '#333' } }
        });

        const seriesData = tuningData.map(t => {
            const arr = keys.map(k => {
                const val = t.params[k];
                return typeof val === 'string' ? parseFloat(val) : val;
            });
            arr.push(t.value);
            return arr;
        });

        const losses = tuningData.map(t => t.value);
        const minLoss = Math.min(...losses);
        const maxLoss = Math.max(...losses);

        return (
            <ReactECharts
                option={{
                    parallel: { top: 30, bottom: 20, left: 30, right: 40 },
                    parallelAxis: parallelAxis,
                    title: { text: "Hyperparameter Impact", textStyle: { fontSize: 11, color: '#888' }, left: 'center' },
                    visualMap: {
                        type: 'continuous',
                        min: minLoss,
                        max: maxLoss,
                        show: false,
                        dimension: keys.length,
                        inRange: { color: ['#10b981', '#3b82f6', '#ef4444'] }
                    },
                    series: {
                        type: 'parallel',
                        lineStyle: { width: 2, opacity: 0.5 },
                        data: seriesData
                    }
                }}
                style={{ height: '300px', width: '100%' }}
            />
        );
    };

    const chartData = getChartData(selectedRunId);

    return (
        <div className="flex-1 flex flex-col overflow-hidden h-full bg-background">
            <div className="flex-1 flex overflow-hidden border-b border-border/50">
                {/* Sidebar: Configs List */}
                <Card className="w-64 flex flex-col rounded-none shadow-none border-0 border-r border-border/50 overflow-hidden h-full">
                    <div className="px-3 py-2 border-b border-border/50 flex justify-between items-center bg-muted/5">
                        <div className="font-semibold text-[10px] uppercase tracking-wider text-muted-foreground">
                            Config & Saved Runs
                        </div>
                        <CreateRunDialog
                            isCreateOpen={isCreateOpen} setIsCreateOpen={setIsCreateOpen}
                            newRunType={newRunType} setNewRunType={setNewRunType as any}
                            newRunName={newRunName} setNewRunName={setNewRunName}
                            newRunPreset={newRunPreset} setNewRunPreset={setNewRunPreset}
                            handleCreateNew={handleCreateNew}
                        />
                    </div>
                    <RunsSidebarList
                        runs={runs}
                        selectedRunId={selectedRunId} setSelectedRunId={setSelectedRunId}
                        setRunToRename={setRunToRename} setNewName={setNewName}
                        setRunToFlush={setRunToFlush} setRunToDelete={setRunToDelete}
                    />
                </Card>

                {/* Main Execution Surface */}
                <div className="flex-1 grid grid-cols-4 h-full">
                    {/* Col 1 & 2: Telemetry & Parameters */}
                    <Card className="col-span-2 shadow-none rounded-none border-0 border-r border-border/50 flex flex-col h-full bg-background overflow-hidden">
                        <Tabs defaultValue="telemetry" className="flex flex-col h-full">
                            <div className="px-3 py-1.5 border-b border-border/50 bg-muted/5 flex justify-between items-center">
                                <div>
                                    <h3 className="font-semibold text-xs">Run Config & Telemetry</h3>
                                </div>
                                <TabsList className="h-6">
                                    <TabsTrigger value="telemetry" className="text-[10px] px-2 py-0.5 h-5">Metrics</TabsTrigger>
                                    <TabsTrigger value="parameters" className="text-[10px] px-2 py-0.5 h-5">Parameters</TabsTrigger>
                                </TabsList>
                            </div>
                            <ScrollArea className="flex-1">
                                <TabsContent value="telemetry" className="m-0 p-4 flex flex-col gap-4">
                                    {selectedRun?.type === 'TUNING' ? (
                                        <div className="flex flex-col gap-6">
                                            <ReactECharts
                                                option={{
                                                    grid: { top: 30, right: 20, bottom: 25, left: 40 },
                                                    tooltip: {
                                                        trigger: 'item',
                                                        formatter: (params: any) => `Trial ${params.value[0]}<br/>Score: ${params.value[1]}`
                                                    },
                                                    title: { text: "Optimization History (Optuna)", textStyle: { fontSize: 11, color: '#888' }, left: 'center' },
                                                    xAxis: { type: 'value', name: 'Trial', splitLine: { show: false }, axisLabel: { color: '#888', fontSize: 9 } },
                                                    yAxis: { type: 'value', scale: true, splitLine: { lineStyle: { color: '#333' } }, axisLabel: { color: '#888', fontSize: 9 } },
                                                    series: [{
                                                        name: 'Trial',
                                                        type: 'scatter',
                                                        data: tuningData.map(t => [t.number, t.value]),
                                                        itemStyle: { color: '#3b82f6' }
                                                    }]
                                                }}
                                                style={{ height: '220px', width: '100%' }}
                                            />
                                            {generateParallelChart()}
                                        </div>
                                    ) : (
                                        <>
                                            <ReactECharts
                                                option={{
                                                    grid: { top: 10, right: 10, bottom: 20, left: 30 },
                                                    tooltip: { trigger: 'axis' },
                                                    title: { text: "Loss Curve", textStyle: { fontSize: 10, color: '#888' } },
                                                    xAxis: { type: 'category', data: chartData.steps, axisLabel: { color: '#888', fontSize: 9 } },
                                                    yAxis: { type: 'value', splitLine: { lineStyle: { color: '#333' } }, axisLabel: { color: '#888', fontSize: 9 } },
                                                    series: [
                                                        { data: chartData.loss, type: 'line', symbol: 'none', lineStyle: { color: '#3b82f6', width: 1, opacity: 0.3 } },
                                                        { data: chartData.smoothedLoss, type: 'line', smooth: true, symbol: 'none', lineStyle: { color: '#3b82f6', width: 2 } }
                                                    ]
                                                }}
                                                style={{ height: '150px', width: '100%' }}
                                            />
                                            <ReactECharts
                                                option={{
                                                    grid: { top: 10, right: 10, bottom: 20, left: 30 },
                                                    tooltip: { trigger: 'axis' },
                                                    title: { text: "Score Mean", textStyle: { fontSize: 10, color: '#888' } },
                                                    xAxis: { type: 'category', data: chartData.steps, axisLabel: { color: '#888', fontSize: 9 } },
                                                    yAxis: { type: 'value', splitLine: { lineStyle: { color: '#333' } }, axisLabel: { color: '#888', fontSize: 9 } },
                                                    series: [
                                                        { data: chartData.score, type: 'line', symbol: 'none', lineStyle: { color: '#10b981', width: 1, opacity: 0.3 } },
                                                        { data: chartData.smoothedScore, type: 'line', smooth: true, symbol: 'none', lineStyle: { color: '#10b981', width: 2 } }
                                                    ]
                                                }}
                                                style={{ height: '150px', width: '100%' }}
                                            />
                                        </>
                                    )}
                                </TabsContent>
                                <TabsContent value="parameters" className="m-0 p-4">
                                    <ExecutionParameters selectedRun={selectedRun} getConfigValue={getConfigValue} updateConfigKey={updateConfigKey} />
                                </TabsContent>
                            </ScrollArea>
                        </Tabs>
                    </Card>

                    {/* Col 3: Hydra Payload / Editor */}
                    <ConfigEditor
                        localConfig={localConfig}
                        setLocalConfig={setLocalConfig}
                        selectedRun={selectedRun}
                        handleSaveConfig={handleSaveConfig}
                    />

                    {/* Col 4: Action Controls */}
                    <EngineControls
                        selectedRun={selectedRun}
                        setRunLogs={setRunLogs}
                        selectedLogRunIds={selectedLogRunIds}
                        setSelectedLogRunIds={setSelectedLogRunIds}
                        loadRuns={loadRuns}
                        logoUrl={logoUrl}
                    />
                </div>
            </div>

            {/* Bottom Pane: Logs output and run comparison toggle */}
            <LiveLogsViewer
                runs={runs}
                runLogs={runLogs}
                selectedLogRunIds={selectedLogRunIds}
                toggleLogRun={toggleLogRun}
                handleCopyLogs={handleCopyLogs}
                copiedLogId={copiedLogId}
                logsEndRef={logsEndRef}
            />

            {/* Dialog for renaming */}
            <Dialog open={!!runToRename} onOpenChange={(open) => !open && setRunToRename(null)}>
                <DialogContent className="sm:max-w-[350px]">
                    <DialogHeader>
                        <DialogTitle>Rename Run</DialogTitle>
                    </DialogHeader>
                    <FieldSet>
                        <Field>
                            <FieldLabel>New Name</FieldLabel>
                            <Input value={newName} onChange={e => setNewName(e.target.value)} />
                        </Field>
                    </FieldSet>
                    <DialogFooter>
                        <Button variant="outline" size="sm" onClick={() => setRunToRename(null)}>Cancel</Button>
                        <Button size="sm" onClick={handleRename}>Save</Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            {/* Dialog for deletion */}
            <Dialog open={!!runToDelete} onOpenChange={(open) => !open && setRunToDelete(null)}>
                <DialogContent className="sm:max-w-[400px]">
                    <DialogHeader>
                        <DialogTitle>Delete Config</DialogTitle>
                        <DialogDescription>This deletes the configuration entirely. Cannot be undone.</DialogDescription>
                    </DialogHeader>
                    <DialogFooter>
                        <Button variant="outline" size="sm" onClick={() => setRunToDelete(null)}>Cancel</Button>
                        <Button variant="destructive" size="sm" onClick={handleDelete}>Delete</Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            {/* Dialog for flushing */}
            <Dialog open={!!runToFlush} onOpenChange={(open) => !open && setRunToFlush(null)}>
                <DialogContent className="sm:max-w-[400px]">
                    <DialogHeader>
                        <DialogTitle>Flush Data</DialogTitle>
                        <DialogDescription>Clears all metrics, checkpoints, and logs for this run but keeps the config.</DialogDescription>
                    </DialogHeader>
                    <DialogFooter>
                        <Button variant="outline" size="sm" onClick={() => setRunToFlush(null)}>Cancel</Button>
                        <Button variant="destructive" size="sm" onClick={handleFlush}>Flush</Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </div >
    );
}
