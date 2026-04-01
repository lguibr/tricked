import { useState, useEffect } from 'react';
import { Play, Square, Pause, Trash2, Edit2, Eraser, Plus, TerminalSquare, Copy, Check } from 'lucide-react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import ReactECharts from 'echarts-for-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Slider } from '@/components/ui/slider';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger, DialogFooter } from '@/components/ui/dialog';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Field, FieldDescription, FieldGroup, FieldLabel, FieldSet } from '@/components/ui/field';
import { Toggle } from '@/components/ui/toggle';
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

    // State for viewing logs
    const [selectedLogRunIds, setSelectedLogRunIds] = useState<string[]>([]);
    const [copiedLogId, setCopiedLogId] = useState<string | null>(null);
    const [runLogs, setRunLogs] = useState<Record<string, string[]>>({});
    const [localConfig, setLocalConfig] = useState<string>('');

    useEffect(() => { if (selectedRun) setLocalConfig(selectedRun.config); }, [selectedRunId, runs]);

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
            const newRun = await invoke<Run>('create_run', { name: newRunName, type: newRunType, baseConfigId: null });
            setRuns([...runs, newRun]);
            setIsCreateOpen(false);
            setNewRunName('');
            setNewRunType('SINGLE');
            setSelectedRunId(newRun.id);
        }
    };

    const handleSaveConfig = async () => {
        if (selectedRunId) {
            await invoke('save_config', { id: selectedRunId, config: localConfig });
            loadRuns();
        }
    };

    const getChartData = (runId: string | null) => {
        if (!runId) return { steps: [], loss: [], winRate: [] };
        const lines = runLogs[runId] || [];
        const lossData: number[] = [];
        const steps: string[] = [];

        lines.forEach(line => {
            if (line.includes('Loss:')) {
                const stepMatch = line.match(/Step (\d+)/);
                const lossMatch = line.match(/Loss: ([\d.]+)/);
                if (stepMatch && lossMatch) {
                    steps.push(stepMatch[1]);
                    lossData.push(parseFloat(lossMatch[1]));
                }
            }
        });

        if (lossData.length === 0) {
            return {
                steps: ['0', '1k', '2k', '3k', '4k', '5k'],
                loss: [2.3, 2.1, 1.8, 1.5, 1.2, 0.9],
                winRate: [0.1, 0.22, 0.45, 0.65, 0.82, 0.95]
            };
        }

        return { steps, loss: lossData, winRate: lossData.map(v => Math.max(0, 1 - (v / 3))) };
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
                        <Dialog open={isCreateOpen} onOpenChange={setIsCreateOpen}>
                            <DialogTrigger asChild>
                                <Button variant="ghost" size="icon" className="h-5 w-5"><Plus className="w-3 h-3" /></Button>
                            </DialogTrigger>
                            <DialogContent className="sm:max-w-[425px]">
                                <DialogHeader>
                                    <DialogTitle>Create Run/Experiment</DialogTitle>
                                    <DialogDescription>Setup a new single run or tuning project.</DialogDescription>
                                </DialogHeader>

                                <FieldSet>
                                    <FieldGroup className="mt-2">
                                        <Field>
                                            <FieldLabel>Run Type</FieldLabel>
                                            <div className="flex space-x-2">
                                                <Button size="sm" variant={newRunType === 'SINGLE' ? 'default' : 'outline'} onClick={() => setNewRunType('SINGLE')}>Single</Button>
                                                <Button size="sm" variant={newRunType === 'TUNING' ? 'default' : 'outline'} onClick={() => setNewRunType('TUNING')}>Optuna Tuning</Button>
                                            </div>
                                        </Field>
                                        <Field>
                                            <FieldLabel htmlFor="new-name">Name</FieldLabel>
                                            <Input id="new-name" value={newRunName} onChange={(e) => setNewRunName(e.target.value)} placeholder="e.g. baseline_v2" />
                                        </Field>
                                        <Field>
                                            <FieldLabel htmlFor="base-config">Base Config</FieldLabel>
                                            <select id="base-config" className="flex h-9 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm ring-offset-background focus:outline-none focus:ring-1 focus:ring-ring disabled:cursor-not-allowed disabled:opacity-50">
                                                {runs.map(r => <option key={r.id} value={r.id}>{r.name}</option>)}
                                            </select>
                                            <FieldDescription>Select a config to inherit parameters from.</FieldDescription>
                                        </Field>
                                        {newRunType === 'TUNING' && (
                                            <Field>
                                                <FieldLabel htmlFor="cardinality">Suggestion Range Cardinality</FieldLabel>
                                                <Input id="cardinality" type="number" placeholder="e.g. 50" defaultValue={50} />
                                                <FieldDescription>Number of trials or hyperparameter suggestions to explore.</FieldDescription>
                                            </Field>
                                        )}
                                    </FieldGroup>
                                </FieldSet>

                                <DialogFooter>
                                    <Button variant="outline" onClick={() => setIsCreateOpen(false)}>Cancel</Button>
                                    <Button onClick={handleCreateNew}>Create</Button>
                                </DialogFooter>
                            </DialogContent>
                        </Dialog>
                    </div>
                    <ScrollArea className="flex-1 p-0">
                        <div className="flex flex-col">
                            {runs.map((run) => (
                                <div
                                    key={run.id}
                                    className={`px-3 py-2 border-b border-border/30 relative group cursor-pointer transition-colors ${selectedRunId === run.id ? 'bg-primary/5 border-l-2 border-l-primary' : 'border-l-2 border-l-transparent hover:bg-accent'}`}
                                    onClick={() => setSelectedRunId(run.id)}
                                >
                                    <div className="flex justify-between items-start">
                                        <div className="pr-12">
                                            <h3 className={`font-medium text-xs leading-tight ${selectedRunId === run.id ? 'text-primary' : ''} truncate`}>{run.name}</h3>
                                            <p className={`text-[10px] mt-0.5 font-mono ${run.status === 'RUNNING' ? 'text-green-500' : 'text-muted-foreground'}`}>
                                                {run.type.substring(0, 1)} · {run.status}
                                            </p>
                                        </div>
                                    </div>
                                    <div className="absolute top-1.5 right-1.5 opacity-0 group-hover:opacity-100 flex space-x-0.5 transition-opacity">
                                        {run.status !== 'RUNNING' && (
                                            <>
                                                <TooltipProvider>
                                                    <Tooltip>
                                                        <TooltipTrigger asChild>
                                                            <Button variant="ghost" size="icon" className="h-5 w-5" onClick={(e) => { e.stopPropagation(); setRunToRename(run.id); setNewName(run.name); }}>
                                                                <Edit2 className="h-2.5 w-2.5" />
                                                            </Button>
                                                        </TooltipTrigger>
                                                        <TooltipContent><p className="text-xs">Rename</p></TooltipContent>
                                                    </Tooltip>
                                                </TooltipProvider>

                                                <TooltipProvider>
                                                    <Tooltip>
                                                        <TooltipTrigger asChild>
                                                            <Button variant="ghost" size="icon" className="h-5 w-5 text-yellow-500 hover:bg-yellow-500/20 hover:text-yellow-600" onClick={(e) => { e.stopPropagation(); setRunToFlush(run.id); }}>
                                                                <Eraser className="h-2.5 w-2.5" />
                                                            </Button>
                                                        </TooltipTrigger>
                                                        <TooltipContent><p className="text-xs">Flush Data</p></TooltipContent>
                                                    </Tooltip>
                                                </TooltipProvider>

                                                <TooltipProvider>
                                                    <Tooltip>
                                                        <TooltipTrigger asChild>
                                                            <Button variant="ghost" size="icon" className="h-5 w-5 text-destructive hover:bg-destructive hover:text-white" onClick={(e) => { e.stopPropagation(); setRunToDelete(run.id); }}>
                                                                <Trash2 className="h-2.5 w-2.5" />
                                                            </Button>
                                                        </TooltipTrigger>
                                                        <TooltipContent><p className="text-xs">Delete</p></TooltipContent>
                                                    </Tooltip>
                                                </TooltipProvider>
                                            </>
                                        )}
                                        {run.status === 'RUNNING' && (
                                            <Button variant="ghost" size="icon" className="h-5 w-5 text-destructive hover:bg-destructive hover:text-white" onClick={(e) => { e.stopPropagation(); /* simulate stop */ }}>
                                                <Square className="h-2.5 w-2.5" />
                                            </Button>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </ScrollArea>
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
                                    <ReactECharts
                                        option={{
                                            grid: { top: 10, right: 10, bottom: 20, left: 30 },
                                            tooltip: { trigger: 'axis' },
                                            title: { text: "Loss Curve", textStyle: { fontSize: 10, color: '#888' } },
                                            xAxis: { type: 'category', data: chartData.steps, axisLabel: { color: '#888', fontSize: 9 } },
                                            yAxis: { type: 'value', splitLine: { lineStyle: { color: '#333' } }, axisLabel: { color: '#888', fontSize: 9 } },
                                            series: [{ data: chartData.loss, type: 'line', smooth: true, symbol: 'none', lineStyle: { color: '#3b82f6', width: 2 } }]
                                        }}
                                        style={{ height: '150px', width: '100%' }}
                                    />
                                    <ReactECharts
                                        option={{
                                            grid: { top: 10, right: 10, bottom: 20, left: 30 },
                                            tooltip: { trigger: 'axis' },
                                            title: { text: "Win Rate", textStyle: { fontSize: 10, color: '#888' } },
                                            xAxis: { type: 'category', data: chartData.steps, axisLabel: { color: '#888', fontSize: 9 } },
                                            yAxis: { type: 'value', max: 1.0, splitLine: { lineStyle: { color: '#333' } }, axisLabel: { color: '#888', fontSize: 9 } },
                                            series: [{ data: chartData.winRate, type: 'line', smooth: true, symbol: 'none', lineStyle: { color: '#10b981', width: 2 } }]
                                        }}
                                        style={{ height: '150px', width: '100%' }}
                                    />
                                </TabsContent>
                                <TabsContent value="parameters" className="m-0 p-4">
                                    <FieldSet>
                                        <FieldGroup>
                                            <Field>
                                                <div className="flex justify-between w-full">
                                                    <FieldLabel>Max Steps</FieldLabel>
                                                    <span className="text-[10px] font-mono text-muted-foreground">1,000,000</span>
                                                </div>
                                                <Slider className="py-2" defaultValue={[1000000]} max={10000000} step={100000} disabled={selectedRun?.status === 'RUNNING'} />
                                            </Field>
                                            <div className="h-2" />
                                            <Field>
                                                <div className="flex justify-between w-full">
                                                    <FieldLabel>Gumbel C_Visit</FieldLabel>
                                                    <span className="text-[10px] font-mono text-muted-foreground">50</span>
                                                </div>
                                                <Slider className="py-2" defaultValue={[50]} max={100} step={1} disabled={selectedRun?.status === 'RUNNING'} />
                                            </Field>
                                            <div className="h-2" />
                                            <Field>
                                                <div className="flex justify-between w-full">
                                                    <FieldLabel>Value Weight</FieldLabel>
                                                    <span className="text-[10px] font-mono text-muted-foreground">0.25</span>
                                                </div>
                                                <Slider className="py-2" defaultValue={[25]} max={100} step={1} disabled={selectedRun?.status === 'RUNNING'} />
                                            </Field>
                                        </FieldGroup>
                                    </FieldSet>
                                </TabsContent>
                            </ScrollArea>
                        </Tabs>
                    </Card>

                    {/* Col 3: Hydra Payload / Editor */}
                    <Card className="col-span-1 shadow-none rounded-none border-0 border-r border-border/50 flex flex-col h-full overflow-hidden bg-background">
                        <div className="px-3 py-2 border-b border-border/50 bg-muted/5 flex justify-between items-center">
                            <div>
                                <h3 className="font-semibold text-xs">JSON Payload</h3>
                                <p className="text-[10px] text-muted-foreground">Hydra config format</p>
                            </div>
                            <Button variant="ghost" size="icon" className="h-5 w-5"><Edit2 className="h-2.5 w-2.5" /></Button>
                        </div>
                        <div className="flex-1 flex flex-col bg-zinc-950 p-0 font-mono text-[10px] text-zinc-400 overflow-hidden border-t border-black/50 relative">
                            <textarea
                                className="flex-1 w-full h-full bg-transparent resize-none outline-none p-3 break-all focus:ring-0 text-zinc-300"
                                value={localConfig}
                                onChange={e => setLocalConfig(e.target.value)}
                                spellCheck={false}
                            />
                            {selectedRun && localConfig !== selectedRun.config && (
                                <div className="absolute bottom-2 right-2 flex gap-2">
                                    <Button size="sm" onClick={handleSaveConfig} className="h-6 text-[10px]">Save Config</Button>
                                </div>
                            )}
                        </div>
                    </Card>

                    {/* Col 4: Action Controls */}
                    <Card className="col-span-1 shadow-none rounded-none border-0 flex flex-col h-full bg-background">
                        <div className="px-3 py-2 border-b border-border/50 bg-muted/5">
                            <h3 className="font-semibold text-xs">Engine Control</h3>
                            <p className="text-[10px] text-muted-foreground">Process & Lifecycle</p>
                        </div>
                        <div className="flex-1 flex flex-col items-center justify-center p-4">
                            <div className="w-16 h-16 rounded-lg flex items-center justify-center mb-4 overflow-hidden border border-border/50">
                                <img src={logoUrl} alt="Tricked AI Logo" className="w-full h-full object-cover" />
                            </div>

                            <div className="flex flex-col gap-2 w-full max-w-[120px]">
                                <Button size="sm" className="w-full flex justify-start pl-3" disabled={selectedRun?.status === 'RUNNING'} onClick={async () => {
                                    if (selectedRun) {
                                        setRunLogs(prev => ({ ...prev, [selectedRun.id]: [] }));
                                        if (!selectedLogRunIds.includes(selectedRun.id)) setSelectedLogRunIds([...selectedLogRunIds, selectedRun.id]);
                                        try { await invoke('start_run', { id: selectedRun.id }); loadRuns(); } catch (e) { console.error(e); }
                                    }
                                }}>
                                    <Play className="h-3 w-3 mr-2" /> Start Run
                                </Button>
                                <Button size="sm" variant="secondary" className="w-full flex justify-start pl-3" disabled={selectedRun?.status !== 'RUNNING'} onClick={async () => {
                                    if (selectedRun) { try { await invoke('stop_run', { id: selectedRun.id, force: false }); loadRuns(); } catch (e) { console.error(e); } }
                                }}>
                                    <Pause className="h-3 w-3 mr-2" /> Graceful Stop
                                </Button>
                                <Button size="sm" variant="destructive" className="w-full flex justify-start pl-3" disabled={selectedRun?.status !== 'RUNNING'} onClick={async () => {
                                    if (selectedRun) { try { await invoke('stop_run', { id: selectedRun.id, force: true }); loadRuns(); } catch (e) { console.error(e); } }
                                }}>
                                    <Square className="h-3 w-3 mr-2" /> Kill Run
                                </Button>
                            </div>

                            <div className="mt-8 text-center">
                                <div className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider mb-1">State</div>
                                <div className={`text-xs font-bold ${selectedRun?.status === 'RUNNING' ? 'text-green-500 animate-pulse' : 'text-zinc-500'}`}>
                                    {selectedRun?.status || 'IDLE'}
                                </div>
                            </div>
                        </div>
                    </Card>
                </div>
            </div>

            {/* Bottom Pane: Logs output and run comparison toggle */}
            <Card className="h-64 flex flex-col rounded-none shadow-none border-0 overflow-hidden shrink-0 bg-[#0c0c0c]">
                <div className="flex items-center justify-between px-3 py-1.5 border-b border-black/50 bg-zinc-900 shadow-sm">
                    <div className="flex items-center space-x-2 text-muted-foreground text-[10px] font-mono">
                        <TerminalSquare className="w-3 h-3" />
                        <span>LIVE LOGROUTER</span>
                    </div>
                    <div className="flex items-center gap-2">
                        {runs.filter(r => r.status !== 'WAITING').map(r => (
                            <Toggle
                                key={r.id}
                                size="sm"
                                pressed={selectedLogRunIds.includes(r.id)}
                                onPressedChange={(p) => toggleLogRun(r.id, p)}
                                className="h-6 px-2 text-[10px] data-[state=on]:bg-primary/20 data-[state=on]:text-primary border border-transparent data-[state=on]:border-primary/50"
                            >
                                {r.name}
                            </Toggle>
                        ))}
                    </div>
                </div>
                <div className="flex-1 flex overflow-hidden bg-black text-green-500 selection:bg-green-900 selection:text-white">
                    {selectedLogRunIds.length === 0 ? (
                        <div className="flex-1 flex items-center justify-center text-zinc-600 text-xs font-mono">
                            Select a run to view logs
                        </div>
                    ) : (
                        selectedLogRunIds.map(runId => {
                            const run = runs.find(r => r.id === runId);
                            // Define color blocks for visual separation
                            const color = runId === '1' ? 'border-orange-500/50' : runId === '2' ? 'border-purple-500/50' : 'border-blue-500/50';
                            const lines = runLogs[runId] || [];
                            const logsText = lines.join('\n');

                            return (
                                <ScrollArea key={runId} className={`flex-1 p-2 font-mono text-[10px] leading-relaxed border-l ${color} first:border-l-0 pb-12 relative overflow-y-auto`}>
                                    <div className="sticky top-0 bg-black/90 backdrop-blur-sm py-1 mb-2 font-semibold text-zinc-300 z-10 border-b border-border/20 flex justify-between items-center group/header">
                                        <div><span className="opacity-50">#</span> {run?.name}</div>
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            className="h-5 w-5 opacity-0 group-hover/header:opacity-100 text-muted-foreground hover:text-white"
                                            onClick={() => handleCopyLogs(runId, logsText)}
                                        >
                                            {copiedLogId === runId ? <Check className="h-3 w-3 text-green-500" /> : <Copy className="h-3 w-3" />}
                                        </Button>
                                    </div>
                                    <div className="space-y-0.5 whitespace-pre-wrap">
                                        {lines.length === 0 ? <span className="text-zinc-600 italic">Waiting for connection...</span> :
                                            lines.map((line, idx) => (
                                                <div key={idx}>
                                                    {line.includes('[WARN]') ? <span className="text-yellow-500">{line}</span> :
                                                        line.includes('[ERR]') ? <span className="text-red-500">{line}</span> :
                                                            line.includes('[INFO]') ? <span><span className="text-blue-400">[INFO]</span>{line.split('[INFO]')[1]}</span> :
                                                                line}
                                                </div>
                                            ))
                                        }
                                    </div>
                                </ScrollArea>
                            );
                        })
                    )}
                </div>
            </Card>

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
        </div>
    );
}
