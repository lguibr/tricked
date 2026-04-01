import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Play, Square, Pause } from 'lucide-react';
import { invoke } from '@tauri-apps/api/core';

interface EngineControlsProps {
    selectedRun: any;
    setRunLogs: React.Dispatch<React.SetStateAction<Record<string, string[]>>>;
    selectedLogRunIds: string[];
    setSelectedLogRunIds: (ids: string[]) => void;
    loadRuns: () => Promise<void>;
    logoUrl: string;
}

export function EngineControls({
    selectedRun,
    setRunLogs,
    selectedLogRunIds,
    setSelectedLogRunIds,
    loadRuns,
    logoUrl
}: EngineControlsProps) {

    const handleStart = async () => {
        if (!selectedRun) return;
        setRunLogs(prev => ({ ...prev, [selectedRun.id]: [] }));
        if (!selectedLogRunIds.includes(selectedRun.id)) {
            setSelectedLogRunIds([...selectedLogRunIds, selectedRun.id]);
        }
        try {
            await invoke('start_run', { id: selectedRun.id });
            loadRuns();
        } catch (e) {
            console.error(e);
        }
    };

    const handleStop = async (force: boolean) => {
        if (!selectedRun) return;
        try {
            await invoke('stop_run', { id: selectedRun.id, force });
            loadRuns();
        } catch (e) {
            console.error(e);
        }
    };

    return (
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
                    <Button size="sm" className="w-full flex justify-start pl-3" disabled={selectedRun?.status === 'RUNNING'} onClick={handleStart}>
                        <Play className="h-3 w-3 mr-2" /> Start Run
                    </Button>
                    <Button size="sm" variant="secondary" className="w-full flex justify-start pl-3" disabled={selectedRun?.status !== 'RUNNING'} onClick={() => handleStop(false)}>
                        <Pause className="h-3 w-3 mr-2" /> Graceful Stop
                    </Button>
                    <Button size="sm" variant="destructive" className="w-full flex justify-start pl-3" disabled={selectedRun?.status !== 'RUNNING'} onClick={() => handleStop(true)}>
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
    );
}
