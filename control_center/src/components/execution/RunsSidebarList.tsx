import { ScrollArea } from '@/components/ui/scroll-area';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Button } from '@/components/ui/button';
import { Edit2, Eraser, Trash2 } from 'lucide-react';

interface RunsSidebarListProps {
    runs: any[];
    selectedRunId: string | null;
    setSelectedRunId: (id: string) => void;
    setRunToRename: (id: string) => void;
    setNewName: (name: string) => void;
    setRunToFlush: (id: string) => void;
    setRunToDelete: (id: string) => void;
}

export function RunsSidebarList({
    runs, selectedRunId, setSelectedRunId,
    setRunToRename, setNewName, setRunToFlush, setRunToDelete
}: RunsSidebarListProps) {
    return (
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
                        </div>
                    </div>
                ))}
            </div>
        </ScrollArea>
    );
}
