import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Edit2 } from 'lucide-react';

interface ConfigEditorProps {
    localConfig: string;
    setLocalConfig: (config: string) => void;
    selectedRun: any;
    handleSaveConfig: () => Promise<void>;
}

export function ConfigEditor({ localConfig, setLocalConfig, selectedRun, handleSaveConfig }: ConfigEditorProps) {
    return (
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
    );
}
