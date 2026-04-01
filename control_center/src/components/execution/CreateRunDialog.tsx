import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger, DialogFooter } from '@/components/ui/dialog';
import { Field, FieldDescription, FieldGroup, FieldLabel, FieldSet } from '@/components/ui/field';
import { Plus } from 'lucide-react';

interface CreateRunDialogProps {
    isCreateOpen: boolean;
    setIsCreateOpen: (v: boolean) => void;
    newRunType: 'SINGLE' | 'TUNING';
    setNewRunType: (v: 'SINGLE' | 'TUNING') => void;
    newRunName: string;
    setNewRunName: (v: string) => void;
    newRunPreset: string;
    setNewRunPreset: (v: string) => void;
    handleCreateNew: () => void;
}

export function CreateRunDialog({
    isCreateOpen, setIsCreateOpen,
    newRunType, setNewRunType,
    newRunName, setNewRunName,
    newRunPreset, setNewRunPreset,
    handleCreateNew
}: CreateRunDialogProps) {
    return (
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
                            <FieldLabel htmlFor="base-config">Base Config / Hydra Payload</FieldLabel>
                            <select id="base-config" value={newRunPreset} onChange={(e) => setNewRunPreset(e.target.value)} className="flex h-9 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm ring-offset-background focus:outline-none focus:ring-1 focus:ring-neutral-700 disabled:cursor-not-allowed disabled:opacity-50">
                                <option value="default">Default Core Settings</option>
                                <option value="small">Small Network Profiling Check</option>
                                <option value="big">Big ResNet SOTA Config</option>
                            </select>
                            <FieldDescription>Select a config schema to bootstrap parameters from.</FieldDescription>
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
    );
}
