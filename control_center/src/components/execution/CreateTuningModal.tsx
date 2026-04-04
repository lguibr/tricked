import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Field, FieldLabel, FieldGroup } from "@/components/ui/field";
import { Slider } from "@/components/ui/slider";
import { ScrollArea } from "@/components/ui/scroll-area";

interface CreateTuningModalProps {
    isOpen: boolean;
    setIsOpen: (v: boolean) => void;
    studyType: "HARDWARE" | "MCTS" | "LEARNING";
    runsViewRefresh?: () => void;
}

export function CreateTuningModal({
    isOpen,
    setIsOpen,
    studyType,
    runsViewRefresh,
}: CreateTuningModalProps) {
    const [trials, setTrials] = useState<number[]>([30]);
    const [timeout, setTimeoutVal] = useState<number[]>([400]);
    const [maxSteps, setMaxSteps] = useState<number[]>([15]);

    // Network Limits (used across all studies currently as base template)
    const [resnetBlocks, setResnetBlocks] = useState<number[]>([4]);
    const [resnetChannels, setResnetChannels] = useState<number[]>([64]);

    // HARDWARE Study Bounds
    const [workersMin, setWorkersMin] = useState<number[]>([1]);
    const [workersMax, setWorkersMax] = useState<number[]>([32]);
    const [batchMin, setBatchMin] = useState<number[]>([64]);
    const [batchMax, setBatchMax] = useState<number[]>([2048]);

    // MCTS Study Bounds
    const [simsMin, setSimsMin] = useState<number[]>([10]);
    const [simsMax, setSimsMax] = useState<number[]>([1000]);
    const [gumbelMin, setGumbelMin] = useState<number[]>([4]);
    const [gumbelMax, setGumbelMax] = useState<number[]>([32]);

    // LEARNING Study Bounds
    const [lrMin, setLrMin] = useState<number[]>([0.005]);
    const [lrMax, setLrMax] = useState<number[]>([0.1]);

    const handleCreate = async () => {
        try {
            const bounds: Record<string, any> = {};
            if (studyType === "HARDWARE") {
                bounds["num_processes"] = { min: workersMin[0], max: workersMax[0] };
                bounds["train_batch_size"] = { min: batchMin[0], max: batchMax[0] };
            } else if (studyType === "MCTS") {
                bounds["simulations"] = { min: simsMin[0], max: simsMax[0] };
                bounds["max_gumbel_k"] = { min: gumbelMin[0], max: gumbelMax[0] };
            } else if (studyType === "LEARNING") {
                bounds["lr_init"] = { min: lrMin[0], max: lrMax[0] };
            }

            await invoke("start_study", {
                studyType,
                trials: trials[0],
                maxSteps: maxSteps[0],
                timeout: timeout[0],
                resnetBlocks: resnetBlocks[0],
                resnetChannels: resnetChannels[0],
                bounds,
            });

            if (runsViewRefresh) runsViewRefresh();
            setIsOpen(false);
        } catch (e) {
            console.error(e);
            alert(e);
        }
    };

    const getDomainContent = () => {
        if (studyType === "HARDWARE") {
            return (
                <div className="flex flex-col gap-3 p-3 bg-[#0a0a0c] rounded border border-border/20">
                    <Field>
                        <div className="flex justify-between mb-1">
                            <FieldLabel className="text-[10px] text-zinc-400">Worker Processes Limit (Min / Max)</FieldLabel>
                            <span className="text-[10px] font-mono text-zinc-500">{workersMin[0]} - {workersMax[0]}</span>
                        </div>
                        <div className="flex gap-4">
                            <Slider value={workersMin} onValueChange={(v) => { if (v[0] <= workersMax[0]) setWorkersMin(v); }} min={1} max={128} step={1} className="py-1 flex-1" />
                            <Slider value={workersMax} onValueChange={(v) => { if (v[0] >= workersMin[0]) setWorkersMax(v); }} min={1} max={128} step={1} className="py-1 flex-1" />
                        </div>
                    </Field>
                    <Field>
                        <div className="flex justify-between mb-1">
                            <FieldLabel className="text-[10px] text-zinc-400">Train Batch Size Limit (Min / Max)</FieldLabel>
                            <span className="text-[10px] font-mono text-zinc-500">{batchMin[0]} - {batchMax[0]}</span>
                        </div>
                        <div className="flex gap-4">
                            <Slider value={batchMin} onValueChange={(v) => { if (v[0] <= batchMax[0]) setBatchMin(v); }} min={64} max={4096} step={64} className="py-1 flex-1" />
                            <Slider value={batchMax} onValueChange={(v) => { if (v[0] >= batchMin[0]) setBatchMax(v); }} min={64} max={4096} step={64} className="py-1 flex-1" />
                        </div>
                    </Field>
                </div>
            );
        }
        if (studyType === "MCTS") {
            return (
                <div className="flex flex-col gap-3 p-3 bg-[#0a0a0c] rounded border border-border/20">
                    <Field>
                        <div className="flex justify-between mb-1">
                            <FieldLabel className="text-[10px] text-zinc-400">Simulations (Min / Max)</FieldLabel>
                            <span className="text-[10px] font-mono text-zinc-500">{simsMin[0]} - {simsMax[0]}</span>
                        </div>
                        <div className="flex gap-4">
                            <Slider value={simsMin} onValueChange={(v) => { if (v[0] <= simsMax[0]) setSimsMin(v); }} min={10} max={2000} step={10} className="py-1 flex-1" />
                            <Slider value={simsMax} onValueChange={(v) => { if (v[0] >= simsMin[0]) setSimsMax(v); }} min={10} max={2000} step={10} className="py-1 flex-1" />
                        </div>
                    </Field>
                    <Field>
                        <div className="flex justify-between mb-1">
                            <FieldLabel className="text-[10px] text-zinc-400">Max Gumbel K (Min / Max)</FieldLabel>
                            <span className="text-[10px] font-mono text-zinc-500">{gumbelMin[0]} - {gumbelMax[0]}</span>
                        </div>
                        <div className="flex gap-4">
                            <Slider value={gumbelMin} onValueChange={(v) => { if (v[0] <= gumbelMax[0]) setGumbelMin(v); }} min={4} max={64} step={1} className="py-1 flex-1" />
                            <Slider value={gumbelMax} onValueChange={(v) => { if (v[0] >= gumbelMin[0]) setGumbelMax(v); }} min={4} max={64} step={1} className="py-1 flex-1" />
                        </div>
                    </Field>
                </div>
            );
        }
        if (studyType === "LEARNING") {
            return (
                <div className="flex flex-col gap-3 p-3 bg-[#0a0a0c] rounded border border-border/20">
                    <Field>
                        <div className="flex justify-between mb-1">
                            <FieldLabel className="text-[10px] text-zinc-400">Learning Rate Limit (Min / Max)</FieldLabel>
                            <span className="text-[10px] font-mono text-zinc-500">{lrMin[0]} - {lrMax[0]}</span>
                        </div>
                        <div className="flex gap-4">
                            <Slider value={lrMin} onValueChange={(v) => { if (v[0] <= lrMax[0]) setLrMin(v); }} min={0.001} max={0.1} step={0.001} className="py-1 flex-1" />
                            <Slider value={lrMax} onValueChange={(v) => { if (v[0] >= lrMin[0]) setLrMax(v); }} min={0.001} max={0.1} step={0.001} className="py-1 flex-1" />
                        </div>
                    </Field>
                </div>
            );
        }
    };

    return (
        <Dialog open={isOpen} onOpenChange={setIsOpen}>
            <DialogContent className="sm:max-w-[600px] max-h-[90vh] flex flex-col pt-6 pb-4">
                <DialogHeader className="shrink-0">
                    <DialogTitle>Configure {studyType} Optuna Scan</DialogTitle>
                    <DialogDescription>
                        Specify exactly which variables Optuna is allowed to sweep, and its absolute boundary limits.
                    </DialogDescription>
                </DialogHeader>

                <ScrollArea className="flex-1 px-4 -mx-4">
                    <FieldGroup className="gap-3">
                        <div className="flex flex-col gap-2 p-3 bg-zinc-950 rounded-lg border border-border/40">
                            <span className="text-xs font-bold text-zinc-300 uppercase tracking-wider">Optuna Global Controls</span>
                            <div className="grid grid-cols-3 gap-4 mt-2 p-3 bg-[#0a0a0c] rounded border border-border/20">
                                <Field>
                                    <div className="flex justify-between w-full mb-1">
                                        <FieldLabel className="text-[10px] text-zinc-400">Max Trials</FieldLabel>
                                        <span className="text-[10px] font-mono text-zinc-500">{trials[0]}</span>
                                    </div>
                                    <Slider value={trials} onValueChange={setTrials} min={1} max={500} step={1} className="py-1" />
                                </Field>
                                <Field>
                                    <div className="flex justify-between w-full mb-1">
                                        <FieldLabel className="text-[10px] text-zinc-400">Timeout (Secs)</FieldLabel>
                                        <span className="text-[10px] font-mono text-zinc-500">{timeout[0]}</span>
                                    </div>
                                    <Slider value={timeout} onValueChange={setTimeoutVal} min={10} max={3600} step={10} className="py-1" />
                                </Field>
                                <Field>
                                    <div className="flex justify-between w-full mb-1">
                                        <FieldLabel className="text-[10px] text-zinc-400">Steps Per Trial</FieldLabel>
                                        <span className="text-[10px] font-mono text-zinc-500">{maxSteps[0]}</span>
                                    </div>
                                    <Slider value={maxSteps} onValueChange={setMaxSteps} min={1} max={100} step={1} className="py-1" />
                                </Field>
                            </div>
                        </div>

                        <div className="flex flex-col gap-2 p-3 bg-zinc-950 rounded-lg border border-border/40">
                            <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider">{studyType} Hyperparameter Bounds</span>
                            {getDomainContent()}
                        </div>

                        <div className="flex flex-col gap-2 p-3 bg-zinc-950 rounded-lg border border-border/40">
                            <span className="text-xs font-bold text-zinc-300 uppercase tracking-wider">Constant Network Limits</span>
                            <div className="grid grid-cols-2 gap-4 mt-2 p-3 bg-[#0a0a0c] rounded border border-border/20">
                                <Field>
                                    <div className="flex justify-between mb-1">
                                        <FieldLabel className="text-[10px] text-zinc-400">ResNet Blocks</FieldLabel>
                                        <span className="text-[10px] font-mono text-zinc-500">{resnetBlocks[0]}</span>
                                    </div>
                                    <Slider value={resnetBlocks} onValueChange={setResnetBlocks} min={2} max={30} step={1} className="py-1" />
                                </Field>
                                <Field>
                                    <div className="flex justify-between mb-1">
                                        <FieldLabel className="text-[10px] text-zinc-400">ResNet Channels</FieldLabel>
                                        <span className="text-[10px] font-mono text-zinc-500">{resnetChannels[0]}</span>
                                    </div>
                                    <Slider value={resnetChannels} onValueChange={setResnetChannels} min={32} max={512} step={32} className="py-1" />
                                </Field>
                            </div>
                        </div>
                    </FieldGroup>
                </ScrollArea>

                <DialogFooter className="shrink-0 mt-4">
                    <Button variant="outline" onClick={() => setIsOpen(false)}>Cancel</Button>
                    <Button onClick={handleCreate} className="bg-emerald-600 hover:bg-emerald-700 text-white">Execute {studyType} Scan</Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}
