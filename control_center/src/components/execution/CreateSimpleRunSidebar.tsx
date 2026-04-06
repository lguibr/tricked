import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Field, FieldLabel, FieldSet, FieldGroup } from "@/components/ui/field";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ParameterForm, GroupDef } from "./ParameterForm";
import { AlertCircle } from "lucide-react";

interface Run {
    id: string;
    name: string;
    status: string;
    type: string;
    config: string;
    tag?: string;
}

export function CreateSimpleRunSidebar({
    onClose,
    loadRuns,
}: {
    onClose: () => void;
    loadRuns: () => void;
}) {
    const [name, setName] = useState("");
    const [error, setError] = useState("");

    const [config, setConfig] = useState<Record<string, any>>({
        num_processes: 8,
        train_batch_size: 1024,
        simulations: 800,
        max_gumbel_k: 16,
        lr_init: 0.02,
        num_blocks: 10,
        hidden_dimension_size: 128,
    });

    const parameterGroups: GroupDef[] = [
        {
            title: "Hardware Compute",
            color: "text-zinc-300",
            fields: [
                {
                    key: "num_processes",
                    label: "Worker Threads",
                    min: 1,
                    max: 128,
                    step: 1,
                },
                {
                    key: "train_batch_size",
                    label: "Batch Size",
                    min: 64,
                    max: 4096,
                    step: 64,
                },
            ],
        },
        {
            title: "MCTS Engine",
            color: "text-zinc-300",
            fields: [
                {
                    key: "simulations",
                    label: "Simulations",
                    min: 10,
                    max: 2000,
                    step: 10,
                },
                { key: "max_gumbel_k", label: "Gumbel K", min: 4, max: 64, step: 1 },
            ],
        },
        {
            title: "Learning Architecture",
            color: "text-zinc-300",
            fields: [
                {
                    key: "lr_init",
                    label: "Learning Rate",
                    min: 0.001,
                    max: 0.1,
                    step: 0.001,
                },
                { key: "num_blocks", label: "ResNet Blocks", min: 2, max: 30, step: 1 },
                {
                    key: "hidden_dimension_size",
                    label: "ResNet Channels",
                    min: 32,
                    max: 512,
                    step: 32,
                },
            ],
        },
    ];

    const handleCreate = async () => {
        if (!name.trim()) {
            setError("Run name is required.");
            return;
        }
        setError("");

        try {
            const createdRun = await invoke<Run>("create_run", {
                name,
                type: "SINGLE",
                preset: "default",
            });

            try {
                const baseConfig = JSON.parse(createdRun.config || "{}");
                Object.assign(baseConfig, config);

                await invoke("update_run_config", {
                    id: createdRun.id,
                    config: JSON.stringify(baseConfig, null, 2),
                });
            } catch (err) {
                console.warn("Failed to parse or save base config", err);
            }

            onClose();
            setName("");
            loadRuns();
        } catch (e) {
            console.error(e);
            setError(String(e));
        }
    };

    return (
        <div className="flex flex-col h-full overflow-hidden bg-[#09090b]">
            <div className="px-5 py-3 border-b border-border/10 flex justify-between items-center bg-black/20">
                <h3 className="text-xs font-bold text-zinc-100 uppercase tracking-widest">
                    New Simple Run
                </h3>
            </div>
            <ScrollArea className="flex-1 p-4">
                <FieldSet className="flex flex-col gap-4">
                    <FieldGroup className="gap-2">
                        <Field>
                            <FieldLabel
                                htmlFor="new-name"
                                className="text-[10px] text-zinc-400"
                            >
                                Experiment Name <span className="text-red-500">*</span>
                            </FieldLabel>
                            <Input
                                id="new-name"
                                value={name}
                                onChange={(e) => {
                                    setName(e.target.value);
                                    if (error) setError("");
                                }}
                                placeholder="e.g. baseline_v2"
                                className={`bg-zinc-900 border-border/30 text-sm ${error ? "border-red-500/50 focus-visible:ring-red-500/20" : ""
                                    }`}
                            />
                            {error && (
                                <div className="flex items-center gap-1 mt-1 text-red-500 text-[10px]">
                                    <AlertCircle className="w-3 h-3" />
                                    <span>{error}</span>
                                </div>
                            )}
                        </Field>
                        <div className="h-px bg-border/20 my-2" />

                        <ParameterForm
                            mode="single"
                            value={config}
                            onChange={setConfig}
                            groups={parameterGroups}
                        />
                    </FieldGroup>
                </FieldSet>
            </ScrollArea>
            <div className="p-4 border-t border-border/20 flex gap-2 shrink-0 bg-black/20">
                <Button
                    variant="outline"
                    className="flex-1 text-xs h-8"
                    onClick={onClose}
                >
                    Cancel
                </Button>
                <Button className="flex-1 text-xs h-8" onClick={handleCreate}>
                    Create Run
                </Button>
            </div>
        </div>
    );
}
