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
import { Input } from "@/components/ui/input";
import { Field, FieldLabel, FieldSet, FieldGroup } from "@/components/ui/field";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ParameterForm, GroupDef } from "./ParameterForm";

interface Run {
  id: string;
  name: string;
  status: string;
  type: string;
  config: string;
  tag?: string;
}

export function CreateSimpleRunModal({
  isOpen,
  setIsOpen,
  loadRuns,
}: {
  isOpen: boolean;
  setIsOpen: (v: boolean) => void;
  loadRuns: () => void;
}) {
  const [name, setName] = useState("");

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
    if (!name.trim()) return;
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

      setIsOpen(false);
      setName("");
      loadRuns();
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogContent className="sm:max-w-[550px] max-h-[90vh] flex flex-col pt-6 pb-4">
        <DialogHeader className="shrink-0">
          <DialogTitle>New Simple Run</DialogTitle>
          <DialogDescription>
            Generates a new configuration directory bound to a single training
            trajectory.
          </DialogDescription>
        </DialogHeader>

        <ScrollArea className="flex-1 px-4 -mx-4">
          <FieldSet className="py-2">
            <FieldGroup className="gap-4">
              <Field>
                <FieldLabel htmlFor="new-name">Experiment Name</FieldLabel>
                <Input
                  id="new-name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g. baseline_v2"
                />
              </Field>
              <hr className="border-border/50 my-2" />

              <ParameterForm
                mode="single"
                value={config}
                onChange={setConfig}
                groups={parameterGroups}
              />
            </FieldGroup>
          </FieldSet>
        </ScrollArea>

        <DialogFooter className="shrink-0 mt-4">
          <Button variant="outline" onClick={() => setIsOpen(false)}>
            Cancel
          </Button>
          <Button onClick={handleCreate}>Create Single Run</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
