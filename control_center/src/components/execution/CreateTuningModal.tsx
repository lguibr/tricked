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
import { FieldGroup } from "@/components/ui/field";
import { ParameterForm, GroupDef } from "./ParameterForm";

interface CreateTuningModalProps {
  isOpen: boolean;
  setIsOpen: (v: boolean) => void;
  runsViewRefresh?: () => void;
}

export function CreateTuningModal({
  isOpen,
  setIsOpen,
  runsViewRefresh,
}: CreateTuningModalProps) {
  const [config, setConfig] = useState<Record<string, any>>({
    trials: 50,
    timeout: 1800,
    maxSteps: 50,
    resnetBlocks: 4,
    resnetChannels: 64,
    num_processes: { min: 1, max: 32 },
    train_batch_size: { min: 64, max: 2048 },
    simulations: { min: 10, max: 1000 },
    max_gumbel_k: { min: 4, max: 32 },
    lr_init: { min: 0.005, max: 0.1 },
  });

  const singleGroups: GroupDef[] = [
    {
      title: "Optuna Global Controls",
      color: "text-zinc-300",
      fields: [
        { key: "trials", label: "Max Trials", min: 10, max: 1000, step: 10 },
        {
          key: "timeout",
          label: "Timeout (Secs)",
          min: 10,
          max: 7200,
          step: 60,
        },
        {
          key: "maxSteps",
          label: "Steps Per Trial",
          min: 1,
          max: 100,
          step: 1,
        },
      ],
    },
    {
      title: "Network Capacity (Fixed)",
      color: "text-zinc-300",
      fields: [
        {
          key: "resnetBlocks",
          label: "ResNet Blocks",
          min: 2,
          max: 30,
          step: 1,
        },
        {
          key: "resnetChannels",
          label: "ResNet Channels",
          min: 32,
          max: 512,
          step: 32,
        },
      ],
    },
  ];

  const boundGroups: GroupDef[] = [
    {
      title: "Hardware Bounds",
      color: "text-emerald-400",
      fields: [
        {
          key: "num_processes",
          label: "Worker Processes",
          min: 1,
          max: 128,
          step: 1,
        },
        {
          key: "train_batch_size",
          label: "Train Batch Size",
          min: 64,
          max: 4096,
          step: 64,
        },
      ],
    },
    {
      title: "MCTS Bounds",
      color: "text-blue-400",
      fields: [
        {
          key: "simulations",
          label: "Simulations",
          min: 10,
          max: 2000,
          step: 10,
        },
        {
          key: "max_gumbel_k",
          label: "Max Gumbel K",
          min: 4,
          max: 64,
          step: 1,
        },
      ],
    },
    {
      title: "Learning Bounds",
      color: "text-purple-400",
      fields: [
        {
          key: "lr_init",
          label: "Learning Rate",
          min: 0.001,
          max: 0.1,
          step: 0.001,
        },
      ],
    },
  ];

  const handleCreate = async () => {
    try {
      const bounds: Record<string, any> = {
        num_processes: config.num_processes,
        train_batch_size: config.train_batch_size,
        simulations: config.simulations,
        max_gumbel_k: config.max_gumbel_k,
        lr_init: config.lr_init,
      };

      await invoke("start_study", {
        trials: config.trials,
        maxSteps: config.maxSteps,
        timeout: config.timeout,
        resnetBlocks: config.resnetBlocks,
        resnetChannels: config.resnetChannels,
        bounds,
      });

      if (runsViewRefresh) runsViewRefresh();
      setIsOpen(false);
    } catch (e) {
      console.error(e);
      alert(e);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogContent className="sm:max-w-[700px] max-h-[90vh] flex flex-col pt-6 pb-4">
        <DialogHeader className="shrink-0">
          <DialogTitle>Configure Holistic Optuna Scan</DialogTitle>
          <DialogDescription>
            Specify the exploration boundaries for the multi-objective unified
            tuning run.
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto px-4 -mx-4 pb-12 pt-2">
          <FieldGroup className="gap-4">
            <ParameterForm
              mode="single"
              value={config}
              onChange={setConfig}
              groups={singleGroups}
            />
            <ParameterForm
              mode="bounds"
              value={config}
              onChange={setConfig}
              groups={boundGroups}
            />
          </FieldGroup>
        </div>

        <DialogFooter className="shrink-0 mt-4 pt-4 border-t border-border/10">
          <Button variant="outline" onClick={() => setIsOpen(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleCreate}
            className="bg-emerald-600 hover:bg-emerald-700 text-white"
          >
            Execute Holistic Scan
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
