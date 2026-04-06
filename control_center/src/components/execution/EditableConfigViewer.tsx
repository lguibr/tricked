import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Button } from "@/components/ui/button";
import { ParameterForm, GroupDef } from "./ParameterForm";
import { Save, Loader2 } from "lucide-react";
import type { Run } from "@/bindings/Run";

export const ENGINE_PARAM_GROUPS: GroupDef[] = [
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
      {
        key: "inference_batch_size_limit",
        label: "Inference Limit",
        min: 16,
        max: 1024,
        step: 16,
      },
      {
        key: "checkpoint_interval",
        label: "Checkpoint Interval",
        min: 10,
        max: 1000,
        step: 10,
      },
    ],
  },
  {
    title: "MCTS Engine",
    color: "text-blue-400",
    fields: [
      {
        key: "simulations",
        label: "Simulations",
        min: 10,
        max: 5000,
        step: 10,
      },
      { key: "max_gumbel_k", label: "Max Gumbel K", min: 4, max: 64, step: 1 },
      {
        key: "inference_timeout_ms",
        label: "Inference Timeout (ms)",
        min: 10,
        max: 1000,
        step: 10,
      },
    ],
  },
  {
    title: "Learning Architecture",
    color: "text-emerald-400",
    fields: [
      {
        key: "lr_init",
        label: "Learning Rate",
        min: 0.001,
        max: 0.1,
        step: 0.001,
      },
      {
        key: "discount_factor",
        label: "Discount Factor",
        min: 0.9,
        max: 0.999,
        step: 0.001,
      },
      {
        key: "td_lambda",
        label: "TD Lambda",
        min: 0.5,
        max: 1.0,
        step: 0.01,
      },
      {
        key: "weight_decay",
        label: "Weight Decay",
        min: 0.0,
        max: 0.1,
        step: 0.0001,
      },
      { key: "unroll_steps", label: "Unroll Steps", min: 1, max: 15, step: 1 },
      {
        key: "temporal_difference_steps",
        label: "TD Steps",
        min: 1,
        max: 15,
        step: 1,
      },
      {
        key: "buffer_capacity_limit",
        label: "Buffer Capacity",
        min: 10000,
        max: 1000000,
        step: 10000,
      },
    ],
  },
  {
    title: "Network Capacity",
    color: "text-purple-400",
    fields: [
      { key: "num_blocks", label: "ResNet Blocks", min: 2, max: 30, step: 1 },
      {
        key: "hidden_dimension_size",
        label: "ResNet Channels",
        min: 32,
        max: 512,
        step: 32,
      },
      {
        key: "support_size",
        label: "Value Support Size",
        min: 10,
        max: 1000,
        step: 10,
      },
    ],
  },
];

export function EditableConfigViewer({ run }: { run: Run }) {
  const [config, setConfig] = useState<Record<string, any>>({});
  const [isSaving, setIsSaving] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    try {
      setConfig(JSON.parse(run.config));
      setHasChanges(false);
    } catch (e) {
      console.error("Failed to parse run config", e);
    }
  }, [run.config]);

  const handleChange = (newConfig: Record<string, any>) => {
    setConfig(newConfig);
    setHasChanges(true);
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await invoke("save_config", {
        id: run.id,
        config: JSON.stringify(config, null, 2),
      });
      setHasChanges(false);
    } catch (e) {
      console.error(e);
      alert(e);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="relative group/config mt-1 bg-[#0a0a0c]/50 border border-border/20 rounded-md p-3 flex flex-col gap-3 max-h-[400px] overflow-y-auto">
      <div className="flex items-center justify-between">
        <span className="text-xs font-bold uppercase tracking-widest text-[#f59e0b]">
          Edit Configuration
        </span>
        {hasChanges && (
          <Button
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              handleSave();
            }}
            disabled={isSaving}
            className="h-6 text-[10px] bg-[#f59e0b] hover:bg-[#d97706] text-black px-2 shadow-md shadow-[#f59e0b]/20"
          >
            {isSaving ? (
              <Loader2 className="w-3 h-3 mr-1 animate-spin" />
            ) : (
              <Save className="w-3 h-3 mr-1" />
            )}
            Save Changes
          </Button>
        )}
      </div>

      <ParameterForm
        mode="single"
        value={config}
        onChange={handleChange}
        groups={ENGINE_PARAM_GROUPS}
      />
    </div>
  );
}
