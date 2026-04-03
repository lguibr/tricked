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
import {
  Field,
  FieldLabel,
  FieldSet,
  FieldGroup,
  FieldDescription,
} from "@/components/ui/field";
import { Slider } from "@/components/ui/slider";
import { ScrollArea } from "@/components/ui/scroll-area";

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
  const [preset, setPreset] = useState("default");

  // Exact scaler overrides
  const [simulations, setSimulations] = useState([800]);
  const [gumbel, setGumbel] = useState([16]);
  const [lr, setLr] = useState([0.02]);
  const [workers, setWorkers] = useState([8]);
  const [blocks, setBlocks] = useState([10]);
  const [batchSize, setBatchSize] = useState([1024]);

  const handleCreate = async () => {
    if (!name.trim()) return;
    try {
      const createdRun = await invoke<Run>("create_run", {
        name,
        type: "SINGLE",
        preset,
      });

      try {
        const baseConfig = JSON.parse(createdRun.config || "{}");
        baseConfig["simulations"] = simulations[0];
        baseConfig["max_gumbel_k"] = gumbel[0];
        baseConfig["lr_init"] = lr[0];
        baseConfig["num_processes"] = workers[0];
        baseConfig["num_blocks"] = blocks[0];
        baseConfig["train_batch_size"] = batchSize[0];

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
              <Field>
                <FieldLabel htmlFor="base-config">
                  Base Config / Hydra Payload
                </FieldLabel>
                <select
                  id="base-config"
                  value={preset}
                  onChange={(e) => setPreset(e.target.value)}
                  className="flex h-9 w-full items-center justify-between rounded-md border border-input bg-zinc-950 px-3 py-1 text-sm shadow-sm ring-offset-background disabled:cursor-not-allowed disabled:opacity-50 text-zinc-100 outline-none focus:ring-1 focus:ring-primary"
                >
                  <option value="default" className="bg-zinc-900 text-zinc-100">
                    Default Core Settings
                  </option>
                  <option value="small" className="bg-zinc-900 text-zinc-100">
                    Small Network Profiling Check
                  </option>
                  <option value="big" className="bg-zinc-900 text-zinc-100">
                    Big ResNet SOTA Config
                  </option>
                </select>
                <FieldDescription>
                  Select a config schema to bootstrap parameters from.
                </FieldDescription>
              </Field>

              <hr className="border-border/50 my-2" />

              <Field>
                <div className="flex justify-between w-full mb-1">
                  <FieldLabel>MCTS Simulations</FieldLabel>
                  <span className="text-[10px] font-mono text-muted-foreground">
                    {simulations[0]}
                  </span>
                </div>
                <Slider
                  className="py-2"
                  value={simulations}
                  onValueChange={setSimulations}
                  min={10}
                  max={2000}
                  step={10}
                />
              </Field>

              <Field>
                <div className="flex justify-between w-full mb-1">
                  <FieldLabel>Max Gumbel K</FieldLabel>
                  <span className="text-[10px] font-mono text-muted-foreground">
                    {gumbel[0]}
                  </span>
                </div>
                <Slider
                  className="py-2"
                  value={gumbel}
                  onValueChange={setGumbel}
                  min={4}
                  max={64}
                  step={1}
                />
              </Field>

              <Field>
                <div className="flex justify-between w-full mb-1">
                  <FieldLabel>Learning Rate (lr_init)</FieldLabel>
                  <span className="text-[10px] font-mono text-muted-foreground">
                    {lr[0]}
                  </span>
                </div>
                <Slider
                  className="py-2"
                  value={lr}
                  onValueChange={setLr}
                  min={0.0001}
                  max={0.1}
                  step={0.0001}
                />
              </Field>

              <Field>
                <div className="flex justify-between w-full mb-1">
                  <FieldLabel>Train Batch Size</FieldLabel>
                  <span className="text-[10px] font-mono text-muted-foreground">
                    {batchSize[0]}
                  </span>
                </div>
                <Slider
                  className="py-2"
                  value={batchSize}
                  onValueChange={setBatchSize}
                  min={64}
                  max={4096}
                  step={64}
                />
              </Field>

              <Field>
                <div className="flex justify-between w-full mb-1">
                  <FieldLabel>Worker Threads</FieldLabel>
                  <span className="text-[10px] font-mono text-muted-foreground">
                    {workers[0]}
                  </span>
                </div>
                <Slider
                  className="py-2"
                  value={workers}
                  onValueChange={setWorkers}
                  min={1}
                  max={128}
                  step={1}
                />
              </Field>

              <Field>
                <div className="flex justify-between w-full mb-1">
                  <FieldLabel>Neural Net Blocks</FieldLabel>
                  <span className="text-[10px] font-mono text-muted-foreground">
                    {blocks[0]}
                  </span>
                </div>
                <Slider
                  className="py-2"
                  value={blocks}
                  onValueChange={setBlocks}
                  min={2}
                  max={30}
                  step={1}
                />
              </Field>
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
