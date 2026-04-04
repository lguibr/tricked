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

  const [globalScale, setGlobalScale] = useState([3]);
  const [mctsScale, setMctsScale] = useState([3]);
  const [hwScale, setHwScale] = useState([3]);
  const [learnScale, setLearnScale] = useState([3]);

  // Exact scaler overrides
  const [simulations, setSimulations] = useState([800]);
  const [gumbel, setGumbel] = useState([16]);
  const [lr, setLr] = useState([0.02]);
  const [workers, setWorkers] = useState([8]);
  const [blocks, setBlocks] = useState([10]);
  const [batchSize, setBatchSize] = useState([1024]);

  const SCALE_NAMES: Record<number, string> = { 1: "Micro", 2: "Small", 3: "Base", 4: "Large", 5: "Massive", 6: "Unknown" };
  const MCTS_SIMS: Record<number, number> = { 1: 50, 2: 200, 3: 800, 4: 1200, 5: 2000 };
  const MCTS_GUMBEL: Record<number, number> = { 1: 8, 2: 12, 3: 16, 4: 32, 5: 64 };
  const HW_WORKERS: Record<number, number> = { 1: 1, 2: 4, 3: 8, 4: 16, 5: 64 };
  const HW_BATCH: Record<number, number> = { 1: 64, 2: 256, 3: 1024, 4: 2048, 5: 4096 };
  const LR_VALS: Record<number, number> = { 1: 0.1, 2: 0.05, 3: 0.02, 4: 0.01, 5: 0.005 };
  const BLOCKS: Record<number, number> = { 1: 2, 2: 4, 3: 10, 4: 15, 5: 30 };

  const handleMctsScale = (v: number[]) => {
    setMctsScale(v);
    setSimulations([MCTS_SIMS[v[0]]]);
    setGumbel([MCTS_GUMBEL[v[0]]]);
  };

  const handleHwScale = (v: number[]) => {
    setHwScale(v);
    setWorkers([HW_WORKERS[v[0]]]);
    setBatchSize([HW_BATCH[v[0]]]);
  };

  const handleLearnScale = (v: number[]) => {
    setLearnScale(v);
    setLr([LR_VALS[v[0]]]);
    setBlocks([BLOCKS[v[0]]]);
  };

  const handleGlobalScale = (v: number[]) => {
    setGlobalScale(v);
    handleMctsScale(v);
    handleHwScale(v);
    handleLearnScale(v);
  };

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
              <hr className="border-border/50 my-2" />

              <Field>
                <div className="flex justify-between w-full mb-1">
                  <FieldLabel className="text-emerald-400">Global Architecture Scale</FieldLabel>
                  <span className="text-[10px] font-mono text-emerald-400 font-bold px-2 py-0.5 bg-emerald-950/30 rounded border border-emerald-900/50">
                    {SCALE_NAMES[globalScale[0]] || "Custom"}
                  </span>
                </div>
                <Slider
                  className="py-2"
                  value={globalScale}
                  onValueChange={handleGlobalScale}
                  min={1} max={5} step={1}
                />
              </Field>

              <hr className="border-border/50 mt-4 mb-2" />

              {/* HARDWARE DOMAIN */}
              <div className="flex flex-col gap-3 p-3 bg-zinc-950 rounded-lg border border-border/40">
                <div className="flex flex-col gap-1">
                  <div className="flex justify-between w-full">
                    <span className="text-xs font-bold text-zinc-300 uppercase tracking-wider">Hardware Compute</span>
                    <span className="text-[10px] font-mono text-zinc-500">{SCALE_NAMES[hwScale[0]] || "Custom"}</span>
                  </div>
                  <Slider value={hwScale} onValueChange={handleHwScale} min={1} max={5} step={1} className="py-2 opacity-80" />
                </div>
                <div className="grid grid-cols-2 gap-4 mt-2 p-3 bg-[#0a0a0c] rounded border border-border/20">
                  <Field>
                    <div className="flex justify-between w-full mb-1">
                      <FieldLabel className="text-[10px] text-zinc-400">Worker Threads</FieldLabel>
                      <span className="text-[10px] font-mono text-zinc-500">{workers[0]}</span>
                    </div>
                    <Slider value={workers} onValueChange={(v) => { setWorkers(v); setHwScale([6]); }} min={1} max={128} step={1} className="py-1" />
                  </Field>
                  <Field>
                    <div className="flex justify-between w-full mb-1">
                      <FieldLabel className="text-[10px] text-zinc-400">Batch Size</FieldLabel>
                      <span className="text-[10px] font-mono text-zinc-500">{batchSize[0]}</span>
                    </div>
                    <Slider value={batchSize} onValueChange={(v) => { setBatchSize(v); setHwScale([6]); }} min={64} max={4096} step={64} className="py-1" />
                  </Field>
                </div>
              </div>

              {/* MCTS DOMAIN */}
              <div className="flex flex-col gap-3 p-3 bg-zinc-950 rounded-lg border border-border/40 mt-2">
                <div className="flex flex-col gap-1">
                  <div className="flex justify-between w-full">
                    <span className="text-xs font-bold text-zinc-300 uppercase tracking-wider">MCTS Engine Engine</span>
                    <span className="text-[10px] font-mono text-zinc-500">{SCALE_NAMES[mctsScale[0]] || "Custom"}</span>
                  </div>
                  <Slider value={mctsScale} onValueChange={handleMctsScale} min={1} max={5} step={1} className="py-2 opacity-80" />
                </div>
                <div className="grid grid-cols-2 gap-4 mt-2 p-3 bg-[#0a0a0c] rounded border border-border/20">
                  <Field>
                    <div className="flex justify-between w-full mb-1">
                      <FieldLabel className="text-[10px] text-zinc-400">Simulations</FieldLabel>
                      <span className="text-[10px] font-mono text-zinc-500">{simulations[0]}</span>
                    </div>
                    <Slider value={simulations} onValueChange={(v) => { setSimulations(v); setMctsScale([6]); }} min={10} max={2000} step={10} className="py-1" />
                  </Field>
                  <Field>
                    <div className="flex justify-between w-full mb-1">
                      <FieldLabel className="text-[10px] text-zinc-400">Gumbel K</FieldLabel>
                      <span className="text-[10px] font-mono text-zinc-500">{gumbel[0]}</span>
                    </div>
                    <Slider value={gumbel} onValueChange={(v) => { setGumbel(v); setMctsScale([6]); }} min={4} max={64} step={1} className="py-1" />
                  </Field>
                </div>
              </div>

              {/* LEARNING DOMAIN */}
              <div className="flex flex-col gap-3 p-3 bg-zinc-950 rounded-lg border border-border/40 mt-2">
                <div className="flex flex-col gap-1">
                  <div className="flex justify-between w-full">
                    <span className="text-xs font-bold text-zinc-300 uppercase tracking-wider">Learning Architecture</span>
                    <span className="text-[10px] font-mono text-zinc-500">{SCALE_NAMES[learnScale[0]] || "Custom"}</span>
                  </div>
                  <Slider value={learnScale} onValueChange={handleLearnScale} min={1} max={5} step={1} className="py-2 opacity-80" />
                </div>
                <div className="grid grid-cols-2 gap-4 mt-2 p-3 bg-[#0a0a0c] rounded border border-border/20">
                  <Field>
                    <div className="flex justify-between w-full mb-1">
                      <FieldLabel className="text-[10px] text-zinc-400">Learning Rate</FieldLabel>
                      <span className="text-[10px] font-mono text-zinc-500">{lr[0]}</span>
                    </div>
                    <Slider value={lr} onValueChange={(v) => { setLr(v); setLearnScale([6]); }} min={0.001} max={0.1} step={0.001} className="py-1" />
                  </Field>
                  <Field>
                    <div className="flex justify-between w-full mb-1">
                      <FieldLabel className="text-[10px] text-zinc-400">ResNet Blocks</FieldLabel>
                      <span className="text-[10px] font-mono text-zinc-500">{blocks[0]}</span>
                    </div>
                    <Slider value={blocks} onValueChange={(v) => { setBlocks(v); setLearnScale([6]); }} min={2} max={30} step={1} className="py-1" />
                  </Field>
                </div>
              </div>
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
