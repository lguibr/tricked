import { useState, useEffect } from "react";
import { VscPlay, VscSync, VscSettingsGear, VscClose } from "react-icons/vsc";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ParameterForm } from "@/components/execution/ParameterForm";
import { useTuningStore } from "@/store/useTuningStore";
import { useAppStore } from "@/store/useAppStore";
import { applyPresetToGroup } from "./TuningPresets";
import { getSingleGroups, getBoundGroups } from "./TuningGroups";

export function CreateStudySidebar({ onClose }: { onClose: () => void }) {
  const config = useTuningStore((state) => state.config);
  const setConfig = useTuningStore((state) => state.setConfig);
  const isActive = useTuningStore((state) => state.isActive);
  const tuneComplete = useTuningStore((state) => state.tuneComplete);
  const startScan = useTuningStore((state) => state.startScan);
  const studyName = useTuningStore((state: any) => state.studyName);
  const setStudyName = useTuningStore((state: any) => state.setStudyName);
  const initialRefineConfig = useTuningStore(
    (state) => state.initialRefineConfig,
  );
  const setInitialRefineConfig = useTuningStore(
    (state) => state.setInitialRefineConfig,
  );

  const [presetLevel, setPresetLevel] = useState(3);
  const [groupPresets, setGroupPresets] = useState<number[]>([3, 3, 3, 3, 3]);

  // If we have a refine config, set the bounds around it tightly on mount
  useEffect(() => {
    if (initialRefineConfig) {
      const tightBounds = { ...config };
      const setTightBound = (key: string, val: any) => {
        if (typeof val === "number") {
          const spread = Math.max(val * 0.1, 1);
          tightBounds[key] = { min: val - spread, max: val + spread };
        }
      };
      // For bounds
      setTightBound("simulations", initialRefineConfig.simulations);
      setTightBound("max_gumbel_k", initialRefineConfig.max_gumbel_k);
      setTightBound("train_batch_size", initialRefineConfig.train_batch_size);
      setTightBound("num_processes", initialRefineConfig.num_processes);
      if (initialRefineConfig.lr_init) {
        tightBounds.lr_init = {
          min: initialRefineConfig.lr_init * 0.5,
          max: initialRefineConfig.lr_init * 1.5,
        };
      }
      setConfig(tightBounds);
      setInitialRefineConfig(null);
    }
  }, [initialRefineConfig]);

  const handleGroupPresetChange = (groupIndex: number, level: number) => {
    const prev = [...groupPresets];
    prev[groupIndex] = level;
    setGroupPresets(prev);

    const newConfig = { ...config };
    applyPresetToGroup(newConfig, groupIndex, level);
    setConfig(newConfig);

    if (prev.every((p) => p === level)) {
      setPresetLevel(level);
    } else {
      setPresetLevel(0);
    }
  };

  const handleGlobalPresetChange = (level: number) => {
    setPresetLevel(level);
    setGroupPresets([level, level, level, level, level]);
    const newConfig = { ...config };
    for (let i = 0; i < 5; i++) {
      applyPresetToGroup(newConfig, i, level);
    }
    setConfig(newConfig);
  };

  const singleGroups = getSingleGroups(groupPresets);
  const boundGroups = getBoundGroups(groupPresets);

  return (
    <div className="flex flex-col h-full bg-[#0a0a0a]">
      <div className="px-3 py-2 border-b border-border/10 flex justify-between items-center bg-black/20 shrink-0">
        <h3 className="text-xs font-bold text-zinc-100 uppercase tracking-widest flex items-center gap-1.5">
          <VscSettingsGear className="text-emerald-500" />
          New Tuning Study
        </h3>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6 text-zinc-400 hover:text-white hover:bg-white/10"
          onClick={onClose}
        >
          <VscClose size={14} />
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
        <div className="flex flex-col gap-3">
          <Card
            className={`bg-[#080808] border p-3 flex flex-col gap-3 shrink-0 transition-colors shadow-inner ${tuneComplete ? "border-emerald-500/30" : "border-white/5"}`}
          >
            <div className="p-4 border-b border-border/5 shrink-0 bg-[#080808]">
              <div className="flex items-center gap-2 mb-1">
                <VscSettingsGear className="w-4 h-4 text-purple-400" />
                <h2 className="text-[11px] font-black uppercase tracking-widest text-zinc-100">
                  Optimizer Study Config
                </h2>
              </div>
              <p className="text-[10px] text-zinc-500 mb-4">
                Adjust hyperparameter search bounds. Save & initialize study
                parameters directly from this tab.
              </p>

              <div className="flex flex-col gap-2 mt-4">
                <label className="text-[9px] font-bold uppercase tracking-widest text-zinc-400 mb-1">
                  Study Name / Identifier
                </label>
                <input
                  type="text"
                  className="w-full bg-[#111] border border-white/10 rounded px-2 py-1 text-[11px] font-mono text-zinc-200 outline-none focus:border-purple-500/50"
                  placeholder="tuning_study_XYZ"
                  value={studyName || ""}
                  onChange={(e) => setStudyName(e.target.value)}
                />
              </div>
            </div>

            <p className="text-[9px] text-zinc-400 leading-tight uppercase font-mono tracking-tight pb-1">
              Multi-objective optimization mapping hardware throughput limits
              and training convergence potential simultaneously.
            </p>

            <div className="flex flex-col gap-3">
              {!isActive && (
                <div className="flex flex-col gap-2 p-3 bg-zinc-900/50 border border-border/20 rounded-md">
                  <div className="flex justify-between items-center">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-400">
                      Global Bounds Preset
                    </span>
                    <span className="text-[10px] font-mono text-zinc-500">
                      Level{" "}
                      {presetLevel === 0 ? "Custom" : `${presetLevel} / 5`}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="5"
                    step="1"
                    value={presetLevel || 3}
                    onChange={(e) =>
                      handleGlobalPresetChange(parseInt(e.target.value))
                    }
                    className="w-full accent-emerald-500"
                  />
                </div>
              )}

              {!isActive && (
                <div className="flex flex-col gap-3 pt-1">
                  <ParameterForm
                    mode="single"
                    value={config}
                    onChange={setConfig}
                    groups={singleGroups}
                    onGroupPresetChange={(idx, level) => {
                      if (idx === 1) handleGroupPresetChange(0, level);
                    }}
                  />
                  <ParameterForm
                    mode="bounds"
                    value={config}
                    onChange={setConfig}
                    groups={boundGroups}
                    onGroupPresetChange={(idx, level) => {
                      handleGroupPresetChange(idx + 1, level);
                    }}
                  />
                </div>
              )}

              <div className="flex gap-1.5 mt-2">
                <Button
                  disabled={isActive}
                  onClick={async () => {
                    await startScan();
                    useAppStore.getState().loadRuns();
                    onClose();
                  }}
                  className="flex-1 bg-emerald-600/20 text-emerald-400 hover:bg-emerald-600/40 border border-emerald-500/40 shadow-none text-[9px] h-7 font-black tracking-widest uppercase"
                >
                  {isActive ? (
                    <VscSync className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                  ) : (
                    <VscPlay className="w-3.5 h-3.5 mr-1.5" />
                  )}
                  {isActive ? "Starting..." : "Initialize Search"}
                </Button>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
