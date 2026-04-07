import { useState, useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  VscLightbulb,
  VscPlay,
  VscDebugStop,
  VscSync,
  VscTrash,
  VscPassFilled,
  VscSettingsGear,
  VscTypeHierarchy,
  VscServerProcess,
  VscRepoForked,
  VscPulse,
} from "react-icons/vsc";
import { LiveLogsViewer } from "./LiveLogsViewer";
import { OptunaStudyDashboard } from "@/components/OptunaStudyDashboard";
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from "@/components/ui/resizable";
import { ParameterForm, GroupDef } from "./ParameterForm";

import { useAppStore } from "@/store/useAppStore";

export function StudiesWorkspace() {
  const runLogs = useAppStore((state) => state.runLogs);
  const [isActive, setIsActive] = useState(false);
  const [tuneComplete, setTuneComplete] = useState(false);

  const logsRef = useRef<Record<string, HTMLDivElement | null>>({});

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
    discount_factor: { min: 0.9, max: 0.999 },
    td_lambda: { min: 0.5, max: 1.0 },
    weight_decay: { min: 0.0, max: 0.1 },
  });

  const singleGroups: GroupDef[] = [
    {
      title: "Optuna Global Controls",
      color: "text-zinc-300",
      icon: VscSettingsGear,
      fields: [
        {
          key: "trials",
          label: "Max Trials",
          min: 10,
          max: 1000,
          step: 10,
          tooltip: "Maximum number of tuning trials to perform.",
        },
        {
          key: "timeout",
          label: "Timeout (s)",
          min: 10,
          max: 7200,
          step: 60,
          tooltip:
            "Maximum time permitted for the study before early stopping.",
        },
        {
          key: "maxSteps",
          label: "Steps/Trial",
          min: 1,
          max: 100,
          step: 1,
          tooltip: "Maximum steps optimized in each individual trial.",
        },
      ],
    },
    {
      title: "1. Neural Architecture & Topology",
      color: "text-purple-400",
      icon: VscTypeHierarchy,
      fields: [
        {
          key: "resnetBlocks",
          label: "ResNet Blocks",
          min: 2,
          max: 30,
          step: 1,
          tooltip:
            "Number of residual blocks spanning the deep neural network.",
        },
        {
          key: "resnetChannels",
          label: "ResNet Channels",
          min: 32,
          max: 512,
          step: 32,
          tooltip: "Number of hidden dimension channels defining model width.",
        },
      ],
    },
  ];

  const boundGroups: GroupDef[] = [
    {
      title: "2. MDP & Value Estimation",
      color: "text-emerald-400",
      icon: VscLightbulb,
      fields: [
        {
          key: "discount_factor",
          label: "Discount Range",
          min: 0.9,
          max: 0.999,
          step: 0.001,
          tooltip: "Optuna will search this discount factor space.",
        },
        {
          key: "td_lambda",
          label: "TD Lambda Range",
          min: 0.5,
          max: 1.0,
          step: 0.01,
          tooltip: "Optuna will search this TD lambda space.",
        },
      ],
    },
    {
      title: "3. Search Dynamics",
      color: "text-blue-400",
      icon: VscRepoForked,
      fields: [
        {
          key: "simulations",
          label: "Simulations",
          min: 10,
          max: 2000,
          step: 10,
          tooltip: "Tree search explorations.",
        },
        {
          key: "max_gumbel_k",
          label: "Max Gumbel K",
          min: 4,
          max: 64,
          step: 1,
          tooltip: "Action subset size for Gumbel sampling.",
        },
      ],
    },
    {
      title: "4. Optimization & Gradient",
      color: "text-red-400",
      icon: VscPulse,
      fields: [
        {
          key: "lr_init",
          label: "Learning Rate",
          min: 0.0001,
          max: 0.1,
          step: 0.0001,
          tooltip: "Optuna will search this learning rate space.",
        },
        {
          key: "weight_decay",
          label: "Weight Decay Range",
          min: 0.0,
          max: 0.1,
          step: 0.0001,
          tooltip: "Optuna will search this L2 regularization space.",
        },
        {
          key: "train_batch_size",
          label: "Train Batch Size",
          min: 64,
          max: 4096,
          step: 64,
          tooltip:
            "Range for learning batch sizes sent through backpropagation.",
        },
      ],
    },
    {
      title: "5. Systems Concurrency",
      color: "text-amber-400",
      icon: VscServerProcess,
      fields: [
        {
          key: "num_processes",
          label: "Worker Processes",
          min: 1,
          max: 128,
          step: 1,
          tooltip: "Range for exploring data generation concurrency.",
        },
      ],
    },
  ];

  const refreshStatus = async () => {
    try {
      setTuneComplete(
        await invoke<boolean>("get_study_status", { studyType: "UNIFIED" }),
      );
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    const checkActive = async () => {
      try {
        const active = await invoke<boolean>("get_active_study");
        setIsActive(active);
        refreshStatus();
      } catch (e) {
        console.error(e);
      }
    };
    checkActive();
    const int = setInterval(checkActive, 2000);
    return () => clearInterval(int);
  }, []);

  const handleStartScan = async () => {
    try {
      const bounds: Record<string, any> = {
        num_processes: config.num_processes,
        train_batch_size: config.train_batch_size,
        simulations: config.simulations,
        max_gumbel_k: config.max_gumbel_k,
        lr_init: config.lr_init,
        discount_factor: config.discount_factor,
        td_lambda: config.td_lambda,
        weight_decay: config.weight_decay,
      };

      await invoke("start_study", {
        trials: config.trials,
        maxSteps: config.maxSteps,
        timeout: config.timeout,
        resnetBlocks: config.resnetBlocks,
        resnetChannels: config.resnetChannels,
        bounds,
      });

      refreshStatus();
      setIsActive(true);
    } catch (e) {
      console.error(e);
      alert(e);
    }
  };

  const handleStop = async () => {
    try {
      await invoke("stop_study", { force: false });
      setIsActive(false);
    } catch (e) {
      console.error(e);
    }
  };

  const handleFlush = async () => {
    if (!confirm(`Are you sure you want to wipe the global study database?`))
      return;
    try {
      await invoke("flush_study", { studyType: "UNIFIED" });
      await refreshStatus();
    } catch (e) {
      console.error(e);
      alert(e);
    }
  };

  return (
    <div className="flex h-full w-full bg-[#020202] text-zinc-200">
      {/* Configuration Column */}
      <div className="w-[340px] border-r border-white/5 p-3 flex flex-col gap-4 overflow-y-auto shrink-0 bg-[#050505] custom-scrollbar shadow-xl z-10">
        <div className="flex flex-col gap-0.5 shrink-0 px-1 hover:bg-white/[0.02] rounded pb-2 border-b border-white/5">
          <h2 className="text-sm font-black tracking-widest uppercase text-zinc-100 flex items-center gap-1.5">
            <VscSettingsGear className="text-emerald-500" />
            Tuning Lab
          </h2>
          <span className="text-[9px] text-zinc-500 uppercase tracking-widest font-black flex items-center justify-between">
            Optuna Hub Integrations
          </span>
        </div>

        <div className="flex flex-col gap-3 flex-1">
          <Card
            className={`bg-[#080808] border p-3 flex flex-col gap-3 shrink-0 transition-colors shadow-inner ${tuneComplete ? "border-emerald-500/30" : "border-white/5"}`}
          >
            <div className="flex items-center justify-between text-emerald-400 border-b border-white/5 pb-2">
              <div className="flex items-center gap-1.5">
                <VscLightbulb className="w-4 h-4" />
                <h3 className="font-bold text-[10px] uppercase tracking-widest">
                  Holistic Tuning
                </h3>
              </div>
              {tuneComplete && (
                <VscPassFilled className="w-4 h-4 text-emerald-500" />
              )}
            </div>

            <p className="text-[9px] text-zinc-400 leading-tight uppercase font-mono tracking-tight pb-1">
              Multi-objective optimization mapping hardware throughput limits
              and training convergence potential simultaneously.
            </p>

            <div className="flex flex-col gap-3">
              {!isActive && (
                <div className="flex flex-col gap-3 pt-1">
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
                </div>
              )}

              <div className="flex gap-1.5">
                <Button
                  disabled={isActive}
                  onClick={handleStartScan}
                  className="flex-1 bg-emerald-600/20 text-emerald-400 hover:bg-emerald-600/40 border border-emerald-500/40 shadow-none text-[9px] h-7 font-black tracking-widest uppercase"
                >
                  {isActive ? (
                    <VscSync className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                  ) : (
                    <VscPlay className="w-3.5 h-3.5 mr-1.5" />
                  )}
                  {isActive
                    ? "Scan Running..."
                    : tuneComplete
                      ? "Restart Scan"
                      : "Start Scan"}
                </Button>
                {tuneComplete && !isActive && (
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => handleFlush()}
                    className="h-7 w-7 bg-transparent border-red-500/30 text-red-500 hover:bg-red-500/20"
                  >
                    <VscTrash className="w-3.5 h-3.5" />
                  </Button>
                )}
              </div>
            </div>
            {isActive && (
              <Button
                onClick={handleStop}
                variant="destructive"
                className="w-full shrink-0 text-[9px] h-7 font-black tracking-widest uppercase mt-1"
              >
                <VscDebugStop className="w-3.5 h-3.5 mr-1.5" /> Kill Active
                Process
              </Button>
            )}
          </Card>
        </div>
      </div>

      {/* Dashboard & Terminal View */}
      <div className="flex-1 flex flex-col bg-[#020202] overflow-hidden">
        <ResizablePanelGroup direction="vertical">
          {/* Top: Optuna Dashboard */}
          <ResizablePanel defaultSize={70} minSize={30}>
            <OptunaStudyDashboard />
          </ResizablePanel>

          <ResizableHandle className="h-0.5 bg-white/10 hover:bg-emerald-500/50 transition-colors z-50 cursor-row-resize shadow-[0_0_5px_rgba(255,255,255,0.1)]" />

          {/* Bottom: Terminal */}
          <ResizablePanel defaultSize={30} minSize={10}>
            <div className="flex flex-col h-full">
              <div className="h-6 px-3 flex items-center border-b border-white/5 bg-[#050505] shrink-0">
                <span className="text-[8.5px] font-black uppercase tracking-widest text-zinc-500 flex items-center gap-1.5">
                  <VscServerProcess className="text-zinc-600" />
                  Live Process Diagnostics
                </span>
              </div>
              <div className="flex-1 relative overflow-hidden bg-[#020202]">
                <LiveLogsViewer
                  runs={[
                    {
                      id: "STUDY",
                      name: "Tuning Engine",
                      type: "STUDY",
                      status: isActive ? "RUNNING" : "STOPPED",
                      config: "{}",
                    },
                  ]}
                  runLogs={runLogs}
                  selectedLogRunIds={["STUDY"]}
                  toggleLogRun={() => {}}
                  handleCopyLogs={(_id, logs) =>
                    navigator.clipboard.writeText(logs)
                  }
                  copiedLogId={null}
                  logsEndRef={logsRef}
                  runColors={{ STUDY: "#10b981" }}
                />
              </div>
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>
    </div>
  );
}
