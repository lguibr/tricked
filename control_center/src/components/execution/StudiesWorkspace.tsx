import { useState, useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Brain,
  Play,
  Square,
  Loader2,
  Trash2,
  CheckCircle2,
  Sliders,
  Network,
  Cpu,
  GitBranch,
  Activity,
} from "lucide-react";
import { LiveLogsViewer } from "./LiveLogsViewer";
import { OptunaStudyDashboard } from "@/components/OptunaStudyDashboard";
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from "@/components/ui/resizable";
import { ParameterForm, GroupDef } from "./ParameterForm";

interface StudiesWorkspaceProps {
  runLogs: Record<string, string[]>;
}

export function StudiesWorkspace({ runLogs }: StudiesWorkspaceProps) {
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
  });

  const singleGroups: GroupDef[] = [
    {
      title: "Optuna Global Controls",
      color: "text-zinc-300",
      icon: Sliders,
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
          label: "Timeout (Secs)",
          min: 10,
          max: 7200,
          step: 60,
          tooltip:
            "Maximum time permitted for the study before early stopping.",
        },
        {
          key: "maxSteps",
          label: "Steps Per Trial",
          min: 1,
          max: 100,
          step: 1,
          tooltip: "Maximum steps optimized in each individual trial.",
        },
      ],
    },
    {
      title: "Network Capacity",
      color: "text-zinc-300",
      icon: Network,
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
      title: "Hardware Bounds",
      color: "text-emerald-400",
      icon: Cpu,
      fields: [
        {
          key: "num_processes",
          label: "Worker Processes",
          min: 1,
          max: 128,
          step: 1,
          tooltip: "Range for exploring data generation concurrency.",
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
      title: "MCTS Bounds",
      color: "text-blue-400",
      icon: GitBranch,
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
      title: "Learning Bounds",
      color: "text-purple-400",
      icon: Activity,
      fields: [
        {
          key: "lr_init",
          label: "Learning Rate",
          min: 0.001,
          max: 0.1,
          step: 0.001,
          tooltip: "Optuna will search this learning rate space.",
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
    <div className="flex h-full w-full bg-[#050505] text-zinc-200">
      {/* Configuration Column */}
      <div className="w-[400px] border-r border-border/20 p-6 flex flex-col gap-6 overflow-y-auto shrink-0 bg-[#0a0a0a] custom-scrollbar">
        <div className="flex flex-col gap-1 shrink-0">
          <h2 className="text-xl font-black tracking-widest uppercase text-zinc-100">
            Tuning Lab
          </h2>
          <span className="text-[10px] text-zinc-500 uppercase tracking-widest font-semibold flex items-center justify-between">
            Optuna Hub Integrations
          </span>
        </div>

        <div className="flex flex-col gap-4">
          <Card
            className={`bg-[#101010] border p-5 flex flex-col gap-4 shrink-0 transition-colors ${tuneComplete ? "border-emerald-500/50" : "border-border/10"}`}
          >
            <div className="flex items-center justify-between text-emerald-400">
              <div className="flex items-center gap-3">
                <Brain className="w-5 h-5" />
                <h3 className="font-bold text-sm uppercase tracking-wider">
                  Holistic Tuning
                </h3>
              </div>
              {tuneComplete && (
                <CheckCircle2 className="w-4 h-4 text-emerald-500" />
              )}
            </div>

            <p className="text-xs text-zinc-400 leading-relaxed pb-2 border-b border-border/10">
              Multi-objective optimization mapping hardware throughput limits
              and training convergence potential simultaneously.
            </p>

            <div className="flex flex-col gap-4">
              {!isActive && (
                <div className="flex flex-col gap-4 pt-1">
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

              <div className="flex gap-2">
                <Button
                  disabled={isActive}
                  onClick={handleStartScan}
                  className="flex-1 bg-emerald-600/10 text-emerald-400 hover:bg-emerald-600/30 border border-emerald-500/20 shadow-none text-xs"
                >
                  {isActive ? (
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <Play className="w-3 h-3 mr-2" />
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
                    className="bg-transparent border-red-500/20 text-red-500 hover:bg-red-500/10"
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                )}
              </div>
            </div>
            {isActive && (
              <Button
                onClick={handleStop}
                variant="destructive"
                className="w-full shrink-0 text-xs mt-2"
              >
                <Square className="w-3 h-3 mr-2" /> Kill Active Process
              </Button>
            )}
          </Card>
        </div>
      </div>

      {/* Dashboard & Terminal View */}
      <div className="flex-1 flex flex-col bg-[#030303] overflow-hidden">
        <ResizablePanelGroup direction="vertical">
          {/* Top: Optuna Dashboard */}
          <ResizablePanel defaultSize={70} minSize={30}>
            <OptunaStudyDashboard />
          </ResizablePanel>

          <ResizableHandle className="h-1 bg-border/20 hover:bg-primary/50 transition-colors z-50 cursor-row-resize" />

          {/* Bottom: Terminal */}
          <ResizablePanel defaultSize={30} minSize={10}>
            <div className="flex flex-col h-full">
              <div className="h-8 px-4 flex items-center border-b border-border/20 bg-zinc-950/80 shrink-0">
                <span className="text-[10px] font-bold uppercase tracking-widest text-zinc-500">
                  Live Process Diagnostics
                </span>
              </div>
              <div className="flex-1 relative overflow-hidden bg-black">
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
