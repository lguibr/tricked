import { useState, useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Field, FieldLabel } from "@/components/ui/field";
import {
  Cpu,
  BrainCircuit,
  Play,
  Square,
  Loader2,
  Trash2,
  CheckCircle2,
  Workflow,
} from "lucide-react";
import { LiveLogsViewer } from "./LiveLogsViewer";
import { OptunaStudyDashboard } from "@/components/OptunaStudyDashboard";
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from "@/components/ui/resizable";

const usePersistedState = (key: string, defaultValue: number) => {
  const [state, setState] = useState(() => {
    const saved = localStorage.getItem(key);
    return saved ? parseInt(saved, 10) : defaultValue;
  });
  useEffect(() => {
    localStorage.setItem(key, state.toString());
  }, [key, state]);
  return [state, setState] as const;
};

interface StudiesWorkspaceProps {
  runLogs: Record<string, string[]>;
}

export function StudiesWorkspace({ runLogs }: StudiesWorkspaceProps) {
  const [isActive, setIsActive] = useState(false);
  const [trials, setTrials] = usePersistedState("optuna_trials", 30);
  const [maxSteps, setMaxSteps] = usePersistedState("optuna_max_steps", 15);
  const [timeoutVal, setTimeoutVal] = usePersistedState("optuna_timeout", 400);

  const [hwComplete, setHwComplete] = useState(false);
  const [mctsComplete, setMctsComplete] = useState(false);
  const [lrComplete, setLrComplete] = useState(false);

  const logsRef = useRef<Record<string, HTMLDivElement | null>>({});

  const refreshStatus = async () => {
    try {
      setHwComplete(
        await invoke<boolean>("get_study_status", { studyType: "HARDWARE" }),
      );
      setMctsComplete(
        await invoke<boolean>("get_study_status", { studyType: "MCTS" }),
      );
      setLrComplete(
        await invoke<boolean>("get_study_status", { studyType: "LEARNING" }),
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

  const handleStart = async (type: "HARDWARE" | "MCTS" | "LEARNING") => {
    try {
      await invoke("start_study", {
        studyType: type,
        trials,
        maxSteps,
        timeout: timeoutVal,
      });
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

  const handleFlush = async (type: "HARDWARE" | "MCTS" | "LEARNING") => {
    if (!confirm(`Are you sure you want to wipe the ${type} study database?`))
      return;
    try {
      await invoke("flush_study", { studyType: type });
      await refreshStatus();
    } catch (e) {
      console.error(e);
      alert(e);
    }
  };

  return (
    <div className="flex h-full w-full bg-[#050505] text-zinc-200">
      {/* Configuration Column */}
      <div className="w-[350px] border-r border-border/20 p-6 flex flex-col gap-5 overflow-y-auto shrink-0 bg-[#0a0a0a] custom-scrollbar">
        <div className="flex flex-col gap-1 shrink-0">
          <h2 className="text-xl font-black tracking-widest uppercase text-zinc-100">
            Tuning Lab
          </h2>
          <span className="text-[10px] text-zinc-500 uppercase tracking-widest font-semibold flex items-center justify-between">
            Optuna Hub Integrations
          </span>
        </div>

        <Card
          className={`bg-[#101010] border p-5 flex flex-col gap-4 shrink-0 transition-colors ${hwComplete ? "border-emerald-500/50" : "border-border/10"}`}
        >
          <div className="flex items-center justify-between text-emerald-400">
            <div className="flex items-center gap-3">
              <Cpu className="w-5 h-5" />
              <h3 className="font-bold text-sm uppercase tracking-wider">
                Hardware Limits
              </h3>
            </div>
            {hwComplete && (
              <CheckCircle2 className="w-4 h-4 text-emerald-500" />
            )}
          </div>
          <p className="text-xs text-zinc-400 leading-relaxed">
            Isolate system limits to maximize iterations per second across CPU
            architectures.
          </p>
          <div className="flex gap-2">
            <Button
              disabled={isActive}
              onClick={() => handleStart("HARDWARE")}
              className="flex-1 bg-emerald-600/10 text-emerald-400 hover:bg-emerald-600/30 border border-emerald-500/20 shadow-none text-xs"
            >
              {isActive ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Play className="w-3 h-3 mr-2" />
              )}
              {hwComplete ? "Restart Scan" : "Start Scan"}
            </Button>
            {hwComplete && (
              <Button
                variant="outline"
                size="icon"
                onClick={() => handleFlush("HARDWARE")}
                className="bg-transparent border-red-500/20 text-red-500 hover:bg-red-500/10"
              >
                <Trash2 className="w-4 h-4" />
              </Button>
            )}
          </div>
        </Card>

        <Card
          className={`bg-[#101010] border p-5 flex flex-col gap-4 shrink-0 transition-colors ${mctsComplete ? "border-blue-500/50" : "border-border/10"}`}
        >
          <div className="flex items-center justify-between text-blue-400">
            <div className="flex items-center gap-3">
              <Workflow className="w-5 h-5" />
              <h3 className="font-bold text-sm uppercase tracking-wider">
                MCTS Brain
              </h3>
            </div>
            {mctsComplete && <CheckCircle2 className="w-4 h-4 text-blue-500" />}
          </div>
          <p className="text-xs text-zinc-400 leading-relaxed">
            Optimizes tree exploration paths, Dirichlet noise, and PUCT
            parameters dynamically based off hardware capacity.
          </p>
          <div className="flex gap-2">
            <Button
              disabled={isActive || !hwComplete}
              onClick={() => handleStart("MCTS")}
              className="flex-1 bg-blue-600/10 text-blue-400 hover:bg-blue-600/30 border border-blue-500/20 shadow-none text-xs disabled:opacity-30 disabled:cursor-not-allowed"
            >
              {isActive ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Play className="w-3 h-3 mr-2" />
              )}
              {mctsComplete ? "Restart Scan" : "Start Scan"}
            </Button>
            {mctsComplete && (
              <Button
                variant="outline"
                size="icon"
                onClick={() => handleFlush("MCTS")}
                className="bg-transparent border-red-500/20 text-red-500 hover:bg-red-500/10"
              >
                <Trash2 className="w-4 h-4" />
              </Button>
            )}
          </div>
        </Card>

        <Card
          className={`bg-[#101010] border p-5 flex flex-col gap-4 shrink-0 transition-colors ${lrComplete ? "border-purple-500/50" : "border-border/10"}`}
        >
          <div className="flex items-center justify-between text-purple-400">
            <div className="flex items-center gap-3">
              <BrainCircuit className="w-5 h-5" />
              <h3 className="font-bold text-sm uppercase tracking-wider">
                Learning Velocity
              </h3>
            </div>
            {lrComplete && <CheckCircle2 className="w-4 h-4 text-purple-500" />}
          </div>
          <p className="text-xs text-zinc-400 leading-relaxed">
            Optimizes mathematical momentum and temperature decay for steepest
            descent.
          </p>
          <div className="flex gap-2">
            <Button
              disabled={isActive || !mctsComplete}
              onClick={() => handleStart("LEARNING")}
              className="flex-1 bg-purple-600/10 text-purple-400 hover:bg-purple-600/30 border border-purple-500/20 shadow-none text-xs disabled:opacity-30 disabled:cursor-not-allowed"
            >
              {isActive ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Play className="w-3 h-3 mr-2" />
              )}
              {lrComplete ? "Restart Scan" : "Start Scan"}
            </Button>
            {lrComplete && (
              <Button
                variant="outline"
                size="icon"
                onClick={() => handleFlush("LEARNING")}
                className="bg-transparent border-red-500/20 text-red-500 hover:bg-red-500/10"
              >
                <Trash2 className="w-4 h-4" />
              </Button>
            )}
          </div>
        </Card>

        <div className="flex flex-col gap-4 mt-auto bg-black/20 p-4 rounded-xl border border-white/5 shrink-0">
          <h4 className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">
            Global Parameters
          </h4>
          <Field className="grid gap-1">
            <FieldLabel className="text-xs text-zinc-400 font-medium">
              Trials per Process
            </FieldLabel>
            <Input
              type="number"
              value={trials}
              onChange={(e) => setTrials(parseInt(e.target.value))}
              className="bg-zinc-900 border-border/20 h-8 text-xs"
            />
          </Field>
          <Field className="grid gap-1">
            <FieldLabel className="text-xs text-zinc-400 font-medium">
              Max Epochs (Per Trial)
            </FieldLabel>
            <Input
              type="number"
              value={maxSteps}
              onChange={(e) => setMaxSteps(parseInt(e.target.value))}
              className="bg-zinc-900 border-border/20 h-8 text-xs"
            />
          </Field>
          <Field className="grid gap-1">
            <FieldLabel className="text-xs text-zinc-400 font-medium">
              Timeout Pruning Limit (Secs)
            </FieldLabel>
            <Input
              type="number"
              value={timeoutVal}
              onChange={(e) => setTimeoutVal(parseInt(e.target.value))}
              className="bg-zinc-900 border-border/20 h-8 text-xs"
            />
          </Field>
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
                />
              </div>
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>
    </div>
  );
}
