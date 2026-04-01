import { useState, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { TerminalSquare, Activity, Settings } from 'lucide-react';

import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Toggle } from '@/components/ui/toggle';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { ExecutionTab } from '@/components/ExecutionTab';
import { MetricsDashboard } from '@/components/MetricsDashboard';
import { OptunaStudyDashboard } from '@/components/OptunaStudyDashboard';
import { Breadcrumb, BreadcrumbItem, BreadcrumbLink, BreadcrumbList, BreadcrumbPage, BreadcrumbSeparator } from "@/components/ui/breadcrumb";
import logoUrl from '@/assets/logo.svg';

interface Run {
  id: string;
  name: string;
  status: string;
  type: string;
  config: string;
}

function App() {
  const [activeTab, setActiveTab] = useState('execution');
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedDashboardRuns, setSelectedDashboardRuns] = useState<string[]>([]);
  const [runLogs, setRunLogs] = useState<Record<string, string[]>>({});
  const [runColors, setRunColors] = useState<Record<string, string>>({});
  const dashboardLogsEndRef = useRef<HTMLDivElement | null>(null);

  const DEFAULT_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#ef4444', '#14b8a6'];

  useEffect(() => {
    if (dashboardLogsEndRef.current) {
      dashboardLogsEndRef.current.scrollIntoView({ behavior: "auto" });
    }
  }, [runLogs, selectedDashboardRuns]);

  useEffect(() => {
    let active = true;
    const fetchRuns = async () => {
      try {
        const result = await invoke<Run[]>('list_runs');
        if (active) {
          setRuns(result);
        }
      } catch (e) {
        console.error(e);
      }
    };
    fetchRuns();
    const interval = setInterval(fetchRuns, 3000);

    let unlisten: (() => void) | undefined;
    import('@tauri-apps/api/event').then(({ listen }) => {
      listen('log_event', (event: any) => {
        const { run_id, line } = event.payload;
        setRunLogs(prev => {
          const updated = [...(prev[run_id] || []), line].slice(-500);
          return { ...prev, [run_id]: updated };
        });
      }).then(u => unlisten = u);
    });

    return () => {
      active = false;
      clearInterval(interval);
      if (unlisten) unlisten();
    };
  }, []);

  const toggleDashboardRun = (id: string, pressed: boolean) => {
    if (pressed) setSelectedDashboardRuns(prev => [...prev, id]);
    else setSelectedDashboardRuns(prev => prev.filter(r => r !== id));
  };

  return (
    <div className="dark h-screen bg-background text-foreground flex flex-col font-sans overflow-hidden">
      {/* Top Header */}
      <header className="border-b border-border/50 px-4 py-2 flex items-center justify-between bg-muted/5 z-10 flex-shrink-0">
        <div className="flex items-center space-x-3">
          <img src={logoUrl} alt="Tricked AI Logo" className="w-6 h-6" />
          <Breadcrumb>
            <BreadcrumbList>
              <BreadcrumbItem>
                <BreadcrumbLink href="#" className="text-xs font-semibold">Tricked AI Control Center</BreadcrumbLink>
              </BreadcrumbItem>
              <BreadcrumbSeparator />
              <BreadcrumbItem>
                <BreadcrumbPage className="text-xs">{activeTab === 'execution' ? 'Execution & Setup' : 'Telemetry Dashboards'}</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
        </div>
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-[300px]">
          <TabsList className="grid w-full grid-cols-2 h-8">
            <TabsTrigger value="execution" className="text-xs"><Settings className="w-3 h-3 mr-1.5" /> Execution</TabsTrigger>
            <TabsTrigger value="dashboards" className="text-xs"><Activity className="w-3 h-3 mr-1.5" /> Dashboards</TabsTrigger>
          </TabsList>
        </Tabs>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 flex overflow-hidden">
        <div className={activeTab === 'execution' ? "flex-1 flex overflow-hidden" : "hidden"}>
          <ExecutionTab />
        </div>

        <div className={activeTab === 'dashboards' ? "flex-1 flex overflow-hidden bg-background" : "hidden"}>
          {/* Left Sidebar */}
          <div className="w-96 border-r border-border/50 flex flex-col bg-[#0c0c0c] overflow-hidden shrink-0">
            {/* Run Selection Panel */}
            <div className="px-3 py-3 border-b border-white/10 bg-zinc-950 flex-shrink-0">
              <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider mb-2 block">Compare Runs</span>
              <div className="flex flex-wrap gap-2">
                {runs.map((r, idx) => {
                  const runColor = runColors[r.id] || DEFAULT_COLORS[idx % DEFAULT_COLORS.length];
                  return (
                    <div key={r.id} className="flex items-center space-x-1 mb-1">
                      <input
                        type="color"
                        value={runColor}
                        onChange={(e) => setRunColors(prev => ({ ...prev, [r.id]: e.target.value }))}
                        className="w-5 h-5 p-0 border-0 rounded cursor-pointer bg-transparent"
                      />
                      <Toggle
                        pressed={selectedDashboardRuns.includes(r.id)}
                        onPressedChange={(p) => toggleDashboardRun(r.id, p)}
                        size="sm"
                        className="h-6 px-2 text-[10px] data-[state=on]:bg-primary/20 data-[state=on]:text-primary border border-zinc-800 data-[state=on]:border-primary/50 text-zinc-400"
                      >
                        <div className={`w-1.5 h-1.5 rounded-full mr-1.5 ${r.status === 'RUNNING' ? 'bg-green-500 animate-pulse' : r.status === 'COMPLETED' ? 'bg-blue-500' : 'bg-zinc-600'}`} />
                        {r.name}
                      </Toggle>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Terminal Header */}
            <div className="px-3 py-1.5 border-b border-white/10 flex items-center space-x-2 text-muted-foreground text-[10px] font-mono bg-zinc-900">
              <TerminalSquare className="w-3 h-3" />
              <span>STDOUT & STDERR</span>
            </div>
            <ScrollArea className="flex-1 p-2 font-mono text-[10px] leading-tight text-zinc-400">
              <div className="space-y-0.5 whitespace-pre-wrap">
                {selectedDashboardRuns.length === 0 ? (
                  <span className="text-zinc-600 italic">Select a run to view live terminal</span>
                ) : (
                  selectedDashboardRuns.map(runId => (
                    <div key={runId} className="mb-4">
                      <div className="text-zinc-500 font-bold mb-1 border-b border-white/10 pb-1"># {runs.find(r => r.id === runId)?.name}</div>
                      {(runLogs[runId] || []).length === 0 ? (
                        <span className="text-zinc-600 italic">Waiting for connection...</span>
                      ) : (
                        (runLogs[runId] || []).map((line, idx) => (
                          <div key={idx}>
                            {line.includes('[WARN]') ? <span className="text-yellow-500">{line}</span> :
                              line.includes('[ERR]') ? <span className="text-red-500">{line}</span> :
                                line.includes('[INFO]') ? <span><span className="text-blue-400">[INFO]</span>{line.split('[INFO]')[1]}</span> :
                                  line}
                          </div>
                        ))
                      )}
                    </div>
                  ))
                )}
                <div ref={dashboardLogsEndRef} />
              </div>
            </ScrollArea>
            <div className="p-1 border-t border-white/10 bg-zinc-900">
              <Input className="h-6 text-[10px] bg-black border-zinc-800 placeholder:text-zinc-600 px-2 rounded-sm" placeholder="Filter logs (regex)..." />
            </div>
          </div>

          {/* Dashboard Visuals */}
          <div className="flex-1 flex flex-col h-full overflow-hidden">
            <Tabs defaultValue="metrics" className="flex flex-col h-full w-full">
              <div className="flex justify-between items-center bg-muted/5 border-b border-border/50 px-3 py-1.5 shrink-0">
                <TabsList className="h-6">
                  <TabsTrigger value="metrics" className="text-[10px] px-2 py-0.5 h-5">WandB Metrics</TabsTrigger>
                  <TabsTrigger value="optuna" className="text-[10px] px-2 py-0.5 h-5">Optuna Studies</TabsTrigger>
                  <TabsTrigger value="config" className="text-[10px] px-2 py-0.5 h-5">Hydra Payload</TabsTrigger>
                </TabsList>
              </div>

              <div className="flex-1 overflow-hidden relative">
                <TabsContent value="metrics" className="m-0 h-full w-full">
                  <MetricsDashboard runIds={selectedDashboardRuns} runColors={runColors} />
                </TabsContent>
                <TabsContent value="optuna" className="m-0 h-full absolute inset-0">
                  <OptunaStudyDashboard />
                </TabsContent>
                <TabsContent value="config" className="m-0 h-full absolute inset-0 p-4 overflow-auto bg-black font-mono text-[10px] text-zinc-300">
                  {selectedDashboardRuns.map(runId => {
                    const run = runs.find(r => r.id === runId);
                    return (
                      <div key={runId} className="mb-4">
                        <h3 className="font-bold text-white mb-2"># {run?.name} config</h3>
                        <pre>{run?.config}</pre>
                      </div>
                    );
                  })}
                  {selectedDashboardRuns.length === 0 && <div className="text-zinc-600 flex items-center justify-center h-full">Select a run above to view hydra payload</div>}
                </TabsContent>
              </div>
            </Tabs>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
