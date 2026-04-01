import { useState } from 'react';
import { TerminalSquare, Activity, Settings } from 'lucide-react';

import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { ExecutionTab } from '@/components/ExecutionTab';
import { Breadcrumb, BreadcrumbItem, BreadcrumbLink, BreadcrumbList, BreadcrumbPage, BreadcrumbSeparator } from "@/components/ui/breadcrumb";
import logoUrl from '@/assets/logo.svg';

function App() {
  const [activeTab, setActiveTab] = useState('execution');

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
        {activeTab === 'execution' && (
          <ExecutionTab />
        )}

        {activeTab === 'dashboards' && (
          <div className="flex-1 flex overflow-hidden bg-background">
            {/* Terminal Sidebar */}
            <div className="w-96 border-r border-border/50 flex flex-col bg-[#0c0c0c] overflow-hidden shrink-0">
              <div className="px-3 py-1.5 border-b border-white/10 flex items-center space-x-2 text-muted-foreground text-[10px] font-mono bg-zinc-900">
                <TerminalSquare className="w-3 h-3" />
                <span>STDOUT & STDERR</span>
              </div>
              <ScrollArea className="flex-1 p-2 font-mono text-[10px] leading-tight text-zinc-400">
                <div className="space-y-0.5">
                  <p><span className="text-blue-400">[INFO]</span> Starting Tricked AI Engine...</p>
                  <p><span className="text-green-500">[OPTUNA]</span> Trial 12 finished with score: 1.45</p>
                  <p><span className="text-blue-400">[PROFILE]</span> mcts_search avg: 142ms, allocations: 0</p>
                  <p><span className="text-yellow-500">[WARN]</span> Replay buffer near capacity (98%)</p>
                  <p><span className="text-blue-400">[INFO]</span> GPU Utilization stabilized at 99.2%</p>
                  <br /><br /><br />
                </div>
              </ScrollArea>
              <div className="p-1 border-t border-white/10 bg-zinc-900">
                <Input className="h-6 text-[10px] bg-black border-zinc-800 placeholder:text-zinc-600 px-2 rounded-sm" placeholder="Filter logs (regex)..." />
              </div>
            </div>

            {/* Dashboard Visuals */}
            <div className="flex-1 flex flex-col h-full overflow-hidden">
              <div className="flex justify-between items-center bg-muted/5 border-b border-border/50 px-3 py-1.5 shrink-0">
                <div className="flex items-center space-x-3">
                  <div className="flex items-center space-x-1.5"><div className="w-2 h-2 rounded-full bg-orange-500"></div><span className="text-[10px] font-medium tracking-tight">Run 1</span></div>
                  <div className="flex items-center space-x-1.5"><div className="w-2 h-2 rounded-full bg-purple-500"></div><span className="text-[10px] font-medium tracking-tight">Run 2</span></div>
                  <div className="flex items-center space-x-1.5"><div className="w-2 h-2 rounded-full bg-pink-500"></div><span className="text-[10px] font-medium tracking-tight">Run 3</span></div>
                </div>
                <div className="flex items-center space-x-2">
                  <Input placeholder="Regex Coloring Runs" className="w-48 h-6 text-[10px] px-2 rounded-sm bg-background border-border/50" />
                </div>
              </div>

              {/* 3x3 Grid */}
              <div className="flex-1 grid grid-cols-3 grid-rows-3 bg-border/50 gap-[1px]">
                <div className="flex flex-col bg-background"><div className="py-1.5 px-2 border-b border-border/50 bg-muted/5"><h3 className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">SCORE</h3></div><div className="flex-1 flex items-center justify-center text-xs">📈</div></div>
                <div className="flex flex-col bg-background"><div className="py-1.5 px-2 border-b border-border/50 bg-muted/5"><h3 className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">GAMES/SEC</h3></div><div className="flex-1 flex items-center justify-center text-xs">📈</div></div>
                <div className="flex flex-col bg-background"><div className="py-1.5 px-2 border-b border-border/50 bg-muted/5"><h3 className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">GAME WINS</h3></div><div className="flex-1 flex items-center justify-center text-xs">📈</div></div>
                <div className="flex flex-col bg-background"><div className="py-1.5 px-2 border-b border-border/50 bg-muted/5"><h3 className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">TOTAL LOSS</h3></div><div className="flex-1 flex items-center justify-center text-xs">📈</div></div>
                <div className="flex flex-col bg-background"><div className="py-1.5 px-2 border-b border-border/50 bg-muted/5"><h3 className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">VALUE LOSS</h3></div><div className="flex-1 flex items-center justify-center text-xs">📈</div></div>
                <div className="flex flex-col bg-background"><div className="py-1.5 px-2 border-b border-border/50 bg-muted/5"><h3 className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">POLICY LOSS</h3></div><div className="flex-1 flex items-center justify-center text-xs">📈</div></div>
                <div className="flex flex-col col-span-3 bg-background"><div className="py-1.5 px-2 border-b border-border/50 bg-muted/5"><h3 className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">MCTS SEARCH DEPTH TRAJECTORY</h3></div><div className="flex-1 flex items-center justify-center text-xs">📈 iframe placeholder</div></div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
