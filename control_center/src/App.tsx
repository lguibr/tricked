import { useState } from 'react';
import { Play, Square, Pause, Copy, Trash2, TerminalSquare, Activity, Settings } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { ScrollArea } from '@/components/ui/scroll-area';

function App() {
  const [activeTab, setActiveTab] = useState('execution');

  return (
    <div className="dark min-h-screen bg-background text-foreground flex flex-col font-sans">
      {/* Top Header */}
      <header className="border-b border-border px-6 py-4 flex items-center justify-between shadow-sm sticky top-0 bg-background z-10">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded bg-primary flex items-center justify-center">
            <span className="text-primary-foreground font-bold text-xl">T</span>
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">Tricked AI Control Center</h1>
            <p className="text-xs text-muted-foreground">Tauri Native Optimization Hub</p>
          </div>
        </div>
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-[400px]">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="execution"><Settings className="w-4 h-4 mr-2" /> Planner & Execution</TabsTrigger>
            <TabsTrigger value="dashboards"><Activity className="w-4 h-4 mr-2" /> Dashboards</TabsTrigger>
          </TabsList>
        </Tabs>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 flex overflow-hidden">
        {activeTab === 'execution' && (
          <div className="flex-1 flex">
            {/* Sidebar: Configs List */}
            <div className="w-64 border-r border-border bg-card/50 flex flex-col">
              <div className="p-4 border-b border-border font-semibold text-sm uppercase tracking-wider text-muted-foreground">
                Config & Saved Runs
              </div>
              <ScrollArea className="flex-1 p-4">
                <div className="space-y-3">
                  {/* Mock Config Item */}
                  <div className="border border-primary bg-primary/10 rounded-md p-3 relative group">
                    <h3 className="font-medium text-sm text-primary">baseline_run</h3>
                    <p className="text-xs text-muted-foreground mt-1">Status: WAITING</p>
                    <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 flex space-x-1 transition-opacity">
                      <Button variant="ghost" size="icon" className="h-6 w-6"><Copy className="h-3 w-3" /></Button>
                      <Button variant="ghost" size="icon" className="h-6 w-6 text-destructive hover:bg-destructive hover:text-white"><Trash2 className="h-3 w-3" /></Button>
                    </div>
                  </div>
                  {/* Mock Config Item */}
                  <div className="border border-border bg-card hover:bg-accent rounded-md p-3 relative group cursor-pointer transition-colors">
                    <h3 className="font-medium text-sm">optuna_search_01</h3>
                    <p className="text-xs text-muted-foreground mt-1 text-green-500">Status: RUNNING</p>
                    <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 flex space-x-1 transition-opacity">
                      <Button variant="ghost" size="icon" className="h-6 w-6"><Copy className="h-3 w-3" /></Button>
                      <Button variant="ghost" size="icon" className="h-6 w-6 text-destructive hover:bg-destructive hover:text-white"><Trash2 className="h-3 w-3" /></Button>
                    </div>
                  </div>
                </div>
              </ScrollArea>
              <div className="p-4 border-t border-border">
                <Button className="w-full" variant="outline">+ New Config</Button>
              </div>
            </div>

            {/* Main Execution Surface (3 Cols) */}
            <div className="flex-1 grid grid-cols-3 gap-6 p-6 overflow-y-auto">
              {/* Col 1: Tuning Parameters */}
              <Card className="col-span-1 shadow-md border-border/60">
                <CardHeader>
                  <CardTitle>Tune Parameters</CardTitle>
                  <CardDescription>Initial settings & range suggestions</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-3">
                    <div className="flex justify-between"><Label>Max Steps</Label><span className="text-xs text-muted-foreground">1,000,000</span></div>
                    <Slider defaultValue={[1000000]} max={10000000} step={100000} />
                  </div>
                  <div className="space-y-3">
                    <div className="flex justify-between"><Label>Gumbel C_Visit</Label><span className="text-xs text-muted-foreground">50</span></div>
                    <Slider defaultValue={[50]} max={100} step={1} />
                  </div>
                  <div className="space-y-3">
                    <div className="flex justify-between"><Label>Value Weight</Label><span className="text-xs text-muted-foreground">0.25</span></div>
                    <Slider defaultValue={[25]} max={100} step={1} />
                  </div>
                </CardContent>
              </Card>

              {/* Col 2: Hydra Payload / Editor */}
              <Card className="col-span-1 shadow-md border-border/60 bg-muted/20">
                <CardHeader>
                  <CardTitle>Hydra Payload</CardTitle>
                  <CardDescription>Raw Config JSON</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="w-full h-[400px] bg-card border border-border rounded-md p-4 font-mono text-xs overflow-auto text-muted-foreground">
                    <pre>
                      {`{
  "run_name": "baseline_run",
  "tuning_params": {
    "max_steps": 1000000,
    "gumbel_c_visit": 50,
    "value_weight": 0.25
  },
  "architecture": "hexagonal",
  "auto_tune": true
}`}
                    </pre>
                  </div>
                </CardContent>
              </Card>

              {/* Col 3: Action Controls */}
              <Card className="col-span-1 flex flex-col items-center justify-center p-12 text-center shadow-md bg-gradient-to-b from-card to-background border-border/60">
                <div className="w-32 h-32 rounded-full bg-primary/10 flex items-center justify-center mb-8 border-4 border-primary/20">
                  <span className="text-5xl font-black text-primary">T</span>
                </div>
                <h2 className="text-2xl font-bold mb-2">TRICKED AI</h2>
                <p className="text-muted-foreground mb-12 text-sm">Native Engine Core</p>

                <div className="flex space-x-4 w-full justify-center">
                  <Button size="lg" className="rounded-full w-16 h-16 shadow-lg shadow-primary/20"><Play className="h-8 w-8" /></Button>
                  <Button size="lg" variant="secondary" className="rounded-full w-16 h-16"><Pause className="h-8 w-8" /></Button>
                  <Button size="lg" variant="destructive" className="rounded-full w-16 h-16 shadow-lg"><Square className="h-7 w-7" /></Button>
                </div>
              </Card>
            </div>
          </div>
        )}

        {activeTab === 'dashboards' && (
          <div className="flex-1 flex">
            {/* Terminal Sidebar */}
            <div className="w-96 border-r border-border flex flex-col bg-[#0c0c0c]">
              <div className="p-3 border-b border-border/20 flex items-center space-x-2 text-muted-foreground text-sm font-mono bg-zinc-900">
                <TerminalSquare className="w-4 h-4" />
                <span>STDOUT & STDERR</span>
              </div>
              <ScrollArea className="flex-1 p-4 font-mono text-[11px] text-zinc-400">
                <div className="space-y-1">
                  <p><span className="text-blue-400">[INFO]</span> Starting Tricked AI Engine...</p>
                  <p><span className="text-green-500">[OPTUNA]</span> Trial 12 finished with score: 1.45</p>
                  <p><span className="text-blue-400">[PROFILE]</span> mcts_search avg: 142ms, allocations: 0</p>
                  <p><span className="text-yellow-500">[WARN]</span> Replay buffer near capacity (98%)</p>
                  <p><span className="text-blue-400">[INFO]</span> GPU Utilization stabilized at 99.2%</p>
                  {/* More mock logs... */}
                  <br /><br /><br />
                </div>
              </ScrollArea>
              <div className="p-2 border-t border-border/20 bg-zinc-900">
                <Input className="h-8 text-xs bg-zinc-800 border-zinc-700 placeholder:text-zinc-500" placeholder="Filter logs (regex)..." />
              </div>
            </div>

            {/* Dashboard Visuals */}
            <div className="flex-1 flex flex-col bg-muted/10 p-6">
              <div className="flex justify-between items-center mb-6">
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-2"><div className="w-3 h-3 rounded-full bg-orange-500"></div><span className="text-sm font-medium">Run 1</span></div>
                  <div className="flex items-center space-x-2"><div className="w-3 h-3 rounded-full bg-purple-500"></div><span className="text-sm font-medium">Run 2</span></div>
                  <div className="flex items-center space-x-2"><div className="w-3 h-3 rounded-full bg-pink-500"></div><span className="text-sm font-medium">Run 3</span></div>
                </div>
                <div className="flex items-center space-x-3">
                  <Input placeholder="Regex Coloring Runs" className="w-64" />
                </div>
              </div>

              {/* 3x3 Grid */}
              <div className="flex-1 grid grid-cols-3 grid-rows-3 gap-4">
                {/* Mock Chart Cards */}
                <Card className="flex flex-col"><CardHeader className="py-3"><CardTitle className="text-sm">SCORE</CardTitle></CardHeader><CardContent className="flex-1 bg-card/50 flex items-center justify-center">📈 [Chart Render]</CardContent></Card>
                <Card className="flex flex-col"><CardHeader className="py-3"><CardTitle className="text-sm">GAMES/SEC</CardTitle></CardHeader><CardContent className="flex-1 bg-card/50 flex items-center justify-center">📈 [Chart Render]</CardContent></Card>
                <Card className="flex flex-col"><CardHeader className="py-3"><CardTitle className="text-sm">GAME WINS</CardTitle></CardHeader><CardContent className="flex-1 bg-card/50 flex items-center justify-center">📈 [Chart Render]</CardContent></Card>
                <Card className="flex flex-col"><CardHeader className="py-3"><CardTitle className="text-sm">TOTAL LOSS</CardTitle></CardHeader><CardContent className="flex-1 bg-card/50 flex items-center justify-center">📈 [Chart Render]</CardContent></Card>
                <Card className="flex flex-col"><CardHeader className="py-3"><CardTitle className="text-sm">VALUE LOSS</CardTitle></CardHeader><CardContent className="flex-1 bg-card/50 flex items-center justify-center">📈 [Chart Render]</CardContent></Card>
                <Card className="flex flex-col"><CardHeader className="py-3"><CardTitle className="text-sm">POLICY LOSS</CardTitle></CardHeader><CardContent className="flex-1 bg-card/50 flex items-center justify-center">📈 [Chart Render]</CardContent></Card>
                <Card className="flex flex-col col-span-3"><CardHeader className="py-3"><CardTitle className="text-sm">MCTS SEARCH DEPTH TRAJECTORY</CardTitle></CardHeader><CardContent className="flex-1 bg-card/50 flex items-center justify-center">📈 [Large Chart/W&B iframe placeholder]</CardContent></Card>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
