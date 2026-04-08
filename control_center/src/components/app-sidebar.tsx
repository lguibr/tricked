import {
  VscAdd,
  VscGraph,
  VscBeaker,
  VscServer,
  VscPlayCircle,
} from "react-icons/vsc";
import { Button } from "@/components/ui/button";
import { RunsSidebarList } from "@/components/execution/RunsSidebarList";
import logoUrl from "@/assets/logo.svg";
import { CreateSimpleRunSidebar } from "@/components/execution/CreateSimpleRunSidebar";
import { useAppStore } from "@/store/useAppStore";
import { StudiesSidebar } from "@/components/execution/StudiesSidebar";
import { PlaygroundSidebar } from "@/components/playground/PlaygroundSidebar";
import { VaultSidebar } from "@/components/execution/VaultSidebar";

export function AppSidebar() {
  const viewMode = useAppStore((state) => state.viewMode);
  const setViewMode = useAppStore((state) => state.setViewMode);
  const isCreatingRun = useAppStore((state) => state.isCreatingRun);
  const setIsCreatingRun = useAppStore((state) => state.setIsCreatingRun);

  const navItem = (mode: string, label: string, Icon: any) => {
    const active = viewMode === mode;
    return (
      <button
        onClick={() => setViewMode(mode as any)}
        className={`flex flex-col items-center justify-center p-1.5 rounded-sm transition-colors border ${active ? "bg-primary/10 text-primary border-primary/20 shadow-[0_0_10px_rgba(var(--primary),0.1)]" : "text-zinc-500 hover:bg-white/5 border-transparent hover:text-zinc-300"}`}
      >
        <Icon className="w-4 h-4 mb-0.5" />
        <span className="text-[8px] uppercase tracking-widest font-bold">
          {label}
        </span>
      </button>
    );
  };

  const renderSidebarContent = () => {
    if (viewMode === "runs") {
      return (
        <div className="flex w-full min-w-0 flex-col flex-1 overflow-hidden">
          <div className="px-3 py-1.5 text-[8.5px] font-bold text-zinc-600 bg-[#080808] uppercase tracking-widest border-b border-white/5 flex justify-between items-center shrink-0">
            Experiment Library
            <span className="text-[7px] bg-white/10 px-1 py-0.5 rounded text-zinc-400">
              REMOTE
            </span>
          </div>
          <div className="flex-1 overflow-hidden flex flex-col">
            {isCreatingRun ? (
              <CreateSimpleRunSidebar onClose={() => setIsCreatingRun(false)} />
            ) : (
              <RunsSidebarList />
            )}
          </div>
        </div>
      );
    }

    if (viewMode === "studies") return <StudiesSidebar />;
    if (viewMode === "playground") return <PlaygroundSidebar />;
    if (viewMode === "vault") return <VaultSidebar />;

    return null;
  };

  return (
    <div className="flex flex-col h-full w-full border-r border-border/20 bg-[#050505]">
      {/* Header */}
      <div className="border-b border-white/10 p-2 flex flex-col gap-2 shrink-0 bg-[#0a0a0a]">
        <div className="flex items-center gap-2 px-1">
          <div className="flex size-6 items-center justify-center rounded-sm bg-primary/20 text-primary border border-primary/30 shrink-0 shadow-[0_0_8px_rgba(16,185,129,0.2)]">
            <img src={logoUrl} alt="Logo" className="size-4" />
          </div>
          <div className="flex flex-col leading-none">
            <span className="text-[11px] font-black tracking-widest text-zinc-100 uppercase">
              Tricked AI
            </span>
            <span className="text-[8px] text-primary/80 uppercase tracking-widest font-bold">
              Control Center
            </span>
          </div>
        </div>

        {/* Navigation Grid */}
        <div className="grid grid-cols-4 gap-1 bg-black/50 p-1 rounded border border-white/5">
          {navItem("runs", "Metrics", VscGraph)}
          {navItem("studies", "Tuning", VscBeaker)}
          {navItem("vault", "Vault", VscServer)}
          {navItem("playground", "Arena", VscPlayCircle)}
        </div>

        {viewMode === "runs" && (
          <Button
            size="sm"
            className="w-full h-6 text-[9px] uppercase tracking-widest font-bold shadow-md shadow-primary/20 hover:shadow-primary/40 transition-shadow bg-primary/20 text-primary hover:bg-primary/30 border border-primary/30"
            onClick={() => {
              setViewMode("runs");
              setIsCreatingRun(true);
            }}
          >
            <VscAdd className="w-3 h-3 mr-1" /> New Simple Run
          </Button>
        )}
      </div>

      {/* Main List Area */}
      <div className="flex min-h-0 flex-1 flex-col overflow-auto bg-[#030303]">
        {renderSidebarContent()}
      </div>
    </div>
  );
}
