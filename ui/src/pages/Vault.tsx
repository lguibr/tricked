import { useEffect } from 'react';
import { BoardVisualizer } from '@/components/BoardVisualizer';
import { TrajectoryTable } from '@/components/vault/TrajectoryTable';
import { ReplayScrubber } from '@/components/vault/ReplayScrubber';
import { useEngineStore } from '@/store/useEngineStore';
import { ShieldAlert } from 'lucide-react';

export function Vault() {
  const setReplaying = useEngineStore((state) => state.setReplaying);
  const isReplaying = useEngineStore((state) => state.isReplaying);
  const activeReplayData = useEngineStore((state) => state.activeReplayData);
  const replayCursor = useEngineStore((state) => state.replayCursor);

  const currentReplayState = isReplaying && activeReplayData?.steps
    ? activeReplayData.steps[replayCursor]
    : null;

  useEffect(() => {
    // Prevent websocket from overriding the board state while in the Vault
    setReplaying(true);
    return () => setReplaying(false);
  }, [setReplaying]);

  return (
    <div className="flex flex-col min-h-[calc(100vh-4rem)] gap-8 p-8 max-w-7xl mx-auto w-full pb-20">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-4xl font-black tracking-tight text-white shadow-sm drop-shadow-md">THE VAULT</h2>
        <button
          onClick={() => useEngineStore.getState().refreshData()}
          className="bg-primary/10 text-primary border border-primary/30 px-6 py-2 rounded-md font-bold uppercase tracking-wider hover:bg-primary/20 hover:scale-105 transition-all text-sm flex items-center gap-2"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" /><path d="M3 3v5h5" /></svg>
          Refresh Replays
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-12 items-start">
        {/* Left Column: Explorer */}
        <div className="flex flex-col gap-4">
          <div className="bg-red-500/10 border border-red-500/30 p-3 rounded-lg flex items-start gap-2">
            <ShieldAlert className="w-4 h-4 text-red-500 shrink-0 mt-0.5 animate-pulse" />
            <div>
              <h4 className="font-semibold text-red-200 text-xs uppercase tracking-wider">Experience Forensics</h4>
              <p className="text-[11px] text-red-300/80 mt-0.5 leading-relaxed">
                Replay historical agent rollouts isolated from the Redis cache.
              </p>
            </div>
          </div>

          <TrajectoryTable />
        </div>

        {/* Right Column: Visualization & Timeline */}
        <div className="flex flex-col gap-4 items-center bg-black/20 p-6 rounded-xl border border-white/5 shadow-inner">
          <div className="w-full max-w-[320px] mx-auto opacity-90 transition-opacity">
            <BoardVisualizer showPolicy={true} showHoles={true} gameStateOverride={currentReplayState} />
          </div>

          {currentReplayState && (
            <div className="w-full max-w-[320px] flex gap-2 justify-between text-xs font-mono text-primary/80 mt-2 mb-2">
              <div className="flex flex-col items-center p-2 bg-black/40 rounded-lg border border-white/10 w-1/3 text-center">
                <span className="text-white/40 uppercase text-[9px] font-bold mb-1">Tray</span>
                <span className="font-semibold text-cyan-400">
                  {currentReplayState.available?.filter((id: number) => id >= 0).join(', ') || 'N/A'}
                </span>
              </div>
              <div className="flex flex-col items-center p-2 bg-black/40 rounded-lg border border-white/10 w-1/3 text-center">
                <span className="text-white/40 uppercase text-[9px] font-bold mb-1">Piece</span>
                <span className="font-semibold text-yellow-400">
                  {currentReplayState.piece_id >= 0 ? currentReplayState.piece_id : 'Pass'}
                </span>
              </div>
              <div className="flex flex-col items-center p-2 bg-black/40 rounded-lg border border-white/10 w-1/3 text-center">
                <span className="text-white/40 uppercase text-[9px] font-bold mb-1">Action</span>
                <span className="font-semibold text-emerald-400">
                  {currentReplayState.action !== undefined ? currentReplayState.action : 'N/A'}
                </span>
              </div>
            </div>
          )}

          <div className="w-full max-w-lg mx-auto h-24 relative mt-2">
            {isReplaying ? (
              <div className="absolute inset-0 top-0 left-0 w-full animate-in fade-in slide-in-from-bottom-4 duration-300">
                <ReplayScrubber />
              </div>
            ) : (
              <div className="absolute inset-0 w-full p-4 border border-border border-dashed rounded-lg text-center text-muted-foreground flex flex-col items-center justify-center bg-black/40">
                <span className="text-xs font-bold uppercase tracking-widest text-primary/70">Awaiting Trajectory</span>
                <span className="text-[10px] mt-1 opacity-60">Select a game from the Redis table to begin forensic playback.</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
