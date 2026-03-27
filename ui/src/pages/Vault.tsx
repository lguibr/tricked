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
      <div className="flex items-center justify-between">
        <h2 className="text-4xl font-black tracking-tight text-white shadow-sm drop-shadow-md">THE VAULT</h2>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
        {/* Left Column: Explorer */}
        <div className="lg:col-span-1 flex flex-col gap-6">
          <div className="bg-red-500/10 border border-red-500/30 p-4 rounded-xl flex items-start gap-3">
            <ShieldAlert className="w-5 h-5 text-red-500 shrink-0 mt-0.5 animate-pulse" />
            <div>
              <h4 className="font-semibold text-red-200 text-sm">Experience Forensics</h4>
              <p className="text-xs text-red-300/80 mt-1">
                Forensic playback of agent deaths. Hole logits (pulsing red) indicate paths the MCTS flagged as
                unfillable traps.
              </p>
            </div>
          </div>

          <TrajectoryTable />
        </div>

        {/* Right Column: Visualization & Timeline */}
        <div className="lg:col-span-2 flex flex-col gap-8 items-center">
          <div className="w-full max-w-xl mx-auto opacity-90 hover:opacity-100 transition-opacity">
            <BoardVisualizer showPolicy={true} showHoles={true} gameStateOverride={currentReplayState} />
          </div>

          <div className="w-full max-w-2xl mx-auto h-32 relative">
            {isReplaying ? (
              <div className="absolute inset-0 top-0 left-0 w-full animate-in fade-in slide-in-from-bottom-4 duration-500">
                <ReplayScrubber />
              </div>
            ) : (
              <div className="absolute inset-0 w-full p-6 border border-border border-dashed rounded-xl text-center text-muted-foreground flex flex-col items-center justify-center">
                <span className="text-sm font-semibold uppercase tracking-wider">No Trajectory Loaded</span>
                <span className="text-xs mt-2">Select a game from the Redis cache to begin playback.</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
