import { useEffect, useState } from 'react';
import { BoardVisualizer } from '@/components/BoardVisualizer';
import { HeartbeatMonitor } from '@/components/HeartbeatMonitor';
import { OrchestratorControls } from '@/components/OrchestratorControls';
import { useEngineStore } from '@/store/useEngineStore';
import { Button } from '@/components/ui/button';
import { Bot } from 'lucide-react';
import { getPieceData } from '@/lib/math';

export function MissionControl() {
  const [activeSlot, setActiveSlot] = useState(0);
  const connectWebSocket = useEngineStore((s) => s.connectWebSocket);
  const playHumanMove = useEngineStore((s) => s.playHumanMove);
  const playAiMove = useEngineStore((s) => s.playAiMove);
  const isTraining = useEngineStore((s) => s.isTraining);
  const gameState = useEngineStore((s) => s.gameState);

  const available = gameState?.available || [-1, -1, -1];
  const pieceMasks = gameState?.piece_masks;

  useEffect(() => {
    connectWebSocket();
  }, [connectWebSocket]);

  return (
    <div className="flex flex-col lg:flex-row min-h-[calc(100vh-4rem)] gap-8 p-8 max-w-7xl mx-auto w-full items-center justify-center">
      <div className="flex-1 w-full max-w-2xl flex flex-col gap-6">
        <div className="flex flex-col mb-2">
          <h2 className="text-4xl font-black tracking-tight text-white mb-2 shadow-sm drop-shadow-md">MISSION CONTROL</h2>
          <p className="text-muted-foreground text-sm max-w-lg leading-relaxed">
            Welcome to the live orchestration interface. This board visualizes the 96-triangle environment. The piece tray on the right displays your available shapes.
            Click <strong className="text-primary">Play AI Move</strong> to trigger the AlphaZero engine to search for and execute the optimal piece placement using its neural network.
          </p>
        </div>
        <div className="flex gap-4">
          <Button
            onClick={playAiMove}
            disabled={!isTraining}
            className="flex-1 w-full bg-cyan-600 hover:bg-cyan-500 text-white font-bold py-6 text-lg rounded-sm shadow-[0_0_15px_rgba(0,251,251,0.2)] transition-all"
          >
            <Bot className="mr-2 h-6 w-6" /> PLAY AI MOVE
          </Button>
          <div className="flex flex-1 gap-2 border border-white/10 bg-black/20 p-2 rounded-sm justify-between">
            {available.map((pId: number, slot: number) => {
              const pData = getPieceData(pId, pieceMasks);
              return (
                <div
                  key={slot}
                  className={`w-16 h-16 flex items-center justify-center border-2 cursor-pointer transition-all ${activeSlot === slot ? 'border-primary bg-primary/20' : 'border-white/10 hover:border-white/30'}`}
                  onClick={() => setActiveSlot(slot)}
                >
                  {pData ? (
                    <svg viewBox={pData.viewBox} className="w-12 h-12 fill-cyan-400 stroke-cyan-200">
                      {pData.polys.map((pts: string, i: number) => (
                        <polygon key={i} points={pts} strokeWidth="1" />
                      ))}
                    </svg>
                  ) : <span className="text-white/20 text-xs">Empty</span>}
                </div>
              );
            })}
          </div>
        </div>
        <BoardVisualizer showPolicy={true} showHoles={true} onPlayMove={(idx) => playHumanMove(activeSlot, idx)} />
      </div>
      <div className="flex flex-col w-full lg:w-96 gap-6">
        <HeartbeatMonitor />
        <OrchestratorControls />
      </div>
    </div>
  );
}
