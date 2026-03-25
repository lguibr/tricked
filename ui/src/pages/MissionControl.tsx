import { useEffect } from 'react';
import { BoardVisualizer } from '@/components/BoardVisualizer';
import { HeartbeatMonitor } from '@/components/HeartbeatMonitor';
import { OrchestratorControls } from '@/components/OrchestratorControls';
import { useEngineStore } from '@/store/useEngineStore';

export function MissionControl() {
  const connectWebSocket = useEngineStore((s) => s.connectWebSocket);

  useEffect(() => {
    connectWebSocket();
  }, [connectWebSocket]);

  return (
    <div className="flex flex-col lg:flex-row min-h-[calc(100vh-4rem)] gap-8 p-8 max-w-7xl mx-auto w-full items-center justify-center">
      <div className="flex-1 w-full max-w-2xl flex flex-col gap-6">
        <h2 className="text-4xl font-black tracking-tight text-white mb-2 shadow-sm drop-shadow-md">MISSION CONTROL</h2>
        <BoardVisualizer showPolicy={true} showHoles={true} />
      </div>
      <div className="flex flex-col w-full lg:w-96 gap-6">
        <HeartbeatMonitor />
        <OrchestratorControls />
      </div>
    </div>
  );
}
