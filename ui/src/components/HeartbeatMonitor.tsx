
import { Activity } from 'lucide-react';
import { useEngineStore } from '@/store/useEngineStore';

export function HeartbeatMonitor() {
  const trainingInfo = useEngineStore((state) => state.trainingInfo);
  const isTraining = useEngineStore((state) => state.isTraining);



  const currentGPS = trainingInfo?.games_per_second || 0;

  return (
    <div className="flex flex-col bg-background/80 border border-white/10 p-4 rounded-none shadow-lg backdrop-blur-md w-full max-w-sm">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Activity className={`w-5 h-5 ${isTraining ? 'text-primary animate-pulse' : 'text-muted-foreground'}`} />
          <span className="font-semibold text-sm uppercase tracking-wider text-muted-foreground">Engine Vitals</span>
        </div>
        <div className="text-right">
          <span className={`text-4xl font-black ${isTraining ? 'text-primary drop-shadow-[0_0_15px_rgba(0,251,251,0.5)]' : 'text-muted-foreground'}`}>
            {currentGPS.toFixed(2)}
          </span>
          <span className="text-sm font-semibold tracking-widest text-muted-foreground ml-2">GAMES / SEC</span>
        </div>
      </div>
      <div className="h-1 w-full bg-black/40 rounded-full overflow-hidden mt-4">
        <div className={`h-full ${isTraining ? 'bg-primary animate-pulse shadow-[0_0_10px_rgba(0,251,251,0.8)]' : 'bg-transparent'}`} style={{ width: '100%' }} />
      </div>
    </div>
  );
}
