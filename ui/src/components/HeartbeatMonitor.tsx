import { useMemo } from 'react';
import { LineChart, Line, YAxis, ResponsiveContainer } from 'recharts';
import { Activity } from 'lucide-react';
import { useEngineStore } from '@/store/useEngineStore';

export function HeartbeatMonitor() {
  const trainingInfo = useEngineStore((state) => state.trainingInfo);
  const isTraining = useEngineStore((state) => state.isTraining);

  const data = useMemo(() => {
    // We mock the sparkline locally by storing a rolling buffer of 20 points
    // if 'gps_history' is not available from backend. But we'll just parse the live val
    return trainingInfo?.gps_history || Array.from({ length: 20 }).map(() => ({ val: 0 }));
  }, [trainingInfo]);

  const currentGPS = trainingInfo?.games_per_second || 0;

  return (
    <div className="flex flex-col bg-background/80 border border-white/10 p-4 rounded-none shadow-lg backdrop-blur-md w-full max-w-sm">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          <Activity className={`w-5 h-5 ${isTraining ? 'text-primary animate-pulse' : 'text-muted-foreground'}`} />
          <span className="font-semibold text-sm uppercase tracking-wider text-muted-foreground">Engine Vitals</span>
        </div>
        <div className="text-right">
          <span className={`text-2xl font-black ${isTraining ? 'text-primary' : 'text-muted-foreground'}`}>
            {currentGPS.toFixed(1)}
          </span>
          <span className="text-xs text-muted-foreground ml-1">GPS</span>
        </div>
      </div>

      <div className="h-16 w-full mt-2 opacity-80 min-w-0 min-h-[64px] flex-1">
        <ResponsiveContainer width="99%" height={64}>
          <LineChart data={data}>
            <YAxis domain={['auto', 'auto']} hide />
            <Line
              type="monotone"
              dataKey="val"
              stroke={isTraining ? '#00fbfb' : '#374151'}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
