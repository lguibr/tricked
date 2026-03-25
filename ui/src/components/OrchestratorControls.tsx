import { Play, Pause, PlaySquare, ExternalLink } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useEngineStore } from '@/store/useEngineStore';

export function OrchestratorControls() {
  const isTraining = useEngineStore((state) => state.isTraining);
  const startTraining = useEngineStore((state) => state.startTraining);
  const pauseTraining = useEngineStore((state) => state.pauseTraining);
  const resumeTraining = useEngineStore((state) => state.resumeTraining);

  return (
    <div className="flex flex-col space-y-4 bg-background/80 border border-white/10 p-6 rounded-none shadow-lg backdrop-blur-md w-full max-w-sm">
      <h3 className="font-semibold text-sm uppercase tracking-wider text-muted-foreground mb-2">Orchestrator</h3>

      <div className="grid grid-cols-2 gap-3">
        {!isTraining ? (
          <Button
            onClick={() => startTraining({})}
            className="w-full bg-primary hover:bg-primary/90 text-background font-bold h-12 shadow-[0_0_15px_rgba(0,251,251,0.3)] transition-all"
          >
            <Play className="mr-2 h-5 w-5" /> START
          </Button>
        ) : (
          <Button
            onClick={pauseTraining}
            variant="destructive"
            className="w-full font-bold h-12 shadow-[0_0_15px_rgba(239,68,68,0.3)] transition-all"
          >
            <Pause className="mr-2 h-5 w-5" /> PAUSE
          </Button>
        )}

        <Button
          onClick={resumeTraining}
          disabled={isTraining}
          variant="outline"
          className="w-full font-bold h-12 border-border/80 hover:bg-secondary/10"
        >
          <PlaySquare className="mr-2 h-5 w-5" /> RESUME
        </Button>
      </div>

      <Button
        variant="secondary"
        className="w-full mt-4 h-10 font-semibold bg-muted hover:bg-muted/80 text-foreground"
        onClick={() => window.open('http://localhost:8081', '_blank')}
      >
        <ExternalLink className="mr-2 h-4 w-4" /> Open W&B Dashboard
      </Button>
    </div>
  );
}
