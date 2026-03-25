import { useState, useEffect } from 'react';
import { Slider } from '@/components/ui/slider';
import { useEngineStore } from '@/store/useEngineStore';
import { Play, Pause, SkipBack, SkipForward } from 'lucide-react';
import { Button } from '@/components/ui/button';

export function ReplayScrubber() {
  const activeReplayData = useEngineStore((state) => state.activeReplayData);
  const replayCursor = useEngineStore((state) => state.replayCursor);
  const setReplayCursor = useEngineStore((state) => state.setReplayCursor);
  const [isPlaying, setIsPlaying] = useState(false);

  const maxSteps = activeReplayData?.steps?.length || 100;

  useEffect(() => {
    let rafId: number;
    let lastTime = 0;
    const stepInterval = 500;

    const loop = (time: number) => {
      if (!lastTime) lastTime = time;
      const delta = time - lastTime;

      if (delta >= stepInterval) {
        lastTime = time;
        setReplayCursor((c: number) => {
          if (c >= maxSteps - 1) {
            setIsPlaying(false);
            return c;
          }
          return c + 1;
        });
      }

      if (isPlaying) {
        rafId = requestAnimationFrame(loop);
      }
    };

    if (isPlaying) {
      rafId = requestAnimationFrame(loop);
    }

    return () => {
      if (rafId) cancelAnimationFrame(rafId);
    };
  }, [isPlaying, maxSteps, setReplayCursor]);

  return (
    <div className="flex flex-col gap-4 w-full bg-background border border-border/50 p-6 rounded-xl shadow-lg">
      <div className="flex items-center justify-between">
        <h4 className="font-semibold text-sm uppercase tracking-wider text-muted-foreground">Scrubber Timeline</h4>
        <span className="font-mono font-bold text-primary drop-shadow-[0_0_5px_rgba(0,251,251,0.5)]">
          Step {replayCursor} / {maxSteps}
        </span>
      </div>

      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2">
          <Button variant="outline" size="icon" onClick={() => setReplayCursor(0)} className="h-8 w-8">
            <SkipBack className="h-4 w-4" />
          </Button>
          <Button
            variant="default"
            size="icon"
            onClick={() => setIsPlaying(!isPlaying)}
            className="h-10 w-10 bg-primary hover:bg-primary/80 text-background shadow-[0_0_10px_rgba(0,251,251,0.3)] transition-all"
          >
            {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5 ml-1" />}
          </Button>
          <Button variant="outline" size="icon" onClick={() => setReplayCursor(maxSteps - 1)} className="h-8 w-8">
            <SkipForward className="h-4 w-4" />
          </Button>
        </div>

        <div className="flex-1 px-4">
          <Slider
            min={0}
            max={maxSteps - 1}
            step={1}
            value={[replayCursor]}
            onValueChange={(val) => {
              setIsPlaying(false);
              setReplayCursor(Array.isArray(val) ? val[0] : (val as number));
            }}
            className="cursor-pointer"
          />
        </div>
      </div>
    </div>
  );
}
