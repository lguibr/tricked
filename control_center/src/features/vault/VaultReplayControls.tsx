import { VscPlay, VscDebugPause, VscTriangleLeft, VscTriangleRight } from "react-icons/vsc";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";

export function VaultReplayControls({
  currentStep,
  setCurrentStep,
  isPlaying,
  setIsPlaying,
  totalSteps,
}: {
  currentStep: number;
  setCurrentStep: React.Dispatch<React.SetStateAction<number>>;
  isPlaying: boolean;
  setIsPlaying: (p: boolean) => void;
  totalSteps: number;
}) {
  return (
    <div className="h-24 border-t border-white/5 bg-black p-4 flex flex-col gap-3 shrink-0">
      <div className="flex-1 flex items-center gap-4">
        <div className="flex gap-1 shrink-0">
          <Button
            variant="outline"
            size="icon"
            className="h-8 w-8 bg-white/5 border-white/10"
            onClick={() => setCurrentStep((prev) => Math.max(0, prev - 1))}
            disabled={currentStep === 0}
          >
            <VscTriangleLeft className="text-zinc-400 w-4 h-4" />
          </Button>

          <Button
            variant="outline"
            size="icon"
            className="h-8 w-8 bg-emerald-500/10 border-emerald-500/20 hover:bg-emerald-500/20"
            onClick={() => setIsPlaying(!isPlaying)}
          >
            {isPlaying ? (
              <VscDebugPause className="text-emerald-400 w-4 h-4" />
            ) : (
              <VscPlay className="text-emerald-400 w-4 h-4" />
            )}
          </Button>

          <Button
            variant="outline"
            size="icon"
            className="h-8 w-8 bg-white/5 border-white/10"
            onClick={() => setCurrentStep((prev) => Math.min(totalSteps - 1, prev + 1))}
            disabled={currentStep === totalSteps - 1}
          >
            <VscTriangleRight className="text-zinc-400 w-4 h-4" />
          </Button>
        </div>

        <div className="flex-1 px-4">
          <Slider
            value={[currentStep]}
            min={0}
            max={totalSteps - 1}
            step={1}
            onValueChange={(v) => {
              setIsPlaying(false);
              setCurrentStep(v[0]);
            }}
            className="cursor-pointer"
          />
        </div>
        <div className="text-[10px] uppercase font-mono font-bold text-zinc-500 w-12 text-right">
          {currentStep + 1}
          <span className="opacity-50">/{totalSteps}</span>
        </div>
      </div>
    </div>
  );
}
