import { AlertCircle, CheckCircle2, Zap } from 'lucide-react';

interface EstimatorProps {
  d_model: number;
  num_blocks: number;
  batch_size: number;
}

export function ResourceEstimator({ d_model, num_blocks, batch_size }: EstimatorProps) {
  // Rough VRAM estimation heuristic
  const params = d_model * d_model * 12 * num_blocks;
  const bytesPerParam = 4;
  const optimizerMultiplier = 3;

  const modelVram = (params * bytesPerParam * optimizerMultiplier) / 1024 ** 3;
  const activationVram = (batch_size * 1024 * d_model * num_blocks * 2) / 1024 ** 3;
  const estimatedGB = Math.max(0.5, modelVram + activationVram);

  const isWarning = estimatedGB > 22;
  const isCritical = estimatedGB > 78;

  let stateColor = 'text-green-400';
  let Icon = CheckCircle2;
  let message = 'Fits comfortably in 24GB consumer GPUs.';

  if (isCritical) {
    stateColor = 'text-red-500';
    Icon = AlertCircle;
    message = 'Exceeds 80GB limits. Requires multi-GPU setup.';
  } else if (isWarning) {
    stateColor = 'text-yellow-400';
    Icon = Zap;
    message = 'Nearing 24GB limits. May OOM on RTX 3090/4090.';
  }

  return (
    <div
      className={`p-5 border rounded-none bg-background/50 backdrop-blur-md shadow-lg transition-colors duration-300 ${isCritical ? 'border-red-500/50' : isWarning ? 'border-yellow-400/50' : 'border-green-400/50'}`}
    >
      <div className="flex items-center justify-between mb-3">
        <h4 className="font-semibold text-sm uppercase tracking-wider text-muted-foreground flex items-center gap-2">
          <Icon className={`w-5 h-5 ${stateColor}`} />
          Estimated VRAM Footprint
        </h4>
        <span className={`font-mono text-xl tracking-tight font-black ${stateColor}`}>
          ~{estimatedGB.toFixed(1)} GB
        </span>
      </div>
      <div className="h-2 w-full bg-black/60 rounded-full overflow-hidden mt-4 mb-1 border border-white/5">
        <div
          className={`h-full transition-all duration-500 ease-out ${isCritical ? 'bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.8)]' : isWarning ? 'bg-yellow-400 shadow-[0_0_10px_rgba(250,204,21,0.8)]' : 'bg-green-400 shadow-[0_0_10px_rgba(74,222,128,0.8)]'}`}
          style={{ width: `${Math.min(100, (estimatedGB / 24) * 100)}%` }}
        />
      </div>
      <div className="flex justify-between text-[10px] text-muted-foreground/50 mb-4 px-1 font-mono">
        <span>0GB</span>
        <span>24GB (Consumer Limit)</span>
      </div>
      <p className="text-sm text-foreground/80 leading-relaxed font-medium">{message}</p>
    </div>
  );
}
