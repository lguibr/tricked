import { Button } from '@/components/ui/button';

export type Preset = {
  name: string;
  config: {
    d_model: number;
    num_blocks: number;
    learning_rate: number;
    batch_size: number;
  };
};

const PRESETS: Preset[] = [
  { name: 'Tiny / Debug', config: { d_model: 64, num_blocks: 4, learning_rate: 1e-3, batch_size: 128 } },
  { name: 'Standard Run', config: { d_model: 256, num_blocks: 12, learning_rate: 3e-4, batch_size: 1024 } },
  { name: 'Massive Compute', config: { d_model: 512, num_blocks: 24, learning_rate: 1e-4, batch_size: 4096 } },
];

export function PresetPills({ onSelect }: { onSelect: (config: Preset['config']) => void }) {
  return (
    <div className="flex flex-wrap gap-3">
      {PRESETS.map((preset) => (
        <Button
          key={preset.name}
          variant="outline"
          size="sm"
          onClick={() => onSelect(preset.config)}
          className="border-primary/50 text-foreground hover:bg-primary/20 hover:text-white transition-colors"
        >
          {preset.name}
        </Button>
      ))}
    </div>
  );
}
