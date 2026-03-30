import { Button } from '@/components/ui/button';

export type PresetConfig = Partial<{
  hidden_dimension_size: number;
  num_blocks: number;
  support_size: number;
  learning_rate: number;
  simulations: number;
  max_gumbel_k: number;
  unroll_steps: number;
  num_processes: number;
  train_batch_size: number;
  zmq_batch_size: number;
  lr_init: number;
}>;

export type Preset = {
  name: string;
  config: PresetConfig;
};

const MODEL_PRESETS: Preset[] = [
  { name: 'Tiny', config: { hidden_dimension_size: 32, num_blocks: 2, support_size: 100 } },
  { name: 'Small', config: { hidden_dimension_size: 64, num_blocks: 4, support_size: 200 } },
  { name: 'Medium', config: { hidden_dimension_size: 128, num_blocks: 8, support_size: 300 } },
  { name: 'Large', config: { hidden_dimension_size: 256, num_blocks: 12, support_size: 300 } },
  { name: 'Huge', config: { hidden_dimension_size: 512, num_blocks: 16, support_size: 600 } },
  { name: 'Gargantuan', config: { hidden_dimension_size: 1024, num_blocks: 24, support_size: 1000 } },
];

const EXPLORATION_PRESETS: Preset[] = [
  { name: 'Hasty', config: { simulations: 16, max_gumbel_k: 2, unroll_steps: 3 } },
  { name: 'Cautious', config: { simulations: 64, max_gumbel_k: 4, unroll_steps: 5 } },
  { name: 'Brave', config: { simulations: 128, max_gumbel_k: 8, unroll_steps: 7 } },
  { name: 'Wise', config: { simulations: 512, max_gumbel_k: 16, unroll_steps: 10 } },
  { name: 'Omniscient', config: { simulations: 2048, max_gumbel_k: 32, unroll_steps: 15 } },
];

const SCALE_PRESETS: Preset[] = [
  { name: 'Solo', config: { num_processes: 1, train_batch_size: 64, zmq_batch_size: 1, lr_init: 5e-3 } },
  { name: 'Pair', config: { num_processes: 2, train_batch_size: 128, zmq_batch_size: 2, lr_init: 2e-3 } },
  { name: 'Squad', config: { num_processes: 8, train_batch_size: 256, zmq_batch_size: 8, lr_init: 1e-3 } },
  { name: 'Swarm', config: { num_processes: 32, train_batch_size: 512, zmq_batch_size: 32, lr_init: 5e-4 } },
  { name: 'Horde', config: { num_processes: 128, train_batch_size: 1024, zmq_batch_size: 64, lr_init: 3e-4 } },
  { name: 'Legion', config: { num_processes: 512, train_batch_size: 2048, zmq_batch_size: 256, lr_init: 1e-4 } },
];

export function PresetPillsGroup({
  presets,
  onSelect,
  disabled
}: {
  presets: Preset[];
  onSelect: (config: PresetConfig) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex flex-wrap gap-2 mt-4 mb-2">
      {presets.map((preset) => (
        <Button
          key={preset.name}
          variant="outline"
          size="sm"
          onClick={() => onSelect(preset.config)}
          disabled={disabled}
          className="border-primary/30 text-foreground hover:bg-primary/20 hover:text-white transition-colors h-7 px-3 text-xs"
        >
          {preset.name}
        </Button>
      ))}
    </div>
  );
}

export function ModelSizePills({ onSelect, disabled }: { onSelect: (config: PresetConfig) => void; disabled?: boolean }) {
  return <PresetPillsGroup presets={MODEL_PRESETS} onSelect={onSelect} disabled={disabled} />;
}

export function ExplorationPills({ onSelect, disabled }: { onSelect: (config: PresetConfig) => void; disabled?: boolean }) {
  return <PresetPillsGroup presets={EXPLORATION_PRESETS} onSelect={onSelect} disabled={disabled} />;
}

export function ScalePills({ onSelect, disabled }: { onSelect: (config: PresetConfig) => void; disabled?: boolean }) {
  return <PresetPillsGroup presets={SCALE_PRESETS} onSelect={onSelect} disabled={disabled} />;
}
