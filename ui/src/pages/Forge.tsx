import { useState, useEffect } from 'react';
import { useEngineStore } from '@/store/useEngineStore';
import { LogarithmicSlider } from '@/components/forge/LogarithmicSlider';
import { ModelSizePills, ExplorationPills, ScalePills } from '@/components/forge/PresetPills';
import { ResourceEstimator } from '@/components/forge/ResourceEstimator';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Save, Layers, Network, Database } from 'lucide-react';
import { motion } from 'framer-motion';
import { generateExperimentName } from '@/lib/nameGenerator';

export function Forge() {
  const [nameEdited, setNameEdited] = useState(false);
  const isTraining = useEngineStore((s) => s.isTraining);
  const [config, setConfig] = useState<{
    device: string;
    hidden_dimension_size: number;
    num_blocks: number;
    support_size: number;
    buffer_capacity_limit: number;
    simulations: number;
    train_batch_size: number;
    train_epochs: number;
    num_processes: number;
    worker_device: string;
    unroll_steps: number;
    temporal_difference_steps: number;
    zmq_batch_size: number;
    zmq_timeout_ms: number;
    max_gumbel_k: number;
    gumbel_scale: number;
    temp_decay_steps: number;
    difficulty: number;
    temp_boost: boolean;
    experiment_name_identifier: string;
    lr_init: number;
  }>({
    device: 'cuda',
    hidden_dimension_size: 128,
    num_blocks: 4,
    support_size: 300,
    buffer_capacity_limit: 200000,
    simulations: 128,
    train_batch_size: 256,
    train_epochs: 4,
    num_processes: 32,
    worker_device: 'cpu',
    unroll_steps: 7,
    temporal_difference_steps: 10,
    zmq_batch_size: 16,
    zmq_timeout_ms: 5,
    max_gumbel_k: 8,
    gumbel_scale: 1.0,
    temp_decay_steps: 30,
    difficulty: 6,
    temp_boost: false,
    experiment_name_identifier: 'Headless-CUDA-Training',
    lr_init: 0.001,
  });

  const update = (key: keyof typeof config, val: any) => {
    setConfig((prev) => ({ ...prev, [key]: val }));
  };

  useEffect(() => {
    if (!nameEdited) {
      const newName = generateExperimentName(config.num_processes, config.simulations, config.hidden_dimension_size);
      if (newName !== config.experiment_name_identifier) {
        setConfig(prev => ({ ...prev, experiment_name_identifier: newName }));
      }
    }
  }, [config.num_processes, config.simulations, config.hidden_dimension_size, nameEdited, config.experiment_name_identifier]);

  const containerVariants: any = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.1 } },
  };

  const itemVariants: any = {
    hidden: { y: 20, opacity: 0 },
    visible: { y: 0, opacity: 1, transition: { type: 'spring', stiffness: 100 } },
  };

  return (
    <div className="flex flex-col min-h-[calc(100vh-4rem)] gap-8 p-8 max-w-7xl mx-auto w-full pb-20 relative">
      <div className="flex flex-wrap items-center justify-between sticky top-0 bg-background/60 backdrop-blur-3xl z-40 py-4 -my-4 mb-4 border-b border-border shadow-[0_4px_30px_rgba(0,0,0,0.5)] gap-4">
        <div className="flex items-center gap-6">
          <h2 className="text-4xl font-black tracking-tight text-white shadow-sm drop-shadow-md">
            THE FORGE
          </h2>
          <Input
            className="bg-background/80 border-border text-base font-bold text-primary w-[300px] h-auto py-1.5 focus-visible:ring-accent"
            value={config.experiment_name_identifier}
            disabled={isTraining}
            placeholder="Experiment Name"
            onChange={(e) => {
              setNameEdited(true);
              update('experiment_name_identifier', e.target.value);
            }}
          />
        </div>
        <div className="flex gap-4 items-center">
          <Button
            disabled={isTraining}
            onClick={async () => {
              const startTraining = useEngineStore.getState().startTraining;
              let finalExpName = config.experiment_name_identifier.trim();
              if (!finalExpName) {
                finalExpName = generateExperimentName(config.num_processes, config.simulations, config.hidden_dimension_size);
                setConfig(prev => ({ ...prev, experiment_name_identifier: finalExpName }));
              }
              try {
                await startTraining({ ...config, experiment_name_identifier: finalExpName });
              } catch (e) {
                console.error(e);
              }
            }}
            className="bg-primary hover:bg-primary/80 text-background font-bold shadow-[0_0_15px_rgba(0,251,251,0.5)] transition-all">
            <Save className="w-4 h-4 mr-2" /> Start Configuration & Train
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="lg:col-span-2 flex flex-col gap-8"
        >
          <motion.div
            variants={itemVariants}
            className="bg-muted border border-border p-6 rounded-2xl shadow-xl backdrop-blur-xl"
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2 border-b border-border pb-2 flex-grow">
                <Layers className="text-primary w-5 h-5" />
                <h3 className="font-bold text-xl text-white">Residual Architecture</h3>
              </div>
            </div>
            <ModelSizePills
              disabled={isTraining}
              onSelect={(newConfig: any) => {
                setNameEdited(false);
                setConfig((prev) => ({ ...prev, ...newConfig }));
              }}
            />
            <div className="space-y-8 mt-4">
              <div className="grid gap-2">
                <div className="flex justify-between items-center">
                  <div className="space-y-0.5">
                    <Label className="text-sm font-semibold text-muted-foreground">Embedding Dimension (d_model)</Label>
                    <p className="text-[10px] text-muted-foreground/60">Width of the hidden state channels</p>
                  </div>
                  <span className="text-sm font-mono text-primary font-bold">{config.hidden_dimension_size}</span>
                </div>
                <Slider disabled={isTraining}
                  min={32}
                  max={256}
                  step={32}
                  value={[config.hidden_dimension_size]}
                  onValueChange={(v: any) => update('hidden_dimension_size', v[0])}
                />
              </div>
              <div className="grid gap-2">
                <div className="flex justify-between items-center">
                  <div className="space-y-0.5">
                    <Label className="text-sm font-semibold text-muted-foreground">Residual Blocks</Label>
                    <p className="text-[10px] text-muted-foreground/60">Depth of the ResNet towers for representation & dynamics</p>
                  </div>
                  <span className="text-sm font-mono text-primary font-bold">{config.num_blocks}</span>
                </div>
                <Slider disabled={isTraining}
                  min={2}
                  max={24}
                  step={2}
                  value={[config.num_blocks]}
                  onValueChange={(v: any) => update('num_blocks', v[0])}
                />
              </div>
              <div className="grid gap-2">
                <div className="flex justify-between items-center">
                  <div className="space-y-0.5">
                    <Label className="text-sm font-semibold text-muted-foreground">Reward/Value Support Size</Label>
                    <p className="text-[10px] text-muted-foreground/60">Categorical representation limits for predictions</p>
                  </div>
                  <span className="text-sm font-mono text-primary font-bold">{config.support_size}</span>
                </div>
                <Slider disabled={isTraining}
                  min={10}
                  max={1000}
                  step={10}
                  value={[config.support_size]}
                  onValueChange={(v: any) => update('support_size', v[0])}
                />
              </div>
            </div>
          </motion.div>

          <motion.div
            variants={itemVariants}
            className="bg-muted border border-border p-6 rounded-2xl shadow-xl backdrop-blur-xl"
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2 border-b border-border pb-2 flex-grow">
                <Network className="text-accent w-5 h-5" />
                <h3 className="font-bold text-xl text-white">MCTS Search & Routing</h3>
              </div>
            </div>
            <ExplorationPills
              disabled={isTraining}
              onSelect={(newConfig: any) => {
                setNameEdited(false);
                setConfig((prev) => ({ ...prev, ...newConfig }));
              }}
            />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-8 mt-4">
              <LogarithmicSlider disabled={isTraining}
                label="Simulations"
                description="Number of MCTS search paths per action"
                min={16}
                max={4096}
                value={config.simulations}
                onChange={(v) => update('simulations', Math.round(v))}
              />
              <LogarithmicSlider disabled={isTraining}
                label="Buffer Capacity"
                description="Max trajectories in Prioritized Experience Replay"
                min={1000}
                max={1000000}
                value={config.buffer_capacity_limit}
                onChange={(v) => update('buffer_capacity_limit', Math.round(v))}
              />

              <div className="grid gap-2">
                <div className="flex justify-between items-center">
                  <div className="space-y-0.5">
                    <Label className="text-sm font-semibold text-muted-foreground">Unroll Steps</Label>
                    <p className="text-[10px] text-muted-foreground/60">Future temporal steps predicted</p>
                  </div>
                  <span className="text-sm font-mono text-accent font-bold">{config.unroll_steps}</span>
                </div>
                <Slider disabled={isTraining}
                  min={1}
                  max={20}
                  step={1}
                  value={[config.unroll_steps]}
                  onValueChange={(v: any) => update('unroll_steps', v[0])}
                />
              </div>
              <div className="grid gap-2">
                <div className="flex justify-between items-center">
                  <div className="space-y-0.5">
                    <Label className="text-sm font-semibold text-muted-foreground">TD Steps (Bootstrap)</Label>
                    <p className="text-[10px] text-muted-foreground/60">Value horizon bootstrapping</p>
                  </div>
                  <span className="text-sm font-mono text-accent font-bold">{config.temporal_difference_steps}</span>
                </div>
                <Slider disabled={isTraining}
                  min={1}
                  max={30}
                  step={1}
                  value={[config.temporal_difference_steps]}
                  onValueChange={(v: any) => update('temporal_difference_steps', v[0])}
                />
              </div>

              <div className="grid gap-2">
                <div className="flex justify-between items-center">
                  <div className="space-y-0.5">
                    <Label className="text-sm font-semibold text-muted-foreground">Gumbel Max K</Label>
                    <p className="text-[10px] text-muted-foreground/60">Top-K actions sampled during search</p>
                  </div>
                  <span className="text-sm font-mono text-accent font-bold">{config.max_gumbel_k}</span>
                </div>
                <Slider disabled={isTraining}
                  min={2}
                  max={32}
                  step={1}
                  value={[config.max_gumbel_k]}
                  onValueChange={(v: any) => update('max_gumbel_k', v[0])}
                />
              </div>
              <div className="grid gap-2">
                <div className="flex justify-between items-center">
                  <div className="space-y-0.5">
                    <Label className="text-sm font-semibold text-muted-foreground">Gumbel Scale</Label>
                    <p className="text-[10px] text-muted-foreground/60">Prior logit temperature scaling</p>
                  </div>
                  <span className="text-sm font-mono text-accent font-bold">{config.gumbel_scale.toFixed(1)}</span>
                </div>
                <Slider disabled={isTraining}
                  min={0.1}
                  max={5.0}
                  step={0.1}
                  value={[config.gumbel_scale]}
                  onValueChange={(v: any) => update('gumbel_scale', v[0])}
                />
              </div>

              <div className="flex items-center justify-between col-span-1 md:col-span-2 bg-background/40 p-4 rounded-lg border border-border">
                <div className="space-y-0.5">
                  <Label className="text-sm font-bold text-white">Temperature Boost</Label>
                  <p className="text-xs text-muted-foreground">Inject heavy exploration noise during early self-play</p>
                </div>
                <Switch disabled={isTraining} checked={config.temp_boost} onCheckedChange={(c) => update('temp_boost', c)} />
              </div>
            </div>
          </motion.div>

          <motion.div
            variants={itemVariants}
            className="bg-muted border border-border p-6 rounded-2xl shadow-xl backdrop-blur-xl"
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2 border-b border-border pb-2 flex-grow">
                <Database className="text-primary w-5 h-5" />
                <h3 className="font-bold text-xl text-white">System & I/O</h3>
              </div>
            </div>
            <ScalePills
              disabled={isTraining}
              onSelect={(newConfig: any) => {
                setNameEdited(false);
                setConfig((prev) => ({ ...prev, ...newConfig }));
              }}
            />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-8 mt-4">
              <div className="grid gap-2">
                <div className="space-y-0.5">
                  <Label className="text-sm font-semibold text-muted-foreground">Hardware Device (Model)</Label>
                  <p className="text-[10px] text-muted-foreground/60">Accelerator for neural net training & inference</p>
                </div>
                <Select disabled={isTraining} value={config.device} onValueChange={(v: any) => update('device', v)}>
                  <SelectTrigger className="bg-background/50 border-border">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="cuda">NVIDIA CUDA</SelectItem>
                    <SelectItem value="mps">Apple Metal (MPS)</SelectItem>
                    <SelectItem value="cpu">System CPU</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="grid gap-2">
                <div className="space-y-0.5">
                  <Label className="text-sm font-semibold text-muted-foreground">Worker Device (Simulations)</Label>
                  <p className="text-[10px] text-muted-foreground/60">Device used by concurrent MCTS threads</p>
                </div>
                <Select disabled={isTraining} value={config.worker_device} onValueChange={(v: any) => update('worker_device', v)}>
                  <SelectTrigger className="bg-background/50 border-border">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="cuda">NVIDIA CUDA</SelectItem>
                    <SelectItem value="mps">Apple Metal (MPS)</SelectItem>
                    <SelectItem value="cpu">System CPU</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <LogarithmicSlider disabled={isTraining}
                label="Dynamic ZMQ Batch Size"
                description="Max concurrent inferences per batch"
                min={1}
                max={2048}
                value={config.zmq_batch_size}
                onChange={(v) => update('zmq_batch_size', Math.round(v))}
              />
              <div className="grid gap-2">
                <div className="flex justify-between items-center">
                  <div className="space-y-0.5">
                    <Label className="text-sm font-semibold text-muted-foreground">ZMQ Timeout (ms)</Label>
                    <p className="text-[10px] text-muted-foreground/60">Max wait time for dynamic batch filling</p>
                  </div>
                  <span className="text-sm font-mono text-primary font-bold">{config.zmq_timeout_ms}</span>
                </div>
                <Slider disabled={isTraining}
                  min={1}
                  max={50}
                  step={1}
                  value={[config.zmq_timeout_ms]}
                  onValueChange={(v: any) => update('zmq_timeout_ms', v[0])}
                />
              </div>

              {/* Experiment name moved to header */}
            </div>
          </motion.div>
        </motion.div>

        <div className="flex flex-col gap-6 lg:sticky lg:top-24 self-start">
          <ResourceEstimator
            hidden_dimension_size={config.hidden_dimension_size}
            num_blocks={config.num_blocks}
            train_batch_size={config.train_batch_size || 256}
          />

          <div className="bg-muted border border-border p-6 rounded-2xl shadow-xl backdrop-blur-xl">
            <h3 className="font-bold text-lg text-white mb-6 border-b border-border pb-4">Training Hyperparameters</h3>
            <div className="space-y-8">
              <LogarithmicSlider disabled={isTraining}
                label="Learning Rate"
                description="Optimizer step size (Adam)"
                min={1e-5}
                max={1e-2}
                value={config.lr_init}
                onChange={(v) => update('lr_init', v)}
              />
              <LogarithmicSlider disabled={isTraining}
                label="Batch Size"
                description="Number of positions sampled per training step"
                min={64}
                max={2048}
                value={config.train_batch_size}
                onChange={(v) => update('train_batch_size', Math.round(v))}
              />
              <div className="grid gap-2">
                <div className="flex justify-between items-center">
                  <div className="space-y-0.5">
                    <Label className="text-sm font-semibold text-muted-foreground">Training Epochs</Label>
                    <p className="text-[10px] text-muted-foreground/60">Passes over the experience replay buffer per generation</p>
                  </div>
                  <span className="text-sm font-mono text-primary font-bold">{config.train_epochs}</span>
                </div>
                <Slider disabled={isTraining}
                  min={1}
                  max={50}
                  step={1}
                  value={[config.train_epochs]}
                  onValueChange={(v: any) => update('train_epochs', v[0])}
                />
              </div>
              <div className="grid gap-2">
                <div className="flex justify-between items-center">
                  <div className="space-y-0.5">
                    <Label className="text-sm font-semibold text-muted-foreground">Num Processes (Workers)</Label>
                    <p className="text-[10px] text-muted-foreground/60">Concurrent self-play threads generating games</p>
                  </div>
                  <span className="text-sm font-mono text-primary font-bold">{config.num_processes}</span>
                </div>
                <Slider disabled={isTraining}
                  min={0}
                  max={7}
                  step={1}
                  value={[Math.max(0, Math.log2(config.num_processes || 1))]}
                  onValueChange={(v: any) => update('num_processes', Math.pow(2, v[0]))}
                />
              </div>
            </div>
          </div>

          <div className="flex-1 bg-black/40 border border-primary/20 rounded-none p-6 relative shadow-[0_0_30px_rgba(0,251,251,0.1)] shrink-0 flex flex-col group">
            <div className="flex items-center justify-between mb-4 border-b border-white/10 pb-4">
              <h4 className="font-semibold text-sm uppercase tracking-wider text-primary flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                Hydra Payload
              </h4>
              <Button
                variant="ghost"
                size="sm"
                className="text-primary hover:text-white transition-colors h-8 rounded-none border border-primary/20"
                onClick={() => {
                  navigator.clipboard.writeText(JSON.stringify(config, null, 2));
                  const el = document.getElementById('copy-icon');
                  if (el) {
                    el.innerHTML = '<path d="M20 6L9 17l-5-5"/>'; // Checkmark
                    setTimeout(
                      () =>
                      (el.innerHTML =
                        '<rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>'),
                      2000,
                    ); // Copy
                  }
                }}
              >
                <svg
                  id="copy-icon"
                  xmlns="http://www.w3.org/2000/svg"
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
                  <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
                </svg>
                <span className="ml-2 font-mono text-xs hidden sm:inline">COPY JSON</span>
              </Button>
            </div>
            <pre className="text-xs font-mono text-muted-foreground bg-background/50 p-4 rounded-none border border-white/5 overflow-x-auto whitespace-pre-wrap flex-1 overflow-y-auto custom-scrollbar">
              {JSON.stringify(config, null, 2)
                .split('\n')
                .map((line, i) => (
                  <div key={i} className="leading-snug">
                    <span className="text-white/20 select-none mr-4 pr-4 border-r border-white/10 inline-block w-8 text-right">
                      {(i + 1).toString()}
                    </span>
                    <span className={line.includes(':') ? 'text-accent' : 'text-primary'}>{line}</span>
                  </div>
                ))}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}
