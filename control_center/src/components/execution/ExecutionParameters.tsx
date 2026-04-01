import { Field, FieldGroup, FieldLabel, FieldSet, FieldDescription } from '@/components/ui/field';
import { Slider } from '@/components/ui/slider';

interface ExecutionParametersProps {
    selectedRun: any;
    getConfigValue: (key: string, fallback: any) => any;
    updateConfigKey: (key: string, value: any) => void;
}

export function ExecutionParameters({ selectedRun, getConfigValue, updateConfigKey }: ExecutionParametersProps) {
    const isTuning = selectedRun?.type === 'TUNING';

    return (
        <FieldSet>
            <FieldGroup className="gap-6">
                <div className="mb-2 p-2 bg-blue-500/10 border border-blue-500/20 rounded-md">
                    <p className="text-[10px] text-blue-400">
                        {isTuning
                            ? "TUNING MODE: Set the [MIN, MAX] boundaries for the hyperparameter search space."
                            : "SINGLE MODE: Set absolute scaler values for the engine to execute."}
                    </p>
                </div>
                <Field>
                    <div className="flex justify-between w-full mb-1">
                        <FieldLabel>MCTS Simulations</FieldLabel>
                        <span className="text-[10px] font-mono text-muted-foreground">
                            {isTuning
                                ? `${getConfigValue('simulations_range', [16, 800])[0]} - ${getConfigValue('simulations_range', [16, 800])[1]}`
                                : getConfigValue('simulations', 800)}
                        </span>
                    </div>
                    <Slider className="py-2"
                        value={isTuning ? getConfigValue('simulations_range', [16, 800]) : [getConfigValue('simulations', 800)]}
                        onValueChange={(vals) => isTuning ? updateConfigKey('simulations_range', vals) : updateConfigKey('simulations', vals[0])}
                        min={10} max={2000} step={10} disabled={selectedRun?.status === 'RUNNING'}
                    />
                    <FieldDescription className="text-[9px] mt-1 text-zinc-500">How many times the AI plays out future moves in its 'imagination' per turn. Higher values lead to dramatically stronger play but linearly increase search time.</FieldDescription>
                </Field>
                <Field>
                    <div className="flex justify-between w-full mb-1">
                        <FieldLabel>Gumbel C_Visit (max_gumbel_k)</FieldLabel>
                        <span className="text-[10px] font-mono text-muted-foreground">
                            {isTuning
                                ? `${getConfigValue('max_gumbel_k_range', [4, 16])[0]} - ${getConfigValue('max_gumbel_k_range', [4, 16])[1]}`
                                : getConfigValue('max_gumbel_k', 16)}
                        </span>
                    </div>
                    <Slider className="py-2"
                        value={isTuning ? getConfigValue('max_gumbel_k_range', [4, 16]) : [getConfigValue('max_gumbel_k', 16)]}
                        onValueChange={(vals) => isTuning ? updateConfigKey('max_gumbel_k_range', vals) : updateConfigKey('max_gumbel_k', vals[0])}
                        min={4} max={64} step={1} disabled={selectedRun?.status === 'RUNNING'}
                    />
                    <FieldDescription className="text-[9px] mt-1 text-zinc-500">The number of top moves tracked by the Gumbel Alpha Zero algorithm. A higher cap increases diversity in self-play but slows down sequential completion scaling.</FieldDescription>
                </Field>
                <Field>
                    <div className="flex justify-between w-full mb-1">
                        <FieldLabel>Learning Rate (lr_init)</FieldLabel>
                        <span className="text-[10px] font-mono text-muted-foreground">
                            {isTuning
                                ? `${getConfigValue('lr_init_range', [0.0005, 0.05])[0]} - ${getConfigValue('lr_init_range', [0.0005, 0.05])[1]}`
                                : getConfigValue('lr_init', 0.02)}
                        </span>
                    </div>
                    <Slider className="py-2"
                        value={isTuning ? getConfigValue('lr_init_range', [0.0005, 0.05]) : [getConfigValue('lr_init', 0.02)]}
                        onValueChange={(vals) => isTuning ? updateConfigKey('lr_init_range', vals) : updateConfigKey('lr_init', vals[0])}
                        min={0.0001} max={0.1} step={0.0001} disabled={selectedRun?.status === 'RUNNING'}
                    />
                    <FieldDescription className="text-[9px] mt-1 text-zinc-500">The step size for gradient descent. If the loss explodes, lower this. If learning plateaus too early, it might be too small.</FieldDescription>
                </Field>
                <Field>
                    <div className="flex justify-between w-full mb-1">
                        <FieldLabel>Worker Threads (num_processes)</FieldLabel>
                        <span className="text-[10px] font-mono text-muted-foreground">
                            {isTuning
                                ? `${getConfigValue('num_processes_range', [4, 16])[0]} - ${getConfigValue('num_processes_range', [4, 16])[1]}`
                                : getConfigValue('num_processes', 8)}
                        </span>
                    </div>
                    <Slider className="py-2"
                        value={isTuning ? getConfigValue('num_processes_range', [4, 16]) : [getConfigValue('num_processes', 8)]}
                        onValueChange={(vals) => isTuning ? updateConfigKey('num_processes_range', vals) : updateConfigKey('num_processes', vals[0])}
                        min={1} max={128} step={1} disabled={selectedRun?.status === 'RUNNING'}
                    />
                    <FieldDescription className="text-[9px] mt-1 text-zinc-500">How many concurrent self-play games run in parallel. Too many might starve the GPU batching inference queue; too few underutilizes hardware.</FieldDescription>
                </Field>
                <Field>
                    <div className="flex justify-between w-full mb-1">
                        <FieldLabel>Neural Net Blocks (num_blocks)</FieldLabel>
                        <span className="text-[10px] font-mono text-muted-foreground">
                            {isTuning
                                ? `${getConfigValue('num_blocks_range', [4, 10])[0]} - ${getConfigValue('num_blocks_range', [4, 10])[1]}`
                                : getConfigValue('num_blocks', 10)}
                        </span>
                    </div>
                    <Slider className="py-2"
                        value={isTuning ? getConfigValue('num_blocks_range', [4, 10]) : [getConfigValue('num_blocks', 10)]}
                        onValueChange={(vals) => isTuning ? updateConfigKey('num_blocks_range', vals) : updateConfigKey('num_blocks', vals[0])}
                        min={2} max={30} step={1} disabled={selectedRun?.status === 'RUNNING'}
                    />
                    <FieldDescription className="text-[9px] mt-1 text-zinc-500">Number of ResNet blocks in the architecture. Deep networks (15+) learn better representations but are significantly slower during MCTS rollouts.</FieldDescription>
                </Field>
                <Field>
                    <div className="flex justify-between w-full mb-1">
                        <FieldLabel>Train Batch Size</FieldLabel>
                        <span className="text-[10px] font-mono text-muted-foreground">
                            {isTuning
                                ? `${getConfigValue('train_batch_size_range', [256, 1024])[0]} - ${getConfigValue('train_batch_size_range', [256, 1024])[1]}`
                                : getConfigValue('train_batch_size', 1024)}
                        </span>
                    </div>
                    <Slider className="py-2"
                        value={isTuning ? getConfigValue('train_batch_size_range', [256, 1024]) : [getConfigValue('train_batch_size', 1024)]}
                        onValueChange={(vals) => isTuning ? updateConfigKey('train_batch_size_range', vals) : updateConfigKey('train_batch_size', vals[0])}
                        min={64} max={4096} step={64} disabled={selectedRun?.status === 'RUNNING'}
                    />
                    <FieldDescription className="text-[9px] mt-1 text-zinc-500">The amount of replay buffer samples consumed per training step. Larger batches offer more stable gradients but quickly exhaust limited VRAM.</FieldDescription>
                </Field>
            </FieldGroup>
        </FieldSet>
    );
}
