<script>
    import { onMount, onDestroy } from 'svelte';
    
    let stats = {
        models_generations: 12,
        self_play_games: 10420,
        gpu_util_percent: 94,
        vram_util_percent: 88,
        throughput_pos_s: 4200
    };
</script>

<div class="p-6 md:p-10 min-h-full">
    <!-- Header Section -->
    <header class="mb-10 flex flex-col md:flex-row justify-between items-start md:items-center gap-6 border-b border-outline-variant/30 pb-6">
        <div>
            <div class="flex items-center gap-3 mb-2">
                <span class="material-symbols-outlined text-primary-fixed" style="font-variation-settings: 'FILL' 1;">developer_board</span>
                <h1 class="text-3xl font-headline font-bold text-on-surface uppercase tracking-tight glow-cyan">Engine Core</h1>
            </div>
            <p class="text-sm font-body text-on-surface-variant max-w-2xl leading-relaxed">
                Centralized monitoring for the Tricked AI compute cluster. Real-time visualization of MCTS self-play throughput and hardware utilization.
            </p>
        </div>
        
        <div class="flex items-stretch gap-4">
            <div class="bg-surface-container/50 border border-primary-fixed/30 p-4 min-w-[140px] asymmetric-clip backdrop-blur-sm">
                <p class="text-[10px] text-primary-fixed uppercase tracking-widest font-bold mb-1">Status</p>
                <div class="flex items-center gap-2">
                    <span class="relative flex h-3 w-3">
                        <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                        <span class="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
                    </span>
                    <span class="text-on-surface font-mono text-sm">ONLINE</span>
                </div>
            </div>
        </div>
    </header>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        <!-- Global Iteration Pipeline -->
        <section class="lg:col-span-2">
            <h2 class="text-sm font-headline uppercase tracking-widest text-primary-fixed mb-6 border-l-2 border-primary-fixed pl-3">Global Iteration Pipeline</h2>
            
            <div class="relative py-8">
                <!-- Connecting Line -->
                <div class="absolute top-1/2 left-0 w-full h-[2px] bg-outline-variant/30 -translate-y-1/2 z-0 hidden md:block"></div>
                <div class="absolute top-1/2 left-0 w-[60%] h-[2px] bg-primary-fixed -translate-y-1/2 z-0 hidden md:block glow-cyan"></div>
                
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6 relative z-10">
                    
                    <!-- Hex Node: Self-Play -->
                    <div class="relative group">
                        <div class="bg-surface-container aspect-square clip-hex border border-primary-fixed/50 flex flex-col items-center justify-center p-6 text-center transition-all duration-300 hover:bg-primary-fixed/5">
                            <span class="material-symbols-outlined text-primary-fixed text-4xl mb-3 shadow-glow" data-icon="sports_esports">sports_esports</span>
                            <h3 class="font-headline font-bold text-on-surface text-lg">SELF-PLAY</h3>
                            <p class="text-xs text-on-surface-variant mt-2 font-mono">{stats.self_play_games} Games</p>
                        </div>
                        <div class="absolute -bottom-10 left-1/2 -translate-x-1/2 flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                            <span class="w-2 h-2 rounded-full bg-primary-fixed animate-pulse"></span>
                            <span class="text-[10px] font-mono text-primary-fixed">ACTIVE</span>
                        </div>
                    </div>
                    
                    <!-- Hex Node: Train -->
                    <div class="relative group">
                        <div class="bg-surface-container aspect-square clip-hex border border-primary-fixed/50 flex flex-col items-center justify-center p-6 text-center transition-all duration-300 hover:bg-primary-fixed/5">
                            <span class="material-symbols-outlined text-primary-fixed text-4xl mb-3 shadow-glow" data-icon="model_training">model_training</span>
                            <h3 class="font-headline font-bold text-on-surface text-lg">TRAIN</h3>
                            <p class="text-xs text-on-surface-variant mt-2 font-mono">Iteration {stats.models_generations}</p>
                        </div>
                        <div class="absolute -bottom-10 left-1/2 -translate-x-1/2 flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                            <span class="w-2 h-2 rounded-full bg-primary-fixed animate-pulse"></span>
                            <span class="text-[10px] font-mono text-primary-fixed">ACTIVE</span>
                        </div>
                    </div>
                    
                    <!-- Hex Node: Eval -->
                    <div class="relative group">
                        <div class="bg-surface-container aspect-square clip-hex border border-outline-variant/30 flex flex-col items-center justify-center p-6 text-center transition-all duration-300 opacity-60">
                            <span class="material-symbols-outlined text-on-surface-variant text-4xl mb-3" data-icon="query_stats">query_stats</span>
                            <h3 class="font-headline font-bold text-on-surface text-lg">EVAL</h3>
                            <p class="text-xs text-on-surface-variant mt-2 font-mono">Pending</p>
                        </div>
                        <div class="absolute -bottom-10 left-1/2 -translate-x-1/2 flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                            <span class="w-2 h-2 rounded-full bg-outline-variant"></span>
                            <span class="text-[10px] font-mono text-on-surface-variant">WAITING</span>
                        </div>
                    </div>
                    
                </div>
            </div>
            
            <!-- Real-time Logs / Output -->
            <div class="mt-8 bg-[#0a0c10] border border-outline-variant/20 h-48 relative overflow-hidden flex flex-col">
                <div class="scanline absolute inset-0 z-10 pointer-events-none opacity-30"></div>
                <div class="bg-surface-container-highest px-4 py-2 flex justify-between items-center border-b border-outline-variant/30 z-20">
                    <span class="text-[10px] font-mono text-primary-fixed tracking-widest uppercase">system_stdout</span>
                    <span class="material-symbols-outlined text-sm text-on-surface-variant">terminal</span>
                </div>
                <div class="p-4 font-mono text-xs text-on-surface-variant overflow-y-auto space-y-1 z-20">
                    <p><span class="text-emerald-500">[10:42:01]</span> MCTS Root expanded. Generating 800 trajectories...</p>
                    <p><span class="text-emerald-500">[10:42:02]</span> GPU_0 allocated for batch inference (1024 samples)</p>
                    <p><span class="text-emerald-500">[10:42:03]</span> Experience Buffer appending 120 new states.</p>
                    <p><span class="text-primary-fixed">[10:42:04]</span> Self-Play Game #10420 completed. Winner: Player 1</p>
                    <p class="text-on-surface-variant/50 animate-pulse">Awaiting next batch...</p>
                </div>
            </div>
        </section>
        
        <!-- Resource Allocation -->
        <section class="space-y-6">
            <h2 class="text-sm font-headline uppercase tracking-widest text-primary-fixed mb-6 border-l-2 border-primary-fixed pl-3">Resource Allocation</h2>
            
            <!-- Overall Throughput -->
            <div class="bg-surface-container-low border border-outline-variant/30 p-5 relative corner-accent">
                <span class="text-[10px] font-mono text-on-surface-variant uppercase tracking-widest">Throughput</span>
                <div class="flex items-baseline gap-2 mt-2">
                    <span class="text-4xl font-headline font-bold text-on-surface">{stats.throughput_pos_s}</span>
                    <span class="text-xs text-primary-fixed font-mono glow-cyan">pos/s</span>
                </div>
            </div>

            <!-- GPU Panel -->
            <div class="bg-surface-container border border-primary-fixed/20 p-5 relative overflow-hidden">
                <div class="absolute top-0 right-0 w-16 h-16 bg-primary-fixed/5 blur-xl rounded-full"></div>
                
                <div class="flex justify-between items-center mb-4">
                    <div class="flex items-center gap-2">
                        <span class="material-symbols-outlined text-primary-fixed text-sm">memory</span>
                        <h3 class="font-headline text-sm font-bold tracking-wider text-on-surface uppercase">RTX 3080 Ti</h3>
                    </div>
                </div>
                
                <div class="space-y-4">
                    <!-- GPU Util -->
                    <div>
                        <div class="flex justify-between text-xs font-mono mb-1">
                            <span class="text-on-surface-variant">Compute Util</span>
                            <span class="text-primary-fixed">{stats.gpu_util_percent}%</span>
                        </div>
                        <div class="w-full h-1 bg-surface-container-highest overflow-hidden">
                            <div class="h-full bg-primary-fixed glow-cyan transition-all duration-1000" style="width: {stats.gpu_util_percent}%"></div>
                        </div>
                    </div>
                    
                    <!-- VRAM Util -->
                    <div>
                        <div class="flex justify-between text-xs font-mono mb-1">
                            <span class="text-on-surface-variant">VRAM Alloc</span>
                            <span class="text-error">{stats.vram_util_percent}%</span>
                        </div>
                        <div class="w-full h-1 bg-surface-container-highest overflow-hidden">
                            <div class="h-full bg-error drop-shadow-[0_0_5px_rgba(255,84,73,0.8)] transition-all duration-1000" style="width: {stats.vram_util_percent}%"></div>
                        </div>
                    </div>
                </div>
            </div>
            
        </section>
        
    </div>
</div>

<style>
    /* Specific overrides if needed, most handled by global app.css */
    .shadow-glow {
        text-shadow: 0 0 10px var(--color-primary-fixed);
    }
</style>
