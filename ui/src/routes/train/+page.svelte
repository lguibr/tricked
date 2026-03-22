<script lang="ts">
    import { engine } from '$lib/state.svelte';
    import ExperimentOrchestrator from '$lib/components/ExperimentOrchestrator.svelte';
    import ControlsBar from '$lib/components/ControlsBar.svelte';
</script>

<svelte:head>
    <title>Tricked: Orchestrator Control</title>
</svelte:head>

<div class="p-6 md:p-10 min-h-full flex flex-col items-center">
    
    <!-- Title / Header -->
    <div class="w-full max-w-5xl mb-10 border-b border-outline-variant/30 pb-6 flex justify-between items-end">
        <div>
            <div class="flex items-center gap-3 mb-2">
                <span class="material-symbols-outlined text-primary-fixed" style="font-variation-settings: 'FILL' 1;">insights</span>
                <h1 class="text-3xl font-headline font-bold text-on-surface uppercase tracking-tight glow-cyan">Orchestrator Control Center</h1>
            </div>
            <p class="text-sm font-body text-on-surface-variant max-w-2xl leading-relaxed">
                Tune hyperparameters, launch experiments, and manage training state.
            </p>
        </div>
        
        <div class="text-right">
            <p class="text-[10px] font-mono text-primary-fixed tracking-widest uppercase mb-1">Session Status</p>
            {#if engine.loading}
                <div class="text-secondary animate-pulse text-sm font-bold font-headline uppercase tracking-widest">
                    Awaiting Connection...
                </div>
            {:else}
                <div class="text-emerald-500 font-mono text-sm max-w-[200px] truncate" title={engine.trainingInfo?.run_id || 'IDLE'}>
                    [RUN] {engine.trainingInfo?.run_id || 'IDLE'}
                </div>
            {/if}
        </div>
    </div>

    <!-- Main Content Grid -->
    {#if !engine.loading}
        <div class="w-full max-w-5xl flex flex-col gap-8">
            <ExperimentOrchestrator />
            
            <div class="bg-surface-container border border-outline-variant/30 p-6 relative corner-accent">
                <h2 class="text-sm font-headline uppercase tracking-widest text-primary-fixed mb-6 border-l-2 border-primary-fixed pl-3 glow-cyan">Global Controls</h2>
                <ControlsBar />
            </div>
        </div>
    {/if}
</div>
