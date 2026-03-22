<script lang="ts">
    import { engine } from '$lib/state.svelte';
    import TriangoBoard from '$lib/components/TriangoBoard.svelte';
    import VisualCognitionDash from '$lib/components/VisualCognitionDash.svelte';
</script>

<svelte:head>
    <title>Tricked: Visual Cognition</title>
</svelte:head>

<div class="p-6 md:p-10 min-h-full flex flex-col items-center">
    
    <!-- Title / Header -->
    <div class="w-full max-w-7xl mb-10 border-b border-outline-variant/30 pb-6 flex justify-between items-end">
        <div>
            <div class="flex items-center gap-3 mb-2">
                <span class="material-symbols-outlined text-primary-fixed" style="font-variation-settings: 'FILL' 1;">query_stats</span>
                <h1 class="text-3xl font-headline font-bold text-on-surface uppercase tracking-tight glow-cyan">Visual Cognition Dashboard</h1>
            </div>
            <p class="text-sm font-body text-on-surface-variant max-w-2xl leading-relaxed">
                Live monitoring of MCTS traversals, Neural Network evaluations, and game state.
            </p>
        </div>
        
        <div class="text-right">
            <p class="text-[10px] font-mono text-primary-fixed tracking-widest uppercase mb-1">Board State</p>
            <div class="text-on-surface font-mono text-sm max-w-[200px]">
                Score: <span class="text-secondary">{engine.gameState?.score ?? 0}</span> | 
                Pieces: <span class="text-primary-fixed">{engine.gameState?.pieces_left ?? 0}</span>
            </div>
        </div>
    </div>

    <!-- Main Content -->
    <div class="w-full max-w-7xl flex flex-col items-center">
        <!-- The Board and its related neural visualizations are wrapped here -->
        <TriangoBoard />
        
        <div class="w-full mt-10">
            <VisualCognitionDash />
        </div>
    </div>
</div>
