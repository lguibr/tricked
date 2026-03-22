<script lang="ts">
    import { engine } from '$lib/state.svelte';
    import VaultDataGrid from '$lib/components/VaultDataGrid.svelte';
    import PieceBuffer from '$lib/components/PieceBuffer.svelte';
</script>

<svelte:head>
    <title>Tricked: Historic Replay Vault</title>
</svelte:head>

<div class="p-6 md:p-10 min-h-full flex flex-col items-center">
    
    <!-- Title / Header -->
    <div class="w-full max-w-7xl mb-10 border-b border-outline-variant/30 pb-6 flex justify-between items-end">
        <div>
            <div class="flex items-center gap-3 mb-2">
                <span class="material-symbols-outlined text-primary-fixed" style="font-variation-settings: 'FILL' 1;">inventory_2</span>
                <h1 class="text-3xl font-headline font-bold text-on-surface uppercase tracking-tight glow-cyan">Historic Replay Vault</h1>
            </div>
            <p class="text-sm font-body text-on-surface-variant max-w-2xl leading-relaxed">
                Review and analyze past self-play trajectories, value targets, and piece transitions.
            </p>
        </div>
    </div>

    <!-- Main Content -->
    <div class="w-full max-w-7xl flex flex-col items-center gap-10">
        <VaultDataGrid />
        <PieceBuffer />
    </div>
</div>
