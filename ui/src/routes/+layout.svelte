<script lang="ts">
    import '../app.css';
    import { page } from '$app/stores';
    import { onMount, onDestroy } from 'svelte';
    import { engine } from '$lib/state.svelte';
    import EpochTelemetry from '$lib/components/EpochTelemetry.svelte';
    
    onMount(() => {
        engine.mount();
    });

    onDestroy(() => {
        engine.unmount();
    });
</script>

<div class="bg-surface-container-lowest text-on-surface font-body selection:bg-primary-container selection:text-on-primary-fixed overflow-hidden flex flex-col h-screen">
    <!-- TopAppBar -->
    <header class="fixed top-0 w-full border-b-0 bg-surface-dim/60 backdrop-blur-xl z-50 flex justify-between items-center px-6 h-16 shadow-[0_4px_20px_rgba(0,0,0,0.8)]">
        <div class="flex items-center gap-4">
            <span class="material-symbols-outlined text-primary-fixed" style="font-variation-settings: 'FILL' 0;">terminal</span>
            <h1 class="font-headline uppercase tracking-widest font-bold text-primary-fixed drop-shadow-[0_0_8px_rgba(0,255,255,0.5)] text-xl">ORCHESTRATOR_V2.0</h1>
        </div>
        <div class="flex items-center gap-6">
            <nav class="hidden md:flex gap-8 items-center">
                <a href="/compute" class="font-headline text-sm uppercase tracking-widest font-bold transition-all cursor-crosshair {$page.url.pathname === '/compute' ? 'text-primary-fixed-dim' : 'text-on-surface-variant hover:text-primary-fixed hover:bg-primary-fixed/10 px-2 py-1'}">COMPUTE</a>
                <a href="/train" class="font-headline text-sm uppercase tracking-widest font-bold transition-all cursor-crosshair {$page.url.pathname === '/train' ? 'text-primary-fixed-dim' : 'text-on-surface-variant hover:text-primary-fixed hover:bg-primary-fixed/10 px-2 py-1'}">TRAIN</a>
                <a href="/eval" class="font-headline text-sm uppercase tracking-widest font-bold transition-all cursor-crosshair {$page.url.pathname === '/eval' ? 'text-primary-fixed-dim' : 'text-on-surface-variant hover:text-primary-fixed hover:bg-primary-fixed/10 px-2 py-1'}">EVAL</a>
                <a href="/vault" class="font-headline text-sm uppercase tracking-widest font-bold transition-all cursor-crosshair {$page.url.pathname === '/vault' ? 'text-primary-fixed-dim' : 'text-on-surface-variant hover:text-primary-fixed hover:bg-primary-fixed/10 px-2 py-1'}">VAULT</a>
            </nav>
            <div class="h-10 w-10 bg-surface-container-high border border-outline-variant flex items-center justify-center">
                <span class="text-xs font-mono text-primary-fixed">UA_ID</span>
            </div>
        </div>
        <div class="bg-gradient-to-r from-transparent via-primary-fixed/30 to-transparent h-[1px] w-full absolute bottom-0 left-0"></div>
    </header>

    <div class="flex flex-1 pt-16 pb-20 md:pb-0 overflow-hidden">
        <!-- Main Content Slot -->
        <main class="flex-1 relative overflow-y-auto hexa-grid">
            <slot />
        </main>
    </div>

    <!-- BottomNavBar (Mobile Only) -->
    <nav class="lg:hidden fixed bottom-0 left-0 w-full z-50 bg-surface-dim/80 backdrop-blur-md flex justify-around items-center h-20 shadow-[0_-10px_30px_rgba(0,255,255,0.1)] border-t border-primary-fixed/10">
        <a href="/compute" class="flex flex-col items-center justify-center transition-transform active:scale-95 {$page.url.pathname === '/compute' ? 'text-primary-fixed drop-shadow-[0_0_5px_rgba(0,255,255,0.8)]' : 'text-on-surface-variant hover:text-primary-fixed'}">
            <span class="material-symbols-outlined mb-1" data-icon="developer_board">developer_board</span>
            <span class="font-headline text-[10px] font-bold tracking-widest uppercase">Compute</span>
        </a>
        <a href="/train" class="flex flex-col items-center justify-center transition-transform active:scale-95 {$page.url.pathname === '/train' ? 'text-primary-fixed drop-shadow-[0_0_5px_rgba(0,255,255,0.8)]' : 'text-on-surface-variant hover:text-primary-fixed'}">
            <span class="material-symbols-outlined mb-1" data-icon="model_training">model_training</span>
            <span class="font-headline text-[10px] font-bold tracking-widest uppercase">Train</span>
        </a>
        <a href="/eval" class="flex flex-col items-center justify-center transition-transform active:scale-95 {$page.url.pathname === '/eval' ? 'text-primary-fixed drop-shadow-[0_0_5px_rgba(0,255,255,0.8)]' : 'text-on-surface-variant hover:text-primary-fixed'}">
            <span class="material-symbols-outlined mb-1" data-icon="query_stats">query_stats</span>
            <span class="font-headline text-[10px] font-bold tracking-widest uppercase">Eval</span>
        </a>
        <a href="/vault" class="flex flex-col items-center justify-center transition-transform active:scale-95 {$page.url.pathname === '/vault' ? 'text-primary-fixed drop-shadow-[0_0_5px_rgba(0,255,255,0.8)]' : 'text-on-surface-variant hover:text-primary-fixed'}">
            <span class="material-symbols-outlined mb-1" data-icon="inventory_2">inventory_2</span>
            <span class="font-headline text-[10px] font-bold tracking-widest uppercase">Vault</span>
        </a>
    </nav>

    <EpochTelemetry />
</div>
