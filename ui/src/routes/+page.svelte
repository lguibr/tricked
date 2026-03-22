<script lang="ts">
    import { onMount } from 'svelte';
    import { goto } from '$app/navigation';

    onMount(() => {
        goto('/compute');
    });
</script>

<div class="min-h-screen flex items-center justify-center p-8">
    <div class="text-primary-fixed animate-pulse text-xl font-bold font-headline uppercase tracking-widest glow-cyan">
        Initializing Engine Core...
    </div>
</div>
