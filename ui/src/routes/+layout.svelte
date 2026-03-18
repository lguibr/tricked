<script lang="ts">
	import '../app.css';
	let { children } = $props();
</script>

<div class="min-h-screen bg-zinc-950 text-zinc-100 antialiased selection:bg-emerald-500/30">
	{@render children()}
</div>
