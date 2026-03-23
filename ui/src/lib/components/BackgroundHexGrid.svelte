<script lang="ts">
	import { getRowCol, isUp, getPoints, getBoardBit } from '$lib/math';

	let { topGames = [] } = $props<{ topGames: any[] }>();
</script>

<div
	class="fixed inset-0 min-h-screen w-full -z-10 bg-background overflow-hidden flex flex-wrap content-start pointer-events-none opacity-40"
>
	{#each topGames as g}
		<div
			class="w-1/4 sm:w-1/6 md:w-[12.5%] xl:w-[10%] aspect-square flex items-center justify-center p-2 opacity-30 hover:opacity-100 transition-opacity"
		>
			<svg
				viewBox="-300 -300 600 600"
				class="w-full h-full drop-shadow-[0_0_12px_var(--color-secondary)]"
			>
				<g>
					{#each Array(96) as _, i}
						{@const [r, c] = getRowCol(i)}
						{@const up = isUp(r, c)}
						{@const points = getPoints(r, c, up)}
						{#if g.board && getBoardBit(String(g.board), i)}
							<polygon {points} class="fill-secondary/60 stroke-secondary/30 stroke-[1px]" />
						{:else}
							<polygon {points} class="fill-transparent stroke-outline-variant/10 stroke-[0.5px]" />
						{/if}
					{/each}
				</g>
			</svg>
		</div>
	{/each}
</div>
