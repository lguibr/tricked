<script lang="ts">
	import { getRowCol, isUp, getPoints, getBoardBit } from '$lib/math';

	let { topGames } = $props<{ topGames: any[] }>();
</script>

<div
	class="fixed inset-0 min-h-screen w-full -z-10 bg-[#060606] overflow-hidden flex flex-wrap content-start pointer-events-none opacity-20"
>
	{#each topGames as g}
		<div class="w-1/4 sm:w-1/6 md:w-[12.5%] aspect-square flex items-center justify-center p-1">
			<svg
				viewBox="-300 -300 600 600"
				class="w-[85%] h-[85%] drop-shadow-[0_0_8px_rgba(16,185,129,0.4)]"
			>
				<g>
					{#each Array(96) as _, i}
						{@const [r, c] = getRowCol(i)}
						{@const up = isUp(r, c)}
						{@const points = getPoints(r, c, up)}
						{#if g.board && getBoardBit(String(g.board), i)}
							<polygon {points} class="fill-emerald-500/80 stroke-emerald-900/40 stroke-[1px]" />
						{/if}
					{/each}
				</g>
			</svg>
		</div>
	{/each}
</div>
