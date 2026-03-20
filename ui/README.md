# Tricked Svelte UI (`ui`)

## Abstract
The `ui` module is an aggressively optimized, "thin-client" Svelte 5 application. It strictly abides by an immutable data-flow architecture rendering the 96-node Triango hex-grid natively relying entirely on the `tricked_web` Python backend for heavy lifting and hardware AI orchestration.

## Reactive Svelte 5 Architecture
### `lib/state.svelte.ts`
Establishes the `$state()` primitive wrapper bounding all synchronous web operations. This module handles:
1. Long-polling interval synchronizations.
2. Direct binary evaluation of 128-bit `u128` string payloads passed from Rust, translating bitwise topology natively into SVG Hexagon arrays.

### `lib/math.ts`
Implements the inverse geometrical coordinate layouts resolving Flat-Topped hex mechanics into standard Cartesian `{x, y}` Cartesian matrices for rendering. Applies strict $2 \times R \times \cos(30^\circ)$ translation multipliers preventing pixel-bleed rendering.

## Component Segregation
- **`components/HexGrid.svelte`**: A pure stateless renderer iterating dynamically across arbitrary mask subsets.
- **`components/PieceTray.svelte`**: Evaluates Drag-and-Drop collision logic predictively before dispatching the final POST intent.
