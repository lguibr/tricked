# Tricked UI Engine

This directory contains the user interface components and experiments orchestrator built with SvelteKit and TypeScript.

## Structure

- `components/`: UI pieces including Hex Grid visualizations, metric graphs (`VaultDataGrid`), and real-time trackers.
- `routes/`: Standard Next.js/Svelte style routing definitions for landing and vault screens.
- `lib/`: Standard state bindings pushing socket data into reactive maps (`state.svelte.ts`).

## Real-time Requirements

Updates are driven asynchronously from the main loop via websockets connected to the `tricked_web` server, mapping underlying Redis changes to fast DOM updates.
