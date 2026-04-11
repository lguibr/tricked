# UI & Backend Deduplication Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor massive React UI components and the Tauri `runs.rs` backend to eliminate duplicate Echarts rendering logic, consolidate configuration schemas, and organize giant boilerplate files into clean module directories. We will seamlessly incorporate the ECharts "Interactive Bar Pivot" and cross-graph hover synchronization features from our earlier designs into the new modular design.

**Architecture:** 
1. **Frontend Charts**: We use Option A to extract the Echarts rendering loop (`setTimeout` polling), the "double-click to pin bar chart" morph features, and the global hover synchronization handlers into a shared `useChartSync` custom React hook + utility file.
2. **Frontend Configs**: Extract duplicated categorization strings, mappings, and UI badges into `lib/config-schema.ts` and a `ConfigBadge.tsx` component.
3. **Frontend Sidebar**: Break down the massive 780-line `sidebar.tsx` Shadcn layout file into a structured `sidebar/` directory.
4. **Backend**: Decompose `runs.rs` into a `runs/` module isolating lifecycle operations from vault reads.

**Tech Stack:** React, ECharts, Tauri, Rust, Tailwind CSS.

---

### Task 1: Tauri Backend Module Split

**Files:**
- Create: `control_center/src-tauri/src/commands/runs/mod.rs`
- Create: `control_center/src-tauri/src/commands/runs/lifecycle.rs`
- Create: `control_center/src-tauri/src/commands/runs/metrics.rs`
- Create: `control_center/src-tauri/src/commands/runs/vault.rs`
- Modify: `control_center/src-tauri/src/commands/runs.rs` (Delete/Clear)
- Modify: `control_center/src-tauri/src/main.rs` (Update bindings)

**Step 1: Create the runs module skeleton**
Create the inner module directory. Set up `mod.rs` to expose the `@tauri::command` functions publicly.

**Step 2: Move Lifecycle Logic**
Move `create_run`, `rename_run`, `delete_run`, `flush_run`, `list_runs`, and `sync_run_states` into `lifecycle.rs`. Ensure database queries are properly imported.

**Step 3: Move Ancillary Logic**
Move `get_vault_games` to `vault.rs`.
Move `get_run_metrics` to `metrics.rs`.
Fix visibility inside the `mod.rs` file.

**Step 4: Run Rust Check**
Run: `cargo check --manifest-path control_center/src-tauri/Cargo.toml`
Expected: PASS

**Step 5: Commit**
```bash
git add control_center/src-tauri/src/commands/
git commit -m "refactor: split runs.rs into lifecycle, vault, and metrics modules"
```

---

### Task 2: Chart Logic Hook & Pivot Utilities Extraction (Option A)

**Files:**
- Create: `control_center/src/hooks/useChartSync.ts`
- Create: `control_center/src/lib/chart-utils.ts`
- Modify: `control_center/src/components/execution/LossStackedArea.tsx`
- Modify: `control_center/src/components/dashboard/MetricChart.tsx`

**Step 1: Extract Base Utilities & Morph Rules**
In `chart-utils.ts`, extract the pure functions: `hexToHSL`, `getXAxisConfigurator` (the time/step math logic heavily duplicated in both files), and the underlying logic that computes `getPinnedSnapshots` (which morphs line slice data into aggregated bar data).

**Step 2: Create Custom Hook for Pinned Pivot & Sync**
In `useChartSync.ts`, implement the `useChartSync` hook. It must incorporate the exact Echarts Bar Pivot pattern:
- Call `echarts.connect("metricsGroup")` on mount to link graphs.
- Manage internal `useState` for `pinnedTime`.
- Handle the `timeoutId` polling loop that reads `metricsDataRef.current` and triggers `instance.setOption` (dynamically switching between line/bar arrays based on `pinnedTime`).
- Provide an `onEvents` dictionary for `dblclick` (to freeze/unfreeze time) and `mouseover`/`mouseout` (to dispatch global 'highlight' actions to sync bar hover states across all connected graphs simultaneously).
- Ensure `option.tooltip` is silenced or explicitly minimized (e.g. `show: false`, `axisPointer: { type: 'cross' }`) to prevent DOM lag as planned.

**Step 3: Refactor the Components**
Update `LossStackedArea.tsx` and `MetricChart.tsx` to delete all of their internal `useEffect` render blocks, math utilities, and raw `onEvents` objects.
Replace them with a call to `const { chartRef, pinnedTime, getPinnedSnapshots, chartEvents } = useChartSync(...)`. Pass `chartEvents` into the `<ReactECharts onEvents={chartEvents} />` prop.

**Step 4: Run Typecheck**
Run: `cd control_center && npm run typecheck`
Expected: PASS

**Step 5: Commit**
```bash
git add control_center/src/hooks/ control_center/src/lib/ control_center/src/components/
git commit -m "refactor: deduplicate Echarts rendering and bar pivot interactions via useChartSync hook"
```

---

### Task 3: Config Schema & Badge Consolidation

**Files:**
- Create: `control_center/src/lib/config-schema.ts`
- Create: `control_center/src/components/execution/ConfigBadge.tsx`
- Modify: `control_center/src/components/execution/HydraConfigViewer.tsx`
- Modify: `control_center/src/components/execution/CreateSimpleRunSidebar.tsx`

**Step 1: Extract Schema Truth**
In `config-schema.ts`, move `EXPLANATIONS`, `GROUP_THEMES`, `categorizeParams()`, and `getIcon()` out of `HydraConfigViewer`. Make them exported utilities.

**Step 2: Extract visual badge component**
In `ConfigBadge.tsx`, extract the `Tooltip` and `div` rendering of the key-value pair that generates the small colored squares in the viewers.

**Step 3: Update Viewers**
Refactor `HydraConfigViewer` and `CompactTrialParams` to import `categorizeParams` and iterate over `ConfigBadge` components.
Refactor `CreateSimpleRunSidebar` to derive its groups/icons from `config-schema.ts` rather than hardcoding static array layouts if possible.

**Step 4: Run Linter**
Run: `cd control_center && npm run lint`
Expected: PASS

**Step 5: Commit**
```bash
git add control_center/src/lib/ control_center/src/components/
git commit -m "refactor: extract unified tuning parameter schema and badge UI"
```

---

### Task 4: UI Shadcn Sidebar Component Split

**Files:**
- Create: `control_center/src/components/ui/sidebar/index.ts`
- Create: `control_center/src/components/ui/sidebar/context.tsx`
- Create: `control_center/src/components/ui/sidebar/provider.tsx`
- Create: `control_center/src/components/ui/sidebar/parts.tsx`
- Modify: `control_center/src/components/ui/sidebar.tsx` (Delete)

**Step 1: Setup Context**
Move `SidebarContext`, `SidebarContextProps`, and `useSidebar` hook to `context.tsx`.

**Step 2: Setup Provider**
Move the massive `SidebarProvider` block to `provider.tsx`, ensuring it imports the newly split contextual hook.

**Step 3: Setup View Parts**
Move `Sidebar`, `SidebarTrigger`, `SidebarRail`, `SidebarHeader`, `SidebarContent`, `SidebarGroup`, `SidebarMenu`, etc., into `parts.tsx`. 

**Step 4: Centralized Extraction**
In `index.ts`, re-export all components exactly as they were available before:
`export * from "./context"; export * from "./provider"; export * from "./parts";`
Update any generic references to `import { Sidebar } from "@/components/ui/sidebar"` to utilize the new directory structure.

**Step 5: Final Check**
Run: `cd control_center && npm run build`
Expected: PASS

**Step 6: Commit**
```bash
git add control_center/src/components/ui/
git commit -m "refactor: decompose 800-line shadcn sidebar into logical sub-modules"
```
