# ECharts Interactive Hover & Timestamp Bar Pivot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Completely eliminate massive DOM tooltip lag by silencing native ECharts tooltips in favor of a sleek Axis Pointer, and introduce a "Double Click to Pin" interaction that dynamically morphs line graphs into synchronized, high-density bar charts at the hovered timestamp.

**Architecture:** We will disable `option.tooltip` globally across the heavy analytics charts. We will bind `onEvents` to handle `dblclick`. Upon double clicking the graph, we freeze the X-axis time, extract the cross-sectional slice of all active series data, and overwrite the `echartsInstance.setOption` to convert the `type` from `line` to `bar`, rotating the X axis to `category` (Run identifiers). When rendering the bar chart, hovering a specific bar will dispatch an ECharts `highlight` action linked via `metricsGroup` to synchronize hover states across all connected graphs simultaneously.

**Tech Stack:** React, Zustand, echarts-for-react, ECharts grouping API

---

### Task 1: Silence Tooltips & Connect Chart Groups

**Files:**
- Modify: `control_center/src/components/execution/LossStackedArea.tsx`
- Modify: `control_center/src/components/dashboard/MetricChart.tsx`

**Step 1:** In both files, locate `initialOptions.tooltip` and set `show: false`.
**Step 2:** Ensure `axisPointer: { type: 'cross' }` (or similar) is kept to retain visual indexing.
**Step 3:** Mount a `useEffect` inside a primary parent (or inside the charts individually if safely deduped) that calls `echarts.connect('metricsGroup')` so that interactions naturally bleed horizontally.

### Task 2: Build the Double-Click Timestamp Interception

**Files:**
- Modify: `control_center/src/components/execution/LossStackedArea.tsx` (Use as the primary implementation target, replicate for `MetricChart.tsx` if logic holds).

**Step 1:** Create local state: `const [pinnedTime, setPinnedTime] = useState<number | null>(null);`
**Step 2:** Use the `ReactECharts` `onEvents` prop to intercept `dblclick`:
```typescript
const onEvents = {
  dblclick: (params: any) => {
    // If we click an empty area or an axis, capture the X value.
    const timeX = params.value ? params.value[0] : null;
    if (timeX) {
      setPinnedTime(timeX);
    } else {
      setPinnedTime(null); // click again to clear
    }
  }
};
```

### Task 3: Morph Line Series to Sorted Bar Series

**Files:**
- Modify: `control_center/src/components/execution/LossStackedArea.tsx`

**Step 1:** Inside `getSeries()` or `renderLoop()`, check if `pinnedTime` is active.
**Step 2:** If active, compute the exact interpolated value for each run at `pinnedTime`. Sort the runs by descending value.
**Step 3:** Instead of mapping over `smoothedData.map((d) => [d.xVal, d.val])`, construct a single ECharts `bar` series. The X-axis must dynamically swap to `type: "category"` containing the sorted Run IDs.
**Step 4:** Ensure the items explicitly pass `itemStyle: { color: runColors[id] }` so the bars natively map to the correct visually identifiable run.

### Task 4: Hook Cross-Graph Hover Highlighting

**Files:**
- Modify: `control_center/src/components/execution/LossStackedArea.tsx`

**Step 1:** Update `onEvents` to intercept `mouseover` and `mouseout` events when `pinnedTime` is active.
**Step 2:**
```typescript
mouseover: (params: any) => {
  if (pinnedTime && params.componentType === 'series') {
     const runId = params.name; // assuming X-axis category is the Run ID
     // Dispatch ECharts action to highlight runId natively across 'metricsGroup'
     chartRef.current?.getEchartsInstance().dispatchAction({
         type: 'highlight',
         seriesName: runId
     });
  }
}
```
**Step 3:** Ensure `mouseout` dispatches a `downplay` action properly to release the highlight lock.
