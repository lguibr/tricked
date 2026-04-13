// Central HTTP client bridging React to Python SOTA State Machine Backend

const API_BASE = "http://127.0.0.1:8000/api";

export async function fetchRuns() {
  const res = await fetch(`${API_BASE}/runs`);
  if (!res.ok) throw new Error("Failed to fetch runs");
  return res.json();
}

export async function fetchMetrics(runId: string) {
  const res = await fetch(`${API_BASE}/runs/${runId}/metrics`);
  if (!res.ok) throw new Error("Failed to fetch metrics");
  return res.json();
}

export async function startRun(runId: string) {
  const res = await fetch(`${API_BASE}/runs/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: runId }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Failed to start run");
  }
}

export async function stopRun(runId: string, force: boolean = false) {
  const res = await fetch(`${API_BASE}/runs/stop`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: runId, force }),
  });
  if (!res.ok) throw new Error("Failed to stop run");
}

export async function startStudy(runId: string) {
  const res = await fetch(`${API_BASE}/studies/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: runId }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Failed to start study");
  }
}
