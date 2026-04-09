import { describe, it, expect } from "vitest";
import type { ActiveJob } from "../../bindings/ActiveJob";
import type { ProcessInfo } from "../../bindings/ProcessInfo";

// Copied logic from CpuSunburstChart to test it
const mapProcessLogic = (jobs: ActiveJob[]) => {
  const idSet = new Set<string>();
  return jobs
    .map((job) => {
      const color = "#3b82f6";

      const mapProcess = (proc: ProcessInfo): any => {
        const selfCpu = Math.max(0.1, Number(proc.cpu_usage) || 0);
        const baseName = proc.name || "Unknown";
        const nodeName = `${baseName} [${proc.pid}]`;

        let currentId = `${job.id}-${proc.pid}`;
        let counter = 1;
        while (idSet.has(currentId)) {
          currentId = `${job.id}-${proc.pid}-${counter}`;
          counter++;
        }
        idSet.add(currentId);

        const children =
          proc.children
            ?.filter((c) => c.cpu_usage > 0.1 || c.children.length > 0)
            .map((c) => mapProcess(c)) || [];

        if (children.length === 0) {
          return {
            id: currentId,
            name: nodeName,
            value: selfCpu,
            itemStyle: { color: color },
          };
        }

        const selfNodeId = `${currentId}-self`;
        idSet.add(selfNodeId); // Just to be logically safe

        const selfNode = {
          id: selfNodeId,
          name: "Self",
          value: selfCpu,
          itemStyle: { color: color },
        };

        return {
          id: currentId,
          name: nodeName,
          itemStyle: { color: color },
          children: [selfNode, ...children],
        };
      };

      return job.root_process ? mapProcess(job.root_process) : null;
    })
    .filter((d) => d && (d.value > 0 || (d.children && d.children.length > 0)));
};

function getAllIds(nodes: any[], ids: Set<string> = new Set()) {
  for (const node of nodes) {
    if (ids.has(node.id)) {
      throw new Error(`Duplicate ID found: ${node.id}`);
    }
    ids.add(node.id);
    if (node.children) {
      getAllIds(node.children, ids);
    }
  }
  return ids;
}

describe("CpuSunburstChart Data Mapper", () => {
  it("should never produce duplicate IDs even with bad process tree", () => {
    const badJobs: ActiveJob[] = [
      {
        id: "job_1",
        name: "Test",
        job_type: "Test",
        root_process: {
          pid: 100,
          name: "root",
          status: "R",
          cpu_usage: 1.0,
          memory_mb: 10,
          cmd: [],
          children: [
            {
              pid: 100, // DUPLICATE PID!
              name: "thread1",
              status: "R",
              cpu_usage: 2.0,
              memory_mb: 10,
              cmd: [],
              children: [],
            },
            {
              pid: 101,
              name: "thread2",
              status: "R",
              cpu_usage: 3.0,
              memory_mb: 10,
              cmd: [],
              children: [
                {
                  pid: 100, // ANOTHER DUPLICATE PID DEEP IN TREE
                  name: "thread3",
                  status: "R",
                  cpu_usage: 1.0,
                  memory_mb: 10,
                  cmd: [],
                  children: [],
                },
              ],
            },
          ],
        },
      },
      {
        id: "job_1", // DUPLICATE JOB ID
        name: "Test",
        job_type: "Test",
        root_process: {
          pid: 100, // SAME PID
          name: "root",
          status: "R",
          cpu_usage: 1.0,
          memory_mb: 10,
          cmd: [],
          children: [],
        },
      },
    ];

    const result = mapProcessLogic(badJobs);

    // Will throw if there is a duplicate ID
    expect(() => getAllIds(result)).not.toThrow();
  });
});
