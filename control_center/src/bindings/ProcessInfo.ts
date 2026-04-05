export type ProcessInfo = {
  pid: number;
  name: string;
  status: string;
  cpu_usage: number;
  memory_mb: number;
  cmd: string[];
  children: ProcessInfo[];
};

export type ActiveJob = {
  id: string;
  name: string;
  job_type: string;
  root_process: ProcessInfo | null;
};
