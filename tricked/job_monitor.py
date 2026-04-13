import asyncio
import psutil
import json

class JobMonitor:
    def __init__(self, pm):
        self.pm = pm
        self._running = False
        self.latest_jobs = []
        self._proc_cache = {}

    async def start(self):
        self._running = True
        # Run poll in background thread to avoid blocking hotpath EVENT LOOP
        while self._running:
            await asyncio.to_thread(self._poll)
            await asyncio.sleep(2.5)

    def stop(self):
        self._running = False

    def _poll(self):
        jobs = []
        # Clear out dead processes from cache to prevent memory leaks
        alive_pids = set()
        
        if self.pm.active_run:
            pid = self.pm.active_run["pid"]
            try:
                if psutil.pid_exists(pid):
                    if pid not in self._proc_cache:
                        self._proc_cache[pid] = psutil.Process(pid)
                    root = self._proc_cache[pid]
                    root_info = self._get_proc_info(root, alive_pids)
                    if root_info:
                        jobs.append({
                            "id": self.pm.active_run.get("run_id", "active"),
                            "name": self.pm.active_run.get("run_id", "active"),
                            "job_type": self.pm.active_run.get("type", "UNKNOWN"),
                            "root_process": root_info
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
                
        # Cleanup cache
        self._proc_cache = {p: self._proc_cache[p] for p in alive_pids if p in self._proc_cache}
                
        # Bulk fetch GPU PID memory
        gpu_pids = {}
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
            for p in gpu_procs:
                gpu_pids[p.pid] = p.usedGpuMemory / 1024 / 1024 if p.usedGpuMemory else 0.0
        except:
            pass
            
        def _enrich_gpu(node):
            if node["pid"] in gpu_pids:
                node["vram_mb"] = gpu_pids[node["pid"]]
            for child in node.get("children", []):
                _enrich_gpu(child)
                
        for job in jobs:
            _enrich_gpu(job["root_process"])
            
        self.latest_jobs = jobs

    def _get_proc_info(self, proc: psutil.Process, alive_pids: set):
        try:
            with proc.oneshot():
                pid = proc.pid
                alive_pids.add(pid)
                name = proc.name()
                status = proc.status()
                # Because proc is cached across loops, this now accurately yields >0 usage!
                cpu_usage = proc.cpu_percent()
                mem = proc.memory_info().rss / 1024 / 1024
                cmdline = proc.cmdline()
                
            children = []
            for child in proc.children():
                if child.pid not in self._proc_cache:
                    self._proc_cache[child.pid] = child
                c_info = self._get_proc_info(self._proc_cache[child.pid], alive_pids)
                if c_info:
                    children.append(c_info)
                    
            return {
                "pid": pid,
                "name": name,
                "status": status,
                "cpu_usage": cpu_usage,
                "memory_mb": mem,
                "cmd": cmdline,
                "children": children
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return None
