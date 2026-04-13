import asyncio
import psutil
import pynvml
import time
from tricked.proto_out.tricked_pb2 import HardwareMetrics

class HardwareMonitor:
    def __init__(self):
        self.state = HardwareMetrics(
            cpu_usage=0.0,
            cpu_cores_usage=[],
            ram_usage_pct=0.0,
            ram_used_mb=0.0,
            gpu_util=0.0,
            vram_used_mb=0.0,
            disk_usage_pct=0.0,
            network_rx_mbps=0.0,
            network_tx_mbps=0.0,
            disk_read_mbps=0.0,
            disk_write_mbps=0.0,
            machine_identifier="Python Orchestrator"
        )
        self._running = False
        try:
            pynvml.nvmlInit()
            self.nvml_initialized = True
        except:
            self.nvml_initialized = False

        self.last_net = psutil.net_io_counters()
        self.last_disk = psutil.disk_io_counters()
        self.last_time = time.time()

    async def start(self):
        self._running = True
        while self._running:
            await self._poll()
            await asyncio.sleep(1)

    def stop(self):
        self._running = False
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except: pass

    async def _poll(self):
        # CPU
        cpu_usage = psutil.cpu_percent()
        cpu_cores = psutil.cpu_percent(percpu=True)
        
        # RAM
        mem = psutil.virtual_memory()
        ram_used_mb = mem.used / 1024 / 1024
        ram_pct = mem.percent
        
        # Disk IO
        disk_pct = psutil.disk_usage('/').percent
        
        now = time.time()
        dt = now - self.last_time
        if dt <= 0: dt = 0.0001
        self.last_time = now
        
        net = psutil.net_io_counters()
        disk = psutil.disk_io_counters()
        
        rx_mbps = ((net.bytes_recv - self.last_net.bytes_recv) / dt) / 1024 / 1024
        tx_mbps = ((net.bytes_sent - self.last_net.bytes_sent) / dt) / 1024 / 1024
        read_mbps = ((disk.read_bytes - self.last_disk.read_bytes) / dt) / 1024 / 1024
        write_mbps = ((disk.write_bytes - self.last_disk.write_bytes) / dt) / 1024 / 1024
        
        self.last_net = net
        self.last_disk = disk
        
        # GPU
        gpu_util = 0.0
        vram_mb = 0.0
        if self.nvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                vram_mb = meminfo.used / 1024 / 1024
                gpu_util = float(util.gpu)
            except: pass
            
        self.state = HardwareMetrics(
            cpu_usage=cpu_usage,
            cpu_cores_usage=cpu_cores,
            ram_usage_pct=ram_pct,
            ram_used_mb=ram_used_mb,
            gpu_util=gpu_util,
            vram_used_mb=vram_mb,
            disk_usage_pct=disk_pct,
            network_rx_mbps=rx_mbps,
            network_tx_mbps=tx_mbps,
            disk_read_mbps=read_mbps,
            disk_write_mbps=write_mbps,
            machine_identifier="Python Orchestrator"
        )

# Global singleton
hardware_monitor = HardwareMonitor()
