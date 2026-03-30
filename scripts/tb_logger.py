import os

# Monkey-patch os.makedirs to fix python 3.13 + tensorboardX FileExistsError
_orig_makedirs = os.makedirs
os.makedirs = lambda name, mode=0o777, exist_ok=False: _orig_makedirs(
    name, mode, exist_ok=True
)

import json
import redis
import psutil

try:
    import pynvml

    pynvml.nvmlInit()
    has_nvml = True
except Exception:
    has_nvml = False

from tensorboardX import SummaryWriter

r = redis.Redis(host="localhost", port=6379, decode_responses=True)
print("🔗 Connected to Redis. TensorBoard Logger listening for metrics...")

current_exp = r.get("tricked_current_exp")
writer = None


def get_writer():
    global writer, current_exp
    if writer is None:
        if not current_exp:
            current_exp = "tricked_headless"
        writer = SummaryWriter(log_dir=f"runs/{current_exp}")
    return writer


pubsub = r.pubsub()
pubsub.subscribe(
    "tricked_training",
    "tricked_events",
    "tricked_games",
    "tricked_metrics",
    "tricked_config",
)

import time

train_step = 0
eval_step = 0
metric_step = 0
hardware_step = 0
start_time = time.time()
try:
    last_hardware_log = time.time()
    last_disk = psutil.disk_io_counters()
    last_net = psutil.net_io_counters()

    while True:
        message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
        if message and message["type"] == "message":
            channel = message["channel"]
            try:
                data = json.loads(message["data"])
            except:
                continue

            if channel == "tricked_config":
                if "experiment_name_identifier" in data:
                    new_exp = data["experiment_name_identifier"]
                    if new_exp != current_exp:
                        current_exp = new_exp
                        print(f"🔄 Switching TensorBoard log dir to runs/{current_exp}")
                        if writer is not None:
                            writer.close()
                        writer = SummaryWriter(log_dir=f"runs/{current_exp}")
                        train_step = 0
                        eval_step = 0
                        metric_step = 0
                        hardware_step = 0

                config_str = json.dumps(data, indent=2)
                get_writer().add_text(
                    "Metadata/Config", f"```json\n{config_str}\n```", train_step
                )

            elif channel == "tricked_training":
                if "loss" in data:
                    get_writer().add_scalar("train/loss", data.get("loss"), train_step)
                if "policy_loss" in data:
                    get_writer().add_scalar(
                        "train/policy_loss", data.get("policy_loss"), train_step
                    )
                if "value_loss" in data:
                    get_writer().add_scalar(
                        "train/value_loss", data.get("value_loss"), train_step
                    )
                if "reward_loss" in data:
                    get_writer().add_scalar(
                        "train/reward_loss", data.get("reward_loss"), train_step
                    )
                train_step += 1
            elif channel == "tricked_games":
                if "score" in data:
                    get_writer().add_scalar("eval/score", data.get("score"), eval_step)
                if "steps" in data:
                    get_writer().add_scalar("eval/steps", data.get("steps"), eval_step)
                eval_step += 1
            elif channel == "tricked_metrics":
                if "value" in data and "name" in data:
                    metric_name = data.get("name")
                    if "/" not in metric_name:
                        metric_name = f"metrics/{metric_name}"
                    get_writer().add_scalar(metric_name, data.get("value"), metric_step)
                    metric_step += 1

        now = time.time()
        if now - last_hardware_log >= 5.0 and writer is not None:
            try:
                dt = now - last_hardware_log
                last_hardware_log = now

                get_writer().add_scalar(
                    "hardware/cpu_utilization", psutil.cpu_percent(), hardware_step
                )
                get_writer().add_scalar(
                    "hardware/ram_usage_gb",
                    psutil.virtual_memory().used / (1024**3),
                    hardware_step,
                )

                curr_disk = psutil.disk_io_counters()
                if curr_disk and last_disk:
                    disk_read_mbps = (
                        (curr_disk.read_bytes - last_disk.read_bytes) / dt / (1024**2)
                    )
                    disk_write_mbps = (
                        (curr_disk.write_bytes - last_disk.write_bytes) / dt / (1024**2)
                    )
                    get_writer().add_scalar(
                        "hardware/disk_read_mbps", disk_read_mbps, hardware_step
                    )
                    get_writer().add_scalar(
                        "hardware/disk_write_mbps", disk_write_mbps, hardware_step
                    )
                last_disk = curr_disk

                curr_net = psutil.net_io_counters()
                if curr_net and last_net:
                    net_recv_mbps = (
                        (curr_net.bytes_recv - last_net.bytes_recv) / dt / (1024**2)
                    )
                    net_sent_mbps = (
                        (curr_net.bytes_sent - last_net.bytes_sent) / dt / (1024**2)
                    )
                    get_writer().add_scalar(
                        "hardware/net_recv_mbps", net_recv_mbps, hardware_step
                    )
                    get_writer().add_scalar(
                        "hardware/net_sent_mbps", net_sent_mbps, hardware_step
                    )
                last_net = curr_net

                if has_nvml:
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        temp = pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )

                        get_writer().add_scalar(
                            "hardware/gpu_utilization", gpu_util, hardware_step
                        )
                        get_writer().add_scalar(
                            "hardware/vram_usage_gb",
                            mem_info.used / (1024**3),
                            hardware_step,
                        )
                        get_writer().add_scalar(
                            "hardware/gpu_temperature", temp, hardware_step
                        )

                        try:
                            tx_kbs = pynvml.nvmlDeviceGetPcieThroughput(handle, 0)
                            rx_kbs = pynvml.nvmlDeviceGetPcieThroughput(handle, 1)
                            get_writer().add_scalar(
                                "hardware/pcie_tx_mbps", tx_kbs / 1024.0, hardware_step
                            )
                            get_writer().add_scalar(
                                "hardware/pcie_rx_mbps", rx_kbs / 1024.0, hardware_step
                            )
                        except Exception:
                            pass
                    except Exception:
                        pass
                hardware_step += 1
                get_writer().flush()
            except Exception as e:
                print(f"🔥 CRITICAL HARDWARE LOGGING ERROR: {e}", flush=True)
                import traceback

                traceback.print_exc()
except KeyboardInterrupt:
    print("🛑 Shutting down TensorBoard logger")
    if writer is not None:
        writer.close()
