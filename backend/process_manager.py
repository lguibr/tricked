import os
import signal
import subprocess
import threading
import time
import sqlite3
import psutil
import collections
import queue

class ProcessManager:
    def __init__(self, db_path: str, project_root: str):
        self.db_path = db_path
        self.project_root = project_root
        
        # Ensure database tables exist
        try:
            import sqlite3
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = sqlite3.connect(self.db_path, timeout=5)
            conn.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, name TEXT, type TEXT, status TEXT, config TEXT, start_time DATETIME DEFAULT CURRENT_TIMESTAMP, tags TEXT, artifacts_dir TEXT)')
            conn.execute('CREATE TABLE IF NOT EXISTS metrics (step INTEGER, total_loss REAL, policy_loss REAL, value_loss REAL, reward_loss REAL, lr REAL, game_score_min INTEGER, game_score_max INTEGER, game_score_med INTEGER, game_score_mean REAL, win_rate REAL, game_lines_cleared REAL, game_count INTEGER, ram_usage_mb REAL, gpu_usage_pct REAL, cpu_usage_pct REAL, disk_usage_pct REAL, vram_usage_mb REAL, mcts_depth_mean REAL, mcts_search_time_mean REAL, elapsed_time REAL, network_tx_mbps REAL, network_rx_mbps REAL, disk_read_mbps REAL, disk_write_mbps REAL, policy_entropy REAL, gradient_norm REAL, representation_drift REAL, mean_td_error REAL, queue_saturation_ratio REAL, sps_vs_tps REAL, queue_latency_us REAL, sumtree_contention_us REAL, action_space_entropy REAL, layer_gradient_norms TEXT, spatial_heatmap TEXT, difficulty INTEGER, run_id TEXT)')
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error initializing database: {e}")
            
        self.active_run = None # {"run_id": str, "pid": int, "type": "STUDY" | "TRAIN"}
        self.log_buffer = collections.deque(maxlen=500)
        self.log_subscribers = [] # List of asyncio.Queue
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._reconciler_thread = threading.Thread(target=self._reconcile_loop, daemon=True)
        
    def _tail_subprocess(self, proc, log_file_path):
        with open(log_file_path, "a", buffering=1) as f:
            for line in iter(proc.stdout.readline, b''):
                decoded = line.decode('utf-8', errors='replace').strip()
                if decoded:
                    f.write(decoded + "\n")
                    f.flush()
                    self.log_buffer.append(decoded)
                    # Push to all active websocket subscribers
                    for sub in list(self.log_subscribers):
                        try:
                            sub.put_nowait([decoded])
                        except Exception:
                            pass
        
    def start(self):
        self._reconciler_thread.start()
        
    def stop(self):
        self._stop_event.set()
        if self.active_run:
            self.stop_run(self.active_run["run_id"])
            
    def start_run(self, run_id: str, is_study: bool = False):
        with self._lock:
            # Check if something is actually running
            if self.active_run is not None:
                if psutil.pid_exists(self.active_run["pid"]):
                    raise ValueError(f"Run {self.active_run['run_id']} is already active!")

            import sys
            python_path = sys.executable
            
            cmd = [python_path, "backend/main.py"]
            
            # Open log file for stdout/stderr capturing
            run_dir = os.path.join(self.project_root, "backend", "workspace", "runs", run_id)
            os.makedirs(run_dir, exist_ok=True)
            log_file_path = os.path.join(run_dir, "output.log")
            
            # Fetch config from DB and save to disk
            try:
                conn = sqlite3.connect(self.db_path, timeout=5)
                cur = conn.cursor()
                cur.execute("SELECT config FROM runs WHERE id = ?", (run_id,))
                row = cur.fetchone()
                conn.close()
                
                if row is None or not row[0]:
                    raise ValueError(f"Run {run_id} does not exist in local DB or has no config.")
                    
                import json
                raw_config = json.loads(row[0])
                    
                # If config is already nested, this just copies or updates it. But if it's flat, we restructure it safely.
                HARDWARE_KEYS = {"device", "num_processes", "worker_device", "inference_batch_size_limit", "inference_timeout_ms"}
                ARCH_KEYS = {"hidden_dimension_size", "num_blocks", "value_support_size", "reward_support_size", "spatial_channel_count", "hole_predictor_dim"}
                OPT_KEYS = {"buffer_capacity_limit", "train_batch_size", "discount_factor", "td_lambda", "weight_decay", "lr_init", "unroll_steps", "temporal_difference_steps", "reanalyze_ratio", "max_steps"}
                MCTS_KEYS = {"simulations", "max_gumbel_k", "gumbel_scale"}
                ENV_KEYS = {"difficulty", "temp_decay_steps", "temp_boost"}

                nested_config = {
                    "hardware": raw_config.get("hardware", {}),
                    "architecture": raw_config.get("architecture", {}),
                    "optimizer": raw_config.get("optimizer", {}),
                    "mcts": raw_config.get("mcts", {}),
                    "environment": raw_config.get("environment", {}),
                    "checkpoint_interval": raw_config.get("checkpoint_interval", 10),
                }

                for k, v in raw_config.items():
                    if k in ["hardware", "architecture", "optimizer", "mcts", "environment", "checkpoint_interval"]:
                        continue
                    if k in HARDWARE_KEYS:
                        nested_config["hardware"][k] = v
                    elif k in ARCH_KEYS:
                        nested_config["architecture"][k] = v
                    elif k in OPT_KEYS:
                        nested_config["optimizer"][k] = v
                    elif k in MCTS_KEYS:
                        nested_config["mcts"][k] = v
                    elif k in ENV_KEYS:
                        nested_config["environment"][k] = v
                    else:
                        nested_config[k] = v
                        
                with open(os.path.join(run_dir, "base_config.json"), "w") as f:
                    json.dump(nested_config, f, indent=2)
            except Exception as e:
                import traceback
                print(f"Error saving config to disk for run {run_id}: {e}")
                traceback.print_exc()
                raise e

            if is_study:
                cmd.extend(["tune", "--id", run_id, "--db", self.db_path])
            else:
                config_path = os.path.join(run_dir, "base_config.json")
                cmd.extend(["train", "--config", config_path, "--id", run_id, "--db", self.db_path])
                
            env = os.environ.copy()
            env["PYTHONPATH"] = self.project_root
            ext_path = os.path.join(self.project_root, "backend", "extensions")
            env["LD_LIBRARY_PATH"] = ext_path + ":" + env.get("LD_LIBRARY_PATH", "")
            
            self.log_buffer.clear()
            
            # We open in 'a' mode so it appends, and unbuffered ideally, but line-buffered is fine
            # We don't write directly here, we tail it
            
            # Spawn OS Process natively decoupled from FASTAPI worker
            import ctypes
            import signal
            def set_pdeathsig():
                try:
                    libc = ctypes.CDLL("libc.so.6")
                    libc.prctl(1, signal.SIGTERM) # PR_SET_PDEATHSIG = 1
                except:
                    pass

            proc = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=set_pdeathsig
            )
            
            # Start daemon thread to tail logs without blocking
            threading.Thread(target=self._tail_subprocess, args=(proc, log_file_path), daemon=True).start()
            
            self.active_run = {
                "run_id": run_id,
                "pid": proc.pid,
                "proc": proc,
                "type": "STUDY" if is_study else "TRAIN"
            }
            
            # Update DB immediately to avoid Reconciler shooting it
            try:
                conn = sqlite3.connect(self.db_path, timeout=10)
                cur = conn.cursor()
                cur.execute("UPDATE runs SET status = 'RUNNING' WHERE id = ?", (run_id,))
                conn.commit()
            except Exception as e:
                print(f"Error marking RUNNING in DB: {e}")
            finally:
                conn.close()
                

    def stop_run(self, run_id: str, force: bool = False):
        with self._lock:
            if self.active_run and self.active_run["run_id"] == run_id:
                pid = self.active_run["pid"]
                if psutil.pid_exists(pid):
                    try:
                        parent = psutil.Process(pid)
                        # Kill all children first
                        for child in parent.children(recursive=True):
                            child.kill()
                        if force:
                            parent.kill()
                        else:
                            parent.terminate()
                    except psutil.NoSuchProcess:
                        pass
                self.active_run = None
                
            # Update DB
            try:
                conn = sqlite3.connect(self.db_path, timeout=10)
                cur = conn.cursor()
                # Status is STOPPED if intentionally stopped, FAILED if crashed
                cur.execute("UPDATE runs SET status = 'STOPPED' WHERE id = ?", (run_id,))
                conn.commit()
            except Exception as e:
                print(f"Error marking STOPPED in DB: {e}")
            finally:
                conn.close()

    def _reconcile_loop(self):
        while not self._stop_event.is_set():
            time.sleep(3)
            with self._lock:
                try:
                    conn = sqlite3.connect(self.db_path, timeout=5)
                    cur = conn.cursor()
                    cur.execute("SELECT id, status FROM runs WHERE status = 'RUNNING' OR status = 'STARTING'")
                    db_running = cur.fetchall()
                    
                    # 1. IF DB thinks something is running, but ACTIVE_RUN is none or dead -> MARK FAILED
                    for run_id, status in db_running:
                        is_active_here = (self.active_run is not None and self.active_run["run_id"] == run_id)
                        is_alive = False
                        if is_active_here:
                            # Reaps zombie natively and returns exit code if dead
                            if self.active_run["proc"].poll() is None:
                                is_alive = True
                        
                        if not is_alive:
                            print(f"[Reconciler] Healing DB: {run_id} is dead but marked {status}. Setting to FAILED.")
                            cur.execute("UPDATE runs SET status = 'FAILED' WHERE id = ?", (run_id,))
                            conn.commit()
                            if is_active_here:
                                self.active_run = None

                    # 2. IF ACTIVE_RUN is alive, but DB says it's STOPPED/FAILED -> KILL ZOMBIE
                    if self.active_run is not None:
                        pid = self.active_run["pid"]
                        r_id = self.active_run["run_id"]
                        if self.active_run["proc"].poll() is None:
                            cur.execute("SELECT status FROM runs WHERE id = ?", (r_id,))
                            row = cur.fetchone()
                            if row is None or row[0] not in ('RUNNING', 'STARTING'):
                                print(f"[Reconciler] Found Zombie PID {pid} for run {r_id} (DB status {row[0] if row else 'NONE'}). Executing SIGKILL.")
                                try:
                                    parent = psutil.Process(pid)
                                    for child in parent.children(recursive=True):
                                        child.kill()
                                    parent.kill()
                                except:
                                    pass
                                self.active_run = None
                        else:
                            self.active_run = None
                            
                except Exception as e:
                    print(f"Reconciler error: {e}")
                finally:
                    conn.close()
