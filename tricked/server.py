import psycopg2
import json
import os
import asyncio
import redis.asyncio as redis_async
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from .process_manager import ProcessManager
from .telemetry import hardware_monitor
from .job_monitor import JobMonitor

from contextlib import asynccontextmanager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_URL = os.environ.get("DB_URL", "postgresql://tricked_user:tricked_password@localhost:5432/tricked_workspace")
REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379")

# Fake DB_PATH property just to keep process manager happy if it still needs it
DB_PATH = os.path.join(PROJECT_ROOT, "backend", "workspace", "tricked_workspace.db")

pm = ProcessManager(DB_PATH, PROJECT_ROOT)
job_monitor = JobMonitor(pm)

@asynccontextmanager
async def lifespan(app: FastAPI):
    pm.start()
    asyncio.create_task(hardware_monitor.start())
    asyncio.create_task(job_monitor.start())
    yield
    pm.stop()
    hardware_monitor.stop()
    job_monitor.stop()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schemas ---
class StartRunReq(BaseModel):
    id: str

class StartStudyReq(BaseModel):
    id: str
    
class StopRunReq(BaseModel):
    id: str
    force: bool = False

class CreateRunReq(BaseModel):
    id: str
    name: str
    type: str
    config: str
    tags: List[str]

# --- Endpoints ---

from fastapi import Response

@app.get("/api/hardware")
def get_hardware():
    return Response(content=hardware_monitor.state.SerializeToString(), media_type="application/x-protobuf")

@app.websocket("/api/ws/hardware")
async def ws_hardware(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_bytes(hardware_monitor.state.SerializeToString())
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass

@app.get("/api/runs")
def list_runs():
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute("SELECT id, name, type, status, config, start_time, tags FROM runs ORDER BY start_time DESC")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        runs = []
        for r in rows:
            tags = []
            try:
                tags = json.loads(r[6]) if r[6] else []
            except: pass
            runs.append({
                "id": r[0],
                "name": r[1],
                "type": r[2],
                "status": r[3],
                "config": r[4],
                "start_time": r[5] or "",
                "tag": tags[0] if tags else None
            })
        return runs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/api/ws/runs")
async def ws_runs(websocket: WebSocket):
    await websocket.accept()
    try:
        last_hash = ""
        while True:
            runs = list_runs()
            # Simple diff check by serializing to JSON
            current_hash = json.dumps(runs)
            if current_hash != last_hash:
                await websocket.send_json(runs)
                last_hash = current_hash
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass

@app.websocket("/api/ws/jobs")
async def ws_jobs(websocket: WebSocket):
    await websocket.accept()
    try:
        last_hash = ""
        while True:
            jobs = job_monitor.latest_jobs
            current_hash = json.dumps(jobs)
            if current_hash != last_hash:
                await websocket.send_json(jobs)
                last_hash = current_hash
            await asyncio.sleep(1.5)
    except WebSocketDisconnect:
        pass

@app.get("/api/runs/{run_id}/checkpoints")
def list_checkpoints(run_id: str):
    run_dir = os.path.join(PROJECT_ROOT, "backend", "workspace", "runs", run_id)
    if not os.path.exists(run_dir):
        return []
    cps = [os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.startswith("checkpoint_") and f.endswith(".safetensors")]
    # Extract the step number and sort by it
    def extract_step(path):
        import re
        m = re.search(r"checkpoint_step_(\d+)\.safetensors", path)
        return int(m.group(1)) if m else -1
    return sorted(cps, key=extract_step)

@app.get("/api/vault/global")
def get_vault_games():
    try:
        import tricked_engine
        json_str = tricked_engine.get_global_vault_games()
        if json_str == "[]":
            return []
        import json
        return json.loads(json_str)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vault/flush")
def flush_global_vault():
    try:
        import tricked_engine
        tricked_engine.flush_global_vault()
        return {"status": "ok"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vault/empty")
def empty_all_vaults():
    try:
        import tricked_engine
        tricked_engine.empty_all_vaults()
        return {"status": "ok"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class RemoveVaultGameReq(BaseModel):
    score: float
    steps: int
    run_name: str

@app.post("/api/vault/remove")
def remove_vault_game(req: RemoveVaultGameReq):
    try:
        import tricked_engine
        tricked_engine.remove_vault_game(req.score, req.steps, req.run_name)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from tricked.proto_out.tricked_pb2 import TelemetryPayload, MetricRow, MetricHistory
from fastapi import Response

@app.websocket("/api/ws/runs/{run_id}/metrics")
async def ws_metrics(websocket: WebSocket, run_id: str):
    await websocket.accept()
    last_step = -1
    
    # 1. Fetch History from PostgreSQL
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        query = """SELECT step, total_loss, policy_loss, value_loss, reward_loss, lr, 
                   game_score_min, game_score_max, game_score_med, game_score_mean, win_rate, 
                   game_lines_cleared, game_count, ram_usage_mb, gpu_usage_pct, cpu_usage_pct, 
                   disk_usage_pct, vram_usage_mb, mcts_depth_mean, mcts_search_time_mean, 
                   elapsed_time, network_tx_mbps, network_rx_mbps, disk_read_mbps, disk_write_mbps, 
                   policy_entropy, gradient_norm, representation_drift, mean_td_error, 
                   queue_saturation_ratio, sps_vs_tps, queue_latency_us, sumtree_contention_us, 
                   action_space_entropy, layer_gradient_norms, spatial_heatmap, difficulty 
                   FROM metrics WHERE run_id = %s ORDER BY step ASC"""
        cur.execute(query, (run_id,))
        rows = cur.fetchall()
        
        if rows:
            payload = MetricHistory()
            for r in rows:
                row = MetricRow()
                row.step = r[0] if r[0] else 0
                row.total_loss = float(r[1] or 0.0)
                row.policy_loss = float(r[2] or 0.0)
                row.value_loss = float(r[3] or 0.0)
                row.reward_loss = float(r[4] or 0.0)
                row.lr = float(r[5] or 0.0)
                row.game_score_min = float(r[6] or 0.0)
                row.game_score_max = float(r[7] or 0.0)
                row.game_score_med = float(r[8] or 0.0)
                row.game_score_mean = float(r[9] or 0.0)
                row.win_rate = float(r[10] or 0.0)
                row.game_lines_cleared = float(r[11] or 0.0)
                row.game_count = float(r[12] or 0.0)
                row.ram_usage_mb = float(r[13] or 0.0)
                row.gpu_usage_pct = float(r[14] or 0.0)
                row.cpu_usage_pct = float(r[15] or 0.0)
                row.disk_usage_pct = float(r[16] or 0.0)
                row.vram_usage_mb = float(r[17] or 0.0)
                row.mcts_depth_mean = float(r[18] or 0.0)
                row.mcts_search_time_mean = float(r[19] or 0.0)
                row.elapsed_time = float(r[20] or 0.0)
                row.network_tx_mbps = float(r[21] or 0.0)
                row.network_rx_mbps = float(r[22] or 0.0)
                row.disk_read_mbps = float(r[23] or 0.0)
                row.disk_write_mbps = float(r[24] or 0.0)
                row.policy_entropy = float(r[25] or 0.0)
                row.gradient_norm = float(r[26] or 0.0)
                row.representation_drift = float(r[27] or 0.0)
                row.mean_td_error = float(r[28] or 0.0)
                row.queue_saturation_ratio = float(r[29] or 0.0)
                row.sps_vs_tps = float(r[30] or 0.0)
                row.queue_latency_us = float(r[31] or 0.0)
                row.sumtree_contention_us = float(r[32] or 0.0)
                row.action_space_entropy = float(r[33] or 0.0)
                row.difficulty = float(r[36] or 0.0)
                
                if r[34]:
                    try: row.layer_gradient_norms = r[34]
                    except: pass
                if r[35]:
                    try: row.spatial_heatmap.extend(json.loads(r[35]))
                    except: pass
                    
                last_step = max(last_step, row.step)
                payload.metrics.append(row)

            await websocket.send_bytes(payload.SerializeToString())
            
        cur.close()
        conn.close()
    except Exception as e:
        import traceback; traceback.print_exc()

    # 2. Redis PubSub for real-time
    r = redis_async.Redis.from_url(REDIS_URL)
    pubsub = r.pubsub(ignore_subscribe_messages=True)
    await pubsub.subscribe(f"telemetry:metrics:{run_id}")
    
    try:
        while True:
            # Listen to Redis updates
            if pm.active_run and pm.active_run["run_id"] == run_id:
                message = await pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    data_bytes = message['data']
                    row = MetricRow()
                    row.ParseFromString(data_bytes)
                    if row.step > last_step:
                        payload = MetricHistory()
                        payload.metrics.append(row)
                        await websocket.send_bytes(payload.SerializeToString())
                        last_step = row.step
            else:
                # Polling wait to not spin loop if inactive
                await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        import traceback; traceback.print_exc()
    finally:
        await pubsub.unsubscribe()
        await pubsub.close()

@app.websocket("/api/ws/runs/{run_id}/logs")
async def ws_logs(websocket: WebSocket, run_id: str):
    await websocket.accept()
    
    # Send the immediate backlog from RAM
    if pm.active_run and pm.active_run["run_id"] == run_id:
        if pm.log_buffer:
            await websocket.send_json(list(pm.log_buffer))
    else:
        # Read from file if it exists
        log_file_path = os.path.join(PROJECT_ROOT, "backend", "workspace", "runs", run_id, "output.log")
        if os.path.exists(log_file_path):
            try:
                with open(log_file_path, "r") as f:
                    lines = f.readlines()
                    if lines:
                        await websocket.send_json([line.strip() for line in lines[-500:]])
            except Exception:
                pass
            
    client_queue = asyncio.Queue()
    pm.log_subscribers.append(client_queue)
    
    try:
        while True:
            try:
                # Wait for new log lines streamed from the thread
                new_lines = await asyncio.wait_for(client_queue.get(), timeout=1.0)
                if pm.active_run and pm.active_run["run_id"] == run_id:
                    await websocket.send_json(new_lines)
            except asyncio.TimeoutError:
                # Timeout happened, but we are INSIDE the while loop now, 
                # so it simply loops again and keeps the connection alive!
                pass
    except WebSocketDisconnect:
        pass
    finally:
        if client_queue in pm.log_subscribers:
            pm.log_subscribers.remove(client_queue)

@app.post("/api/runs/start")
def start_run(req: StartRunReq):
    try:
        pm.start_run(req.id, is_study=False)
        return {"status": "ok"}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/studies/start")
def start_study(req: StartStudyReq):
    try:
        pm.start_run(req.id, is_study=True)
        return {"status": "ok"}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/runs/stop")
def stop_run(req: StopRunReq):
    pm.stop_run(req.id, force=req.force)
    return {"status": "ok"}

class RenameRunReq(BaseModel):
    id: str
    newName: str

class DeleteRunReq(BaseModel):
    id: str

class FlushRunReq(BaseModel):
    id: str

class CreateRunReqBasic(BaseModel):
    name: str
    type: str
    preset: str = ""

class SaveConfigReq(BaseModel):
    id: str
    config: str

@app.post("/api/runs/rename")
def rename_run(req: RenameRunReq):
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("UPDATE runs SET name=%s WHERE id=%s", (req.newName, req.id))
    conn.close()
    return {"status": "ok"}

@app.post("/api/runs/delete")
def delete_run(req: DeleteRunReq):
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("DELETE FROM runs WHERE id=%s", (req.id,))
    cur.execute("DELETE FROM metrics WHERE run_id=%s", (req.id,))
    conn.close()
    return {"status": "ok"}

@app.post("/api/runs/flush")
def flush_run(req: FlushRunReq):
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("DELETE FROM metrics WHERE run_id=%s", (req.id,))
    cur.execute("UPDATE runs SET status='STOPPED' WHERE id=%s", (req.id,))
    conn.close()
    return {"status": "ok"}

import uuid

@app.post("/api/runs/create")
def create_run(req: CreateRunReqBasic):
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    cur = conn.cursor()
    new_id = str(uuid.uuid4())
    cur.execute("INSERT INTO runs (id, name, type, status, config) VALUES (%s, %s, %s, 'CREATED', '')", 
                (new_id, req.name, req.type))
    conn.close()
    return {"id": new_id, "name": req.name, "type": req.type, "status": "CREATED", "config": "", "start_time": ""}

@app.post("/api/runs/save_config")
def save_config(req: SaveConfigReq):
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("UPDATE runs SET config=%s WHERE id=%s", (req.config, req.id))
    conn.close()
    return {"status": "ok"}

# --- Playground ---
import sys
import os
import torch
import tricked_engine

class StartPlaygroundReq(BaseModel):
    difficulty: int
    clutter: int
    
class ApplyMovePlaygroundReq(BaseModel):
    boardLow: str
    boardHigh: str
    available: List[int]
    score: int
    difficulty: int
    linesCleared: int
    slot: int
    pieceMaskLow: str
    pieceMaskHigh: str
    
@app.post("/api/playground/start")
def playground_start(req: StartPlaygroundReq):
    out = tricked_engine.playground_start_game(req.difficulty, req.clutter)
    return json.loads(out)
    
@app.post("/api/playground/apply_move")
def playground_apply(req: ApplyMovePlaygroundReq):
    out = tricked_engine.playground_apply_move(
        req.boardLow, req.boardHigh, req.available, 
        req.score, req.difficulty, req.linesCleared,
        req.slot, req.pieceMaskLow, req.pieceMaskHigh
    )
    if out is None:
        return None
    return json.loads(out)

class CommitToVaultReq(BaseModel):
    source_run_id: str
    source_run_name: str
    run_type: str
    difficulty: int
    episode_score: float
    steps: List[dict]
    lines_cleared: int
    mcts_depth_mean: float
    mcts_search_time_mean: float

@app.post("/api/playground/commit_to_vault")
def commit_to_vault(req: CommitToVaultReq):
    try:
        import json
        import tricked_engine
        
        # We need to map the frontend steps (which use string 'board_low' and 'available') 
        # to the native [u64,u64] and 'available_pieces', with zeros for neural targets
        mapped_steps = []
        for s in req.steps:
            mapped_steps.append({
                "board_state": [int(s.get("board_low") or "0"), int(s.get("board_high") or "0")],
                "available_pieces": s.get("available") or [-1, -1, -1],
                "action_taken": s.get("action_taken") or 0,
                "piece_identifier": s.get("piece_identifier") or 0,
                "value_prefix_received": s.get("value_prefix_received") or 0.0,
                "policy_target": s.get("policy_target") or [],
                "value_target": s.get("value_target") or 0.0,
                "is_terminal": s.get("is_terminal") or False
            })
            
        data_dict = req.model_dump()
        data_dict["steps"] = mapped_steps
        
        tricked_engine.commit_human_game(json.dumps(data_dict))
        return {"status": "ok"}
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class EvaluateBoardReq(BaseModel):
    boardLow: str
    boardHigh: str
    available: List[int]
    checkpointPath: str
    
@app.post("/api/evaluation/step")
def evaluation_step(req: EvaluateBoardReq):
    import torch
    import os
    target_path = req.checkpointPath

    if target_path.endswith(".safetensors"):
        pt_path = target_path.replace(".safetensors", ".pt")
        if not os.path.exists(pt_path):
            from tricked.models.muzero import MuZeroNet, InitialInferenceModel
            import json
            from safetensors.torch import load_file
            
            # Read network dimensions
            run_dir = os.path.dirname(req.checkpointPath)
            base_config_path = os.path.join(run_dir, "base_config.json")
            dim, blocks = 64, 4
            if os.path.exists(base_config_path):
                with open(base_config_path, "r") as f:
                    try:
                        cfg = json.load(f)
                        dim = cfg.get("architecture", {}).get("hidden_dimension_size", 64)
                        blocks = cfg.get("architecture", {}).get("num_blocks", 4)
                    except: pass
                    
            net = MuZeroNet(dim, blocks)
            state_dict = torch.load(target_path, map_location="cpu", weights_only=True)
            net.load_state_dict(state_dict)
            net.eval()
            
            scripted = torch.jit.script(InitialInferenceModel(net).eval())
            scripted.save(pt_path)
            
        target_path = pt_path

    out = tricked_engine.evaluate_board(
        req.boardLow, req.boardHigh, req.available, target_path
    )
    return json.loads(out)
