import os
import time
import json
import redis
import psycopg2
from psycopg2.extras import execute_values
from tricked.proto_out.tricked_pb2 import MetricRow

REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379")
DB_URL = os.environ.get("DB_URL", "postgresql://tricked_user:tricked_password@localhost:5432/tricked_workspace")

def main():
    r = redis.Redis.from_url(REDIS_URL)
    pubsub = r.pubsub(ignore_subscribe_messages=True)
    pubsub.psubscribe('telemetry:metrics:*')

    batch = []
    last_flush = time.time()

    print(f"db_writer connected to REDIS_URL={REDIS_URL} DB_URL={DB_URL}. Listening on telemetry:metrics:* ...")

    while True:
        try:
            message = pubsub.get_message(timeout=1.0)
            
            if message and message['type'] == 'pmessage':
                data_bytes = message['data']
                row = MetricRow()
                try:
                    row.ParseFromString(data_bytes)
                    batch.append(row)
                except Exception as e:
                    print(f"Error parsing protobuf: {e}")

            if len(batch) >= 100 or (time.time() - last_flush > 5 and len(batch) > 0):
                try:
                    conn = psycopg2.connect(DB_URL)
                    conn.autocommit = True
                    cur = conn.cursor()

                    insert_query = """
                        INSERT INTO metrics (
                            run_id, step, total_loss, policy_loss, value_loss, reward_loss,
                            lr, game_score_min, game_score_max, game_score_med, game_score_mean,
                            win_rate, game_lines_cleared, game_count, ram_usage_mb, gpu_usage_pct,
                            cpu_usage_pct, disk_usage_pct, vram_usage_mb, mcts_depth_mean,
                            mcts_search_time_mean, elapsed_time, network_tx_mbps, network_rx_mbps,
                            disk_read_mbps, disk_write_mbps, policy_entropy, gradient_norm, 
                            representation_drift, mean_td_error, queue_saturation_ratio, 
                            sps_vs_tps, queue_latency_us, sumtree_contention_us, action_space_entropy, 
                            layer_gradient_norms, spatial_heatmap, difficulty
                        ) VALUES %s
                    """

                    rows_data = []
                    for r in batch:
                        sh = list(r.spatial_heatmap)
                        sh_json = json.dumps(sh)
                        
                        rows_data.append((
                            r.run_id, r.step, 
                            r.total_loss if r.HasField('total_loss') else None,
                            r.policy_loss if r.HasField('policy_loss') else None,
                            r.value_loss if r.HasField('value_loss') else None,
                            r.reward_loss if r.HasField('reward_loss') else None,
                            r.lr if r.HasField('lr') else None,
                            r.game_score_min if r.HasField('game_score_min') else None,
                            r.game_score_max if r.HasField('game_score_max') else None,
                            r.game_score_med if r.HasField('game_score_med') else None,
                            r.game_score_mean if r.HasField('game_score_mean') else None,
                            r.win_rate if r.HasField('win_rate') else None,
                            r.game_lines_cleared if r.HasField('game_lines_cleared') else None,
                            r.game_count if r.HasField('game_count') else None,
                            r.ram_usage_mb if r.HasField('ram_usage_mb') else None,
                            r.gpu_usage_pct if r.HasField('gpu_usage_pct') else None,
                            r.cpu_usage_pct if r.HasField('cpu_usage_pct') else None,
                            r.disk_usage_pct if r.HasField('disk_usage_pct') else None,
                            r.vram_usage_mb if r.HasField('vram_usage_mb') else None,
                            r.mcts_depth_mean if r.HasField('mcts_depth_mean') else None,
                            r.mcts_search_time_mean if r.HasField('mcts_search_time_mean') else None,
                            r.elapsed_time if r.HasField('elapsed_time') else None,
                            r.network_tx_mbps if r.HasField('network_tx_mbps') else None,
                            r.network_rx_mbps if r.HasField('network_rx_mbps') else None,
                            r.disk_read_mbps if r.HasField('disk_read_mbps') else None,
                            r.disk_write_mbps if r.HasField('disk_write_mbps') else None,
                            r.policy_entropy if r.HasField('policy_entropy') else None,
                            r.gradient_norm if r.HasField('gradient_norm') else None,
                            r.representation_drift if r.HasField('representation_drift') else None,
                            r.mean_td_error if r.HasField('mean_td_error') else None,
                            r.queue_saturation_ratio if r.HasField('queue_saturation_ratio') else None,
                            r.sps_vs_tps if r.HasField('sps_vs_tps') else None,
                            r.queue_latency_us if r.HasField('queue_latency_us') else None,
                            r.sumtree_contention_us if r.HasField('sumtree_contention_us') else None,
                            r.action_space_entropy if r.HasField('action_space_entropy') else None,
                            r.layer_gradient_norms if r.HasField('layer_gradient_norms') else None,
                            sh_json,
                            r.difficulty if r.HasField('difficulty') else None,
                        ))

                    execute_values(cur, insert_query, rows_data)
                    conn.close()

                except Exception as e:
                    print(f"Error executing db insertion: {e}")
                finally:
                    batch.clear()
                    last_flush = time.time()
                    
        except KeyboardInterrupt:
            print("Shutting down db_writer...")
            break
        except Exception as e:
            print(f"Redis polling error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
