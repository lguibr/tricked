use rusqlite::Connection;
fn main() {
    let conn = Connection::open("control_center/src-tauri/tricked_workspace.db").unwrap();
    let mut stmt = conn.prepare("SELECT step, total_loss, policy_loss, value_loss, reward_loss, lr, game_score_min, game_score_max, game_score_med, game_score_mean, win_rate, game_lines_cleared, game_count, ram_usage_mb, gpu_usage_pct, cpu_usage_pct, disk_usage_pct, vram_usage_mb, mcts_depth_mean, mcts_search_time_mean, elapsed_time, network_tx_mbps, network_rx_mbps, disk_read_mbps, disk_write_mbps, policy_entropy, gradient_norm, representation_drift, mean_td_error, queue_saturation_ratio, sps_vs_tps, action_space_entropy, layer_gradient_norms, spatial_heatmap, difficulty FROM metrics ORDER BY step ASC").unwrap();
    let rows = stmt.query_map([], |row| {
        Ok(format!("step: {}", row.get::<_, i32>(0)?))
    }).unwrap();
    for (i, r) in rows.enumerate() {
        match r {
            Ok(s) => println!("Row {}: {}", i, s),
            Err(e) => println!("Row {} ERR: {:?}", i, e),
        }
    }
}
