use rusqlite::Connection;
use std::path::PathBuf;
use tricked_shared::models::{MetricRow, Run};

pub fn get_db_path() -> PathBuf {
    if let Ok(test_path) = std::env::var("TEST_DB") {
        return PathBuf::from(test_path);
    }
    let cwd = std::env::current_dir().unwrap();
    let root = if cwd.ends_with("src-tauri") {
        cwd.parent().unwrap().parent().unwrap().to_path_buf()
    } else if cwd.ends_with("control_center") {
        cwd.parent().unwrap().to_path_buf()
    } else {
        cwd
    };
    root.join("tricked_workspace.db")
}

pub fn init_db() -> Connection {
    let db_path = get_db_path();
    let conn = Connection::open(&db_path).expect("Failed to open database file");

    // We ignore errors here because during heavy writes the DB might be locked
    // and we don't want to panic the entire Tauri backend just for PRAGMA/CREATE TABLE.
    let _ = conn
        .execute_batch(
            "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         CREATE TABLE IF NOT EXISTS runs (
             id TEXT PRIMARY KEY,
             name TEXT NOT NULL,
             type TEXT NOT NULL,
             status TEXT NOT NULL,
             config JSON,
             tags JSON,
             artifacts_dir TEXT,
             start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
             end_time DATETIME
         );
         CREATE TABLE IF NOT EXISTS metrics (
             run_id TEXT NOT NULL,
             step INTEGER NOT NULL,
             total_loss REAL,
             policy_loss REAL,
             value_loss REAL,
             reward_loss REAL,
             lr REAL,
             game_score_min REAL,
             game_score_max REAL,
             game_score_med REAL,
             game_score_mean REAL,
             win_rate REAL,
             game_lines_cleared INTEGER,
             game_count INTEGER,
             ram_usage_mb REAL,
             gpu_usage_pct REAL,
             cpu_usage_pct REAL,
             disk_usage_pct REAL,
             vram_usage_mb REAL,
             mcts_depth_mean REAL,
             mcts_search_time_mean REAL,
             elapsed_time REAL,
             network_tx_mbps REAL DEFAULT 0.0,
             network_rx_mbps REAL DEFAULT 0.0,
             disk_read_mbps REAL DEFAULT 0.0,
             disk_write_mbps REAL DEFAULT 0.0,
             FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
         );",
        )
        .ok();

    // Auto-migrate legacy DBs lacking elapsed_time and IO
    let _ = conn.execute(
        "ALTER TABLE metrics ADD COLUMN elapsed_time REAL DEFAULT 0.0",
        [],
    );
    let _ = conn.execute(
        "ALTER TABLE metrics ADD COLUMN network_tx_mbps REAL DEFAULT 0.0",
        [],
    );
    let _ = conn.execute(
        "ALTER TABLE metrics ADD COLUMN network_rx_mbps REAL DEFAULT 0.0",
        [],
    );
    let _ = conn.execute(
        "ALTER TABLE metrics ADD COLUMN disk_read_mbps REAL DEFAULT 0.0",
        [],
    );
    let _ = conn.execute(
        "ALTER TABLE metrics ADD COLUMN disk_write_mbps REAL DEFAULT 0.0",
        [],
    );

    // Auto-migrate new metrics fields
    let _ = conn.execute(
        "ALTER TABLE metrics ADD COLUMN policy_entropy REAL DEFAULT 0.0",
        [],
    );
    let _ = conn.execute(
        "ALTER TABLE metrics ADD COLUMN gradient_norm REAL DEFAULT 0.0",
        [],
    );
    let _ = conn.execute(
        "ALTER TABLE metrics ADD COLUMN representation_drift REAL DEFAULT 0.0",
        [],
    );
    let _ = conn.execute(
        "ALTER TABLE metrics ADD COLUMN mean_td_error REAL DEFAULT 0.0",
        [],
    );
    let _ = conn.execute(
        "ALTER TABLE metrics ADD COLUMN queue_saturation_ratio REAL DEFAULT 0.0",
        [],
    );
    let _ = conn.execute(
        "ALTER TABLE metrics ADD COLUMN sps_vs_tps REAL DEFAULT 0.0",
        [],
    );
    let _ = conn.execute("ALTER TABLE metrics ADD COLUMN spatial_heatmap TEXT", []);

    conn
}

pub fn get_metrics(conn: &Connection, run_id: &str) -> rusqlite::Result<Vec<MetricRow>> {
    let mut stmt = conn.prepare("SELECT step, total_loss, policy_loss, value_loss, reward_loss, lr, game_score_min, game_score_max, game_score_med, game_score_mean, win_rate, game_lines_cleared, game_count, ram_usage_mb, gpu_usage_pct, cpu_usage_pct, disk_usage_pct, vram_usage_mb, mcts_depth_mean, mcts_search_time_mean, elapsed_time, network_tx_mbps, network_rx_mbps, disk_read_mbps, disk_write_mbps, policy_entropy, gradient_norm, representation_drift, mean_td_error, queue_saturation_ratio, sps_vs_tps, action_space_entropy, layer_gradient_norms, spatial_heatmap, difficulty FROM metrics WHERE run_id = ?1 ORDER BY step ASC")?;
    let rows = stmt.query_map(rusqlite::params![run_id], |row| {
        let spatial_heatmap_str: Option<String> = row.get(33).unwrap_or(None);
        let spatial_heatmap = if let Some(s) = spatial_heatmap_str {
            serde_json::from_str(&s).unwrap_or(None)
        } else {
            None
        };
        Ok(MetricRow {
            step: row.get(0)?,
            total_loss: row.get(1).unwrap_or(None),
            policy_loss: row.get(2).unwrap_or(None),
            value_loss: row.get(3).unwrap_or(None),
            reward_loss: row.get(4).unwrap_or(None),
            lr: row.get(5).unwrap_or(None),
            game_score_min: row.get(6).unwrap_or(None),
            game_score_max: row.get(7).unwrap_or(None),
            game_score_med: row.get(8).unwrap_or(None),
            game_score_mean: row.get(9).unwrap_or(None),
            win_rate: row.get(10).unwrap_or(None),
            game_lines_cleared: row.get(11).unwrap_or(None),
            game_count: row.get(12).unwrap_or(None),
            ram_usage_mb: row.get(13).unwrap_or(None),
            gpu_usage_pct: row.get(14).unwrap_or(None),
            cpu_usage_pct: row.get(15).unwrap_or(None),
            disk_usage_pct: row.get(16).unwrap_or(None),
            vram_usage_mb: row.get(17).unwrap_or(None),
            mcts_depth_mean: row.get(18).unwrap_or(None),
            mcts_search_time_mean: row.get(19).unwrap_or(None),
            elapsed_time: row.get(20).unwrap_or(None),
            network_tx_mbps: row.get(21).unwrap_or(None),
            network_rx_mbps: row.get(22).unwrap_or(None),
            disk_read_mbps: row.get(23).unwrap_or(None),
            disk_write_mbps: row.get(24).unwrap_or(None),
            policy_entropy: row.get(25).unwrap_or(None),
            gradient_norm: row.get(26).unwrap_or(None),
            representation_drift: row.get(27).unwrap_or(None),
            mean_td_error: row.get(28).unwrap_or(None),
            queue_saturation_ratio: row.get(29).unwrap_or(None),
            sps_vs_tps: row.get(30).unwrap_or(None),
            action_space_entropy: row.get(31).unwrap_or(None),
            layer_gradient_norms: row.get(32).unwrap_or(None),
            spatial_heatmap,
            difficulty: row.get(34).unwrap_or(None),
        })
    })?;

    let mut metrics = Vec::new();
    for r in rows {
        if let Ok(m) = r {
            metrics.push(m);
        }
    }
    Ok(metrics)
}

pub fn list_runs(conn: &Connection) -> rusqlite::Result<Vec<Run>> {
    let mut stmt = conn.prepare(
        "SELECT id, name, type, status, config, start_time, tags FROM runs ORDER BY start_time DESC",
    )?;
    let rows = stmt.query_map([], |row| {
        let tags_str: String = row.get(6).unwrap_or_else(|_| "[]".to_string());
        let tags: Vec<String> = serde_json::from_str(&tags_str).unwrap_or_else(|_| Vec::new());
        Ok(Run {
            id: row.get(0)?,
            name: row.get(1)?,
            r#type: row.get(2)?,
            status: row.get(3)?,
            config: row.get(4)?,
            start_time: row
                .get::<usize, Option<String>>(5)?
                .unwrap_or_else(|| "".to_string()),
            tag: tags.first().cloned(),
        })
    })?;

    let mut runs = Vec::new();
    for r in rows {
        if let Ok(run) = r {
            runs.push(run);
        }
    }
    Ok(runs)
}
