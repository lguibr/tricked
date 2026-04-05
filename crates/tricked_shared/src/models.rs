use serde::{Deserialize, Serialize};
use ts_rs::TS;

#[derive(Clone, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../../control_center/src/bindings/")]
pub struct TelemetryData {
    pub run_id: String,
    pub step: usize,
    pub total_loss: f32,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub reward_loss: f32,
    pub lr: f64,
    pub game_score_min: f32,
    pub game_score_max: f32,
    pub game_score_med: f32,
    pub game_score_mean: f32,
    pub winrate_mean: f32,
    pub game_lines_cleared: u32,
    pub game_count: usize,
    pub ram_usage_mb: f32,
    pub gpu_usage_pct: f32,
    pub cpu_usage_pct: f32,
    pub io_usage: f32,
    pub disk_usage_pct: f64,
    pub vram_usage_mb: f32,
    pub mcts_depth_mean: f32,
    pub mcts_search_time_mean: f32,
    pub network_tx_mbps: f64,
    pub network_rx_mbps: f64,
    pub disk_read_mbps: f64,
    pub disk_write_mbps: f64,
}

#[derive(Clone, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../../control_center/src/bindings/")]
pub struct Run {
    pub id: String,
    pub name: String,
    pub r#type: String,
    pub status: String,
    pub config: String,
    pub start_time: String,
    pub tag: Option<String>,
}

#[derive(Clone, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../../control_center/src/bindings/")]
pub struct ProcessInfo {
    pub pid: u32,
    pub name: String,
    pub status: String,
    pub cpu_usage: f32,
    pub memory_mb: f64,
    pub cmd: Vec<String>,
    pub children: Vec<ProcessInfo>,
}

#[derive(Clone, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../../control_center/src/bindings/")]
pub struct ActiveJob {
    pub id: String,
    pub name: String,
    pub job_type: String,
    pub root_process: Option<ProcessInfo>,
}

#[derive(Clone, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../../control_center/src/bindings/")]
pub struct HardwareMetrics {
    pub cpu_usage: f32,
    pub ram_usage_pct: f64,
    pub ram_used_mb: f64,
    pub gpu_util: f32,
    pub vram_used_mb: f32,
    pub disk_usage_pct: f64,
    pub network_rx_mbps: f64,
    pub network_tx_mbps: f64,
    pub disk_read_mbps: f64,
    pub disk_write_mbps: f64,
}
