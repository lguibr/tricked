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
    pub elapsed_time: f64,
    pub network_tx_mbps: f64,
    pub network_rx_mbps: f64,
    pub disk_read_mbps: f64,
    pub disk_write_mbps: f64,
    pub policy_entropy: f32,
    pub gradient_norm: f32,
    pub representation_drift: f32,
    pub mean_td_error: f32,
    pub queue_saturation_ratio: f32,
    pub sps_vs_tps: f32,
    pub action_space_entropy: f32,
    pub layer_gradient_norms: String,
    pub spatial_heatmap: Vec<f32>,
    pub difficulty: f32,
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
    pub cpu_cores_usage: Option<Vec<f32>>,
    pub ram_usage_pct: f64,
    pub ram_used_mb: f64,
    pub gpu_util: f32,
    pub vram_used_mb: f32,
    pub disk_usage_pct: f64,
    pub network_rx_mbps: f64,
    pub network_tx_mbps: f64,
    pub disk_read_mbps: f64,
    pub disk_write_mbps: f64,
    pub machine_identifier: String,
}

#[derive(Serialize, Deserialize, Clone, TS)]
#[ts(export, export_to = "../../../control_center/src/bindings/")]
pub struct MetricRow {
    pub step: i32,
    pub total_loss: Option<f64>,
    pub policy_loss: Option<f64>,
    pub value_loss: Option<f64>,
    pub reward_loss: Option<f64>,
    pub lr: Option<f64>,
    pub game_score_min: Option<f64>,
    pub game_score_max: Option<f64>,
    pub game_score_med: Option<f64>,
    pub game_score_mean: Option<f64>,
    pub win_rate: Option<f64>,
    pub game_lines_cleared: Option<i32>,
    pub game_count: Option<i32>,
    pub ram_usage_mb: Option<f64>,
    pub gpu_usage_pct: Option<f64>,
    pub cpu_usage_pct: Option<f64>,
    pub disk_usage_pct: Option<f64>,
    pub vram_usage_mb: Option<f64>,
    pub mcts_depth_mean: Option<f64>,
    pub mcts_search_time_mean: Option<f64>,
    pub elapsed_time: Option<f64>,
    pub network_tx_mbps: Option<f64>,
    pub network_rx_mbps: Option<f64>,
    pub disk_read_mbps: Option<f64>,
    pub disk_write_mbps: Option<f64>,
    pub policy_entropy: Option<f64>,
    pub gradient_norm: Option<f64>,
    pub representation_drift: Option<f64>,
    pub mean_td_error: Option<f64>,
    pub queue_saturation_ratio: Option<f64>,
    pub sps_vs_tps: Option<f64>,
    pub action_space_entropy: Option<f64>,
    pub layer_gradient_norms: Option<String>,
    pub spatial_heatmap: Option<Vec<f64>>,
    pub difficulty: Option<f64>,
}

#[derive(Clone, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../../control_center/src/bindings/")]
pub struct LogEvent {
    pub run_id: String,
    pub line: String,
}
