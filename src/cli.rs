use crate::config::Config;
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    Train {
        /// The name of the experiment for logging and paths
        #[arg(long, default_value = "cli_run")]
        experiment_name: String,

        /// Path to the Tricked AI centralized SQLite workspace database
        #[arg(long)]
        workspace_db: Option<String>,

        /// The Run ID of the current training session (for telemetry and config loading)
        #[arg(long)]
        run_id: Option<String>,

        /// Overrides for hyperparameters
        #[arg(long)]
        lr_init: Option<f64>,
        #[arg(long)]
        simulations: Option<i64>,
        #[arg(long)]
        unroll_steps: Option<usize>,
        #[arg(long)]
        temporal_difference_steps: Option<usize>,
        #[arg(long)]
        reanalyze_ratio: Option<f32>,
        #[arg(long)]
        support_size: Option<i64>,
        #[arg(long)]
        temp_decay_steps: Option<i64>,

        /// Max training steps to run before exiting (0 = infinite)
        #[arg(long, default_value = "0")]
        max_steps: usize,
    },
    Tune {
        /// Path to the base configuration JSON
        #[arg(long)]
        config: String,

        /// Number of trials to run
        #[arg(long, default_value = "50")]
        trials: usize,

        /// Number of training steps per trial
        #[arg(long, default_value = "50")]
        max_steps: usize,

        /// Timeout in seconds before pruning
        #[arg(long, default_value = "1800")]
        timeout: u64,

        #[arg(long, default_value = "10")]
        resnet_blocks: usize,

        #[arg(long, default_value = "256")]
        resnet_channels: usize,

        /// JSON-encoded bounds for hyperparameter tuning
        #[arg(long, default_value = "{}")]
        bounds: String,

        /// Workspace Database for logging
        #[arg(long)]
        workspace_db: Option<String>,
    },
}

pub struct TuneConfig {
    pub config_path: String,
    pub trials: usize,
    pub max_steps: usize,
    pub timeout: u64,
    pub resnet_blocks: usize,
    pub resnet_channels: usize,
    pub bounds: String,
    pub workspace_db: Option<String>,
}

pub enum ParsedCommand {
    Train(Box<Config>, usize),
    Tune(TuneConfig),
}

pub fn parse_and_build_config() -> ParsedCommand {
    let cli = Cli::parse();
    match cli.command {
        Commands::Train {
            experiment_name,
            workspace_db,
            run_id,
            lr_init,
            simulations,
            unroll_steps,
            temporal_difference_steps,
            reanalyze_ratio,
            support_size,
            temp_decay_steps,
            max_steps,
        } => {
            let mut cfg = if let (Some(db_path), Some(run_id_str)) = (&workspace_db, &run_id) {
                // Connect to SQLite to fetch the raw JSON config for this run_id
                let conn = rusqlite::Connection::open(db_path).unwrap_or_else(|e| {
                    panic!("Failed to open workspace DB at {}: {}", db_path, e);
                });

                let config_json: String = conn
                    .query_row(
                        "SELECT config FROM runs WHERE id = ?1",
                        rusqlite::params![run_id_str],
                        |row| row.get(0),
                    )
                    .unwrap_or_else(|e| {
                        panic!("Failed to find run_id {} in runs table: {}", run_id_str, e);
                    });

                let mut parsed: Config =
                    serde_json::from_str(&config_json).expect("Failed to parse config from SQLite");
                parsed.experiment_name_identifier = run_id_str.clone();

                let mut custom_base_dir = String::new();
                // optionally load artifacts_dir from db if exist
                if let Ok(Some(d)) = conn.query_row(
                    "SELECT artifacts_dir FROM runs WHERE id = ?1",
                    rusqlite::params![run_id_str],
                    |row| row.get::<_, Option<String>>(0),
                ) {
                    custom_base_dir = d;
                }

                if custom_base_dir.is_empty() {
                    custom_base_dir = format!("runs/{}", run_id_str);
                }

                parsed.paths = crate::config::ExperimentPaths {
                    base_directory: custom_base_dir.clone(),
                    model_checkpoint_path: format!("{}/weights.safetensors", custom_base_dir),
                    metrics_file_path: format!("{}/metrics.csv", custom_base_dir),
                    experiment_name_identifier: run_id_str.clone(),
                    workspace_db: Some(db_path.clone()),
                };
                parsed
            } else {
                Config {
                    experiment_name_identifier: experiment_name.clone(),
                    paths: crate::config::ExperimentPaths::new(&experiment_name),
                    device: "cuda".to_string(),
                    hidden_dimension_size: 256,
                    num_blocks: 10,
                    value_support_size: 300,
                    reward_support_size: 300,
                    spatial_channel_count: 20,
                    hole_predictor_dim: 64,
                    buffer_capacity_limit: 1_000_000,
                    simulations: 200,
                    train_batch_size: 256,
                    discount_factor: 0.99,
                    td_lambda: 0.95,
                    weight_decay: 0.0,
                    checkpoint_interval: 100,
                    num_processes: 4,
                    worker_device: "cpu".to_string(),
                    unroll_steps: 15,
                    temporal_difference_steps: 15,
                    inference_batch_size_limit: 64,
                    inference_timeout_ms: 5,
                    max_gumbel_k: 16,
                    gumbel_scale: 1.0,
                    temp_decay_steps: 10000,
                    difficulty: 6,
                    temp_boost: true,
                    lr_init: 0.0003,
                    reanalyze_ratio: 0.0,
                }
            };

            if let Some(v) = lr_init {
                cfg.lr_init = v;
            }
            if let Some(v) = simulations {
                cfg.simulations = v;
            }
            if let Some(v) = unroll_steps {
                cfg.unroll_steps = v;
            }
            if let Some(v) = temporal_difference_steps {
                cfg.temporal_difference_steps = v;
            }
            if let Some(v) = reanalyze_ratio {
                cfg.reanalyze_ratio = v;
            }
            if let Some(v) = support_size {
                cfg.value_support_size = v;
                cfg.reward_support_size = v;
            }
            if let Some(v) = temp_decay_steps {
                cfg.temp_decay_steps = v;
            }

            ParsedCommand::Train(Box::new(cfg), max_steps)
        }
        Commands::Tune {
            config,
            trials,
            max_steps,
            timeout,
            resnet_blocks,
            resnet_channels,
            bounds,
            workspace_db,
        } => ParsedCommand::Tune(TuneConfig {
            config_path: config,
            trials,
            max_steps,
            timeout,
            resnet_blocks,
            resnet_channels,
            bounds,
            workspace_db,
        }),
    }
}
