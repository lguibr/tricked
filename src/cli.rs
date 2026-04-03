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

        /// Optional path to a JSON/YAML config file
        #[arg(short, long)]
        config: Option<String>,

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
}

pub fn parse_and_build_config() -> (Config, usize) {
    let cli = Cli::parse();
    let Commands::Train {
        experiment_name,
        config,
        lr_init,
        simulations,
        unroll_steps,
        temporal_difference_steps,
        reanalyze_ratio,
        support_size,
        temp_decay_steps,
        max_steps,
    } = cli.command;

    let mut cfg = if let Some(path) = config {
        let file = std::fs::File::open(&path).expect("Failed to open config file");
        let mut parsed: Config = if path.ends_with(".yaml") || path.ends_with(".yml") {
            serde_yaml::from_reader(file).expect("Failed to parse YAML config")
        } else {
            serde_json::from_reader(file).expect("Failed to parse JSON config")
        };
        parsed.experiment_name_identifier = experiment_name.clone();

        let custom_base_dir = std::path::Path::new(&path)
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| format!("runs/{}", experiment_name));

        parsed.paths = crate::config::ExperimentPaths {
            base_directory: custom_base_dir.clone(),
            model_checkpoint_path: format!(
                "{}/{}_weights.safetensors",
                custom_base_dir, experiment_name
            ),
            metrics_file_path: format!("{}/{}_metrics.csv", custom_base_dir, experiment_name),
            experiment_name_identifier: experiment_name.clone(),
        };
        parsed
    } else {
        Config {
            experiment_name_identifier: experiment_name.clone(),
            paths: crate::config::ExperimentPaths::new(&experiment_name),
            device: "cuda".to_string(),
            hidden_dimension_size: 256,
            num_blocks: 10,
            support_size: 300,
            buffer_capacity_limit: 1_000_000,
            simulations: 200,
            train_batch_size: 256,
            train_epochs: 1000,
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
        cfg.support_size = v;
    }
    if let Some(v) = temp_decay_steps {
        cfg.temp_decay_steps = v;
    }

    (cfg, max_steps)
}
