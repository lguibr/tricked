use tricked_engine::cli::{self, ParsedCommand};
use tricked_engine::train::{runner, tune};

#[hotpath::main]
fn main() {
    match cli::parse_and_build_config() {
        ParsedCommand::Train(cfg, max_steps) => {
            runner::run_training(cfg, max_steps);
        }
        ParsedCommand::Tune(tune_cfg) => {
            tune::run_tuning_pipeline(tune_cfg);
        }
    }
}
