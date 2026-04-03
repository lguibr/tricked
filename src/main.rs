use tricked_engine::cli;
use tricked_engine::train::runner;

#[hotpath::main]
fn main() {
    let (cfg, max_steps) = cli::parse_and_build_config();
    runner::run_training(cfg, max_steps);
}
