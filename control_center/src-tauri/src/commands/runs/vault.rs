use crate::db;

#[derive(serde::Serialize)]
pub struct FrontendGameStep {
    pub board_low: String,
    pub board_high: String,
    pub available: Vec<i32>,
    pub action_taken: i64,
    pub piece_identifier: i64,
}

#[derive(serde::Serialize)]
pub struct FrontendVaultGame {
    pub source_run_id: String,
    pub source_run_name: String,
    pub run_type: String,
    pub difficulty_setting: i32,
    pub episode_score: f32,
    pub steps: Vec<FrontendGameStep>,
    pub lines_cleared: u32,
    pub mcts_depth_mean: f32,
    pub mcts_search_time_mean: f32,
}

#[tauri::command]
pub fn get_vault_games() -> Result<Vec<FrontendVaultGame>, String> {
    let conn = db::init_db();
    let runs = db::list_runs(&conn).map_err(|e| e.to_string())?;

    let db_path = db::get_db_path();
    let root = db_path.parent().unwrap();

    let mut all_games = Vec::new();

    for run in runs {
        let vault_file = root.join("runs").join(&run.id).join("vault.bincode");
        if !vault_file.exists() {
            continue;
        }

        if let Ok(file) = std::fs::File::open(vault_file) {
            let reader = std::io::BufReader::new(file);
            if let Ok(games) = bincode::deserialize_from::<
                _,
                Vec<tricked_engine::train::buffer::OwnedGameData>,
            >(reader)
            {
                for g in games {
                    let mut frontend_steps = Vec::with_capacity(g.steps.len());
                    for step in g.steps {
                        frontend_steps.push(FrontendGameStep {
                            board_low: step.board_state[0].to_string(),
                            board_high: step.board_state[1].to_string(),
                            available: step.available_pieces.to_vec(),
                            action_taken: step.action_taken,
                            piece_identifier: step.piece_identifier,
                        });
                    }

                    all_games.push(FrontendVaultGame {
                        source_run_id: run.id.clone(),
                        source_run_name: run.name.clone(),
                        run_type: run.r#type.clone(),
                        difficulty_setting: g.difficulty_setting,
                        episode_score: g.episode_score,
                        steps: frontend_steps,
                        lines_cleared: g.lines_cleared,
                        mcts_depth_mean: g.mcts_depth_mean,
                        mcts_search_time_mean: g.mcts_search_time_mean,
                    });
                }
            }
        }
    }

    all_games.sort_by(|a, b| {
        b.episode_score
            .partial_cmp(&a.episode_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    all_games.truncate(100);

    Ok(all_games)
}
