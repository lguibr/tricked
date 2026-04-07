use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::thread;

use crate::train::buffer::core::OwnedGameData;
use crossbeam_channel::{unbounded, Receiver, Sender};

#[derive(Clone, Copy, Debug)]
pub struct Score(pub f32);

impl PartialEq for Score {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for Score {}

impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Score {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

struct VaultItem {
    score: Score,
    data: OwnedGameData,
}

impl PartialEq for VaultItem {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for VaultItem {}

impl PartialOrd for VaultItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for VaultItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.cmp(&other.score)
    }
}

pub struct VaultManager {
    pub sender: Sender<OwnedGameData>,
}

impl VaultManager {
    pub fn new(artifacts_dir: String) -> Self {
        let (tx, rx): (Sender<OwnedGameData>, Receiver<OwnedGameData>) = unbounded();

        let path = Path::new(&artifacts_dir).join("vault.bincode");
        let dir = Path::new(&artifacts_dir);
        if !dir.exists() {
            let _ = std::fs::create_dir_all(dir);
        }

        thread::Builder::new()
            .name("vault_manager".into())
            .spawn(move || {
                let mut heap: BinaryHeap<Reverse<VaultItem>> = BinaryHeap::new();
                let mut updates = 0;

                while let Ok(game) = rx.recv() {
                    let s = Score(game.episode_score);
                    heap.push(Reverse(VaultItem {
                        score: s,
                        data: game,
                    }));
                    if heap.len() > 100 {
                        heap.pop();
                    }

                    updates += 1;
                    // Sporadically save to disk to avoid I/O bottlenecks
                    if updates >= 5 {
                        updates = 0;
                        if let Ok(file) = File::create(&path) {
                            let writer = BufWriter::new(file);
                            let sorted_games: Vec<&OwnedGameData> =
                                heap.iter().map(|item| &item.0.data).collect();
                            let _ = bincode::serialize_into(writer, &sorted_games);
                        }
                    }
                }
            })
            .unwrap();

        Self { sender: tx }
    }
}

#[cfg(test)]
mod test_vault {
    use super::*;
    #[test]
    fn test_vault_manager_top_100_serialization() {
        let mut heap: BinaryHeap<Reverse<VaultItem>> = BinaryHeap::new();
        for i in 0..150 {
            heap.push(Reverse(VaultItem {
                score: Score(i as f32),
                data: OwnedGameData {
                    difficulty_setting: 0,
                    episode_score: i as f32,
                    steps: vec![],
                    lines_cleared: i as u32,
                    mcts_depth_mean: 0.0,
                    mcts_search_time_mean: 0.0,
                },
            }));
            if heap.len() > 100 {
                heap.pop();
            }
        }
        assert_eq!(heap.len(), 100);
        let min_score = heap.peek().unwrap().0.score.0;
        assert_eq!(
            min_score, 50.0,
            "Vault must evict lowest scores properly maintaining top 100!"
        );
    }
}
